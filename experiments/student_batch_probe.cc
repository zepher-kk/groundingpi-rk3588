// RKLLM native multi-batch probe for the FP16 student (language stage only).
//
// Two modes, same model / embeddings / sampling:
//   s : n_batch=1, one request at a time (baseline)
//   b : n_batch=2, two independent embedding requests in one rkllm_run
//
// API contract (see outputs/gpi-3s-distill/llm-batch/api-contract.md):
//   param.extend_param.n_batch = N at init
//   RKLLMInput in[N]; rkllm_run(handle, &in[0], ...)
//   callback receives RKLLMResult result[N]; result[slot].token_id < 0 => that slot finished
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "rkllm.h"

namespace {

constexpr int kDim = 2048;
constexpr int kMaxNew = 8;
constexpr int32_t kImEnd = 151645;

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct SlotOut {
    std::vector<int32_t> tokens;
    double first_ms = 0;
    bool done = false;
};

struct BatchCtx {
    int n = 1;
    double started = 0;
    std::vector<SlotOut> slots;
    bool error = false;
};

int on_result(RKLLMResult* result, void* userdata, LLMCallState state) {
    auto* ctx = static_cast<BatchCtx*>(userdata);
    if (!ctx) return 0;
    const double t = now_ms();
    if (state == RKLLM_RUN_ERROR) { ctx->error = true; return 0; }
    if (state == RKLLM_RUN_FINISH) {
        for (auto& s : ctx->slots) s.done = true;
        return 0;
    }
    if (state != RKLLM_RUN_NORMAL || result == nullptr) return 0;
    for (int i = 0; i < ctx->n; ++i) {
        const int32_t tok = result[i].token_id;
        if (tok < 0) { ctx->slots[i].done = true; continue; }
        if (ctx->slots[i].tokens.empty()) ctx->slots[i].first_ms = t - ctx->started;
        ctx->slots[i].tokens.push_back(tok);
        if (tok == kImEnd) ctx->slots[i].done = true;
    }
    return 0;
}

struct Req {
    std::string sid;
    std::vector<float> embeds;
    int ntok = 0;
};

bool read_index(const std::string& dir, std::vector<Req>* reqs, const std::vector<std::string>& order) {
    std::map<std::string, Req> by_sid;
    std::ifstream in(dir + "/index.tsv");
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string sid, file, ntok;
        std::getline(ss, sid, '\t');
        std::getline(ss, file, '\t');
        std::getline(ss, ntok, '\t');
        Req r;
        r.sid = sid;
        r.ntok = atoi(ntok.c_str());
        std::ifstream f(dir + "/" + file, std::ios::binary);
        if (!f) return false;
        r.embeds.resize((size_t)r.ntok * kDim);
        f.read(reinterpret_cast<char*>(r.embeds.data()), (std::streamsize)r.embeds.size() * 4);
        if (f.gcount() != (std::streamsize)r.embeds.size() * 4) return false;
        by_sid[sid] = std::move(r);
    }
    for (const auto& sid : order) {
        auto it = by_sid.find(sid);
        if (it == by_sid.end()) return false;
        reqs->push_back(it->second);
    }
    return true;
}

bool load_order(const std::string& path, std::vector<std::string>* order) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string sid;
        std::getline(ss, sid, '\t');
        order->push_back(sid);
    }
    return !order->empty();
}

class Llm {
public:
    ~Llm() { if (h_) rkllm_destroy(h_); }
    bool init(const std::string& model, int n_batch) {
        RKLLMParam p = rkllm_createDefaultParam();
        p.model_path = model.c_str();
        p.max_context_len = 512;
        p.max_new_tokens = kMaxNew;
        p.top_k = 1;
        p.top_p = 1.0f;
        p.temperature = 0.0f;
        p.repeat_penalty = 1.0f;
        p.skip_special_token = false;
        p.extend_param.base_domain_id = 1;
        p.extend_param.n_batch = (uint8_t)n_batch;
        RKLLMCallback cb{};
        cb.result_callback = &on_result;
        cb.result_userdata = nullptr;
        n_batch_ = n_batch;
        const double t0 = now_ms();
        if (rkllm_init(&h_, &p, &cb) != 0 || !h_) { h_ = nullptr; return false; }
        init_ms_ = now_ms() - t0;
        return true;
    }
    bool run(const std::vector<Req>& reqs, int n, BatchCtx* ctx, double* wall_ms, int* status) {
        ctx->n = n;
        ctx->slots.assign(n, SlotOut{});
        ctx->error = false;
        std::vector<RKLLMInput> in(n);
        memset(in.data(), 0, sizeof(RKLLMInput) * n);
        for (int i = 0; i < n; ++i) {
            in[i].role = "user";
            in[i].enable_thinking = false;
            in[i].input_type = RKLLM_INPUT_EMBED;
            in[i].embed_input.embed = const_cast<float*>(reqs[i].embeds.data());
            in[i].embed_input.n_tokens = (size_t)reqs[i].ntok;
        }
        RKLLMInferParam infer{};
        memset(&infer, 0, sizeof(infer));
        infer.mode = RKLLM_INFER_GENERATE;
        infer.keep_history = 0;
        if (rkllm_clear_kv_cache(h_, 0, nullptr, nullptr) != 0) { *status = -2; return false; }
        ctx->started = now_ms();
        const double t0 = ctx->started;
        const int rc = rkllm_run(h_, &in[0], &infer, ctx);
        *wall_ms = now_ms() - t0;
        *status = rc;
        return rc == 0;
    }
    void shutdown() { if (h_) { rkllm_destroy(h_); h_ = nullptr; } }
    double init_ms() const { return init_ms_; }

private:
    LLMHandle h_ = nullptr;
    int n_batch_ = 1;
    double init_ms_ = 0;
};

int32_t parse_tokens(const std::string& s, std::vector<int32_t>* out) {
    out->clear();
    std::stringstream ss(s);
    std::string p;
    while (std::getline(ss, p, ',')) if (!p.empty()) out->push_back(atoi(p.c_str()));
    return (int32_t)out->size();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <embdir> s|b <order.tsv> <out.tsv> <model.rkllm> [n_batch]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const std::string mode = argv[2];
    std::vector<std::string> order;
    if (!load_order(argv[3], &order)) { fprintf(stderr, "FATAL cannot read order %s\n", argv[3]); return 1; }
    const std::string out_path = argv[4];
    const std::string model = argv[5];
    const int n_batch = (mode == "b") ? (argc > 6 ? atoi(argv[6]) : 2) : 1;

    std::vector<Req> reqs;
    if (!read_index(dir, &reqs, order)) { fprintf(stderr, "FATAL cannot load embeddings from %s\n", dir.c_str()); return 1; }

    Llm llm;
    if (!llm.init(model, n_batch)) { fprintf(stderr, "FATAL rkllm_init failed (n_batch=%d)\n", n_batch); return 3; }
    fprintf(stderr, "[batch] model=%s n_batch=%d init_ms=%.0f reqs=%zu\n",
            model.c_str(), n_batch, llm.init_ms(), reqs.size());

    std::ofstream out(out_path);
    out << "sid\tslot\tstatus\tn_tokens\ttokens\tttft_ms\twall_ms\trow_ms\n";
    size_t done = 0, failed = 0;
    const double t_all = now_ms();
    for (size_t i = 0; i < reqs.size(); i += n_batch) {
        std::vector<Req> group(reqs.begin() + (long)i, reqs.begin() + (long)std::min(reqs.size(), i + n_batch));
        BatchCtx ctx;
        double wall = 0;
        int status = 0;
        const double t_req = now_ms();
        const bool ok = llm.run(group, (int)group.size(), &ctx, &wall, &status);
        const double row_ms = now_ms() - t_req;
        for (size_t s = 0; s < group.size(); ++s) {
            const auto& r = group[s];
            const SlotOut& o = ctx.slots[s];
            std::string toks;
            for (size_t k = 0; k < o.tokens.size(); ++k) toks += (k ? "," : "") + std::to_string(o.tokens[k]);
            out << r.sid << "\t" << s << "\t" << (ok && !ctx.error ? "ok" : "error") << "\t"
                << o.tokens.size() << "\t" << toks << "\t" << o.first_ms << "\t" << wall << "\t" << row_ms << "\n";
            ++done;
            if (!(ok && !ctx.error)) ++failed;
        }
        out.flush();
        if (group.size() == 1) {
            printf("REQ %s status=%d ntok=%zu wall=%.1f\n", group[0].sid.c_str(), status,
                   ctx.slots[0].tokens.size(), wall);
        } else {
            printf("BATCH %s+%s status=%d slots=%zu/%zu wall=%.1f\n", group[0].sid.c_str(),
                   group[1].sid.c_str(), status, ctx.slots[0].tokens.size(), ctx.slots[1].tokens.size(), wall);
        }
        fflush(stdout);
    }
    const double total = now_ms() - t_all;
    printf("RUN_DONE mode=%s n=%zu failed=%zu total_ms=%.1f\n", mode.c_str(), done, failed, total);
    llm.shutdown();
    return failed ? 6 : 0;
}
