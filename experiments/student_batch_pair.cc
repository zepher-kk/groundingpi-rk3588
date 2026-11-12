// End-to-end pair probe: two JPEGs, vision per image, then either
//   serial : n_batch=1, A then B            (current deployment)
//   batch  : n_batch=2, both embeddings in one rkllm_run
// Same process, same vision context, two sequential phases (never co-resident LLM).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "embed.h"
#include "rkllm.h"
#include "tokenizer.h"
#include "vision.h"

namespace {

constexpr int kDim = 2048, kSlots = 144, kCanvas = 384, kMaxNew = 8;
constexpr int32_t kImStart = 151644, kImEnd = 151645, kObjectRefStart = 151646, kObjectRefEnd = 151647;
constexpr int32_t kBoxStart = 151648, kVisionStart = 151652, kVisionEnd = 151653, kImagePad = 151655;

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

double pil_bicubic(double x) {
    x = std::fabs(x);
    if (x < 1.0) return (1.5 * x - 2.5) * x * x + 1.0;
    if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    return 0.0;
}

std::vector<double> weights(double c, double sup, double ss, int n, int* lo) {
    int a = (int)std::floor(c - sup + 0.5);
    if (a < 0) a = 0;
    int b = (int)std::floor(c + sup + 0.5);
    if (b > n) b = n;
    std::vector<double> w;
    double tot = 0;
    for (int x = a; x < b; ++x) { double v = pil_bicubic((x - c + 0.5) * ss); w.push_back(v); tot += v; }
    if (tot != 0) for (double& v : w) v /= tot;
    *lo = a;
    return w;
}

cv::Mat resize_pil(const cv::Mat& src, int ow, int oh) {
    const int iw = src.cols, ih = src.rows;
    const double sx = (double)iw / ow, sy = (double)ih / oh;
    const double fx = sx > 1 ? sx : 1, fy = sy > 1 ? sy : 1;
    cv::Mat tmp(ih, ow, CV_8UC3);
    for (int y = 0; y < ih; ++y)
        for (int x = 0; x < ow; ++x) {
            int lo;
            auto w = weights((x + 0.5) * sx, 2.0 * fx, 1.0 / fx, iw, &lo);
            double acc[3] = {0, 0, 0};
            for (int i = 0; i < (int)w.size(); ++i) {
                const uint8_t* p = src.ptr<uint8_t>(y) + (lo + i) * 3;
                acc[0] += w[i] * p[0]; acc[1] += w[i] * p[1]; acc[2] += w[i] * p[2];
            }
            uint8_t* o = tmp.ptr<uint8_t>(y) + x * 3;
            for (int c = 0; c < 3; ++c) o[c] = (uint8_t)std::min(255.0, std::max(0.0, std::floor(acc[c] + 0.5)));
        }
    cv::Mat dst(oh, ow, CV_8UC3);
    for (int y = 0; y < oh; ++y) {
        int lo;
        auto w = weights((y + 0.5) * sy, 2.0 * fy, 1.0 / fy, ih, &lo);
        for (int x = 0; x < ow; ++x) {
            double acc[3] = {0, 0, 0};
            for (int i = 0; i < (int)w.size(); ++i) {
                const uint8_t* p = tmp.ptr<uint8_t>(lo + i) + x * 3;
                acc[0] += w[i] * p[0]; acc[1] += w[i] * p[1]; acc[2] += w[i] * p[2];
            }
            uint8_t* o = dst.ptr<uint8_t>(y) + x * 3;
            for (int c = 0; c < 3; ++c) o[c] = (uint8_t)std::min(255.0, std::max(0.0, std::floor(acc[c] + 0.5)));
        }
    }
    return dst;
}

std::vector<float> preprocess(const cv::Mat& bgr) {
    const int w = bgr.cols, h = bgr.rows;
    const float s = std::min((float)kCanvas / w, (float)kCanvas / h);
    const int rw = std::max(1, (int)std::floor(w * s + 0.5f));
    const int rh = std::max(1, (int)std::floor(h * s + 0.5f));
    const int left = (kCanvas - rw) / 2, top = (kCanvas - rh) / 2;
    cv::Mat rgb, rz;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rz = resize_pil(rgb, rw, rh);
    cv::Mat canvas(kCanvas, kCanvas, CV_8UC3, cv::Scalar(127, 127, 127));
    rz.copyTo(canvas(cv::Rect(left, top, rw, rh)));
    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float stdv[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> out(3 * kCanvas * kCanvas);
    for (int y = 0; y < kCanvas; ++y) {
        const uint8_t* row = canvas.ptr<uint8_t>(y);
        for (int x = 0; x < kCanvas; ++x)
            for (int c = 0; c < 3; ++c)
                out[(size_t)c * kCanvas * kCanvas + y * kCanvas + x] = (row[x * 3 + c] / 255.0f - mean[c]) / stdv[c];
    }
    return out;
}

struct SlotOut { std::vector<int32_t> tokens; double first_ms = 0; };
struct Ctx { int n = 1; double started = 0; std::vector<SlotOut> slots; bool error = false; };

int on_result(RKLLMResult* res, void* ud, LLMCallState state) {
    auto* c = static_cast<Ctx*>(ud);
    if (!c) return 0;
    const double t = now_ms();
    if (state == RKLLM_RUN_ERROR) { c->error = true; return 0; }
    if (state != RKLLM_RUN_NORMAL || !res) return 0;
    for (int i = 0; i < c->n; ++i) {
        const int32_t tok = res[i].token_id;
        if (tok < 0) continue;
        if (c->slots[i].tokens.empty()) c->slots[i].first_ms = t - c->started;
        c->slots[i].tokens.push_back(tok);
    }
    return 0;
}

class Llm {
public:
    bool init(const std::string& model, int n_batch) {
        RKLLMParam p = rkllm_createDefaultParam();
        p.model_path = model.c_str();
        p.max_context_len = 512;
        p.max_new_tokens = kMaxNew;
        p.top_k = 1; p.top_p = 1.0f; p.temperature = 0.0f; p.repeat_penalty = 1.0f;
        p.skip_special_token = false;
        p.extend_param.base_domain_id = 1;
        p.extend_param.n_batch = (uint8_t)n_batch;
        RKLLMCallback cb{};
        cb.result_callback = &on_result;
        const double t0 = now_ms();
        if (rkllm_init(&h_, &p, &cb) != 0 || !h_) { h_ = nullptr; return false; }
        init_ms_ = now_ms() - t0;
        return true;
    }
    bool run(const std::vector<std::vector<float>>& emb, const std::vector<int>& ntok,
             double* wall, Ctx* ctx, int* status) {
        const int n = (int)emb.size();
        ctx->n = n;
        ctx->slots.assign(n, SlotOut{});
        ctx->error = false;
        std::vector<RKLLMInput> in(n);
        memset(in.data(), 0, sizeof(RKLLMInput) * n);
        for (int i = 0; i < n; ++i) {
            in[i].role = "user";
            in[i].input_type = RKLLM_INPUT_EMBED;
            in[i].embed_input.embed = const_cast<float*>(emb[i].data());
            in[i].embed_input.n_tokens = (size_t)ntok[i];
        }
        RKLLMInferParam infer{};
        infer.mode = RKLLM_INFER_GENERATE;
        infer.keep_history = 0;
        if (rkllm_clear_kv_cache(h_, 0, nullptr, nullptr) != 0) { *status = -2; return false; }
        ctx->started = now_ms();
        const int rc = rkllm_run(h_, &in[0], &infer, ctx);
        *wall = now_ms() - ctx->started;
        *status = rc;
        return rc == 0 && !ctx->error;
    }
    void destroy() { if (h_) { rkllm_destroy(h_); h_ = nullptr; } }
    double init_ms() const { return init_ms_; }

private:
    LLMHandle h_ = nullptr;
    double init_ms_ = 0;
};

std::vector<float> build_embed(gp::Tokenizer& tok, gp::EmbeddingTable& emb, gp::VisionStage& vis,
                               const std::string& jpg, const std::string& query, int* ntok, double* vision_ms) {
    auto buf = gp::read_file(jpg);
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
    auto px = preprocess(img);
    std::vector<float> v;
    vis.run(px.data(), px.size(), v, vision_ms);
    std::vector<int32_t> ids = {kImStart, 8948, 198, 2610, 525, 264, 10950, 17847, 13,
                                kImEnd, 198, kImStart, 872, 198};
    auto body = tok.encode("Locate the target referred to by the following description: " + query + ".");
    ids.insert(ids.end(), body.begin(), body.end());
    ids.push_back(kVisionStart);
    const int start = (int)ids.size();
    ids.insert(ids.end(), kSlots, kImagePad);
    for (int32_t t : {kVisionEnd, kImEnd, 198, kImStart, 77091, 198, kObjectRefStart}) ids.push_back(t);
    auto q = tok.encode(query);
    ids.insert(ids.end(), q.begin(), q.end());
    ids.push_back(kObjectRefEnd);
    ids.push_back(kBoxStart);
    std::vector<float> e(ids.size() * kDim);
    for (size_t i = 0; i < ids.size(); ++i) {
        if ((int)i >= start && (int)i < start + kSlots)
            memcpy(&e[i * kDim], &v[(size_t)((int)i - start) * kDim], kDim * 4);
        else
            emb.row(ids[i], &e[i * kDim]);
    }
    *ntok = (int)ids.size();
    return e;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: %s <asset-dir> <model> <jpgA> <qA> <jpgB> <qB> [rounds]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1], model = argv[2];
    const int rounds = argc > 8 ? atoi(argv[8]) : 1;
    gp::Tokenizer tok;
    gp::EmbeddingTable emb;
    gp::VisionStage vis;
    if (!tok.load(dir + "/vocab.bin", dir + "/merges.bin")) return 1;
    if (!emb.open(dir + "/embed_tokens_151669x2048.bf16", 151669, kDim)) return 1;
    if (!vis.init(dir + "/student_vision_trained_fp16.rknn", RKNN_NPU_CORE_0)) return 1;

    int nA = 0, nB = 0;
    double vA = 0, vB = 0;
    auto eA = build_embed(tok, emb, vis, argv[3], argv[4], &nA, &vA);
    auto eB = build_embed(tok, emb, vis, argv[5], argv[6], &nB, &vB);
    vis.release();
    printf("EMBED ntokA=%d ntokB=%d visionA=%.1f visionB=%.1f\n", nA, nB, vA, vB);

    std::vector<int32_t> serA, serB, batA, batB;
    double ser_ms = 0, bat_ms = 0, init1 = 0, init2 = 0;
    for (int r = 0; r < rounds; ++r) {
        {
            Llm l;
            if (!l.init(model, 1)) { fprintf(stderr, "FATAL init n_batch=1\n"); return 3; }
            init1 = l.init_ms();
            Ctx c; double w1 = 0; int st = 0;
            const double t0 = now_ms();
            if (!l.run({eA}, {nA}, &w1, &c, &st)) return 4;
            serA = c.slots[0].tokens;
            Ctx c2; double w2 = 0;
            if (!l.run({eB}, {nB}, &w2, &c2, &st)) return 4;
            serB = c2.slots[0].tokens;
            ser_ms = now_ms() - t0;
            l.destroy();
        }
        {
            Llm l;
            if (!l.init(model, 2)) { fprintf(stderr, "FATAL init n_batch=2\n"); return 3; }
            init2 = l.init_ms();
            Ctx c; double w = 0; int st = 0;
            const double t0 = now_ms();
            if (!l.run({eA, eB}, {nA, nB}, &w, &c, &st)) return 4;
            batA = c.slots[0].tokens;
            batB = c.slots[1].tokens;
            bat_ms = now_ms() - t0;
            l.destroy();
        }
        printf("ROUND %d serial_ms=%.1f batch_ms=%.1f gain=%.2f%% tokens_same=%d/%d\n", r + 1, ser_ms, bat_ms,
               100.0 * (ser_ms / bat_ms - 1.0), (int)(serA == batA), (int)(serB == batB));
        fflush(stdout);
    }
    printf("SUMMARY serial_ms=%.1f batch_ms=%.1f init1_ms=%.0f init2_ms=%.0f "
           "tokA=%zu tokB=%zu\n", ser_ms, bat_ms, init1, init2, batA.size(), batB.size());
    return 0;
}
