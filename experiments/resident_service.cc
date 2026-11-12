// Resident GroundingPI service: Qwen + MoonViT stay loaded for the process lifetime.
//
//   resident_service --requests reqs.tsv [--batches K] [--pipeline] [--warmup N]
//
// Request line: id<TAB>pixels.npy<TAB>width<TAB>height<TAB>query
//           or: {"id":"...","pixels":"...","width":640,"height":480,"query":"..."}
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <algorithm>
#include <chrono>
#include <cmath>

#include "common.h"
#include "embed.h"
#include "llm.h"
#include "npy.h"
#include "protocol.h"
#include "tokenizer.h"
#include "vision.h"

namespace {

struct Request {
    std::string id;
    std::string pixels;
    int width = 0;
    int height = 0;
    std::string query;
};

struct Job {
    uint64_t task_id = 0;
    int request_index = 0;
    std::vector<float> input;
    std::vector<float> output;
    std::vector<float> tokens;
    uint64_t enqueue_ns = 0, dequeue_ns = 0, done_ns = 0;
    gp::StageTimes times[3];
};

class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}
    bool push(Job* job) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [&] { return closed_ || queue_.size() < capacity_; });
        if (closed_) return false;
        queue_.push_back(job);
        not_empty_.notify_one();
        return true;
    }
    Job* pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [&] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) return nullptr;
        Job* job = queue_.front();
        queue_.pop_front();
        not_full_.notify_one();
        return job;
    }
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    std::deque<Job*> queue_;
    size_t capacity_;
    bool closed_ = false;
    std::mutex mutex_;
    std::condition_variable not_empty_, not_full_;
};

std::vector<std::string> split(const std::string& text, char sep) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= text.size()) {
        size_t pos = text.find(sep, start);
        parts.push_back(text.substr(start, pos == std::string::npos ? std::string::npos : pos - start));
        if (pos == std::string::npos) break;
        start = pos + 1;
    }
    return parts;
}

std::string json_field(const std::string& line, const std::string& key, bool numeric = false) {
    std::string needle = "\"" + key + "\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return std::string();
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos) return std::string();
    ++pos;
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) ++pos;
    if (numeric) {
        size_t end = pos;
        while (end < line.size() && (isdigit((unsigned char)line[end]) || line[end] == '-')) ++end;
        return line.substr(pos, end - pos);
    }
    if (pos < line.size() && line[pos] == '"') {
        size_t end = line.find('"', pos + 1);
        if (end == std::string::npos) return std::string();
        return line.substr(pos + 1, end - pos - 1);
    }
    size_t end = line.find_first_of(",}", pos);
    return line.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
}

std::vector<Request> read_requests(const std::string& path) {
    std::vector<Request> rows;
    auto data = gp::read_file(path);
    std::string text(data.begin(), data.end());
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        std::string line = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        start = end == std::string::npos ? text.size() : end + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        Request req;
        if (line[0] == '{') {
            req.id = json_field(line, "id");
            req.pixels = json_field(line, "pixels");
            req.width = atoi(json_field(line, "width", true).c_str());
            req.height = atoi(json_field(line, "height", true).c_str());
            req.query = json_field(line, "query");
        } else {
            auto parts = split(line, '\t');
            if (parts.size() < 5) continue;
            req.id = parts[0];
            req.pixels = parts[1];
            req.width = atoi(parts[2].c_str());
            req.height = atoi(parts[3].c_str());
            req.query = parts[4];
        }
        if (!req.pixels.empty()) rows.push_back(req);
    }
    return rows;
}

long rss_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256]; long kb = -1;
    while (fgets(line, sizeof(line), f))
        if (strncmp(line, "VmRSS:", 6) == 0) { sscanf(line + 6, "%ld", &kb); break; }
    fclose(f);
    return kb;
}

long mem_available_kb() {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256]; long kb = -1;
    while (fgets(line, sizeof(line), f))
        if (strncmp(line, "MemAvailable:", 13) == 0) { sscanf(line + 13, "%ld", &kb); break; }
    fclose(f);
    return kb;
}

void swap_bytes(long* in, long* out) {
    *in = *out = -1;
    FILE* f = fopen("/proc/vmstat", "r");
    if (!f) return;
    char key[64]; long value;
    while (fscanf(f, "%63s %ld", key, &value) == 2) {
        if (strcmp(key, "pswpin") == 0) *in = value;
        if (strcmp(key, "pswpout") == 0) *out = value;
    }
    fclose(f);
}

int thread_count() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256]; int n = -1;
    while (fgets(line, sizeof(line), f))
        if (strncmp(line, "Threads:", 8) == 0) { sscanf(line + 8, "%d", &n); break; }
    fclose(f);
    return n;
}

int fd_count() {
    int count = 0;
    FILE* p = popen("ls /proc/self/fd 2>/dev/null | wc -l", "r");
    if (p) { if (fscanf(p, "%d", &count) != 1) count = -1; pclose(p); }
    return count;
}

struct Options {
    std::string requests;
    std::string models =
        ""
        ""
        "";
    std::string llm = "";
    std::string assets = "";
    std::string dump;
    int batches = 1;
    int warmup = 1;
    bool pipeline = false;
};

bool run_query(const std::vector<float>& visual, const Request& req,
               const gp::Tokenizer& tokenizer, gp::EmbeddingTable& embed,
               gp::LlmSession& llm, double* embed_ms, double* llm_ms, gp::Box* box,
               std::vector<int32_t>* out_tokens, std::string* out_ids) {
    if (visual.size() != 256u * 2560u) return false;
    gp::Timer embed_timer;
    gp::Prompt prompt = gp::build_prompt(tokenizer, req.query);
    std::vector<float> embeds((size_t)prompt.ids.size() * 2560);
    for (size_t k = 0; k < prompt.ids.size(); ++k) {
        float* dst = embeds.data() + k * 2560;
        if ((int)k >= prompt.head_tokens && (int)k < prompt.head_tokens + prompt.image_tokens) {
            memcpy(dst, visual.data() + ((size_t)k - prompt.head_tokens) * 2560, 2560 * 4);
        } else if (!embed.row(prompt.ids[k], dst)) {
            return false;
        }
    }
    *embed_ms = embed_timer.ms();
    auto result = llm.run(embeds.data(), (int)prompt.ids.size());
    *llm_ms = result.run_ms;
    *out_tokens = result.tokens;
    *box = gp::decode_box(result.tokens, req.width, req.height);
    out_ids->clear();
    for (size_t k = 0; k < result.tokens.size(); ++k)
        *out_ids += (k ? "," : "") + std::to_string(result.tokens[k]);
    return result.status == 0 && result.finished && !result.error;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (arg == "--requests") opt.requests = next();
        else if (arg == "--models") opt.models = next();
        else if (arg == "--llm") opt.llm = next();
        else if (arg == "--assets") opt.assets = next();
        else if (arg == "--dump-tokens") opt.dump = next();
        else if (arg == "--batches") opt.batches = atoi(next().c_str());
        else if (arg == "--warmup") opt.warmup = atoi(next().c_str());
        else if (arg == "--pipeline") opt.pipeline = true;
        else { fprintf(stderr, "unknown option %s\n", arg.c_str()); return 2; }
    }
    if (opt.requests.empty()) { fprintf(stderr, "usage: resident_service --requests FILE\n"); return 2; }
    if (opt.batches < 1) { fprintf(stderr, "FATAL batches >= 1\n"); return 2; }
    if (opt.warmup < 0) { fprintf(stderr, "FATAL warmup >= 0\n"); return 2; }

    auto requests = read_requests(opt.requests);
    if (requests.empty()) { fprintf(stderr, "FATAL no requests\n"); return 2; }

    const auto process_start = std::chrono::steady_clock::now();
    auto elapsed_ms = [&] {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - process_start).count();
    };

    long swap_in0 = 0, swap_out0 = 0;
    swap_bytes(&swap_in0, &swap_out0);

    // ---- 1. Qwen first, own IOVA domain ----
    gp::LlmSession llm;
    if (!llm.init(opt.llm, 512, 32, 1)) {
        fprintf(stderr, "FATAL qwen init failed\n");
        return 1;
    }
    const double qwen_init_ms = elapsed_ms();
    const long rss_after_qwen = rss_kb();

    // ---- 2. vision stages ----
    auto models = split(opt.models, ',');
    if (models.size() != 3) { fprintf(stderr, "FATAL need three models\n"); return 1; }
    std::vector<gp::VisionStage> stages(3);
    const int masks[3] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2};
    if (!opt.pipeline) {
        for (int i = 0; i < 3; ++i) {
            if (!stages[(size_t)i].init(models[(size_t)i], masks[i])) {
                fprintf(stderr, "FATAL vision stage %d init failed\n", i);
                for (auto& stage : stages) stage.release();
                return 1;
            }
        }
    }
    const double vision_init_done_ms = elapsed_ms();
    const long rss_after_vision = rss_kb();

    // ---- 3. text resources ----
    gp::Tokenizer tokenizer;
    if (!tokenizer.load(opt.assets + "/vocab.bin", opt.assets + "/merges.bin")) {
        fprintf(stderr, "FATAL tokenizer load failed\n");
        for (auto& stage : stages) stage.release();
        return 1;
    }
    gp::EmbeddingTable embed;
    if (!embed.open(opt.assets + "/embed.bf16", 152670, 2560)) {
        fprintf(stderr, "FATAL embedding table load failed\n");
        for (auto& stage : stages) stage.release();
        return 1;
    }

    // Preload the pixel buffers once (also serves as a validation of the list).
    std::vector<std::vector<float>> pixels(requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
        std::vector<int> shape;
        if (!gp::load_npy_f32(requests[i].pixels, pixels[i], shape)) {
            fprintf(stderr, "FATAL cannot load %s\n", requests[i].pixels.c_str());
            for (auto& stage : stages) stage.release();
            return 1;
        }
    }

    // ---- 4. pipeline workers (batch mode) ----
    BoundedQueue queue0(2), queue1(2), queue2(2);
    std::atomic<bool> error{false};
    std::mutex done_mutex;
    std::condition_variable done_cv;
    size_t completed = 0;
    std::vector<std::thread> threads;
    std::vector<gp::VisionStage> worker_stages(3);
    std::mutex ready_mutex;
    std::condition_variable ready_cv;
    int ready_count = 0;

    auto fail = [&](const char* who) {
        {
            std::lock_guard<std::mutex> lock(done_mutex);
            error.store(true);
        }
        queue0.close(); queue1.close(); queue2.close();
        done_cv.notify_all();
        fprintf(stderr, "[resident] FATAL from %s\n", who);
    };

    if (opt.pipeline) {
        for (int i = 0; i < 3; ++i) {
            threads.emplace_back([&, i] {
                bool ok = worker_stages[(size_t)i].init(models[(size_t)i], masks[i]);
                {
                    std::lock_guard<std::mutex> lock(ready_mutex);
                    ++ready_count;
                }
                ready_cv.notify_all();
                if (!ok) { fail("worker-init"); return; }
                BoundedQueue* in = (i == 0) ? &queue0 : (i == 1 ? &queue1 : &queue2);
                BoundedQueue* out = (i == 0) ? &queue1 : (i == 1 ? &queue2 : nullptr);
                while (true) {
                    Job* job = in->pop();
                    if (!job) break;
                    job->dequeue_ns = gp::now_ns();
                    if (!worker_stages[(size_t)i].run(job->input.data(), job->input.size(),
                                                      job->output, nullptr, &job->times[i])) {
                        fail("worker-run");
                        break;
                    }
                    job->input.swap(job->output);
                    job->output.clear();
                    if (out) {
                        if (!out->push(job)) break;
                    } else {
                        job->done_ns = gp::now_ns();
                        job->tokens.swap(job->input);
                        {
                            std::lock_guard<std::mutex> lock(done_mutex);
                            ++completed;
                        }
                        done_cv.notify_one();
                    }
                }
            });
        }
        {
            std::unique_lock<std::mutex> lock(ready_mutex);
            ready_cv.wait(lock, [&] { return ready_count == 3; });
        }
        if (error.load()) {
            queue0.close(); queue1.close(); queue2.close();
            for (auto& t : threads) t.join();
            return 1;
        }
    }

    const double ready_ms = elapsed_ms();
    fprintf(stderr, "[resident] READY qwen_init=%.0fms vision_init_done=%.0fms ready=%.0fms rss=%ldKB\n",
            qwen_init_ms, vision_init_done_ms, ready_ms, rss_kb());
    printf("{\"event\":\"ready\",\"startup_ms\":%.1f,\"qwen_init_ms\":%.1f,"
           "\"rss_kb\":%ld,\"mem_available_kb\":%ld,\"threads\":%d,\"fds\":%d}\n",
           ready_ms, qwen_init_ms, rss_kb(), mem_available_kb(), thread_count(), fd_count());
    fflush(stdout);

    double first_answer_ms = 0;
    bool have_answer = false;

    // ---- 5. request loop ----
    for (int batch = 0; batch < opt.batches; ++batch) {
        const auto batch_start = std::chrono::steady_clock::now();
        double batch_vision_ms = 0, batch_llm_ms = 0, batch_embed_ms = 0;
        int answers = 0;

        if (opt.pipeline) {
            // One vision job per unique pixel file, then Qwen per request.
            std::vector<size_t> unique;
            std::vector<int> lookup(requests.size(), -1);
            for (size_t i = 0; i < requests.size(); ++i) {
                for (size_t k = 0; k < unique.size(); ++k)
                    if (requests[unique[k]].pixels == requests[i].pixels) { lookup[i] = (int)k; break; }
                if (lookup[i] < 0) { lookup[i] = (int)unique.size(); unique.push_back(i); }
            }
            std::vector<Job> jobs(unique.size());
            {
                std::lock_guard<std::mutex> lock(done_mutex);
                completed = 0;
            }
            for (size_t k = 0; k < unique.size(); ++k) {
                jobs[k].task_id = k;
                jobs[k].request_index = (int)unique[k];
                jobs[k].input = pixels[unique[k]];
            }
            const uint64_t t0 = gp::now_ns();
            for (auto& job : jobs) {
                job.enqueue_ns = gp::now_ns();
                if (!queue0.push(&job)) break;
            }
            {
                std::unique_lock<std::mutex> lock(done_mutex);
                done_cv.wait(lock, [&] { return completed == jobs.size() || error.load(); });
            }
            batch_vision_ms = (double)(gp::now_ns() - t0) / 1e6;
            if (error.load()) {
                for (auto& stage : worker_stages) stage.release();
                queue0.close(); queue1.close(); queue2.close();
                for (auto& t : threads) t.join();
                return 1;
            }
            std::vector<std::vector<float>> image_tokens(jobs.size());
            for (auto& job : jobs) image_tokens[(size_t)job.task_id] = std::move(job.tokens);
            if (!opt.dump.empty()) {
                for (size_t k = 0; k < unique.size(); ++k)
                    gp::write_file(opt.dump + "/" + requests[unique[k]].id + ".f32",
                                   image_tokens[k].data(), image_tokens[k].size() * sizeof(float));
            }
            for (size_t i = 0; i < requests.size(); ++i) {
                const Request& req = requests[i];
                const std::vector<float>& visual = image_tokens[(size_t)lookup[i]];
                double embed_ms = 0, llm_ms = 0;
                gp::Box box;
                std::vector<int32_t> tokens;
                std::string ids;
                if (!run_query(visual, req, tokenizer, embed, llm, &embed_ms, &llm_ms, &box,
                               &tokens, &ids)) {
                    fprintf(stderr, "FATAL query %s\n", req.id.c_str());
                    for (auto& stage : worker_stages) stage.release();
                    queue0.close(); queue1.close(); queue2.close();
                    for (auto& t : threads) t.join();
                    return 1;
                }
                batch_embed_ms += embed_ms;
                batch_llm_ms += llm_ms;
                ++answers;
                const double done_ms = elapsed_ms();
                if (!have_answer) { have_answer = true; first_answer_ms = done_ms; }
                printf("{\"event\":\"answer\",\"batch\":%d,\"id\":\"%s\",\"llm_ms\":%.1f,"
                       "\"valid\":%s,\"none\":%s,\"box\":[%.2f,%.2f,%.2f,%.2f],\"t_ms\":%.1f}\n",
                       batch, req.id.c_str(), llm_ms, box.valid ? "true" : "false",
                       box.none ? "true" : "false", box.x0, box.y0, box.x1, box.y1, done_ms);
                fflush(stdout);
                (void)tokens; (void)ids;
            }
        } else {
            for (size_t i = 0; i < requests.size(); ++i) {
                const Request& req = requests[i];
                const uint64_t req_start = gp::now_ns();
                gp::Timer vision_timer;
                std::vector<float> buffer, tokens;
                const float* current = pixels[i].data();
                size_t count = pixels[i].size();
                double stage_ms[3] = {0, 0, 0};
                for (int s = 0; s < 3; ++s) {
                    if (count < (size_t)stages[(size_t)s].input_elems())
                        count = (size_t)stages[(size_t)s].input_elems();
                    if (!stages[(size_t)s].run(current, count, buffer, &stage_ms[s])) {
                        fprintf(stderr, "FATAL vision run %s\n", req.id.c_str());
                        return 1;
                    }
                    current = buffer.data();
                    count = buffer.size();
                }
                tokens = buffer;
                const double vision_ms = vision_timer.ms();
                if (!opt.dump.empty())
                    gp::write_file(opt.dump + "/" + req.id + ".f32",
                                   tokens.data(), tokens.size() * sizeof(float));
                double embed_ms = 0, llm_ms = 0;
                gp::Box box;
                std::vector<int32_t> gen;
                std::string ids;
                if (!run_query(tokens, req, tokenizer, embed, llm, &embed_ms, &llm_ms, &box, &gen, &ids)) {
                    fprintf(stderr, "FATAL query %s\n", req.id.c_str());
                    return 1;
                }
                batch_vision_ms += vision_ms;
                batch_embed_ms += embed_ms;
                batch_llm_ms += llm_ms;
                ++answers;
                const double done_ms = elapsed_ms();
                if (!have_answer) { have_answer = true; first_answer_ms = done_ms; }
                printf("{\"event\":\"answer\",\"batch\":%d,\"id\":\"%s\",\"vision_ms\":%.1f,"
                       "\"stage_ms\":[%.1f,%.1f,%.1f],\"embed_ms\":%.2f,\"llm_ms\":%.1f,"
                       "\"request_ms\":%.1f,\"valid\":%s,\"none\":%s,"
                       "\"box\":[%.2f,%.2f,%.2f,%.2f],\"t_ms\":%.1f}\n",
                       batch, req.id.c_str(), vision_ms, stage_ms[0], stage_ms[1], stage_ms[2],
                       embed_ms, llm_ms, (double)(gp::now_ns() - req_start) / 1e6,
                       box.valid ? "true" : "false", box.none ? "true" : "false",
                       box.x0, box.y0, box.x1, box.y1, done_ms);
                fflush(stdout);
            }
        }
        const double batch_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - batch_start).count();
        long swap_in = 0, swap_out = 0;
        swap_bytes(&swap_in, &swap_out);
        printf("{\"event\":\"batch\",\"batch\":%d,\"answers\":%d,\"batch_ms\":%.1f,"
               "\"vision_sum_ms\":%.1f,\"embed_sum_ms\":%.1f,\"llm_sum_ms\":%.1f,"
               "\"rss_kb\":%ld,\"mem_available_kb\":%ld,\"swap_in_delta\":%ld,"
               "\"swap_out_delta\":%ld,\"threads\":%d,\"fds\":%d}\n",
               batch, answers, batch_ms, batch_vision_ms, batch_embed_ms, batch_llm_ms,
               rss_kb(), mem_available_kb(), swap_in - swap_in0, swap_out - swap_out0,
               thread_count(), fd_count());
        fflush(stdout);
    }

    printf("{\"event\":\"summary\",\"startup_ms\":%.1f,\"first_answer_ms\":%.1f,"
           "\"rss_kb\":%ld,\"rss_after_qwen_kb\":%ld,\"rss_after_vision_kb\":%ld,"
           "\"fds\":%d}\n",
           ready_ms, first_answer_ms, rss_kb(), rss_after_qwen, rss_after_vision, fd_count());

    if (opt.pipeline) {
        queue0.close(); queue1.close(); queue2.close();
        for (auto& t : threads) t.join();
        for (auto& stage : worker_stages) stage.release();
    }
    for (auto& stage : stages) stage.release();
    return 0;
}
