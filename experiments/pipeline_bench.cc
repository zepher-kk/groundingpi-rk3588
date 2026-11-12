// MoonViT three-stage scheduling experiments.
//
//   mode a: serial, all stages on Core 0
//   mode b: serial, stages on Core 0/1/2
//   mode c: one worker + one context per stage (Core 0/1/2), bounded queues
//
// Scheduling knobs (all optional, mode c only unless noted):
//   --queue-cap N        capacity of every queue (input + 2 inter-stage)
//   --max-concurrent N   global execution permits (1..3); 0 = unlimited
//   --pace-ms F          minimum spacing between consecutive stage00 starts
//   --batches K          run K consecutive timed batches
//   --fault init:N / run:N   application-level fault injection
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <algorithm>
#include <cmath>

#include "common.h"
#include "embed.h"
#include "llm.h"
#include "npy.h"
#include "protocol.h"
#include "tokenizer.h"
#include "vision.h"

namespace {

struct Job {
    uint64_t task_id = 0;
    int image_index = 0;
    int pass = 0;
    std::string image_id;
    std::vector<float> input;
    std::vector<float> output;
    std::vector<float> tokens;
    uint64_t submit_begin_ns = 0;
    uint64_t enqueued_ns = 0;
    uint64_t enqueue_ns[3] = {0, 0, 0};
    uint64_t dequeue_ns[3] = {0, 0, 0};
    uint64_t done_ns = 0;
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
    size_t capacity() const { return capacity_; }

private:
    std::deque<Job*> queue_;
    size_t capacity_;
    bool closed_ = false;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
};

// Cancellable counting semaphore used to cap how many stages run at once.
class PermitPool {
public:
    explicit PermitPool(int capacity) : capacity_(capacity) {}
    bool acquire(const std::atomic<bool>& error) {
        std::unique_lock<std::mutex> lock(mutex_);
        while (used_ >= capacity_) {
            if (error.load()) return false;
            cv_.wait_for(lock, std::chrono::milliseconds(20));
        }
        if (error.load()) return false;
        ++used_;
        return true;
    }
    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (used_ > 0) --used_;
        }
        cv_.notify_one();
    }

private:
    int capacity_;
    int used_ = 0;
    std::mutex mutex_;
    std::condition_variable cv_;
};

struct Options {
    std::string mode = "c";
    std::string images;
    std::string queries;
    std::string llm = "";
    std::string assets = "";
    std::string models =
        ""
        ""
        "";
    std::string timeline;
    std::string tokens;
    std::string json;
    int passes = 1;
    int warmup = 1;
    int queue_cap = 2;
    int max_concurrent = 0;
    double pace_ms = 0;
    int batches = 1;
    bool reverse = false;
    int fault_init = -1;
    int fault_run = -1;
    int fault_run_at = 0;
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

struct ImageRow { std::string id, path; };

std::vector<ImageRow> read_rows(const std::string& path) {
    std::vector<ImageRow> rows;
    auto data = gp::read_file(path);
    std::string text(data.begin(), data.end());
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        std::string line = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        start = end == std::string::npos ? text.size() : end + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto parts = split(line, '\t');
        if (parts.size() < 2) continue;
        rows.push_back({parts[0], parts[1]});
    }
    return rows;
}

struct Sampler {
    std::atomic<bool> stop{false};
    std::thread thread;
    std::vector<int> npu, ddr;

    void start() {
        stop.store(false);
        thread = std::thread([this] {
            while (!stop.load()) {
                npu.push_back(read_load("/sys/class/devfreq/fdab0000.npu/load"));
                ddr.push_back(read_load("/sys/class/devfreq/dmc/load"));
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
    }
    void finish() {
        stop.store(true);
        if (thread.joinable()) thread.join();
    }
    static int read_load(const char* path) {
        FILE* f = fopen(path, "r");
        if (!f) return -1;
        int percent = -1;
        if (fscanf(f, "%d", &percent) != 1) percent = -1;
        fclose(f);
        return percent;
    }
};

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    size_t index = (size_t)std::ceil(p * (double)values.size()) - 1;
    if (index >= values.size()) index = values.size() - 1;
    return values[index];
}

int mean_of(const std::vector<int>& v) {
    double sum = 0; int n = 0;
    for (int x : v) if (x >= 0) { sum += x; ++n; }
    return n ? (int)std::lround(sum / n) : -1;
}

int max_of(const std::vector<int>& v) {
    int m = -1; for (int x : v) if (x > m) m = x; return m;
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

int fd_count() {
    int count = 0;
    FILE* p = popen("ls /proc/self/fd 2>/dev/null | wc -l", "r");
    if (p) { if (fscanf(p, "%d", &count) != 1) count = -1; pclose(p); }
    return count;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (arg == "--mode") opt.mode = next();
        else if (arg == "--images") opt.images = next();
        else if (arg == "--queries") opt.queries = next();
        else if (arg == "--models") opt.models = next();
        else if (arg == "--timeline") opt.timeline = next();
        else if (arg == "--tokens") opt.tokens = next();
        else if (arg == "--json") opt.json = next();
        else if (arg == "--passes") opt.passes = atoi(next().c_str());
        else if (arg == "--warmup") opt.warmup = atoi(next().c_str());
        else if (arg == "--queue-cap") opt.queue_cap = atoi(next().c_str());
        else if (arg == "--max-concurrent") opt.max_concurrent = atoi(next().c_str());
        else if (arg == "--pace-ms") opt.pace_ms = atof(next().c_str());
        else if (arg == "--batches") opt.batches = atoi(next().c_str());
        else if (arg == "--reverse") opt.reverse = true;
        else if (arg == "--fault-init") opt.fault_init = atoi(next().c_str());
        else if (arg == "--fault-run") opt.fault_run = atoi(next().c_str());
        else if (arg == "--fault-run-at") opt.fault_run_at = atoi(next().c_str());
        else { fprintf(stderr, "unknown option %s\n", arg.c_str()); return 2; }
    }

    // ---- parameter validation ----
    if (opt.images.empty()) { fprintf(stderr, "FATAL empty image list\n"); return 2; }
    if (opt.mode != "a" && opt.mode != "b" && opt.mode != "c") {
        fprintf(stderr, "FATAL invalid mode %s\n", opt.mode.c_str()); return 2;
    }
    if (opt.queue_cap < 1) { fprintf(stderr, "FATAL queue cap must be >= 1\n"); return 2; }
    if (opt.passes < 1) { fprintf(stderr, "FATAL passes must be >= 1\n"); return 2; }
    if (opt.warmup < 0) { fprintf(stderr, "FATAL warmup must be >= 0\n"); return 2; }
    if (opt.batches < 1) { fprintf(stderr, "FATAL batches must be >= 1\n"); return 2; }
    if (opt.max_concurrent < 0 || opt.max_concurrent > 3) {
        fprintf(stderr, "FATAL max-concurrent must be 0..3\n"); return 2;
    }
    if (opt.max_concurrent > 0 && opt.mode != "c") {
        fprintf(stderr, "FATAL max-concurrent only applies to mode c\n"); return 2;
    }
    if (opt.pace_ms < 0) { fprintf(stderr, "FATAL pace must be >= 0\n"); return 2; }
    if (opt.fault_init > 2 || opt.fault_run > 2) {
        fprintf(stderr, "FATAL fault stage must be 0..2\n"); return 2;
    }

    auto images = read_rows(opt.images);
    if (images.empty()) { fprintf(stderr, "FATAL no images\n"); return 2; }
    if (opt.reverse) std::reverse(images.begin(), images.end());

    // Preload pixels so IO is outside the measurement window.
    std::vector<std::vector<float>> pixels(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        std::vector<int> shape;
        if (!gp::load_npy_f32(images[i].path, pixels[i], shape)) {
            fprintf(stderr, "FATAL cannot load %s\n", images[i].path.c_str());
            return 1;
        }
    }
    const uint64_t ready_ns = gp::now_ns();

    const int masks[3] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2};
    auto models = split(opt.models, ',');
    if (models.size() != 3) { fprintf(stderr, "FATAL need three models\n"); return 2; }
    std::vector<int> stage_masks(3);
    for (int i = 0; i < 3; ++i) stage_masks[i] = (opt.mode == "a") ? RKNN_NPU_CORE_0 : masks[i];

    std::vector<Job> jobs;
    jobs.reserve(images.size() * (size_t)opt.passes);
    uint64_t task_id = 0;
    for (int pass = 0; pass < opt.passes; ++pass)
        for (size_t i = 0; i < images.size(); ++i) {
            Job job;
            job.task_id = task_id++;
            job.image_index = (int)i;
            job.pass = pass;
            job.image_id = images[i].id;
            job.input = pixels[i];
            jobs.push_back(std::move(job));
        }

    // ---- serial modes ----
    std::vector<gp::VisionStage> stages(3);
    double vision_init_ms = 0;
    if (opt.mode != "c") {
        gp::Timer timer;
        for (int i = 0; i < 3; ++i) {
            if (opt.fault_init == i) {
                fprintf(stderr, "FAULT inject init failure at stage %d\n", i);
                return 1;
            }
            if (!stages[(size_t)i].init(models[(size_t)i], stage_masks[i])) {
                fprintf(stderr, "FATAL stage %d init failed (expected core mask %d)\n",
                        i, stage_masks[i]);
                return 1;
            }
        }
        vision_init_ms = timer.ms();
        fprintf(stderr, "[bench] vision init wall=%.0fms\n", vision_init_ms);
    }

    // ---- pipeline mode ----
    BoundedQueue queue0((size_t)opt.queue_cap), queue1((size_t)opt.queue_cap),
        queue2((size_t)opt.queue_cap);
    std::atomic<bool> error{false};
    std::mutex results_mutex;
    std::condition_variable results_cv;
    size_t completed = 0;
    std::unique_ptr<PermitPool> permits;
    if (opt.max_concurrent > 0) permits.reset(new PermitPool(opt.max_concurrent));

    auto fail = [&](const char* who) {
        {
            std::lock_guard<std::mutex> lock(results_mutex);
            error.store(true);
        }
        queue0.close(); queue1.close(); queue2.close();
        results_cv.notify_all();
        fprintf(stderr, "[bench] FATAL from %s\n", who);
    };

    struct Worker {
        int index = 0;
        std::string model;
        int mask = 0;
        BoundedQueue* in = nullptr;
        BoundedQueue* out = nullptr;
        gp::VisionStage stage;
        bool init_ok = false;
        double init_ms = 0;
    };
    std::vector<Worker> workers(3);
    std::deque<Job> warm_jobs;  // kept alive until workers have joined
    std::vector<std::thread> threads;
    std::mutex ready_mutex;
    std::condition_variable ready_cv;
    int ready_count = 0;

    if (opt.mode == "c") {
        for (int i = 0; i < 3; ++i) {
            workers[(size_t)i].index = i;
            workers[(size_t)i].model = models[(size_t)i];
            workers[(size_t)i].mask = stage_masks[i];
        }
        workers[0].in = &queue0; workers[0].out = &queue1;
        workers[1].in = &queue1; workers[1].out = &queue2;
        workers[2].in = &queue2; workers[2].out = nullptr;

        const uint64_t init_start = gp::now_ns();
        for (int i = 0; i < 3; ++i) {
            threads.emplace_back([&, i] {
                Worker& w = workers[(size_t)i];
                gp::Timer timer;
                w.init_ok = !(opt.fault_init == i) &&
                            w.stage.init(w.model, w.mask);
                w.init_ms = timer.ms();
                {
                    std::lock_guard<std::mutex> lock(ready_mutex);
                    ++ready_count;
                }
                ready_cv.notify_all();
                if (!w.init_ok) { fail("init"); return; }

                while (true) {
                    Job* job = w.in->pop();
                    if (!job) break;
                    job->dequeue_ns[w.index] = gp::now_ns();
                    if (permits && !permits->acquire(error)) break;
                    const bool inject = (opt.fault_run == w.index &&
                                         (int)job->task_id == opt.fault_run_at);
                    bool ok = !inject ? w.stage.run(job->input.data(), job->input.size(),
                                                    job->output, nullptr, &job->times[w.index])
                                      : false;
                    if (permits) permits->release();
                    if (!ok) { fail("run"); break; }
                    job->input.swap(job->output);
                    job->output.clear();
                    if (w.out) {
                        job->enqueue_ns[w.index + 1] = gp::now_ns();
                        if (!w.out->push(job)) break;
                    } else {
                        job->done_ns = gp::now_ns();
                        job->tokens.swap(job->input);
                        {
                            std::lock_guard<std::mutex> lock(results_mutex);
                            ++completed;
                        }
                        results_cv.notify_one();
                    }
                }
            });
        }
        {
            std::unique_lock<std::mutex> lock(ready_mutex);
            ready_cv.wait(lock, [&] { return ready_count == 3; });
        }
        vision_init_ms = (double)(gp::now_ns() - init_start) / 1e6;
        for (const auto& w : workers)
            fprintf(stderr, "[bench] stage%d expected_core=%d init=%.0fms ok=%d\n",
                    w.index, w.mask, w.init_ms, (int)w.init_ok);
        if (error.load()) {
            queue0.close(); queue1.close(); queue2.close();
            for (auto& t : threads) t.join();
            fprintf(stderr, "FATAL pipeline init failed\n");
            return 1;
        }
    }

    // ---- warmup + timed batches ----
    auto run_serial = [&](Job& job) {
        job.submit_begin_ns = gp::now_ns();
        job.enqueued_ns = job.submit_begin_ns;
        for (int i = 0; i < 3; ++i) {
            job.dequeue_ns[i] = gp::now_ns();
            if (!stages[(size_t)i].run(job.input.data(), job.input.size(), job.output,
                                       nullptr, &job.times[i]))
                return false;
            job.input.swap(job.output);
            job.output.clear();
        }
        job.tokens = job.input;
        job.done_ns = gp::now_ns();
        return true;
    };

    // Warmup keeps its own Job objects so workers never see a dead pointer.
    for (int w = 0; w < opt.warmup; ++w) {
        warm_jobs.emplace_back();
        Job& warm = warm_jobs.back();
        warm.input = jobs.front().input;
        if (opt.mode == "c") {
            warm.submit_begin_ns = warm.enqueued_ns = gp::now_ns();
            if (!queue0.push(&warm)) break;
            std::unique_lock<std::mutex> lock(results_mutex);
            results_cv.wait(lock, [&] { return completed > 0 || error.load(); });
            if (completed > 0) --completed;
            if (error.load()) break;
        } else if (!run_serial(warm)) {
            fprintf(stderr, "FATAL warmup failed\n");
            return 1;
        }
    }
    if (error.load()) {
        queue0.close(); queue1.close(); queue2.close();
        for (auto& t : threads) t.join();
        return 1;
    }

    std::vector<double> batch_wall;
    for (int batch = 0; batch < opt.batches; ++batch) {
        for (auto& job : jobs) {
            job.done_ns = 0;
            job.tokens.clear();
            job.input = pixels[(size_t)job.image_index];
        }
        {
            std::lock_guard<std::mutex> lock(results_mutex);
            completed = 0;
        }
        Sampler sampler;
        sampler.start();
        const uint64_t batch_start = gp::now_ns();
        uint64_t first_submit = 0;
        int submitted = 0;

        if (opt.mode == "c") {
            for (size_t k = 0; k < jobs.size(); ++k) {
                if (error.load()) break;
                if (opt.pace_ms > 0 && k > 0) {
                    const uint64_t target = first_submit + (uint64_t)((double)k * opt.pace_ms * 1e6);
                    while (gp::now_ns() < target) {
                        std::this_thread::sleep_for(std::chrono::microseconds(200));
                        if (error.load()) break;
                    }
                }
                Job& job = jobs[k];
                job.submit_begin_ns = gp::now_ns();
                if (k == 0) first_submit = job.submit_begin_ns;
                job.enqueue_ns[0] = job.submit_begin_ns;
                if (!queue0.push(&job)) break;
                job.enqueued_ns = gp::now_ns();
                ++submitted;
            }
            if (!error.load()) {
                std::unique_lock<std::mutex> lock(results_mutex);
                results_cv.wait(lock, [&] { return completed == jobs.size() || error.load(); });
            }
        } else {
            first_submit = gp::now_ns();
            for (auto& job : jobs) {
                if (!run_serial(job)) { fprintf(stderr, "FATAL serial run failed\n"); return 1; }
                ++submitted;
            }
        }
        const uint64_t batch_end = gp::now_ns();
        sampler.finish();

        if (error.load()) {
            queue0.close(); queue1.close(); queue2.close();
            for (auto& t : threads) t.join();
            fprintf(stderr, "FATAL batch %d aborted\n", batch);
            return 1;
        }

        // ---- per-batch statistics ----
        std::vector<double> latency, completion, interval, blocking;
        std::vector<double> service[3], queue_wait[3];
        size_t finished = 0;
        uint64_t first_done = 0, last_done = 0;
        bool missing = false;
        for (const auto& job : jobs) {
            if (job.done_ns == 0) { missing = true; continue; }
            ++finished;
            if (!first_done || job.done_ns < first_done) first_done = job.done_ns;
            if (job.done_ns > last_done) last_done = job.done_ns;
            latency.push_back((double)(job.done_ns - job.submit_begin_ns) / 1e6);
            completion.push_back((double)(job.done_ns - first_submit) / 1e6);
            blocking.push_back((double)(job.enqueued_ns - job.submit_begin_ns) / 1e6);
            for (int i = 0; i < 3; ++i) {
                service[i].push_back((double)(job.times[i].get_end - job.times[i].prep_begin) / 1e6);
                // Queue wait = dequeue - the moment the job was actually enqueued.
                queue_wait[i].push_back((double)(job.dequeue_ns[i] - job.enqueue_ns[i]) / 1e6);
            }
        }
        std::sort(completion.begin(), completion.end());
        for (size_t i = 1; i < completion.size(); ++i) interval.push_back(completion[i] - completion[i - 1]);
        const double wall_ms = (double)(last_done - first_submit) / 1e6;
        batch_wall.push_back(wall_ms);

        printf("{\"batch\":%d,\"mode\":\"%s\",\"tasks\":%zu,\"finished\":%zu,"
               "\"queue_cap\":%d,\"max_concurrent\":%d,\"pace_ms\":%.1f,"
               "\"wall_ms\":%.1f,\"throughput\":%.3f,\"first_ms\":%.1f,"
               "\"latency_p50\":%.1f,\"latency_p95\":%.1f,"
               "\"block_p50\":%.2f,\"block_max\":%.2f,"
               "\"interval_p50\":%.1f,\"interval_p95\":%.1f,"
               "\"completion_p50\":%.1f,\"completion_p95\":%.1f,"
               "\"vision_init_ms\":%.0f,\"ddr_mean\":%d,\"ddr_max\":%d,\"npu_raw\":%d,"
               "\"service_p50\":[%.1f,%.1f,%.1f],\"queue_p50\":[%.1f,%.1f,%.1f],"
               "\"rss_kb\":%ld,\"fds\":%d}\n",
               batch, opt.mode.c_str(), jobs.size(), finished, opt.queue_cap,
               opt.max_concurrent, opt.pace_ms, wall_ms, (double)jobs.size() / (wall_ms / 1000.0),
               (double)(first_done - first_submit) / 1e6,
               percentile(latency, 0.5), percentile(latency, 0.95),
               percentile(blocking, 0.5), blocking.empty() ? 0.0 : *std::max_element(blocking.begin(), blocking.end()),
               percentile(interval, 0.5), percentile(interval, 0.95),
               percentile(completion, 0.5), percentile(completion, 0.95),
               vision_init_ms, mean_of(sampler.ddr), max_of(sampler.ddr), mean_of(sampler.npu),
               percentile(service[0], 0.5), percentile(service[1], 0.5), percentile(service[2], 0.5),
               percentile(queue_wait[0], 0.5), percentile(queue_wait[1], 0.5), percentile(queue_wait[2], 0.5),
               rss_kb(), fd_count());
        fflush(stdout);
        (void)batch_start; (void)batch_end; (void)ready_ns; (void)submitted;

        if (missing) {
            fprintf(stderr, "FATAL batch %d missing tasks\n", batch);
            return 3;
        }
        if (batch == 0 && !opt.timeline.empty()) {
            FILE* f = fopen(opt.timeline.c_str(), "w");
            if (f) {
                fprintf(f, "task\timage\tpass\tsubmit\tenqueued\tdone"
                           "\te0\td0\tpb0\trb0\tre0\tge0\te1\td1\tpb1\trb1\tre1\tge1"
                           "\te2\td2\tpb2\trb2\tre2\tge2\n");
                for (const auto& job : jobs) {
                    fprintf(f, "%llu\t%s\t%d\t%llu\t%llu\t%llu",
                            (unsigned long long)job.task_id, job.image_id.c_str(), job.pass,
                            (unsigned long long)job.submit_begin_ns,
                            (unsigned long long)job.enqueued_ns,
                            (unsigned long long)job.done_ns);
                    for (int i = 0; i < 3; ++i)
                        fprintf(f, "\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu",
                                (unsigned long long)job.enqueue_ns[i],
                                (unsigned long long)job.dequeue_ns[i],
                                (unsigned long long)job.times[i].prep_begin,
                                (unsigned long long)job.times[i].run_begin,
                                (unsigned long long)job.times[i].run_end,
                                (unsigned long long)job.times[i].get_end);
                    fprintf(f, "\n");
                }
                fclose(f);
            }
        }
        if (batch == 0 && !opt.tokens.empty()) {
            for (const auto& job : jobs) {
                char path[512];
                snprintf(path, sizeof(path), "%s/%s_p%d_t%llu.f32", opt.tokens.c_str(),
                         job.image_id.c_str(), job.pass, (unsigned long long)job.task_id);
                gp::write_file(path, job.tokens.data(), job.tokens.size() * sizeof(float));
            }
        }
    }

    // ---- optional LLM phase: release vision, load Qwen once, answer queries ----
    if (!opt.queries.empty()) {
        const uint64_t llm_begin = gp::now_ns();
        auto abort_all = [&]() {
            queue0.close(); queue1.close(); queue2.close();
            for (auto& t : threads) t.join();
            return 1;
        };
        for (auto& stage : stages) stage.release();
        for (auto& w : workers) w.stage.release();
        const double release_ms = (double)(gp::now_ns() - llm_begin) / 1e6;

        gp::LlmSession llm;
        gp::Timer qwen_timer;
        if (!llm.init(opt.llm)) {
            fprintf(stderr, "FATAL llm init\n");
            queue0.close(); queue1.close(); queue2.close();
            for (auto& t : threads) t.join();
            return 1;
        }
        const double qwen_init_ms = qwen_timer.ms();

        gp::Tokenizer tokenizer;
        if (!tokenizer.load(opt.assets + "/vocab.bin", opt.assets + "/merges.bin")) return abort_all();
        gp::EmbeddingTable embed;
        if (!embed.open(opt.assets + "/embed.bf16", 152670, 2560)) return abort_all();

        struct QueryRow { std::string id, image, query; int w = 0, h = 0; };
        std::vector<QueryRow> rows;
        {
            auto data = gp::read_file(opt.queries);
            std::string text(data.begin(), data.end());
            size_t start = 0;
            while (start < text.size()) {
                size_t end = text.find('\n', start);
                std::string line = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
                start = end == std::string::npos ? text.size() : end + 1;
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                auto parts = split(line, '\t');
                if (parts.size() < 5) continue;
                rows.push_back({parts[0], parts[1], parts[2], atoi(parts[3].c_str()), atoi(parts[4].c_str())});
            }
        }
        int query_count = 0, query_valid = 0;
        const uint64_t query_begin = gp::now_ns();
        for (const auto& row : rows) {
            int index = -1;
            for (size_t i = 0; i < images.size(); ++i)
                if (images[i].id == row.image) { index = (int)i; break; }
            if (index < 0) continue;
            const Job* job = nullptr;
            for (const auto& candidate : jobs)
                if (candidate.pass == 0 && candidate.image_index == index) { job = &candidate; break; }
            if (!job || job->tokens.size() != 256u * 2560u) continue;

            gp::Prompt prompt = gp::build_prompt(tokenizer, row.query);
            std::vector<float> embeds((size_t)prompt.ids.size() * 2560);
            for (size_t k = 0; k < prompt.ids.size(); ++k) {
                float* dst = embeds.data() + k * 2560;
                if ((int)k >= prompt.head_tokens && (int)k < prompt.head_tokens + prompt.image_tokens) {
                    memcpy(dst, job->tokens.data() + ((size_t)k - prompt.head_tokens) * 2560, 2560 * 4);
                } else if (!embed.row(prompt.ids[k], dst)) {
                    return abort_all();
                }
            }
            auto result = llm.run(embeds.data(), (int)prompt.ids.size());
            gp::Box box = gp::decode_box(result.tokens, row.w, row.h);
            ++query_count;
            query_valid += box.valid ? 1 : 0;
            std::string ids;
            for (size_t k = 0; k < result.tokens.size(); ++k)
                ids += (k ? "," : "") + std::to_string(result.tokens[k]);
            printf("{\"mode\":\"query\",\"id\":\"%s\",\"image\":\"%s\",\"llm_ms\":%.1f,"
                   "\"first_ms\":%.1f,\"tokens\":[%s],\"valid\":%s,\"none\":%s,"
                   "\"box\":[%.2f,%.2f,%.2f,%.2f]}\n",
                   row.id.c_str(), row.image.c_str(), result.run_ms, result.first_token_ms,
                   ids.c_str(), box.valid ? "true" : "false", box.none ? "true" : "false",
                   box.x0, box.y0, box.x1, box.y1);
            fflush(stdout);
        }
        const double query_ms = (double)(gp::now_ns() - query_begin) / 1e6;
        const double total_ms = (double)(gp::now_ns() - ready_ns) / 1e6;
        printf("{\"mode\":\"full\",\"vision_mode\":\"%s\",\"vision_wall_ms\":%.1f,"
               "\"release_ms\":%.1f,\"qwen_init_ms\":%.0f,\"query_loop_ms\":%.1f,"
               "\"queries\":%d,\"valid\":%d,\"total_from_ready_ms\":%.1f}\n",
               opt.mode.c_str(), batch_wall.empty() ? 0.0 : batch_wall.front(), release_ms,
               qwen_init_ms, query_ms, query_count, query_valid, total_ms);
    }

    if (opt.mode == "c") {
        queue0.close(); queue1.close(); queue2.close();
        for (auto& t : threads) t.join();
    }
    return 0;
}
