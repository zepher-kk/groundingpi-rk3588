// GroundingPI 3s student v1: trained TinyViT+projector (RKNN) -> merged Qwen3-1.7B (RKLLM).
//
// Protocol gpi-3s-protocol-v1:
//   prompt = <im_start>system\n...<im_end>\n<im_start>user\n
//            "Locate the target referred to by the following description: " Q "."
//            <vision_start> 144*<image_pad> <vision_end><im_end>\n<im_start>assistant\n
//            <object_ref_start> Q <object_ref_end><box_start>
//   answer (trained) = 4 coords [151670,152669] + <box_end> + <im_end>, but the trained
//   model actually emits C C C C E in free generation; that is reported, not repaired.
//
// Parsing is shared by every mode and by the offline Python re-scoring:
//   status = box_strict | box_compact | none_strict | none_compact | unexpected_token |
//            incomplete | bad_prefix | trailing_after_end | empty
//   valid  = deployment rule (box_compact/box_strict/none_*), strict = original protocol
//   historical_extractable = old rule "first 4 tokens coords and last token <im_end>"
//
// Modes:
//   ids   <query>                      prompt token ids (parity with the Python path)
//   pre   <jpg> <out.f32>              board preprocess pixels (PIL-equivalent)
//   vraw  <px.f32> <out.f32>           vision RKNN only
//   emb   <embeds.f32> <n>             L1: precomputed prompt embeddings -> LLM
//   pix   <px.f32> <query>             L2: pixels -> vision -> LLM
//   one   <jpg> <query>                L3: JPEG + text, prints parse + timings
//   bench <manifest.tsv> <out.tsv>     64-sample run, one row per submitted request
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "embed.h"
#include "llm.h"
#include "tokenizer.h"
#include "vision.h"

namespace {

constexpr int kDim = 2048;
constexpr int kSlots = 144;
constexpr int kCanvas = 384;
constexpr int32_t kImStart = 151644, kImEnd = 151645;
constexpr int32_t kObjectRefStart = 151646, kObjectRefEnd = 151647;
constexpr int32_t kBoxStart = 151648, kBoxEnd = 151649;
constexpr int32_t kVisionStart = 151652, kVisionEnd = 151653, kImagePad = 151655;
constexpr int32_t kNoneBox = 151669, kCoordBase = 151670, kCoordLast = 152669;
constexpr int kMaxNew = 8;

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
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

bool is_coord(int32_t t) { return t >= kCoordBase && t <= kCoordLast; }

struct Letterbox {
    int rw, rh, left, top;
    float sx, sy;
};

Letterbox letterbox_of(int w, int h) {
    const float scale = std::min((float)kCanvas / w, (float)kCanvas / h);
    Letterbox lb;
    lb.rw = std::max(1, (int)std::floor(w * scale + 0.5f));
    lb.rh = std::max(1, (int)std::floor(h * scale + 0.5f));
    lb.left = (kCanvas - lb.rw) / 2;
    lb.top = (kCanvas - lb.rh) / 2;
    lb.sx = (float)lb.rw / w;
    lb.sy = (float)lb.rh / h;
    return lb;
}

// ---- PIL-compatible bicubic (a=-0.5, antialiased) -------------------------------------
double pil_bicubic(double x) {
    x = std::fabs(x);
    if (x < 1.0) return (1.5 * x - 2.5) * x * x + 1.0;
    if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    return 0.0;
}

std::vector<double> pil_weights(double center, double support, double ss, int in_size,
                                int* xmin_out, int* xmax_out) {
    int xmin = (int)std::floor(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = (int)std::floor(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    std::vector<double> w;
    w.reserve(xmax - xmin);
    double total = 0;
    for (int x = xmin; x < xmax; ++x) {
        const double v = pil_bicubic((x - center + 0.5) * ss);
        w.push_back(v);
        total += v;
    }
    if (total != 0) for (double& v : w) v /= total;
    *xmin_out = xmin;
    *xmax_out = xmax;
    return w;
}

cv::Mat resize_pil_bicubic(const cv::Mat& src, int out_w, int out_h) {
    const int in_w = src.cols, in_h = src.rows;
    const double sx = (double)in_w / out_w, sy = (double)in_h / out_h;
    const double fsx = sx > 1.0 ? sx : 1.0, fsy = sy > 1.0 ? sy : 1.0;
    const double support_x = 2.0 * fsx, support_y = 2.0 * fsy;
    const double ssx = 1.0 / fsx, ssy = 1.0 / fsy;

    cv::Mat tmp(in_h, out_w, CV_8UC3);
    std::vector<int> xmins(out_w), xmaxs(out_w);
    std::vector<std::vector<double>> wxs(out_w);
    for (int x = 0; x < out_w; ++x)
        wxs[x] = pil_weights((x + 0.5) * sx, support_x, ssx, in_w, &xmins[x], &xmaxs[x]);
    for (int y = 0; y < in_h; ++y) {
        const uint8_t* row = src.ptr<uint8_t>(y);
        uint8_t* out = tmp.ptr<uint8_t>(y);
        for (int x = 0; x < out_w; ++x) {
            const auto& w = wxs[x];
            double acc[3] = {0, 0, 0};
            for (int i = 0; i < (int)w.size(); ++i) {
                const uint8_t* p = row + (xmins[x] + i) * 3;
                acc[0] += w[i] * p[0];
                acc[1] += w[i] * p[1];
                acc[2] += w[i] * p[2];
            }
            for (int c = 0; c < 3; ++c) {
                double v = std::floor(acc[c] + 0.5);
                out[x * 3 + c] = (uint8_t)std::min(255.0, std::max(0.0, v));
            }
        }
    }

    cv::Mat dst(out_h, out_w, CV_8UC3);
    for (int y = 0; y < out_h; ++y) {
        int ymin, ymax;
        auto wy = pil_weights((y + 0.5) * sy, support_y, ssy, in_h, &ymin, &ymax);
        uint8_t* out = dst.ptr<uint8_t>(y);
        for (int x = 0; x < out_w; ++x) {
            double acc[3] = {0, 0, 0};
            for (int i = 0; i < (int)wy.size(); ++i) {
                const uint8_t* p = tmp.ptr<uint8_t>(ymin + i) + x * 3;
                acc[0] += wy[i] * p[0];
                acc[1] += wy[i] * p[1];
                acc[2] += wy[i] * p[2];
            }
            for (int c = 0; c < 3; ++c) {
                double v = std::floor(acc[c] + 0.5);
                out[x * 3 + c] = (uint8_t)std::min(255.0, std::max(0.0, v));
            }
        }
    }
    return dst;
}

std::vector<float> preprocess(const cv::Mat& bgr) {
    const int w = bgr.cols, h = bgr.rows;
    const Letterbox lb = letterbox_of(w, h);
    cv::Mat rgb, resized;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    resized = resize_pil_bicubic(rgb, lb.rw, lb.rh);
    cv::Mat canvas(kCanvas, kCanvas, CV_8UC3, cv::Scalar(127, 127, 127));
    resized.copyTo(canvas(cv::Rect(lb.left, lb.top, lb.rw, lb.rh)));

    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float stdv[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> out(3 * kCanvas * kCanvas);
    for (int y = 0; y < kCanvas; ++y) {
        const uint8_t* row = canvas.ptr<uint8_t>(y);
        for (int x = 0; x < kCanvas; ++x) {
            for (int c = 0; c < 3; ++c) {
                out[(size_t)c * kCanvas * kCanvas + y * kCanvas + x] =
                    (row[x * 3 + c] / 255.0f - mean[c]) / stdv[c];
            }
        }
    }
    return out;
}

// ---- shared answer parser ------------------------------------------------------------
struct Parse {
    std::string status = "empty";
    bool valid = false;                 // deployment rule
    bool strict = false;                // original trained protocol (needs <box_end>)
    bool none = false;
    bool has_box_end = false;
    bool ended = false;                 // sequence ends with <im_end>
    bool historical_extractable = false;// old rule: first 4 coords + last token <im_end>
    std::string failure;                // why invalid
    std::vector<int> grid;              // 4 values in [0,999] when valid
    bool box_ok = false;                // geometry valid (finite, x1>x0, y1>y0)
    float box[4] = {0, 0, 0, 0};        // original-image coords
    bool out_of_image = false;
};

Parse parse_answer(const std::vector<int32_t>& t) {
    Parse p;
    if (t.empty()) { p.failure = "no_tokens"; return p; }
    p.has_box_end = std::find(t.begin(), t.end(), kBoxEnd) != t.end();
    p.ended = t.back() == kImEnd;

    // last token must be <im_end>; anything after an <im_end> is a protocol violation
    if (!p.ended) {
        const size_t first_end = std::find(t.begin(), t.end(), kImEnd) - t.begin();
        if (first_end < t.size()) { p.status = "trailing_after_end"; p.failure = "tokens_after_im_end"; return p; }
    } else {
        const size_t first_end = std::find(t.begin(), t.end(), kImEnd) - t.begin();
        if (first_end + 1 != t.size()) { p.status = "trailing_after_end"; p.failure = "tokens_after_im_end"; return p; }
    }

    const size_t n = p.ended ? t.size() - 1 : t.size();   // body without trailing <im_end>
    const bool four_coords = n >= 4 && is_coord(t[0]) && is_coord(t[1]) && is_coord(t[2]) && is_coord(t[3]);
    p.historical_extractable = four_coords && p.ended;

    if (n >= 1 && t[0] == kNoneBox) {
        if (!p.ended) { p.status = "incomplete"; p.failure = "no_im_end"; return p; }
        if (n == 1) { p.status = "none_compact"; p.valid = true; p.none = true; return p; }
        if (n == 2 && t[1] == kBoxEnd) { p.status = "none_strict"; p.valid = true; p.strict = true; p.none = true; return p; }
        p.status = "unexpected_token"; p.failure = "tokens_after_none"; return p;
    }

    if (!four_coords) {
        if (n < 4) { p.status = p.ended ? "bad_prefix" : "incomplete"; p.failure = "fewer_than_4_tokens"; return p; }
        p.status = "bad_prefix"; p.failure = "first_four_not_coords"; return p;
    }

    if (!p.ended) { p.status = "incomplete"; p.failure = "no_im_end"; return p; }
    if (n == 4) { p.status = "box_compact"; p.valid = true; return p; }
    if (n == 5 && t[4] == kBoxEnd) { p.status = "box_strict"; p.valid = true; p.strict = true; return p; }
    p.status = "unexpected_token"; p.failure = "tokens_between_coords_and_end"; return p;
}

void fill_box(Parse& p, const std::vector<int32_t>& t, const Letterbox& lb, int w, int h) {
    if (!p.valid || p.none) return;
    float v[4];
    for (int i = 0; i < 4; ++i) v[i] = (float)(t[i] - kCoordBase) / 999.0f * kCanvas;
    v[0] = (v[0] - lb.left) / lb.sx;
    v[1] = (v[1] - lb.top) / lb.sy;
    v[2] = (v[2] - lb.left) / lb.sx;
    v[3] = (v[3] - lb.top) / lb.sy;
    for (int i = 0; i < 4; ++i) {
        if (!std::isfinite(v[i])) { p.valid = false; p.status = "degenerate_box"; p.failure = "non_finite"; return; }
    }
    p.grid.assign(4, 0);
    for (int i = 0; i < 4; ++i) p.grid[i] = (int)(t[i] - kCoordBase);
    if (v[2] <= v[0] || v[3] <= v[1]) {
        p.valid = false; p.status = "degenerate_box"; p.failure = "non_positive_area";
        for (int i = 0; i < 4; ++i) p.box[i] = v[i];
        return;
    }
    for (int i = 0; i < 4; ++i) p.box[i] = v[i];
    p.box_ok = true;
    p.out_of_image = (v[0] < 0 || v[1] < 0 || v[2] > w || v[3] > h);
}

// ---- student --------------------------------------------------------------------------
class Student {
public:
    bool init(const std::string& dir) {
        if (!tok_.load(dir + "/vocab.bin", dir + "/merges.bin")) {
            fprintf(stderr, "FATAL tokenizer load failed: %s/vocab.bin, %s/merges.bin\n",
                    dir.c_str(), dir.c_str());
            return false;
        }
        if (!emb_.open(dir + "/embed_tokens_151669x2048.bf16", 151669, kDim)) {
            fprintf(stderr, "FATAL embedding table load failed\n");
            return false;
        }
        const std::string baseline = dir + "/student_qwen3-1.7b-fp16.rkllm";
        const char* override_path = getenv("GP_LLM");
        llm_path_ = (override_path && *override_path) ? std::string(override_path) : baseline;
        identity_ = llm_path_ == baseline ? "fp16-baseline" : "override";
        FILE* probe = fopen(llm_path_.c_str(), "rb");
        if (!probe) {
            fprintf(stderr, "FATAL llm file missing: %s\n", llm_path_.c_str());
            return false;
        }
        long bytes = 0;
        if (fseek(probe, 0, SEEK_END) == 0) bytes = ftell(probe);
        fclose(probe);
        if (bytes <= 0) {
            fprintf(stderr, "FATAL llm file unreadable: %s\n", llm_path_.c_str());
            return false;
        }
        fprintf(stderr, "[student] llm=%s bytes=%ld identity=%s\n",
                llm_path_.c_str(), bytes, identity_.c_str());
        if (!llm_.init(llm_path_, 512, kMaxNew, 1)) {
            fprintf(stderr, "FATAL llm init failed: %s\n", llm_path_.c_str());
            return false;
        }
        if (!vision_.init(dir + "/student_vision_trained_fp16.rknn", RKNN_NPU_CORE_0)) {
            fprintf(stderr, "FATAL vision init failed\n");
            return false;
        }
        return true;
    }

    std::vector<int32_t> prompt_ids(const std::string& query) {
        std::vector<int32_t> ids = {kImStart, 8948, 198, 2610, 525, 264, 10950, 17847, 13,
                                    kImEnd, 198, kImStart, 872, 198};
        auto body = tok_.encode("Locate the target referred to by the following description: " + query + ".");
        ids.insert(ids.end(), body.begin(), body.end());
        ids.push_back(kVisionStart);
        image_start_ = (int)ids.size();
        ids.insert(ids.end(), kSlots, kImagePad);
        for (int32_t t : {kVisionEnd, kImEnd, 198, kImStart, 77091, 198, kObjectRefStart}) ids.push_back(t);
        auto q = tok_.encode(query);
        ids.insert(ids.end(), q.begin(), q.end());
        ids.push_back(kObjectRefEnd);
        ids.push_back(kBoxStart);
        return ids;
    }

    struct RunTimes {
        double pre = 0, vision = 0, assemble = 0, llm_total = 0, ttft = 0;
    };

    bool run(const std::vector<float>& pixels, const std::string& query,
             std::vector<int32_t>* out_ids, RunTimes* times, std::string* err) {
        std::vector<int32_t> ids = prompt_ids(query);
        if ((int)ids.size() + kMaxNew > 512) { *err = "prompt_too_long"; return false; }
        std::vector<float> vis;
        double vision_ms = 0;
        const double t0 = now_ms();
        if (!vision_.run(pixels.data(), pixels.size(), vis, &vision_ms)) { *err = "vision_run_failed"; return false; }
        if ((int)vis.size() != kSlots * kDim) { *err = "vision_output_shape"; return false; }
        times->vision = vision_ms;
        std::vector<float> embeds(ids.size() * kDim);
        for (size_t i = 0; i < ids.size(); ++i) {
            if ((int)i >= image_start_ && (int)i < image_start_ + kSlots) {
                const int slot = (int)i - image_start_;
                memcpy(&embeds[i * kDim], &vis[(size_t)slot * kDim], kDim * 4);
            } else if (!emb_.row(ids[i], &embeds[i * kDim])) {
                *err = "embed_row_oob_" + std::to_string(ids[i]);
                return false;
            }
        }
        times->assemble = now_ms() - t0 - vision_ms;
        auto res = llm_.run(embeds.data(), (int)ids.size());
        if (res.status != 0 || res.error) { *err = "llm_run_failed_" + std::to_string(res.status); return false; }
        times->llm_total = res.run_ms;
        times->ttft = res.first_token_ms;
        *out_ids = res.tokens;
        return true;
    }

    bool run_embeds(const std::string& path, int n, std::vector<int32_t>* out, std::string* err) {
        auto raw = gp::read_file(path);
        if (raw.size() != (size_t)n * kDim * 4) { *err = "embed_file_size"; return false; }
        std::vector<float> embeds((float*)raw.data(), (float*)raw.data() + (size_t)n * kDim);
        auto res = llm_.run(embeds.data(), n);
        if (res.status != 0) { *err = "llm_run_failed"; return false; }
        *out = res.tokens;
        return true;
    }

    void shutdown() { vision_.release(); llm_.shutdown(); }
    double llm_init_ms() const { return llm_.init_ms(); }
    const std::string& identity() const { return identity_; }
    const std::string& llm_path() const { return llm_path_; }
    gp::VisionStage& vision() { return vision_; }

private:
    gp::Tokenizer tok_;
    gp::EmbeddingTable emb_;
    gp::LlmSession llm_;
    gp::VisionStage vision_;
    int image_start_ = 0;
    std::string llm_path_;
    std::string identity_;
};

void print_ids(const std::vector<int32_t>& ids) {
    printf("IDS n=%zu", ids.size());
    for (auto t : ids) printf(" %d", t);
    printf("\n");
}

void print_parse(const Parse& p, const std::vector<int32_t>& ids) {
    printf("PARSE status=%s valid=%d strict=%d none=%d box_end=%d ended=%d historic=%d\n",
           p.status.c_str(), (int)p.valid, (int)p.strict, (int)p.none, (int)p.has_box_end,
           (int)p.ended, (int)p.historical_extractable);
    printf("BOX valid=%d none=%d box_ok=%d out_of_image=%d xyxy=%.2f,%.2f,%.2f,%.2f\n",
           (int)p.valid, (int)p.none, (int)p.box_ok, (int)p.out_of_image,
           p.box[0], p.box[1], p.box[2], p.box[3]);
    if (!p.valid) printf("FAIL reason=%s\n", p.failure.c_str());
    print_ids(ids);
}

std::vector<std::string> split_tsv(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cur;
    while (std::getline(ss, cur, '\t')) out.push_back(cur);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <asset-dir> ids|pre|vraw|emb|pix|one|bench ...\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const std::string mode = argv[2];

    if (mode == "ids") {
        gp::Tokenizer tok;
        if (!tok.load(dir + "/vocab.bin", dir + "/merges.bin")) return 1;
        const std::string q = argv[3];
        std::vector<int32_t> ids = {kImStart, 8948, 198, 2610, 525, 264, 10950, 17847, 13,
                                    kImEnd, 198, kImStart, 872, 198};
        auto body = tok.encode("Locate the target referred to by the following description: " + q + ".");
        ids.insert(ids.end(), body.begin(), body.end());
        ids.push_back(kVisionStart);
        ids.insert(ids.end(), kSlots, kImagePad);
        for (int32_t t : {kVisionEnd, kImEnd, 198, kImStart, 77091, 198, kObjectRefStart}) ids.push_back(t);
        auto qq = tok.encode(q);
        ids.insert(ids.end(), qq.begin(), qq.end());
        ids.push_back(kObjectRefEnd);
        ids.push_back(kBoxStart);
        print_ids(ids);
        return 0;
    }

    if (mode == "ptest") {
        // parser unit check: ptest <comma-separated ids> [w] [h]
        std::vector<int32_t> ids;
        std::stringstream ss(argv[3]);
        std::string part;
        while (std::getline(ss, part, ',')) {
            if (part.empty()) continue;
            ids.push_back(atoi(part.c_str()));
        }
        const int w = argc > 4 ? atoi(argv[4]) : 640;
        const int h = argc > 5 ? atoi(argv[5]) : 480;
        Parse p = parse_answer(ids);
        fill_box(p, ids, letterbox_of(w, h), w, h);
        printf("PTEST status=%s valid=%d strict=%d none=%d box_end=%d ended=%d historic=%d box_ok=%d "
               "out_of_image=%d xyxy=%.4f,%.4f,%.4f,%.4f failure=%s\n",
               p.status.c_str(), (int)p.valid, (int)p.strict, (int)p.none, (int)p.has_box_end,
               (int)p.ended, (int)p.historical_extractable, (int)p.box_ok, (int)p.out_of_image,
               p.box[0], p.box[1], p.box[2], p.box[3], p.failure.c_str());
        return 0;
    }

    if (mode == "pre") {        auto buf = gp::read_file(argv[3]);
        if (buf.empty()) { fprintf(stderr, "FATAL cannot read %s\n", argv[3]); return 1; }
        cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (img.empty()) { fprintf(stderr, "FATAL jpeg decode failed: %s\n", argv[3]); return 1; }
        auto pixels = preprocess(img);
        FILE* f = fopen(argv[4], "wb");
        if (!f) { fprintf(stderr, "FATAL cannot write %s\n", argv[4]); return 1; }
        fwrite(pixels.data(), 4, pixels.size(), f);
        fclose(f);
        printf("PRE_OK %dx%d n=%zu\n", img.cols, img.rows, pixels.size());
        return 0;
    }

    if (mode == "vraw") {
        auto raw = gp::read_file(argv[3]);
        if (raw.size() != (size_t)3 * kCanvas * kCanvas * 4) {
            fprintf(stderr, "FATAL pixel file size %zu\n", raw.size());
            return 1;
        }
        gp::VisionStage vision;
        if (!vision.init(dir + "/student_vision_trained_fp16.rknn", RKNN_NPU_CORE_0)) return 1;
        std::vector<float> px((float*)raw.data(), (float*)raw.data() + raw.size() / 4);
        std::vector<float> out;
        double ms = 0;
        if (!vision.run(px.data(), px.size(), out, &ms)) { fprintf(stderr, "FATAL vision run\n"); return 1; }
        FILE* f = fopen(argv[4], "wb");
        fwrite(out.data(), 4, out.size(), f);
        fclose(f);
        printf("VRAW_OK n=%zu ms=%.1f\n", out.size(), ms);
        vision.release();
        return 0;
    }

    Student s;
    if (!s.init(dir)) return 3;
    fprintf(stderr, "INIT ok identity=%s llm_init_ms=%.0f rss=%ldKB\n",
            s.identity().c_str(), s.llm_init_ms(), rss_kb());
    fflush(stdout);

    if (mode == "emb") {
        std::vector<int32_t> ids;
        std::string err;
        const double t = now_ms();
        if (!s.run_embeds(argv[3], atoi(argv[4]), &ids, &err)) {
            printf("EMB_FAIL %s\n", err.c_str());
            s.shutdown();
            return 4;
        }
        Parse p = parse_answer(ids);
        printf("EMB_OK ms=%.1f\n", now_ms() - t);
        print_parse(p, ids);
        s.shutdown();
        return p.valid ? 0 : 5;
    }

    if (mode == "pix") {
        auto raw = gp::read_file(argv[3]);
        if (raw.size() != (size_t)3 * kCanvas * kCanvas * 4) { fprintf(stderr, "FATAL pixel size\n"); return 1; }
        std::vector<float> px((float*)raw.data(), (float*)raw.data() + raw.size() / 4);
        std::vector<int32_t> ids;
        Student::RunTimes times;
        std::string err;
        if (!s.run(px, argv[4], &ids, &times, &err)) { printf("PIX_FAIL %s\n", err.c_str()); return 4; }
        Parse p = parse_answer(ids);
        printf("PIX_TIME vision=%.1f llm_total=%.1f ttft=%.1f\n", times.vision, times.llm_total, times.ttft);
        print_parse(p, ids);
        s.shutdown();
        return p.valid ? 0 : 5;
    }

    if (mode == "one") {
        if (argc < 5) { fprintf(stderr, "usage: one <jpg> <query>\n"); return 2; }
        const std::string jpg = argv[3];
        const std::string query = argv[4];
        const double t_start = now_ms();
        auto buf = gp::read_file(jpg);
        if (buf.empty()) { fprintf(stderr, "FATAL cannot read %s\n", jpg.c_str()); return 1; }
        const double t_read = now_ms();
        cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (img.empty()) { fprintf(stderr, "FATAL jpeg decode failed\n"); return 1; }
        const double t_dec = now_ms();
        auto pixels = preprocess(img);
        const double t_pre = now_ms();
        std::vector<int32_t> ids;
        Student::RunTimes times;
        std::string err;
        if (!s.run(pixels, query, &ids, &times, &err)) { printf("RUN_FAIL %s\n", err.c_str()); s.shutdown(); return 4; }
        const double t_llm = now_ms();
        const Letterbox lb = letterbox_of(img.cols, img.rows);
        Parse p = parse_answer(ids);
        fill_box(p, ids, lb, img.cols, img.rows);
        const double total = now_ms() - t_start;   // includes parsing + result construction
        printf("IMAGE %dx%d lb=%d,%d,%d,%d sx=%.4f sy=%.4f\n", img.cols, img.rows,
               lb.rw, lb.rh, lb.left, lb.top, lb.sx, lb.sy);
        print_parse(p, ids);
        printf("TIME read=%.1f decode=%.1f pre=%.1f vision=%.1f llm_total=%.1f ttft=%.1f parse=%.1f total=%.1f rss=%ldKB\n",
               t_read - t_start, t_dec - t_read, t_pre - t_dec, times.vision, times.llm_total,
               times.ttft, total - (t_llm - t_start), total, rss_kb());
        s.shutdown();
        return p.valid ? 0 : 5;
    }

    if (mode == "bench") {
        if (argc < 5) { fprintf(stderr, "usage: bench <manifest.tsv> <out.tsv>\n"); return 2; }
        const std::string manifest = argv[3];
        const std::string out_path = argv[4];
        std::ifstream in(manifest);
        if (!in) { fprintf(stderr, "FATAL cannot open manifest %s\n", manifest.c_str()); return 1; }
        std::ofstream out(out_path);
        if (!out) { fprintf(stderr, "FATAL cannot write %s\n", out_path.c_str()); return 1; }
        out << "sample_id\tstatus\tvalid\tstrict\tnone\tbox_end\tended\thistoric\tout_of_image"
               "\tx0\ty0\tx1\ty1\tms_decode\tms_pre\tms_vision\tms_llm_total\tms_ttft\tms_total"
               "\tntok\ttokens\tfailure\n";
        std::string line;
        int expected = 0, submitted = 0, completed = 0, runtime_failed = 0, parse_invalid = 0;
        int malformed = 0;
        std::vector<std::string> seen;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            ++expected;
            auto f = split_tsv(line);
            if (f.size() < 10) {
                ++malformed;
                out << "MALFORMED_LINE_" << expected << "\tmalformed_input\t0\t0\t0\t0\t0\t0\t0"
                       "\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t\tmanifest_needs_10_fields\n";
                out.flush();
                continue;
            }
            const std::string sid = f[0], jpg = f[1], query = f[2];
            const int want_w = atoi(f[3].c_str()), want_h = atoi(f[4].c_str());
            if (std::find(seen.begin(), seen.end(), sid) != seen.end()) {
                ++malformed;
                out << sid << "\tduplicate_id\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t\tduplicate_sample_id\n";
                out.flush();
                continue;
            }
            seen.push_back(sid);

            auto buf = gp::read_file(jpg);
            if (buf.empty()) {
                ++submitted; ++runtime_failed;
                out << sid << "\truntime_failed\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t\timage_read_failed\n";
                out.flush();
                continue;
            }
            const double t_start = now_ms();
            cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
            if (img.empty()) {
                ++submitted; ++runtime_failed;
                out << sid << "\truntime_failed\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t\tjpeg_decode_failed\n";
                out.flush();
                continue;
            }
            if (img.cols != want_w || img.rows != want_h) {
                ++submitted; ++runtime_failed;
                out << sid << "\truntime_failed\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t\t"
                    << "image_size_mismatch_" << img.cols << "x" << img.rows << "_expected_"
                    << want_w << "x" << want_h << "\n";
                out.flush();
                continue;
            }
            const double t_dec = now_ms();
            auto pixels = preprocess(img);
            const double t_pre = now_ms();
            std::vector<int32_t> ids;
            Student::RunTimes times;
            std::string err;
            if (!s.run(pixels, query, &ids, &times, &err)) {
                ++submitted; ++runtime_failed;
                out << sid << "\truntime_failed\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t" << (t_dec - t_start)
                    << "\t" << (t_pre - t_dec) << "\t" << times.vision << "\t" << times.llm_total
                    << "\t" << times.ttft << "\t" << (now_ms() - t_start) << "\t0\t\t" << err << "\n";
                out.flush();
                continue;
            }
            const Letterbox lb = letterbox_of(img.cols, img.rows);
            Parse p = parse_answer(ids);
            fill_box(p, ids, lb, img.cols, img.rows);
            const double total = now_ms() - t_start;    // includes parsing + result construction
            ++submitted;
            ++completed;
            if (!p.valid) ++parse_invalid;
            out << sid << "\t" << p.status << "\t" << (int)p.valid << "\t" << (int)p.strict << "\t"
                << (int)p.none << "\t" << (int)p.has_box_end << "\t" << (int)p.ended << "\t"
                << (int)p.historical_extractable << "\t" << (int)p.out_of_image << "\t"
                << p.box[0] << "\t" << p.box[1] << "\t" << p.box[2] << "\t" << p.box[3] << "\t"
                << (t_dec - t_start) << "\t" << (t_pre - t_dec) << "\t" << times.vision << "\t"
                << times.llm_total << "\t" << times.ttft << "\t" << total << "\t" << ids.size() << "\t";
            for (size_t i = 0; i < ids.size(); ++i) out << (i ? "," : "") << ids[i];
            out << "\t" << p.failure << "\n";
            out.flush();
            if (completed % 8 == 0) { printf("BENCH %d/%d rss=%ldKB\n", completed, expected, rss_kb()); fflush(stdout); }
        }
        s.shutdown();
        printf("BENCH_DONE expected=%d submitted=%d completed=%d runtime_failed=%d parse_invalid=%d malformed=%d rss=%ldKB\n",
               expected, submitted, completed, runtime_failed, parse_invalid, malformed, rss_kb());
        if (runtime_failed > 0 || malformed > 0) {
            fprintf(stderr, "BENCH_FAILED runtime_failed=%d malformed=%d\n", runtime_failed, malformed);
            return 6;
        }
        return 0;
    }

    fprintf(stderr, "FATAL unknown mode %s\n", mode.c_str());
    return 2;
}
