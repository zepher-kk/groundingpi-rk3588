// Build prompt embeddings for the language batch probe (vision + text embedding table).
// Output: <dir>/<sid>.f32 (n_tokens x 2048 float32) and <dir>/index.tsv (sid, file, n_tokens)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common.h"
#include "embed.h"
#include "tokenizer.h"
#include "vision.h"

namespace {

constexpr int kDim = 2048, kSlots = 144, kCanvas = 384;
constexpr int32_t kImStart = 151644, kImEnd = 151645, kObjectRefStart = 151646, kObjectRefEnd = 151647;
constexpr int32_t kBoxStart = 151648, kVisionStart = 151652, kVisionEnd = 151653, kImagePad = 151655;

double pil_bicubic(double x) {
    x = std::fabs(x);
    if (x < 1.0) return (1.5 * x - 2.5) * x * x + 1.0;
    if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    return 0.0;
}

std::vector<double> weights(double center, double support, double ss, int n, int* lo, int* hi) {
    int a = (int)std::floor(center - support + 0.5);
    if (a < 0) a = 0;
    int b = (int)std::floor(center + support + 0.5);
    if (b > n) b = n;
    std::vector<double> w;
    double tot = 0;
    for (int x = a; x < b; ++x) { double v = pil_bicubic((x - center + 0.5) * ss); w.push_back(v); tot += v; }
    if (tot != 0) for (double& v : w) v /= tot;
    *lo = a; *hi = b;
    return w;
}

cv::Mat resize_pil(const cv::Mat& src, int ow, int oh) {
    const int iw = src.cols, ih = src.rows;
    const double sx = (double)iw / ow, sy = (double)ih / oh;
    const double fx = sx > 1 ? sx : 1, fy = sy > 1 ? sy : 1;
    cv::Mat tmp(ih, ow, CV_8UC3);
    for (int y = 0; y < ih; ++y) {
        uint8_t* o = tmp.ptr<uint8_t>(y);
        const uint8_t* row = src.ptr<uint8_t>(y);
        for (int x = 0; x < ow; ++x) {
            int lo, hi;
            auto w = weights((x + 0.5) * sx, 2.0 * fx, 1.0 / fx, iw, &lo, &hi);
            double acc[3] = {0, 0, 0};
            for (int i = 0; i < (int)w.size(); ++i) {
                const uint8_t* p = row + (lo + i) * 3;
                acc[0] += w[i] * p[0]; acc[1] += w[i] * p[1]; acc[2] += w[i] * p[2];
            }
            for (int c = 0; c < 3; ++c) o[x * 3 + c] = (uint8_t)std::min(255.0, std::max(0.0, std::floor(acc[c] + 0.5)));
        }
    }
    cv::Mat dst(oh, ow, CV_8UC3);
    for (int y = 0; y < oh; ++y) {
        int lo, hi;
        auto w = weights((y + 0.5) * sy, 2.0 * fy, 1.0 / fy, ih, &lo, &hi);
        uint8_t* o = dst.ptr<uint8_t>(y);
        for (int x = 0; x < ow; ++x) {
            double acc[3] = {0, 0, 0};
            for (int i = 0; i < (int)w.size(); ++i) {
                const uint8_t* p = tmp.ptr<uint8_t>(lo + i) + x * 3;
                acc[0] += w[i] * p[0]; acc[1] += w[i] * p[1]; acc[2] += w[i] * p[2];
            }
            for (int c = 0; c < 3; ++c) o[x * 3 + c] = (uint8_t)std::min(255.0, std::max(0.0, std::floor(acc[c] + 0.5)));
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <asset-dir> <manifest.tsv> <out-dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1], manifest = argv[2], outdir = argv[3];
    gp::Tokenizer tok;
    gp::EmbeddingTable emb;
    gp::VisionStage vision;
    if (!tok.load(dir + "/vocab.bin", dir + "/merges.bin")) { fprintf(stderr, "FATAL tokenizer\n"); return 1; }
    if (!emb.open(dir + "/embed_tokens_151669x2048.bf16", 151669, kDim)) { fprintf(stderr, "FATAL embed table\n"); return 1; }
    if (!vision.init(dir + "/student_vision_trained_fp16.rknn", RKNN_NPU_CORE_0)) { fprintf(stderr, "FATAL vision\n"); return 1; }

    std::ifstream in(manifest);
    if (!in) { fprintf(stderr, "FATAL manifest %s\n", manifest.c_str()); return 1; }
    std::ofstream idx(outdir + "/index.tsv");
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::vector<std::string> f;
        std::string p;
        while (std::getline(ss, p, '\t')) f.push_back(p);
        if (f.size() < 3) continue;
        const std::string sid = f[0], jpg = f[1], query = f[2];
        auto buf2 = gp::read_file(jpg);
        if (buf2.empty()) { fprintf(stderr, "FATAL read %s\n", jpg.c_str()); return 1; }
        cv::Mat img = cv::imdecode(buf2, cv::IMREAD_COLOR);
        if (img.empty()) { fprintf(stderr, "FATAL decode %s\n", jpg.c_str()); return 1; }
        auto px = preprocess(img);
        std::vector<float> vis;
        double ms = 0;
        if (!vision.run(px.data(), px.size(), vis, &ms) || (int)vis.size() != kSlots * kDim) {
            fprintf(stderr, "FATAL vision run %s\n", sid.c_str()); return 1;
        }
        // prompt
        std::vector<int32_t> ids = {kImStart, 8948, 198, 2610, 525, 264, 10950, 17847, 13,
                                    kImEnd, 198, kImStart, 872, 198};
        auto body = tok.encode("Locate the target referred to by the following description: " + query + ".");
        ids.insert(ids.end(), body.begin(), body.end());
        ids.push_back(kVisionStart);
        const int img_start = (int)ids.size();
        ids.insert(ids.end(), kSlots, kImagePad);
        for (int32_t t : {kVisionEnd, kImEnd, 198, kImStart, 77091, 198, kObjectRefStart}) ids.push_back(t);
        auto q = tok.encode(query);
        ids.insert(ids.end(), q.begin(), q.end());
        ids.push_back(kObjectRefEnd);
        ids.push_back(kBoxStart);

        std::vector<float> embeds(ids.size() * kDim);
        for (size_t i = 0; i < ids.size(); ++i) {
            if ((int)i >= img_start && (int)i < img_start + kSlots) {
                memcpy(&embeds[i * kDim], &vis[(size_t)((int)i - img_start) * kDim], kDim * 4);
            } else if (!emb.row(ids[i], &embeds[i * kDim])) {
                fprintf(stderr, "FATAL embed row %d\n", ids[i]); return 1;
            }
        }
        const std::string fname = sid + ".f32";
        FILE* fo = fopen((outdir + "/" + fname).c_str(), "wb");
        fwrite(embeds.data(), 4, embeds.size(), fo);
        fclose(fo);
        idx << sid << "\t" << fname << "\t" << ids.size() << "\n";
        idx.flush();
        printf("PREP %s ntok=%zu vision_ms=%.1f\n", sid.c_str(), ids.size(), ms);
        fflush(stdout);
        ++n;
    }
    vision.release();
    printf("PREP_DONE n=%d\n", n);
    return 0;
}
