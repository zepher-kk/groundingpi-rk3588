// Protocol-v1 board probe: run the student's 190-token prompt on the existing
// Qwen3-1.7B + TinyViT runtimes, with the 144 image slots at positions 32..175.
//
// Unlike student_coexist (fixed 17+144+8 layout), this takes the head/tail rows
// from separate .f32 files so any prompt layout can be measured. Rows come from
// the protocol-v1 template; the base embedding table is frozen during training,
// so these are the exact rows the student feeds.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"
#include "llm.h"
#include "npy.h"
#include "vision.h"

namespace {

long rss_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256]; long kb = -1;
    while (fgets(line, sizeof(line), f))
        if (strncmp(line, "VmRSS:", 6) == 0) { sscanf(line + 6, "%ld", &kb); break; }
    fclose(f);
    return kb;
}

constexpr size_t kDim = 2048;

std::vector<float> read_f32(const std::string& path, size_t rows) {
    auto data = gp::read_file(path);
    if (data.size() != rows * kDim * 4) {
        fprintf(stderr, "bad file %s: %zu bytes, expected %zu\n", path.c_str(), data.size(), rows * kDim * 4);
        return {};
    }
    std::vector<float> out(rows * kDim);
    memcpy(out.data(), data.data(), rows * kDim * 4);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = ".";
    std::string proto_dir = dir + "/protocol-v1";
    if (argc > 1) dir = argv[1];
    if (argc > 2) proto_dir = argv[2];

    const std::string vision_model = dir + "/student_vision_fp16.rknn";
    const std::string llm_model = dir + "/qwen3-1.7b-w8a8.rkllm";
    const std::string pixels_path = dir + "/sample_input.npy";
    const std::string head_path = proto_dir + "/head_32x2048.f32";
    const std::string tail_path = proto_dir + "/tail_14x2048.f32";

    const size_t head_rows = 32, image_rows = 144, tail_rows = 14;
    const size_t total = head_rows + image_rows + tail_rows;

    std::vector<float> input;
    std::vector<int> shape;
    if (!gp::load_npy_f32(pixels_path, input, shape)) {
        fprintf(stderr, "cannot load %s\n", pixels_path.c_str());
        return 1;
    }
    auto head = read_f32(head_path, head_rows);
    auto tail = read_f32(tail_path, tail_rows);
    if (head.empty() || tail.empty()) return 1;

    printf("START rss=%ldKB pixels=%zu rows=%zu image_span=[%zu,%zu]\n", rss_kb(), input.size(),
           total, head_rows, head_rows + image_rows - 1);
    fflush(stdout);

    gp::LlmSession llm;
    gp::Timer t;
    if (!llm.init(llm_model, 512, 8, 1)) {
        printf("QWEN_INIT failed\n");
        return 1;
    }
    printf("QWEN_INIT ok ms=%.0f rss=%ldKB\n", t.ms(), rss_kb());
    fflush(stdout);

    gp::VisionStage vision;
    t.reset();
    if (!vision.init(vision_model, RKNN_NPU_CORE_0)) {
        printf("VISION_INIT failed\n");
        llm.shutdown();
        return 1;
    }
    printf("VISION_INIT ok ms=%.0f rss=%ldKB in=%d out=%d\n", t.ms(), rss_kb(),
           vision.input_elems(), vision.output_elems());
    fflush(stdout);

    std::vector<float> tokens;
    double vision_ms = 0;
    t.reset();
    if (!vision.run(input.data(), input.size(), tokens, &vision_ms)) {
        printf("VISION_RUN failed\n");
        return 1;
    }
    printf("VISION_RUN ok ms=%.1f tokens=%zu rss=%ldKB\n", vision_ms, tokens.size(), rss_kb());
    fflush(stdout);
    if (tokens.size() != image_rows * kDim) {
        printf("VISION_BAD tokens=%zu expected=%zu\n", tokens.size(), image_rows * kDim);
        return 1;
    }

    std::vector<float> embeds(total * kDim);
    memcpy(embeds.data(), head.data(), head_rows * kDim * 4);
    memcpy(embeds.data() + head_rows * kDim, tokens.data(), image_rows * kDim * 4);
    memcpy(embeds.data() + (head_rows + image_rows) * kDim, tail.data(), tail_rows * kDim * 4);

    for (int k = 0; k < 2; ++k) {
        t.reset();
        auto result = llm.run(embeds.data(), (int)total);
        printf("QWEN_RUN %d status=%d tokens=%zu total_ms=%.1f first_ms=%.1f rss=%ldKB\n",
               k, result.status, result.tokens.size(), result.run_ms, result.first_token_ms, rss_kb());
        if (!result.tokens.empty()) {
            printf("QWEN_OUT");
            for (size_t i = 0; i < result.tokens.size() && i < 12; ++i) printf(" %d", result.tokens[i]);
            printf("\n");
        }
        fflush(stdout);
    }

    vision.release();
    printf("VISION_RELEASED rss=%ldKB\n", rss_kb());
    llm.shutdown();
    printf("QWEN_RELEASED rss=%ldKB DONE\n", rss_kb());
    return 0;
}
