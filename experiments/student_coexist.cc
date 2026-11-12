// Student co-residency probe: Qwen3-1.7B (domain 1) + TinyViT vision RKNN in one process.
#include <atomic>
#include <chrono>
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

std::vector<float> read_f32(const std::string& path, size_t expect) {
    auto data = gp::read_file(path);
    if (data.size() != expect * 4) {
        fprintf(stderr, "bad file %s: %zu bytes, expected %zu\n", path.c_str(), data.size(), expect * 4);
        return {};
    }
    std::vector<float> out(expect);
    memcpy(out.data(), data.data(), expect * 4);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = ".";
    if (argc > 1) dir = argv[1];
    const std::string vision_model = dir + "/student_vision_fp16.rknn";
    const std::string llm_model = dir + "/qwen3-1.7b-w8a8.rkllm";
    const std::string pixels_path = dir + "/sample_input.npy";
    const std::string head_tail_path = dir + "/head_tail_25x2048.f32";

    std::vector<float> input;
    std::vector<int> shape;
    if (!gp::load_npy_f32(pixels_path, input, shape)) {
        fprintf(stderr, "cannot load %s\n", pixels_path.c_str());
        return 1;
    }
    auto head_tail = read_f32(head_tail_path, 25u * 2048u);
    if (head_tail.empty()) return 1;

    printf("START rss=%ldKB pixels=%zu\n", rss_kb(), input.size());
    fflush(stdout);

    // 1) Qwen first, own IOVA domain
    gp::LlmSession llm;
    gp::Timer t;
    if (!llm.init(llm_model, 512, 8, 1)) {
        printf("QWEN_INIT failed\n");
        return 1;
    }
    const double qwen_ms = t.ms();
    printf("QWEN_INIT ok ms=%.0f rss=%ldKB\n", qwen_ms, rss_kb());
    fflush(stdout);

    // 2) vision RKNN
    gp::VisionStage vision;
    t.reset();
    if (!vision.init(vision_model, RKNN_NPU_CORE_0)) {
        printf("VISION_INIT failed\n");
        llm.shutdown();
        return 1;
    }
    const double vision_init_ms = t.ms();
    printf("VISION_INIT ok ms=%.0f rss=%ldKB in=%d out=%d\n", vision_init_ms, rss_kb(),
           vision.input_elems(), vision.output_elems());
    fflush(stdout);

    // 3) vision forward
    std::vector<float> tokens;
    double vision_ms = 0;
    t.reset();
    if (!vision.run(input.data(), input.size(), tokens, &vision_ms)) {
        printf("VISION_RUN failed\n");
        return 1;
    }
    printf("VISION_RUN ok ms=%.1f tokens=%zu rss=%ldKB\n", vision_ms, tokens.size(), rss_kb());
    fflush(stdout);

    // 4) assemble head + vision + tail and run Qwen twice
    std::vector<float> embeds(169u * 2048u);
    memcpy(embeds.data(), head_tail.data(), 17u * 2048u * 4);
    memcpy(embeds.data() + 17u * 2048u, tokens.data(), 144u * 2048u * 4);
    memcpy(embeds.data() + 161u * 2048u, head_tail.data() + 17u * 2048u, 8u * 2048u * 4);

    for (int k = 0; k < 2; ++k) {
        auto result = llm.run(embeds.data(), 169);
        printf("QWEN_RUN %d status=%d tokens=%zu total_ms=%.1f first_ms=%.1f rss=%ldKB\n",
               k, result.status, result.tokens.size(), result.run_ms, result.first_token_ms, rss_kb());
        fflush(stdout);
    }

    // 5) release order: vision first, then Qwen
    vision.release();
    printf("VISION_RELEASED rss=%ldKB\n", rss_kb());
    llm.shutdown();
    printf("QWEN_RELEASED rss=%ldKB DONE\n", rss_kb());
    return 0;
}
