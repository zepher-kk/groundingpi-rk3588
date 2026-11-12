// Co-residency probe: load Qwen first, keep it resident, then add the three
// MoonViT stages one by one on Core 0/1/2 and verify both towers still work.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <algorithm>
#include <chrono>
#include <cmath>

#include "common.h"
#include "llm.h"
#include "npy.h"
#include "protocol.h"
#include "vision.h"

namespace {

struct Options {
    std::string pixels;
    std::string known_embeds = "/tmp/cpp_embeds.bin";
    std::string dump_tokens = "/tmp/coexist_tokens.bin";
    std::string assets = "";
    std::string llm = "";
    std::string models =
        ""
        ""
        "";
    int width = 640;
    int height = 480;
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

long resident_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) { sscanf(line + 6, "%ld", &kb); break; }
    }
    fclose(f);
    return kb;
}

void print_box(const char* tag, const gp::Box& box) {
    if (box.none) {
        printf("%s none=true valid=%d\n", tag, (int)box.valid);
    } else {
        printf("%s valid=%d box=[%.2f,%.2f,%.2f,%.2f]\n", tag, (int)box.valid,
               box.x0, box.y0, box.x1, box.y1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (arg == "--pixels") opt.pixels = next();
        else if (arg == "--known-embeds") opt.known_embeds = next();
        else if (arg == "--dump-tokens") opt.dump_tokens = next();
        else if (arg == "--assets") opt.assets = next();
        else if (arg == "--llm") opt.llm = next();
        else if (arg == "--models") opt.models = next();
        else if (arg == "--image-size") {
            auto dims = split(next(), 'x');
            if (dims.size() == 2) { opt.width = atoi(dims[0].c_str()); opt.height = atoi(dims[1].c_str()); }
        } else {
            fprintf(stderr, "unknown option %s\n", arg.c_str());
            return 2;
        }
    }
    if (opt.pixels.empty()) { fprintf(stderr, "usage: coexist_probe --pixels x.npy [--known-embeds f]\n"); return 2; }

    auto embeds = gp::read_file(opt.known_embeds);
    if (embeds.empty() || embeds.size() % (2560 * 4) != 0) {
        fprintf(stderr, "bad embedding file %s (%zu bytes)\n", opt.known_embeds.c_str(), embeds.size());
        return 1;
    }
    const int known_tokens = (int)(embeds.size() / (2560 * 4));
    const float* known = (const float*)embeds.data();
    printf("KNOWN_EMBEDS %d tokens rss=%ldKB\n", known_tokens, resident_kb());
    fflush(stdout);

    // ---- step 1: Qwen first and resident ----
    gp::LlmSession llm;
    if (!llm.init(opt.llm)) {
        printf("QWEN_INIT failed\n");
        return 1;
    }
    printf("QWEN_INIT ok ms=%.0f rss=%ldKB\n", llm.init_ms(), resident_kb());
    fflush(stdout);

    auto before = llm.run(known, known_tokens);
    gp::Box box_before = gp::decode_box(before.tokens, opt.width, opt.height);
    printf("QWEN_BEFORE status=%d tokens=%zu run_ms=%.1f first_ms=%.1f ",
           before.status, before.tokens.size(), before.run_ms, before.first_token_ms);
    print_box("", box_before);
    fflush(stdout);

    // ---- step 2: add the vision stages one at a time ----
    const int masks[3] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2};
    const char* names[3] = {"stage00", "stage09", "stage18"};
    std::vector<gp::VisionStage> stages(3);
    auto models = split(opt.models, ',');
    bool all_loaded = true;
    for (int i = 0; i < 3; ++i) {
        gp::Timer timer;
        bool ok = stages[(size_t)i].init(models[(size_t)i], masks[i]);
        double ms = timer.ms();
        printf("VISION_LOAD %d %s core=%d rc=%d ms=%.1f rss=%ldKB\n",
               i, names[i], i, ok ? 0 : -1, ms, resident_kb());
        fflush(stdout);
        if (!ok) { all_loaded = false; break; }
    }

    // ---- step 3: run one image through the resident vision stages ----
    if (all_loaded) {
        std::vector<float> pixels;
        std::vector<int> shape;
        if (!gp::load_npy_f32(opt.pixels, pixels, shape)) {
            printf("VISION_RUN pixels_load_failed\n");
            return 1;
        }
        std::vector<float> buffer, tokens;
        const float* current = pixels.data();
        size_t count = pixels.size();
        double stage_ms[3] = {0, 0, 0};
        bool ok = true;
        for (int i = 0; i < 3; ++i) {
            if (count < (size_t)stages[(size_t)i].input_elems())
                count = (size_t)stages[(size_t)i].input_elems();
            if (!stages[(size_t)i].run(current, count, buffer, &stage_ms[i])) { ok = false; break; }
            current = buffer.data();
            count = buffer.size();
        }
        if (ok) {
            tokens = buffer;
            gp::write_file(opt.dump_tokens, tokens.data(), tokens.size() * sizeof(float));
            float mn = 1e9f, mx = -1e9f; double sum = 0;
            for (float v : tokens) { mn = v < mn ? v : mn; mx = v > mx ? v : mx; sum += v; }
            printf("VISION_RUN ok tokens=%zu stage_ms=%.1f/%.1f/%.1f min=%.4f max=%.4f mean=%.6f\n",
                   tokens.size(), stage_ms[0], stage_ms[1], stage_ms[2], mn, mx,
                   sum / (double)tokens.size());
        } else {
            printf("VISION_RUN failed\n");
            all_loaded = false;
        }
        fflush(stdout);
    }

    // ---- step 4: Qwen must still generate with the same embeddings ----
    auto after = llm.run(known, known_tokens);
    gp::Box box_after = gp::decode_box(after.tokens, opt.width, opt.height);
    printf("QWEN_AFTER status=%d tokens=%zu run_ms=%.1f first_ms=%.1f ",
           after.status, after.tokens.size(), after.run_ms, after.first_token_ms);
    print_box("", box_after);
    bool same_tokens = before.tokens == after.tokens;
    printf("RESULT co_resident=%s vision_ran=%s qwen_ok=%s same_tokens=%s rss=%ldKB\n",
           all_loaded ? "true" : "false", all_loaded ? "true" : "false",
           (after.status == 0 && after.finished && !after.error) ? "true" : "false",
           same_tokens ? "true" : "false", resident_kb());
    return all_loaded ? 0 : 1;
}
