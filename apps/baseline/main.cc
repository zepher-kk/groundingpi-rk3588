// GroundingPI on RK3588: image + text instruction -> box, all in C++.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>

#include <opencv2/core.hpp>

#include "common.h"
#include "embed.h"
#include "llm.h"
#include "npy.h"
#include "preprocess.h"
#include "protocol.h"
#include "tokenizer.h"
#include "vision.h"

namespace {

struct Options {
    std::string pixels;
    std::string image;
    std::string query;
    std::string query_file;
    // No machine-specific defaults: provide --assets/--llm/--models, or export
    // GPI_ASSET_DIR / GPI_LLM / GPI_MODELS (see configs/baseline.json).
    std::string assets;
    std::string llm;
    std::string dump_tokens;
    std::string prep = "cpu";
    std::string image_size;
    std::string models;
    std::string batch;
    std::string stage_cores;
    int repeat = 1;
    int cores = 1;  // RKNN_NPU_CORE_0
    bool no_llm = false;
    bool keep_vision = false;
    bool dump_prompt = false;
    std::string dump_embeds;
    std::string dump_pixels;
    bool shared = false;
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

int parse_cores(const std::string& value) {    if (value == "all") return RKNN_NPU_CORE_0_1_2;
    if (value == "0") return RKNN_NPU_CORE_0;
    if (value == "1") return RKNN_NPU_CORE_1;
    if (value == "2") return RKNN_NPU_CORE_2;
    return RKNN_NPU_CORE_AUTO;
}

void usage() {
    fprintf(stderr,
            "  --query TEXT      text instruction (or --query-file PATH)\n"
            "  --assets DIR      tokenizer + embedding assets\n"
            "  --llm FILE        rkllm model\n"
            "  --models a,b,c    three vision stages\n"
            "  --prep cpu|rga    preprocessing backend\n"
            "  --cores 0|1|2|all NPU core mask for the vision stages\n"
            "  --repeat N        vision repetitions\n"
            "  --no-llm          stop after the vision stages\n"
            "  --dump-tokens F   write float32 visual tokens to F\n");
}

struct BatchRow {
    std::string id;
    std::string pixels;
    int width = 0;
    int height = 0;
    std::string query;
};

std::vector<BatchRow> read_batch(const std::string& path) {
    std::vector<BatchRow> rows;
    auto data = gp::read_file(path);
    std::string text(data.begin(), data.end());
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        std::string line = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        start = (end == std::string::npos) ? text.size() : end + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto parts = split(line, '\t');
        if (parts.size() < 5) continue;
        BatchRow row;
        row.id = parts[0];
        row.pixels = parts[1];
        row.width = atoi(parts[2].c_str());
        row.height = atoi(parts[3].c_str());
        row.query = parts[4];
        rows.push_back(row);
    }
    return rows;
}

int run_batch(const Options& opt) {
    auto rows = read_batch(opt.batch);
    if (rows.empty()) {
        fprintf(stderr, "empty batch %s\n", opt.batch.c_str());
        return 1;
    }
    gp::Tokenizer tokenizer;
    if (!tokenizer.load(opt.assets + "/vocab.bin", opt.assets + "/merges.bin")) return 1;
    gp::EmbeddingTable embed;
    if (!embed.open(opt.assets + "/embed.bf16", 152670, 2560)) return 1;

    // ---- phase 1: vision for the whole group ----
    std::vector<double> masks;
    (void)masks;
    gp::Timer timer;
    gp::VisionPipeline vision;
    auto models = split(opt.models, ',');
    bool ok = false;
    if (!opt.stage_cores.empty()) {
        auto parts = split(opt.stage_cores, ',');
        std::vector<int> core_masks;
        for (const auto& p : parts) core_masks.push_back(parse_cores(p));
        ok = vision.init(models, core_masks);
    } else {
        ok = vision.init(models, opt.cores);
    }
    if (!ok) return 1;
    const double vision_init_ms = timer.ms();

    std::vector<std::vector<float>> tokens(rows.size());
    std::vector<double> vision_ms(rows.size(), 0), npy_ms(rows.size(), 0);
    std::vector<std::array<double, 3>> stage_ms(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        std::vector<float> input;
        std::vector<int> shape;
        timer.reset();
        if (rows[i].pixels.size() > 4 &&
            rows[i].pixels.substr(rows[i].pixels.size() - 4) == ".npy") {
            if (!gp::load_npy_f32(rows[i].pixels, input, shape)) {
                fprintf(stderr, "cannot load %s\n", rows[i].pixels.c_str());
                return 1;
            }
        } else {
            cv::Mat bgr;
            if (!gp::decode_image(rows[i].pixels, bgr, nullptr)) {
                fprintf(stderr, "cannot decode %s\n", rows[i].pixels.c_str());
                return 1;
            }
            gp::LetterboxInfo info;
            if (opt.prep == "rga") {
                if (!gp::letterbox_rga(bgr, input, &info, nullptr, nullptr)) return 1;
            } else {
                if (!gp::letterbox_cpu(bgr, input, &info, nullptr, nullptr)) return 1;
            }
            rows[i].width = info.width;
            rows[i].height = info.height;
        }
        npy_ms[i] = timer.ms();
        std::vector<double> stages;
        timer.reset();
        if (!vision.run(input.data(), input.size(), tokens[i], &stages)) return 1;
        vision_ms[i] = timer.ms();
        stage_ms[i] = {stages[0], stages[1], stages[2]};
    }

    timer.reset();
    vision.release();
    const double release_ms = timer.ms();

    // ---- phase 2: one resident LLM, one request per row ----
    gp::LlmSession llm;
    if (!llm.init(opt.llm)) return 1;
    double llm_total = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        gp::Prompt prompt = gp::build_prompt(tokenizer, rows[i].query);
        std::vector<float> embeds((size_t)prompt.ids.size() * 2560);
        for (size_t k = 0; k < prompt.ids.size(); ++k) {
            float* dst = embeds.data() + k * 2560;
            if ((int)k >= prompt.head_tokens && (int)k < prompt.head_tokens + prompt.image_tokens) {
                memcpy(dst, tokens[i].data() + ((size_t)k - prompt.head_tokens) * 2560, 2560 * 4);
            } else if (!embed.row(prompt.ids[k], dst)) {
                return 1;
            }
        }
        if (i == 0 && !opt.dump_embeds.empty()) {
            gp::write_file(opt.dump_embeds, embeds.data(), embeds.size() * sizeof(float));
            fprintf(stderr, "[dump] embeds %zu x 2560 -> %s\n", prompt.ids.size(),
                    opt.dump_embeds.c_str());
        }
        auto result = llm.run(embeds.data(), (int)prompt.ids.size());        llm_total += result.run_ms;
        gp::Box box = gp::decode_box(result.tokens, rows[i].width, rows[i].height);
        std::string ids;
        for (size_t k = 0; k < result.tokens.size(); ++k) ids += (k ? "," : "") + std::to_string(result.tokens[k]);
        printf("{\"id\":\"%s\",\"query\":\"%s\",\"vision_ms\":%.2f,\"npy_ms\":%.2f,"
               "\"stage_ms\":[%.2f,%.2f,%.2f],\"llm_ms\":%.2f,\"first_token_ms\":%.2f,"
               "\"gen_tokens\":[%s],\"valid\":%s,\"none\":%s,\"box\":[%.2f,%.2f,%.2f,%.2f]}\n",
               gp::json_escape(rows[i].id).c_str(), gp::json_escape(rows[i].query).c_str(),
               vision_ms[i], npy_ms[i], stage_ms[i][0], stage_ms[i][1], stage_ms[i][2],
               result.run_ms, result.first_token_ms, ids.c_str(),
               box.valid ? "true" : "false", box.none ? "true" : "false",
               box.x0, box.y0, box.x1, box.y1);
        fflush(stdout);
    }
    printf("{\"mode\":\"batch_summary\",\"rows\":%zu,\"vision_init_ms\":%.1f,\"vision_release_ms\":%.1f,"
           "\"llm_init_ms\":%.1f,\"vision_p50_ms\":%.2f,\"llm_p50_ms\":%.2f,\"wall_ms\":%.1f}\n",
           rows.size(), vision_init_ms, release_ms, llm.init_ms(),
           vision_ms[rows.size() / 2], llm_total / (double)rows.size(), timer.ms());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (arg == "--query") opt.query = next();
        else if (arg == "--query-file") opt.query_file = next();
        else if (arg == "--image") opt.image = next();
        else if (arg == "--pixels") opt.pixels = next();
        else if (arg == "--assets") opt.assets = next();
        else if (arg == "--llm") opt.llm = next();
        else if (arg == "--models") opt.models = next();
        else if (arg == "--prep") opt.prep = next();
        else if (arg == "--image-size") opt.image_size = next();
        else if (arg == "--cores") opt.cores = parse_cores(next());
        else if (arg == "--repeat") opt.repeat = atoi(next().c_str());
        else if (arg == "--no-llm") opt.no_llm = true;
        else if (arg == "--keep-vision") opt.keep_vision = true;
        else if (arg == "--dump-prompt") opt.dump_prompt = true;
        else if (arg == "--dump-embeds") opt.dump_embeds = next();
        else if (arg == "--dump-pixels") opt.dump_pixels = next();
        else if (arg == "--shared") opt.shared = true;
        else if (arg == "--batch") opt.batch = next();
        else if (arg == "--stage-cores") opt.stage_cores = next();
        else if (arg == "--dump-tokens") opt.dump_tokens = next();
        else if (arg == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown option %s\n", arg.c_str()); usage(); return 2; }
    }
    if (opt.assets.empty()) { const char* v = getenv("GPI_ASSET_DIR"); if (v) opt.assets = v; }
    if (opt.llm.empty()) { const char* v = getenv("GPI_LLM"); if (v) opt.llm = v; }
    if (opt.models.empty()) { const char* v = getenv("GPI_MODELS"); if (v) opt.models = v; }
    if (opt.assets.empty() || opt.llm.empty() || opt.models.empty()) {
        fprintf(stderr,
                "missing --assets/--llm/--models (or GPI_ASSET_DIR/GPI_LLM/GPI_MODELS).\n"
                "See configs/baseline.json and scripts/run_baseline.sh.\n");
        usage();
        return 2;
    }

    if (!opt.query_file.empty()) {
        auto raw = gp::read_file(opt.query_file);
        std::string text(raw.begin(), raw.end());
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
        opt.query = text;
    }
    if (!opt.batch.empty()) return run_batch(opt);
    if (opt.query.empty() || (!opt.dump_prompt && opt.image.empty() && opt.pixels.empty())) {
        usage();
        return 2;
    }
    if (opt.repeat < 1) opt.repeat = 1;

    if (opt.dump_prompt) {
        gp::Tokenizer tokenizer;
        if (!tokenizer.load(opt.assets + "/vocab.bin", opt.assets + "/merges.bin")) return 1;
        gp::Prompt prompt = gp::build_prompt(tokenizer, opt.query);
        printf("PROMPT_TOKENS %zu HEAD %d TAIL %d\nPROMPT_IDS",
               prompt.ids.size(), prompt.head_tokens, prompt.tail_tokens);
        for (int32_t id : prompt.ids) printf(" %d", id);
        printf("\n");
        return 0;
    }

    const auto program_begin = std::chrono::steady_clock::now();

    // ---- vision ----
    gp::Timer timer;
    gp::VisionPipeline vision;
    auto models = split(opt.models, ',');
    if (models.size() != 3 || !(opt.shared ? vision.init_shared(models, opt.cores)
                                           : vision.init(models, opt.cores))) {
        fprintf(stderr, "vision init failed\n");
        return 1;
    }
    const double vision_init_ms = timer.ms();

    std::vector<float> input;
    std::vector<int> shape;
    gp::LetterboxInfo letterbox;
    double decode_ms = 0, resize_ms = 0, normalize_ms = 0;
    if (!opt.pixels.empty()) {
        timer.reset();
        if (!gp::load_npy_f32(opt.pixels, input, shape)) {
            fprintf(stderr, "cannot load %s\n", opt.pixels.c_str());
            return 1;
        }
        decode_ms = timer.ms();
        if (!opt.image_size.empty()) {
            auto parts = split(opt.image_size, 'x');
            if (parts.size() == 2) {
                letterbox.width = atoi(parts[0].c_str());
                letterbox.height = atoi(parts[1].c_str());
            }
        }
        if (letterbox.width <= 0 || letterbox.height <= 0) {
            letterbox.width = 448;
            letterbox.height = 448;
        }
        const float scale = std::min(448.0f / (float)letterbox.width, 448.0f / (float)letterbox.height);
        letterbox.resized_w = std::max(1, (int)std::lround((float)letterbox.width * scale));
        letterbox.resized_h = std::max(1, (int)std::lround((float)letterbox.height * scale));
        letterbox.left = (448 - letterbox.resized_w) / 2;
        letterbox.top = (448 - letterbox.resized_h) / 2;
    } else {
        cv::Mat bgr;
        if (!gp::decode_image(opt.image, bgr, &decode_ms)) {
            fprintf(stderr, "cannot decode %s\n", opt.image.c_str());
            return 1;
        }
        bool ok = (opt.prep == "rga")
                      ? gp::letterbox_rga(bgr, input, &letterbox, &resize_ms, &normalize_ms)
                      : gp::letterbox_cpu(bgr, input, &letterbox, &resize_ms, &normalize_ms);
        if (!ok) return 1;
    }

    if (!opt.dump_pixels.empty()) {
        gp::write_file(opt.dump_pixels, input.data(), input.size() * sizeof(float));
        fprintf(stderr, "[dump] pixels %zu floats -> %s\n", input.size(), opt.dump_pixels.c_str());
    }

    std::vector<float> tokens;
    std::vector<double> stage_ms;    double best_ms = 1e18;
    for (int r = 0; r < opt.repeat; ++r) {
        timer.reset();
        if (!vision.run(input.data(), input.size(), tokens, &stage_ms)) {
            fprintf(stderr, "vision run failed\n");
            return 1;
        }
        double total = timer.ms();
        if (total < best_ms) best_ms = total;
    }
    const double vision_ms = best_ms;

    if (!opt.dump_tokens.empty()) {
        gp::write_file(opt.dump_tokens, tokens.data(), tokens.size() * sizeof(float));
        fprintf(stderr, "[dump] %s (%zu floats)\n", opt.dump_tokens.c_str(), tokens.size());
    }

    if (opt.no_llm && opt.dump_embeds.empty()) {
        printf("{\"mode\":\"vision\",\"vision_ms\":%.2f,\"stage_ms\":[%.2f,%.2f,%.2f],"
               "\"vision_init_ms\":%.1f,\"decode_ms\":%.2f,\"resize_ms\":%.2f,"
               "\"normalize_ms\":%.2f,\"token_shape\":[256,2560]}\n",
               vision_ms, stage_ms[0], stage_ms[1], stage_ms[2], vision_init_ms,
               decode_ms, resize_ms, normalize_ms);
        return 0;
    }

    // ---- prompt + embeddings ----
    gp::Tokenizer tokenizer;
    if (!tokenizer.load(opt.assets + "/vocab.bin", opt.assets + "/merges.bin")) {
        fprintf(stderr, "tokenizer load failed\n");
        return 1;
    }
    gp::EmbeddingTable embed;
    if (!embed.open(opt.assets + "/embed.bf16", 152670, 2560)) {
        fprintf(stderr, "embedding table load failed\n");
        return 1;
    }
    timer.reset();
    gp::Prompt prompt = gp::build_prompt(tokenizer, opt.query);
    const double prompt_ms = timer.ms();

    if ((int)tokens.size() != prompt.image_tokens * 2560) {
        fprintf(stderr, "unexpected token count %zu\n", tokens.size());
        return 1;
    }
    std::vector<float> embeds((size_t)prompt.ids.size() * 2560);
    for (size_t i = 0; i < prompt.ids.size(); ++i) {
        float* dst = embeds.data() + i * 2560;
        if ((int)i >= prompt.head_tokens && (int)i < prompt.head_tokens + prompt.image_tokens) {
            const float* src = tokens.data() + ((size_t)i - prompt.head_tokens) * 2560;
            memcpy(dst, src, 2560 * sizeof(float));
        } else if (!embed.row(prompt.ids[i], dst)) {
            fprintf(stderr, "embedding row %d failed\n", prompt.ids[i]);
            return 1;
        }
    }

    // ---- llm ----
    if (!opt.dump_embeds.empty()) {
        gp::write_file(opt.dump_embeds, embeds.data(), embeds.size() * sizeof(float));
        fprintf(stderr, "[dump] embeds %zu x 2560 -> %s\n", prompt.ids.size(), opt.dump_embeds.c_str());
    }
    if (opt.no_llm) {
        printf("{\"mode\":\"vision\",\"vision_ms\":%.2f,\"stage_ms\":[%.2f,%.2f,%.2f],"
               "\"vision_init_ms\":%.1f,\"decode_ms\":%.2f,\"resize_ms\":%.2f,"
               "\"normalize_ms\":%.2f,\"token_shape\":[256,2560]}\n",
               vision_ms, stage_ms[0], stage_ms[1], stage_ms[2], vision_init_ms,
               decode_ms, resize_ms, normalize_ms);
        return 0;
    }
    timer.reset();
    if (!opt.keep_vision) vision.release();
    const double vision_release_ms = timer.ms();

    gp::LlmSession llm;
    if (!llm.init(opt.llm)) return 1;
    gp::LlmSession::Result result = llm.run(embeds.data(), (int)prompt.ids.size());

    gp::Box box = gp::decode_box(result.tokens, letterbox.width, letterbox.height);
    const double total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - program_begin)
            .count();

    std::string ids;
    for (size_t i = 0; i < result.tokens.size(); ++i) {
        ids += (i ? ", " : "");
        ids += std::to_string(result.tokens[i]);
    }
    printf("{\"mode\":\"full\",\"query\":\"%s\",\"vision_ms\":%.2f,\"stage_ms\":[%.2f,%.2f,%.2f],"
           "\"vision_init_ms\":%.1f,\"decode_ms\":%.2f,\"resize_ms\":%.2f,\"normalize_ms\":%.2f,"
           "\"prompt_tokens\":%d,\"prompt_ms\":%.3f,\"vision_release_ms\":%.1f,"
           "\"llm_init_ms\":%.1f,\"llm_ms\":%.2f,"
           "\"first_token_ms\":%.2f,\"gen_tokens\":[%s],\"valid\":%s,\"none\":%s,"
           "\"box\":[%.2f,%.2f,%.2f,%.2f],\"image_size\":[%d,%d],\"elapsed_ms\":%.1f}\n",
           gp::json_escape(opt.query).c_str(), vision_ms, stage_ms[0], stage_ms[1], stage_ms[2],
           vision_init_ms, decode_ms, resize_ms, normalize_ms, (int)prompt.ids.size(),
           prompt_ms, vision_release_ms, llm.init_ms(), result.run_ms, result.first_token_ms, ids.c_str(),
           box.valid ? "true" : "false", box.none ? "true" : "false",
           box.x0, box.y0, box.x1, box.y1, letterbox.width, letterbox.height, total_ms);
    return 0;
}
