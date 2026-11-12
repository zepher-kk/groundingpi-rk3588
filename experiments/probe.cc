// Probe RKNN tensor attributes (standard + native) and optionally run a model.
//
//   probe model.rknn [model2.rknn ...]
//   probe --run model.rknn input.npy
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rknn_api.h"

static const char* fmt_name(rknn_tensor_format f) {
    switch (f) {
        case RKNN_TENSOR_NCHW: return "NCHW";
        case RKNN_TENSOR_NHWC: return "NHWC";
        case RKNN_TENSOR_NC1HWC2: return "NC1HWC2";
        case RKNN_TENSOR_UNDEFINED: return "UNDEFINED";
        default: return "UNKNOWN";
    }
}

static const char* type_name(rknn_tensor_type t) {
    switch (t) {
        case RKNN_TENSOR_FLOAT32: return "FLOAT32";
        case RKNN_TENSOR_FLOAT16: return "FLOAT16";
        case RKNN_TENSOR_INT8: return "INT8";
        case RKNN_TENSOR_UINT8: return "UINT8";
        case RKNN_TENSOR_INT32: return "INT32";
        default: return "OTHER";
    }
}

static std::vector<uint8_t> load_file(const char* path) {
    std::vector<uint8_t> data;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return data; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    data.resize((size_t)n);
    if (fread(data.data(), 1, (size_t)n, f) != (size_t)n) data.clear();
    fclose(f);
    return data;
}

static void show(const char* tag, const rknn_tensor_attr& a) {
    printf("  %-10s idx=%d %-40s fmt=%-8s type=%-7s dims=[", tag, a.index, a.name,
           fmt_name(a.fmt), type_name(a.type));
    for (uint32_t i = 0; i < a.n_dims; ++i) printf("%s%u", i ? "," : "", a.dims[i]);
    printf("] n_elems=%u size=%u w_stride=%u size_stride=%u scale=%.8g zp=%d\n",
           a.n_elems, a.size, a.w_stride, a.size_with_stride, (double)a.scale, (int)a.zp);
}

int main(int argc, char** argv) {
    bool run = false;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--run") == 0) run = true;
        else args.emplace_back(argv[i]);
    }
    if (args.empty()) { fprintf(stderr, "usage: probe [--run] model.rknn [input.npy]\n"); return 2; }

    for (size_t k = 0; k < args.size(); ++k) {
        if (args[k].size() > 5 && args[k].substr(args[k].size() - 5) == ".npy") continue;
        auto model = load_file(args[k].c_str());
        if (model.empty()) return 1;
        rknn_context ctx = 0;
        int rc = rknn_init(&ctx, model.data(), (uint32_t)model.size(), 0, nullptr);
        printf("=== %s  init=%d\n", args[k].c_str(), rc);
        if (rc != 0) continue;
        rknn_input_output_num io{};
        rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
        printf("  n_in=%u n_out=%u\n", io.n_input, io.n_output);
        std::vector<rknn_tensor_attr> ins(io.n_input), outs(io.n_output);
        for (uint32_t i = 0; i < io.n_input; ++i) {
            ins[i] = {}; ins[i].index = i;
            rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &ins[i], sizeof(rknn_tensor_attr));
            show("input", ins[i]);
            rknn_tensor_attr n{}; n.index = i;
            if (rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &n, sizeof(n)) == 0) show("native_in", n);
        }
        for (uint32_t i = 0; i < io.n_output; ++i) {
            outs[i] = {}; outs[i].index = i;
            rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &outs[i], sizeof(rknn_tensor_attr));
            show("output", outs[i]);
            rknn_tensor_attr n{}; n.index = i;
            if (rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &n, sizeof(n)) == 0) show("native_out", n);
        }

        if (run && k + 1 < args.size()) {
            auto pixels = load_file(args[k + 1].c_str());
            size_t want = (size_t)ins[0].n_elems;
            std::vector<float> in(want, 0.f);
            if (!pixels.empty()) {
                // minimal .npy reader: skip magic + header, copy float32 payload
                if (pixels.size() > 10 && memcmp(pixels.data(), "\x93NUMPY", 6) == 0) {
                    uint16_t hlen = (uint16_t)(pixels[8] | (pixels[9] << 8));
                    size_t off = 10 + (size_t)hlen;
                    size_t n = (pixels.size() - off) / 4;
                    if (n < want) want = n;
                    memcpy(in.data(), pixels.data() + off, want * 4);
                } else {
                    size_t n = pixels.size() / 4;
                    if (n < want) want = n;
                    memcpy(in.data(), pixels.data(), want * 4);
                }
            }
            rknn_input input{};
            input.index = 0; input.type = RKNN_TENSOR_FLOAT32; input.fmt = RKNN_TENSOR_NCHW;
            input.size = (uint32_t)(want * 4); input.buf = in.data();
            std::vector<float> nhwc;
            if (ins[0].fmt == RKNN_TENSOR_NHWC && ins[0].n_dims == 4) {
                // source is NCHW float32; the model wants NHWC float32
                const int H = (int)ins[0].dims[1], W = (int)ins[0].dims[2], C = (int)ins[0].dims[3];
                nhwc.resize((size_t)H * W * C);
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        for (int c = 0; c < C; ++c)
                            nhwc[((size_t)h * W + w) * C + c] = in[((size_t)c * H + h) * W + w];
                input.fmt = RKNN_TENSOR_NHWC;
                input.buf = nhwc.data();
                input.size = (uint32_t)(nhwc.size() * 4);
            }
            std::vector<__fp16> half;
            if (getenv("PROBE_FP16") && ins[0].fmt == RKNN_TENSOR_NHWC && ins[0].n_dims == 4) {
                const int H = (int)ins[0].dims[1], W = (int)ins[0].dims[2], C = (int)ins[0].dims[3];
                half.resize((size_t)H * W * C);
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        for (int c = 0; c < C; ++c)
                            half[((size_t)h * W + w) * C + c] = (__fp16)in[((size_t)c * H + h) * W + w];
                input.fmt = RKNN_TENSOR_NHWC;
                input.type = RKNN_TENSOR_FLOAT16;
                input.buf = half.data();
                input.size = (uint32_t)(half.size() * 2);
            }
            rc = rknn_inputs_set(ctx, 1, &input);
            printf("  input n=%zu min=%.6f max=%.6f mean=%.6f\n", want,
                   [&]{ float mn=1e9f,mx=-1e9f; double s=0; for (size_t i=0;i<want;++i){float v=in[i]; mn=v<mn?v:mn; mx=v>mx?v:mx; s+=v;} return mn; }(),
                   [&]{ float mx=-1e9f; for (size_t i=0;i<want;++i) if (in[i]>mx) mx=in[i]; return mx; }(),
                   [&]{ double s=0; for (size_t i=0;i<want;++i) s+=in[i]; return s/(double)want; }());
            printf("  inputs_set=%d\n", rc);
            rc = rknn_run(ctx, nullptr);
            printf("  run=%d\n", rc);
            std::vector<rknn_output> out(io.n_output);
            for (uint32_t i = 0; i < io.n_output; ++i) {
                out[i].index = i; out[i].want_float = 1; out[i].is_prealloc = 0;
            }
            rc = rknn_outputs_get(ctx, io.n_output, out.data(), nullptr);
            printf("  outputs_get=%d\n", rc);
            if (rc == 0) {
                for (uint32_t i = 0; i < io.n_output; ++i) {
                    const float* p = (const float*)out[i].buf;
                    size_t n = out[i].size / 4;
                    float mn = 1e9f, mx = -1e9f; double sum = 0;
                    for (size_t j = 0; j < n; ++j) { mn = p[j] < mn ? p[j] : mn; mx = p[j] > mx ? p[j] : mx; sum += p[j]; }
                    printf("  out%d n=%zu min=%.6f max=%.6f mean=%.6f\n", i, n, mn, mx, sum / n);
                    if (getenv("PROBE_DUMP")) {
                        char path[512];
                        snprintf(path, sizeof(path), "%s.out%d.bin", getenv("PROBE_DUMP"), (int)i);
                        FILE* f = fopen(path, "wb");
                        if (f) { fwrite(p, 4, n, f); fclose(f); }
                    }
                }
                rknn_outputs_release(ctx, io.n_output, out.data());
            }
            ++k;
        }
        rknn_destroy(ctx);
        printf("\n");
    }
    return 0;
}
