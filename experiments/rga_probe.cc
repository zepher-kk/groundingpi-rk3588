// Minimal RGA capability probe: resize / colour convert with several formats.
#include <cstdio>
#include <cstring>
#include <vector>

#include <im2d.hpp>
#include <rga.h>

static const char* fmt_name(int f) {
    switch (f) {
        case RK_FORMAT_RGB_888: return "RGB888";
        case RK_FORMAT_BGR_888: return "BGR888";
        case RK_FORMAT_RGBA_8888: return "RGBA8888";
        case RK_FORMAT_BGRA_8888: return "BGRA8888";
        default: return "?";
    }
}

int main() {
    const int W = 640, H = 480, NW = 448, NH = 336;
    std::vector<uint8_t> src((size_t)W * H * 4, 128);
    std::vector<uint8_t> dst((size_t)NW * NH * 4, 0);
    for (int i = 0; i < W * H; ++i) { src[i * 4] = (uint8_t)(i % 251); src[i * 4 + 1] = 64; src[i * 4 + 2] = 200; }

    std::fprintf(stderr, "header=%d version=%s\n", (int)imcheckHeader(RGA_CURRENT_API_HEADER_VERSION), RGA_API_VERSION);

    const int formats[] = {RK_FORMAT_RGB_888, RK_FORMAT_BGR_888, RK_FORMAT_RGBA_8888, RK_FORMAT_BGRA_8888};
    for (int f : formats) {
        const int bpp = (f == RK_FORMAT_RGB_888 || f == RK_FORMAT_BGR_888) ? 3 : 4;
        rga_buffer_t s = wrapbuffer_virtualaddr_t(src.data(), W, H, W, H, f);
        rga_buffer_t d = wrapbuffer_virtualaddr_t(dst.data(), NW, NH, NW, NH, f);
        IM_STATUS st = imresize(s, d);
        std::fprintf(stderr, "resize %-9s (%d bpp): %d\n", fmt_name(f), bpp, (int)st);
    }
    // same-size colour conversion
    {
        rga_buffer_t s = wrapbuffer_virtualaddr_t(src.data(), W, H, W, H, RK_FORMAT_BGR_888);
        rga_buffer_t d = wrapbuffer_virtualaddr_t(dst.data(), W, H, W, H, RK_FORMAT_RGB_888);
        std::fprintf(stderr, "cvtcolor BGR->RGB: %d\n", (int)imcvtcolor(s, d, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888));
    }
    // plain copy
    {
        rga_buffer_t s = wrapbuffer_virtualaddr_t(src.data(), W, H, W, H, RK_FORMAT_RGB_888);
        rga_buffer_t d = wrapbuffer_virtualaddr_t(dst.data(), W, H, W, H, RK_FORMAT_RGB_888);
        std::fprintf(stderr, "imcopy RGB888: %d\n", (int)imcopy(s, d));
    }
    return 0;
}
