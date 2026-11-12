// Probe the letterbox sub-rectangle pattern that fails in the pipeline.
#include <cstdio>
#include <cstring>
#include <vector>

#include <im2d.hpp>
#include <rga.h>

#include "rknn_api.h"

int main(int argc, char** argv) {
    rknn_context ctx = 0;
    std::vector<uint8_t> model;
    if (argc > 1) {
        FILE* f = fopen(argv[1], "rb");
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        model.resize((size_t)n);
        if (fread(model.data(), 1, (size_t)n, f) != (size_t)n) return 1;
        fclose(f);
        int rc = rknn_init(&ctx, model.data(), (uint32_t)model.size(), 0, nullptr);
        fprintf(stderr, "rknn_init=%d\n", rc);
    }
    const int W = 640, H = 480, NW = 448, NH = 336, LEFT = 0, TOP = 56;
    std::vector<uint8_t> src((size_t)W * H * 3, 100);
    std::vector<uint8_t> canvas((size_t)448 * 448 * 3, 127);

    // (a) destination is a sub-rectangle of the canvas, hstride = NH
    {
        uint8_t* off = canvas.data() + ((size_t)TOP * 448 + LEFT) * 3;
        rga_buffer_t s = wrapbuffer_virtualaddr_t(src.data(), W, H, W, H, RK_FORMAT_RGB_888);
        rga_buffer_t d = wrapbuffer_virtualaddr_t(off, NW, NH, 448, NH, RK_FORMAT_RGB_888);
        fprintf(stderr, "a) subrect hstride=NH: %d\n", (int)imresize(s, d));
    }
    // (b) same but hstride = 448 (full canvas height)
    {
        uint8_t* off = canvas.data() + ((size_t)TOP * 448 + LEFT) * 3;
        rga_buffer_t s = wrapbuffer_virtualaddr_t(src.data(), W, H, W, H, RK_FORMAT_RGB_888);
        rga_buffer_t d = wrapbuffer_virtualaddr_t(off, NW, NH, 448, 448, RK_FORMAT_RGB_888);
        fprintf(stderr, "b) subrect hstride=448: %d\n", (int)imresize(s, d));
    }
    // (c) full canvas, destination rect 448x448 (stretch) with offset zero
    {
        rga_buffer_t s = wrapbuffer_virtualaddr_t(src.data(), W, H, W, H, RK_FORMAT_RGB_888);
        rga_buffer_t d = wrapbuffer_virtualaddr_t(canvas.data(), NW, NH, 448, NH, RK_FORMAT_RGB_888);
        fprintf(stderr, "c) offset0 hstride=NH: %d\n", (int)imresize(s, d));
    }
    // (d) BGR source with subrect
    {
        uint8_t* off = canvas.data() + ((size_t)TOP * 448 + LEFT) * 3;
        rga_buffer_t s = wrapbuffer_virtualaddr_t(src.data(), W, H, W, H, RK_FORMAT_BGR_888);
        rga_buffer_t d = wrapbuffer_virtualaddr_t(off, NW, NH, 448, NH, RK_FORMAT_BGR_888);
        fprintf(stderr, "d) bgr subrect: %d\n", (int)imresize(s, d));
    }
    // (e) colour convert on a subrect
    {
        uint8_t* off = canvas.data() + ((size_t)TOP * 448 + LEFT) * 3;
        rga_buffer_t d = wrapbuffer_virtualaddr_t(off, NW, NH, 448, NH, RK_FORMAT_RGB_888);
        fprintf(stderr, "e) cvt on subrect: %d\n", (int)imcvtcolor(d, d, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888));
    }
    return 0;
}
