#include "preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <im2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common.h"

namespace gp {

namespace {

void normalize(const uint8_t* rgb, std::vector<float>& nchw) {
    nchw.resize((size_t)3 * kInputSize * kInputSize);
    const size_t plane = (size_t)kInputSize * kInputSize;
    const float* pad = nullptr;
    (void)pad;
    for (int c = 0; c < 3; ++c) {
        float* dst = nchw.data() + (size_t)c * plane;
        const uint8_t* src = rgb + c;
        for (size_t i = 0; i < plane; ++i) {
            dst[i] = (float)src[i * 3] * (2.0f / 255.0f) - 1.0f;
        }
    }
}

void geometry(int w, int h, LetterboxInfo* info) {
    const float scale = std::min((float)kInputSize / (float)w, (float)kInputSize / (float)h);
    info->width = w;
    info->height = h;
    info->resized_w = std::max(1, (int)std::lround((float)w * scale));
    info->resized_h = std::max(1, (int)std::lround((float)h * scale));
    info->left = (kInputSize - info->resized_w) / 2;
    info->top = (kInputSize - info->resized_h) / 2;
}

}  // namespace

bool decode_image(const std::string& path, cv::Mat& bgr, double* ms) {
    Timer timer;
    auto data = read_file(path);
    if (data.empty()) return false;
    bgr = cv::imdecode(data, cv::IMREAD_COLOR);
    if (bgr.empty()) return false;
    if (ms) *ms = timer.ms();
    return true;
}

bool letterbox_cpu(const cv::Mat& bgr, std::vector<float>& nchw, LetterboxInfo* info,
                   double* resize_ms, double* normalize_ms) {
    LetterboxInfo local;
    geometry(bgr.cols, bgr.rows, &local);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    Timer timer;
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(local.resized_w, local.resized_h), 0, 0, cv::INTER_CUBIC);
    std::vector<uint8_t> canvas((size_t)kInputSize * kInputSize * 3, 127);
    for (int y = 0; y < local.resized_h; ++y) {
        memcpy(canvas.data() + ((size_t)(local.top + y) * kInputSize + local.left) * 3,
               resized.ptr(y), (size_t)local.resized_w * 3);
    }
    if (resize_ms) *resize_ms = timer.ms();

    timer.reset();
    normalize(canvas.data(), nchw);
    if (normalize_ms) *normalize_ms = timer.ms();
    if (info) *info = local;
    return true;
}

bool letterbox_rga(const cv::Mat& bgr, std::vector<float>& nchw, LetterboxInfo* info,
                   double* rga_ms, double* normalize_ms) {
    LetterboxInfo local;
    geometry(bgr.cols, bgr.rows, &local);
    std::vector<uint8_t> canvas((size_t)kInputSize * kInputSize * 3, 127);

    Timer timer;
    static bool checked = false;
    if (!checked) {
        IM_STATUS hs = imcheckHeader(RGA_CURRENT_API_HEADER_VERSION);
        fprintf(stderr, "[rga] imcheckHeader=%d (%s)\n", (int)hs, RGA_API_VERSION);
        checked = true;
    }
    const int src_stride = (int)(bgr.step / 3);
    // RGA rejects unaligned BGR888 strides; copy into 48-byte aligned scratch buffers.
    const int src_bytes = bgr.cols * 3;
    const int src_aligned_bytes = ((src_bytes + 47) / 48) * 48;
    std::vector<uint8_t> src_scratch((size_t)src_aligned_bytes * bgr.rows);
    for (int y = 0; y < bgr.rows; ++y)
        memcpy(src_scratch.data() + (size_t)y * src_aligned_bytes, bgr.ptr(y), (size_t)src_bytes);
    const int row_bytes = local.resized_w * 3;
    const int aligned_bytes = ((row_bytes + 47) / 48) * 48;
    const int aligned_width = aligned_bytes / 3;
    std::vector<uint8_t> scratch((size_t)aligned_bytes * local.resized_h, 127);
    rga_buffer_t src = wrapbuffer_virtualaddr_t(src_scratch.data(), bgr.cols, bgr.rows,
                                                src_aligned_bytes / 3, bgr.rows,
                                                RK_FORMAT_BGR_888);
    rga_buffer_t dst = wrapbuffer_virtualaddr_t(scratch.data(), local.resized_w, local.resized_h,
                                                aligned_width, local.resized_h,
                                                RK_FORMAT_BGR_888);
    IM_STATUS resize_status = imresize(src, dst);
    IM_STATUS cvt_status = (resize_status == IM_STATUS_SUCCESS)
                               ? imcvtcolor(dst, dst, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888)
                               : resize_status;
    if (rga_ms) *rga_ms = timer.ms();
    if (resize_status != IM_STATUS_SUCCESS || cvt_status != IM_STATUS_SUCCESS) {
        fprintf(stderr, "[rga] failed resize=%d cvt=%d src=%dx%d stride=%d dst=%dx%d at %d,%d\n",
                (int)resize_status, (int)cvt_status, bgr.cols, bgr.rows, src_stride,
                local.resized_w, local.resized_h, local.left, local.top);
        return false;
    }
    static bool logged = false;
    if (!logged) {
        fprintf(stderr, "[rga] ok src=%dx%d stride=%d dst=%dx%d at %d,%d aligned=%d\n",
                bgr.cols, bgr.rows, src_stride, local.resized_w, local.resized_h, local.left,
                local.top, aligned_width);
        logged = true;
    }
    for (int y = 0; y < local.resized_h; ++y) {
        memcpy(canvas.data() + ((size_t)(local.top + y) * kInputSize + local.left) * 3,
               scratch.data() + (size_t)y * aligned_bytes, (size_t)row_bytes);
    }

    timer.reset();
    normalize(canvas.data(), nchw);
    if (normalize_ms) *normalize_ms = timer.ms();
    if (info) *info = local;
    return true;
}

}  // namespace gp
