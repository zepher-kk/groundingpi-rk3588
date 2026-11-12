#pragma once
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace gp {

constexpr int kInputSize = 448;

struct LetterboxInfo {
    int width = 0;
    int height = 0;
    int resized_w = 0;
    int resized_h = 0;
    int left = 0;
    int top = 0;
};

// Decodes a JPEG/PNG into BGR.
bool decode_image(const std::string& path, cv::Mat& bgr, double* ms);

// Letterboxes to 448x448 RGB with gray 127 padding and normalises to [-1,1] NCHW.
bool letterbox_cpu(const cv::Mat& bgr, std::vector<float>& nchw, LetterboxInfo* info,
                   double* resize_ms, double* normalize_ms);

// Same, but the resize + colour conversion run on RGA; only the normalisation is CPU.
bool letterbox_rga(const cv::Mat& bgr, std::vector<float>& nchw, LetterboxInfo* info,
                   double* rga_ms, double* normalize_ms);

}  // namespace gp
