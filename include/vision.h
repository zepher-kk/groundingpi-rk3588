#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "rknn_api.h"

namespace gp {

struct StageTimes {
    uint64_t prep_begin = 0, prep_end = 0;
    uint64_t run_begin = 0, run_end = 0;
    uint64_t get_begin = 0, get_end = 0;
};

// One RKNN stage driven through the standard input/output API.
// Inputs are fed as FP16 (converted here), outputs are read as FP32.
class VisionStage {
public:
    ~VisionStage();
    bool init(const std::string& path, int core_mask);
    bool run(const float* input, size_t count, std::vector<float>& output, double* run_ms = nullptr,
             StageTimes* times = nullptr);
    void release();

    int input_elems() const { return (int)in_attr_.n_elems; }
    int output_elems() const { return (int)out_attr_.n_elems; }
    const rknn_tensor_attr& input_attr() const { return in_attr_; }
    const rknn_tensor_attr& output_attr() const { return out_attr_; }
    bool input_nhwc() const { return in_attr_.fmt == RKNN_TENSOR_NHWC && in_attr_.n_dims == 4; }
    int input_h() const { return input_nhwc() ? (int)in_attr_.dims[1] : 0; }
    int input_w() const { return input_nhwc() ? (int)in_attr_.dims[2] : 0; }
    int input_c() const { return input_nhwc() ? (int)in_attr_.dims[3] : 0; }

    rknn_context context() const { return ctx_; }
    uint32_t output_bytes() const {
        return out_attr_.size_with_stride ? out_attr_.size_with_stride : out_attr_.size;
    }
    rknn_tensor_mem* alloc_output_mem();
    bool bind_output(rknn_tensor_mem* mem);
    bool bind_input(rknn_tensor_mem* mem);

private:
    rknn_context ctx_ = 0;
    std::vector<uint8_t> model_;
    rknn_tensor_attr in_attr_{};
    rknn_tensor_attr out_attr_{};
    std::vector<__fp16> in_half_;
    std::vector<float> out_buf_;
    bool ready_ = false;
    bool shared_in_ = false;
    bool shared_out_ = false;
};

class VisionPipeline {
public:
    bool init(const std::vector<std::string>& models, int core_mask);
    bool init(const std::vector<std::string>& models, const std::vector<int>& masks);
    // Binds each stage's output buffer as the next stage's input (no copy).
    bool init_shared(const std::vector<std::string>& models, int core_mask);
    // Runs all stages; input is the first stage's input buffer (NCHW float32 for
    // the patch stage, 2D [C,T] for the following ones).
    bool run(const float* input, size_t count, std::vector<float>& tokens,
             std::vector<double>* stage_ms);
    void release();
    bool shared() const { return !shared_mems_.empty(); }
    const std::vector<VisionStage>& stages() const { return stages_; }

private:
    std::vector<VisionStage> stages_;
    std::vector<rknn_tensor_mem*> shared_mems_;
};

}  // namespace gp
