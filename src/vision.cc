#include "vision.h"

#include <cstdio>
#include <cstring>

#include "common.h"

namespace gp {

VisionStage::~VisionStage() {
    if (ctx_) rknn_destroy(ctx_);
}

bool VisionStage::init(const std::string& path, int core_mask) {
    model_ = read_file(path);
    if (model_.empty()) {
        fprintf(stderr, "[vision] cannot read %s\n", path.c_str());
        return false;
    }
    if (rknn_init(&ctx_, model_.data(), (uint32_t)model_.size(), 0, nullptr) != 0) {
        fprintf(stderr, "[vision] rknn_init failed: %s\n", path.c_str());
        return false;
    }
    if (core_mask >= 0) {
        int rc = rknn_set_core_mask(ctx_, (rknn_core_mask)core_mask);
        if (rc != 0) {
            fprintf(stderr, "[vision] set_core_mask failed: expected=%d rc=%d %s\n",
                    core_mask, rc, path.c_str());
            rknn_destroy(ctx_);
            ctx_ = 0;
            return false;
        }
        fprintf(stderr, "[vision] core mask bound: expected=%d rc=0\n", core_mask);
    }

    in_attr_ = {};
    in_attr_.index = 0;
    if (rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in_attr_, sizeof(in_attr_)) != 0) return false;
    out_attr_ = {};
    out_attr_.index = 0;
    if (rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_attr_, sizeof(out_attr_)) != 0) return false;

    in_half_.resize(in_attr_.n_elems);
    out_buf_.resize(out_attr_.n_elems);
    ready_ = true;
    fprintf(stderr, "[vision] %-32s in=%u (%d dims, fmt %d) out=%u\n",
            path.substr(path.find_last_of('/') + 1).c_str(), in_attr_.n_elems,
            (int)in_attr_.n_dims, (int)in_attr_.fmt, out_attr_.n_elems);
    return true;
}

bool VisionStage::run(const float* input, size_t count, std::vector<float>& output,
                      double* run_ms, StageTimes* times) {
    if (!ready_ || count < in_attr_.n_elems) return false;
    if (times) times->prep_begin = now_ns();
    if (!shared_in_) {
        const int H = input_h(), W = input_w(), C = input_c();
        if (H > 0) {
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    for (int c = 0; c < C; ++c)
                        in_half_[((size_t)h * W + w) * C + c] =
                            (__fp16)input[((size_t)c * H + h) * W + w];
        } else {
            for (uint32_t i = 0; i < in_attr_.n_elems; ++i) in_half_[i] = (__fp16)input[i];
        }

        rknn_input in{};
        in.index = 0;
        in.type = RKNN_TENSOR_FLOAT16;
        in.fmt = in_attr_.fmt;
        in.size = (uint32_t)(in_half_.size() * sizeof(__fp16));
        in.buf = in_half_.data();
        int rc = rknn_inputs_set(ctx_, 1, &in);
        if (rc != 0) {
            in.fmt = RKNN_TENSOR_NCHW;
            rc = rknn_inputs_set(ctx_, 1, &in);
            if (rc != 0) {
                fprintf(stderr, "[vision] inputs_set failed: %d\n", rc);
                return false;
            }
        }
    }

    Timer timer;
    if (times) { times->prep_end = now_ns(); times->run_begin = times->prep_end; }
    if (rknn_run(ctx_, nullptr) != 0) {
        fprintf(stderr, "[vision] rknn_run failed\n");
        return false;
    }
    if (run_ms) *run_ms = timer.ms();
    if (times) times->run_end = now_ns();
    if (shared_out_) {
        output.clear();
        if (times) { times->get_begin = now_ns(); times->get_end = times->get_begin; }
        return true;
    }

    if (times) times->get_begin = now_ns();
    rknn_output out{};
    out.index = 0;
    out.want_float = 1;
    out.is_prealloc = 1;
    out.buf = out_buf_.data();
    out.size = (uint32_t)(out_buf_.size() * sizeof(float));
    if (rknn_outputs_get(ctx_, 1, &out, nullptr) != 0) {
        fprintf(stderr, "[vision] outputs_get failed\n");
        return false;
    }
    output.assign(out_buf_.begin(), out_buf_.end());
    if (times) times->get_end = now_ns();
    return true;
}

bool VisionPipeline::init(const std::vector<std::string>& models, const std::vector<int>& masks) {
    stages_.resize(models.size());
    for (size_t i = 0; i < models.size(); ++i) {
        if (!stages_[i].init(models[i], i < masks.size() ? masks[i] : -1)) return false;
        if (i > 0 && stages_[i].input_elems() != stages_[i - 1].output_elems()) {
            fprintf(stderr, "[vision] stage %zu input %d != previous output %d\n", i,
                    stages_[i].input_elems(), stages_[i - 1].output_elems());
            return false;
        }
    }
    return true;
}

rknn_tensor_mem* VisionStage::alloc_output_mem() {
    return rknn_create_mem(ctx_, output_bytes());
}

bool VisionStage::bind_output(rknn_tensor_mem* mem) {
    rknn_tensor_attr attr = out_attr_;
    attr.index = 0;
    attr.pass_through = 1;
    if (rknn_set_io_mem(ctx_, mem, &attr) != 0) {
        fprintf(stderr, "[vision] bind output failed\n");
        return false;
    }
    shared_out_ = true;
    return true;
}

bool VisionStage::bind_input(rknn_tensor_mem* mem) {
    rknn_tensor_attr attr = in_attr_;
    attr.index = 0;
    attr.pass_through = 1;
    if (rknn_set_io_mem(ctx_, mem, &attr) != 0) {
        fprintf(stderr, "[vision] bind input failed\n");
        return false;
    }
    shared_in_ = true;
    return true;
}

bool VisionPipeline::init_shared(const std::vector<std::string>& models, int core_mask) {
    if (!init(models, core_mask)) return false;
    for (size_t i = 0; i + 1 < stages_.size(); ++i) {
        const uint32_t bytes = stages_[i].output_bytes();
        rknn_tensor_mem* mem = stages_[i].alloc_output_mem();
        if (!mem) return false;
        shared_mems_.push_back(mem);
        if (!stages_[i].bind_output(mem)) return false;
        rknn_tensor_mem* alias = rknn_create_mem_from_fd(
            stages_[i + 1].context(), mem->fd, mem->virt_addr, bytes, 0);
        if (!alias) {
            fprintf(stderr, "[vision] create_mem_from_fd failed at boundary %zu\n", i);
            return false;
        }
        shared_mems_.push_back(alias);
        if (!stages_[i + 1].bind_input(alias)) return false;
        fprintf(stderr, "[vision] shared boundary %zu->%zu: %u bytes\n", i, i + 1, bytes);
    }
    return true;
}

void VisionStage::release() {    if (ctx_) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
    model_.clear();
    model_.shrink_to_fit();
    in_half_.clear();
    in_half_.shrink_to_fit();
    out_buf_.clear();
    out_buf_.shrink_to_fit();
    ready_ = false;
}

void VisionPipeline::release() {
    // Buffers belong to their creating contexts; rknn_destroy frees them.
    shared_mems_.clear();
    for (auto& stage : stages_) stage.release();
}

bool VisionPipeline::init(const std::vector<std::string>& models, int core_mask) {    stages_.resize(models.size());
    for (size_t i = 0; i < models.size(); ++i) {
        if (!stages_[i].init(models[i], core_mask)) return false;
        if (i > 0 && stages_[i].input_elems() != stages_[i - 1].output_elems()) {
            fprintf(stderr, "[vision] stage %zu input %d != previous output %d\n", i,
                    stages_[i].input_elems(), stages_[i - 1].output_elems());
            return false;
        }
    }
    return true;
}

bool VisionPipeline::run(const float* input, size_t count, std::vector<float>& tokens,
                         std::vector<double>* stage_ms) {
    if (stage_ms) stage_ms->assign(stages_.size(), 0.0);
    std::vector<float> buffer;
    const float* current = input;
    size_t current_count = count;
    for (size_t i = 0; i < stages_.size(); ++i) {
        if (current_count < (size_t)stages_[i].input_elems())
            current_count = (size_t)stages_[i].input_elems();
        double ms = 0;
        if (!stages_[i].run(current, current_count, buffer, &ms)) return false;
        if (stage_ms) (*stage_ms)[i] = ms;
        current = buffer.data();
        current_count = buffer.size();
    }
    tokens = buffer;
    return true;
}

}  // namespace gp
