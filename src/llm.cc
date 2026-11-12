#include "llm.h"

#include <chrono>
#include <cstdio>

#include "common.h"

namespace gp {

LlmSession::~LlmSession() { shutdown(); }

void LlmSession::shutdown() {
    if (handle_) {
        rkllm_destroy(handle_);
        handle_ = nullptr;
    }
}

int LlmSession::on_result(RKLLMResult* out, void* userdata, LLMCallState state) {
    auto* self = static_cast<LlmSession*>(userdata);
    if (!self || !self->current_) return 0;
    Result& result = *self->current_;
    if (state == RKLLM_RUN_NORMAL && out) {
        if (result.first_token_ms == 0) {
            result.first_token_ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - self->started_)
                                        .count();
        }
        result.tokens.push_back(out->token_id);
    } else if (state == RKLLM_RUN_FINISH) {
        result.finished = true;
    } else if (state == RKLLM_RUN_ERROR) {
        result.error = true;
    }
    return 0;
}

bool LlmSession::init(const std::string& model, int max_context, int max_new,
                      int base_domain_id) {
    Timer timer;
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = model.c_str();
    param.max_context_len = max_context;
    param.max_new_tokens = max_new;
    param.top_k = 1;
    param.top_p = 1.0f;
    param.temperature = 0.0f;
    param.repeat_penalty = 1.0f;
    param.skip_special_token = false;
    if (base_domain_id > 0) param.extend_param.base_domain_id = base_domain_id;

    RKLLMCallback callback{};
    callback.result_callback = &LlmSession::on_result;
    callback.result_userdata = this;

    if (rkllm_init(&handle_, &param, &callback) != 0 || handle_ == nullptr) {
        fprintf(stderr, "[llm] rkllm_init failed: %s\n", model.c_str());
        handle_ = nullptr;
        return false;
    }
    init_ms_ = timer.ms();
    fprintf(stderr, "[llm] loaded in %.0f ms\n", init_ms_);
    return true;
}

LlmSession::Result LlmSession::run(const float* embeds, int n_tokens) {
    Result result;
    if (!handle_) { result.status = -1; return result; }
    if (rkllm_clear_kv_cache(handle_, 0, nullptr, nullptr) != 0) {
        fprintf(stderr, "[llm] clear_kv_cache failed\n");
        result.status = -2;
        return result;
    }

    current_ = &result;
    started_ = std::chrono::steady_clock::now();
    RKLLMInput input{};
    input.role = "user";
    input.enable_thinking = false;
    input.input_type = RKLLM_INPUT_EMBED;
    input.embed_input.embed = const_cast<float*>(embeds);
    input.embed_input.n_tokens = (size_t)n_tokens;

    RKLLMInferParam infer{};
    infer.mode = RKLLM_INFER_GENERATE;
    infer.keep_history = 0;

    Timer timer;
    result.status = rkllm_run(handle_, &input, &infer, nullptr);
    result.run_ms = timer.ms();
    current_ = nullptr;
    return result;
}

}  // namespace gp
