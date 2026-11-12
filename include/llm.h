#pragma once
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "rkllm.h"

namespace gp {

// Thin wrapper around one resident RKLLM handle fed with embeddings.
class LlmSession {
public:
    struct Result {
        std::vector<int32_t> tokens;
        double run_ms = 0;
        double first_token_ms = 0;
        bool finished = false;
        bool error = false;
        int status = 0;
    };

    ~LlmSession();
    bool init(const std::string& model, int max_context = 512, int max_new = 32,
              int base_domain_id = 0);
    Result run(const float* embeds, int n_tokens);
    void shutdown();
    bool ready() const { return handle_ != nullptr; }
    double init_ms() const { return init_ms_; }

private:
    static int on_result(RKLLMResult* out, void* userdata, LLMCallState state);
    LLMHandle handle_ = nullptr;
    double init_ms_ = 0;
    Result* current_ = nullptr;
    std::chrono::steady_clock::time_point started_{};
};

}  // namespace gp
