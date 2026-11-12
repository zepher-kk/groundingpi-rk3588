#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "tokenizer.h"

namespace gp {

struct Prompt {
    std::vector<int32_t> ids;
    int head_tokens = 0;   // ids before the 256 image pads
    int image_tokens = 0;  // 256
    int tail_tokens = 0;
};

// Builds the exact token sequence of the Python reference prompt:
//   chat prefix + "Locate the target referred to by the following description: <q>."
//   + <|vision_start|> + 256 * <|image_pad|>
//   + <|vision_end|><|im_end|>\n<|im_start|>assistant\n
//   + <|object_ref_start|><q><|object_ref_end|><|box_start|>
Prompt build_prompt(const Tokenizer& tokenizer, const std::string& query);

struct Box {
    bool valid = false;
    bool none = false;
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};

// Decodes generated token ids and maps the 0..999 grid back to image pixels.
Box decode_box(const std::vector<int32_t>& tokens, int width, int height);

// Same box mapping used by the Python evaluator.
Box grid_to_box(const std::vector<int>& values, int width, int height);

}  // namespace gp
