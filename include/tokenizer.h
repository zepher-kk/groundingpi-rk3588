#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace gp {

// Byte-level BPE tokenizer matching the Qwen2 tokenizer used by the Python path.
class Tokenizer {
public:
    bool load(const std::string& vocab_path, const std::string& merges_path);
    std::vector<int32_t> encode(const std::string& text) const;
    bool loaded() const { return !vocab_.empty(); }

    // Exposed for debugging: the pre-tokenizer pieces of a text.
    std::vector<std::string> pretokenize(const std::string& text) const;

private:
    std::vector<int32_t> encode_piece(const std::string& piece) const;
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<std::string, int> merges_;
};

}  // namespace gp
