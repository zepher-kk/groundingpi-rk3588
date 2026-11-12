#pragma once
#include <cstdio>
#include <string>

namespace gp {

// Row-addressable bf16 embedding table (model.embed_tokens.weight export).
class EmbeddingTable {
public:
    ~EmbeddingTable();
    bool open(const std::string& path, int rows, int width);
    // Reads one row and converts bf16 -> float32.
    bool row(int id, float* dst);
    int width() const { return width_; }

private:
    FILE* file_ = nullptr;
    int rows_ = 0;
    int width_ = 0;
};

}  // namespace gp
