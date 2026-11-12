#include "embed.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace gp {

EmbeddingTable::~EmbeddingTable() {
    if (file_) fclose(file_);
}

bool EmbeddingTable::open(const std::string& path, int rows, int width) {
    file_ = fopen(path.c_str(), "rb");
    if (!file_) return false;
    rows_ = rows;
    width_ = width;
    fseek(file_, 0, SEEK_END);
    long size = ftell(file_);
    if (size < (long)rows * width * 2) return false;
    return true;
}

bool EmbeddingTable::row(int id, float* dst) {
    if (!file_ || id < 0 || id >= rows_) return false;
    const long offset = (long)id * width_ * 2;
    if (fseek(file_, offset, SEEK_SET) != 0) return false;
    std::vector<uint16_t> raw((size_t)width_);
    if (fread(raw.data(), 2, (size_t)width_, file_) != (size_t)width_) return false;
    for (int i = 0; i < width_; ++i) {
        uint32_t bits = (uint32_t)raw[(size_t)i] << 16;
        float value;
        memcpy(&value, &bits, 4);
        dst[i] = value;
    }
    return true;
}

}  // namespace gp
