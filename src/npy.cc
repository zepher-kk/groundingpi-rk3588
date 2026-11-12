#include "npy.h"

#include <cstring>
#include <cstdlib>

#include "common.h"

namespace gp {

namespace {

int parse_shape(const std::string& header, std::vector<int>& shape) {
    size_t pos = header.find("'shape':");
    if (pos == std::string::npos) return 0;
    size_t open = header.find('(', pos);
    size_t close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos) return 0;
    std::string body = header.substr(open + 1, close - open - 1);
    int total = 1;
    size_t start = 0;
    while (start < body.size()) {
        size_t comma = body.find(',', start);
        std::string item = body.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        while (!item.empty() && item.front() == ' ') item.erase(item.begin());
        if (!item.empty()) {
            int value = atoi(item.c_str());
            shape.push_back(value);
            total *= value;
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return total;
}

bool header_is_f16(const std::string& header) {
    return header.find("'<f2'") != std::string::npos || header.find("\"<f2\"") != std::string::npos;
}

}  // namespace

bool load_npy_f32(const std::string& path, std::vector<float>& out, std::vector<int>& shape) {
    auto data = read_file(path);
    if (data.size() < 12 || memcmp(data.data(), "\x93NUMPY", 6) != 0) return false;
    const int major = data[6];
    size_t header_len, offset;
    if (major == 1) {
        header_len = (size_t)data[8] | ((size_t)data[9] << 8);
        offset = 10;
    } else {
        header_len = (size_t)data[8] | ((size_t)data[9] << 8) | ((size_t)data[10] << 16) | ((size_t)data[11] << 24);
        offset = 12;
    }
    if (offset + header_len > data.size()) return false;
    std::string header((const char*)data.data() + offset, header_len);
    shape.clear();
    int total = parse_shape(header, shape);
    if (total <= 0) return false;
    const uint8_t* payload = data.data() + offset + header_len;
    const size_t available = data.size() - offset - header_len;
    const bool f16 = header_is_f16(header);
    const size_t want = (size_t)total * (f16 ? 2 : 4);
    if (available < want) return false;
    out.resize((size_t)total);
    if (f16) {
        const __fp16* src = (const __fp16*)payload;
        for (int i = 0; i < total; ++i) out[(size_t)i] = (float)src[i];
    } else {
        memcpy(out.data(), payload, (size_t)total * 4);
    }
    return true;
}

}  // namespace gp
