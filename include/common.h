#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace gp {

class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}
    void reset() { start_ = std::chrono::steady_clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start_).count();
    }
private:
    std::chrono::steady_clock::time_point start_;
};

std::vector<uint8_t> read_file(const std::string& path);
uint64_t now_ns();bool write_file(const std::string& path, const void* data, size_t bytes);
std::string json_escape(const std::string& text);

}  // namespace gp
