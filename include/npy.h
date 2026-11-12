#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gp {

// Load a C-order .npy file of float16/float32 into float32, returning its shape.
bool load_npy_f32(const std::string& path, std::vector<float>& out, std::vector<int>& shape);

}  // namespace gp
