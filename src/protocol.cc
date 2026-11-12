#include "protocol.h"

#include <algorithm>
#include <cmath>

namespace gp {

namespace {
constexpr int32_t kImStart = 151644;
constexpr int32_t kImEnd = 151645;
constexpr int32_t kObjectRefStart = 151646;
constexpr int32_t kObjectRefEnd = 151647;
constexpr int32_t kBoxStart = 151648;
constexpr int32_t kBoxEnd = 151649;
constexpr int32_t kVisionStart = 151652;
constexpr int32_t kVisionEnd = 151653;
constexpr int32_t kImagePad = 151655;
constexpr int32_t kAssistant = 77091;
constexpr int32_t kNewline = 198;
constexpr int32_t kNone = 4064;
constexpr int32_t kCoordBase = 151669;
constexpr int32_t kCoordLast = 152668;

const int32_t kPrefix[] = {kImStart, 8948,        kNewline, 2610, 525, 264,
                           10950,    17847,       13,       kImEnd, kNewline, kImStart,
                           872,      kNewline};
const char kInstruction[] = "Locate the target referred to by the following description: ";
}  // namespace

Prompt build_prompt(const Tokenizer& tokenizer, const std::string& query) {
    Prompt prompt;
    auto& ids = prompt.ids;
    ids.insert(ids.end(), std::begin(kPrefix), std::end(kPrefix));

    const std::string text = std::string(kInstruction) + query + ".";
    auto body = tokenizer.encode(text);
    ids.insert(ids.end(), body.begin(), body.end());
    ids.push_back(kVisionStart);

    prompt.head_tokens = (int)ids.size();
    prompt.image_tokens = 256;
    ids.insert(ids.end(), 256, kImagePad);

    const int32_t tail_fixed[] = {kVisionEnd, kImEnd, kNewline, kImStart, kAssistant,
                                  kNewline,   kObjectRefStart};
    ids.insert(ids.end(), std::begin(tail_fixed), std::end(tail_fixed));
    auto query_ids = tokenizer.encode(query);
    ids.insert(ids.end(), query_ids.begin(), query_ids.end());
    ids.push_back(kObjectRefEnd);
    ids.push_back(kBoxStart);
    prompt.tail_tokens = (int)ids.size() - prompt.head_tokens - prompt.image_tokens;
    return prompt;
}

Box grid_to_box(const std::vector<int>& values, int width, int height) {
    Box box;
    if (values.size() != 4) return box;
    const float scale = std::min(448.0f / (float)width, 448.0f / (float)height);
    const int rw = std::max(1, (int)std::lround((float)width * scale));
    const int rh = std::max(1, (int)std::lround((float)height * scale));
    const int left = (448 - rw) / 2;
    const int top = (448 - rh) / 2;
    const float sx = (float)width / (float)rw;
    const float sy = (float)height / (float)rh;
    float raw[4];
    for (int i = 0; i < 4; ++i) raw[i] = (float)values[(size_t)i] * 448.0f / 999.0f;
    float mapped[4] = {(raw[0] - left) * sx, (raw[1] - top) * sy,
                       (raw[2] - left) * sx, (raw[3] - top) * sy};
    mapped[0] = std::min(std::max(mapped[0], 0.f), (float)width);
    mapped[1] = std::min(std::max(mapped[1], 0.f), (float)height);
    mapped[2] = std::min(std::max(mapped[2], 0.f), (float)width);
    mapped[3] = std::min(std::max(mapped[3], 0.f), (float)height);
    if (mapped[2] <= mapped[0] || mapped[3] <= mapped[1]) {
        box.valid = false;
        return box;
    }
    box.valid = true;
    box.x0 = mapped[0]; box.y0 = mapped[1]; box.x1 = mapped[2]; box.y1 = mapped[3];
    return box;
}

Box decode_box(const std::vector<int32_t>& tokens, int width, int height) {
    Box box;
    std::vector<int32_t> payload = tokens;
    if (!payload.empty() && payload.back() == kImEnd) payload.pop_back();
    if (payload.empty() || payload.back() != kBoxEnd) return box;
    payload.pop_back();  // box_end
    if (payload.size() == 1 && payload[0] == kNone) {
        box.valid = true;
        box.none = true;
        return box;
    }
    if (payload.size() != 4) return box;
    std::vector<int> values;
    for (int32_t id : payload) {
        if (id < kCoordBase || id > kCoordLast) return box;
        values.push_back((int)(id - kCoordBase));
    }
    return grid_to_box(values, width, height);
}

}  // namespace gp
