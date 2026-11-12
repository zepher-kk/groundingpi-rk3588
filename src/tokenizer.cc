#include "tokenizer.h"

#include <array>
#include <cstdio>
#include <cstring>

#include "common.h"

namespace gp {

namespace {

bool is_alpha(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80; }
bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
bool is_space(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
bool is_punct(unsigned char c) { return !is_space(c) && !is_alpha(c) && !is_digit(c); }

void utf8_append(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += (char)cp;
    } else if (cp < 0x800) {
        out += (char)(0xC0 | (cp >> 6));
        out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xE0 | (cp >> 12));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    }
}

const std::array<std::string, 256>& byte_map() {
    static std::array<std::string, 256> table;
    static bool ready = false;
    if (!ready) {
        int extra = 0;
        for (int b = 0; b < 256; ++b) {
            const bool self = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            utf8_append(table[b], self ? (uint32_t)b : (uint32_t)(256 + extra++));
        }
        ready = true;
    }
    return table;
}

std::vector<std::string> utf8_chars(const std::string& text) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = text[i];
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > text.size()) len = 1;
        out.emplace_back(text.substr(i, len));
        i += len;
    }
    return out;
}

const int32_t kBase = 151669;  // <0>

}  // namespace

bool Tokenizer::load(const std::string& vocab_path, const std::string& merges_path) {
    auto vocab = read_file(vocab_path);
    if (vocab.size() < 4) return false;
    uint32_t count;
    memcpy(&count, vocab.data(), 4);
    size_t offset = 4;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t id, len;
        memcpy(&id, vocab.data() + offset, 4); offset += 4;
        memcpy(&len, vocab.data() + offset, 4); offset += 4;
        if (offset + len > vocab.size()) return false;
        vocab_.emplace(std::string((const char*)vocab.data() + offset, len), (int32_t)id);
        offset += len;
    }

    auto merges = read_file(merges_path);
    if (merges.size() < 4) return false;
    uint32_t mcount;
    memcpy(&mcount, merges.data(), 4);
    offset = 4;
    for (uint32_t i = 0; i < mcount; ++i) {
        uint32_t len1, len2;
        memcpy(&len1, merges.data() + offset, 4); offset += 4;
        if (offset + len1 > merges.size()) return false;
        std::string left((const char*)merges.data() + offset, len1); offset += len1;
        memcpy(&len2, merges.data() + offset, 4); offset += 4;
        if (offset + len2 > merges.size()) return false;
        std::string right((const char*)merges.data() + offset, len2); offset += len2;
        merges_.emplace(left + '\x1f' + right, (int)i);
    }
    fprintf(stderr, "[tokenizer] vocab=%zu merges=%zu\n", vocab_.size(), merges_.size());
    return !vocab_.empty() && !merges_.empty();
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
    std::vector<std::string> pieces;
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        unsigned char c = text[i];
        // contractions: 's 't 're 've 'm 'll 'd (case-insensitive)
        if (c == '\'' && i + 1 < n) {
            std::string rest = text.substr(i + 1, std::min<size_t>(2, n - i - 1));
            for (char& ch : rest) ch = (char)tolower((unsigned char)ch);
            if (rest == "s" || rest == "t" || rest == "m" || rest == "d") {
                pieces.push_back(text.substr(i, 2)); i += 2; continue;
            }
            if (rest == "re" || rest == "ve" || rest == "ll") {
                pieces.push_back(text.substr(i, 3)); i += 3; continue;
            }
        }
        // optional single non-letter/digit/non-newline prefix + letters
        if (is_alpha(c)) {
            size_t k = i;
            while (k < n && is_alpha((unsigned char)text[k])) ++k;
            pieces.push_back(text.substr(i, k - i));
            i = k;
            continue;
        }
        if (!is_alpha(c) && !is_digit(c) && c != '\n' && c != '\r' && i + 1 < n &&
            is_alpha((unsigned char)text[i + 1])) {
            size_t k = i + 1;
            while (k < n && is_alpha((unsigned char)text[k])) ++k;
            pieces.push_back(text.substr(i, k - i));
            i = k;
            continue;
        }
        if (is_digit(c)) {
            pieces.push_back(text.substr(i, 1));
            ++i;
            continue;
        }
        // optional space + punctuation run + newlines
        size_t j = i;
        if (c == ' ' && i + 1 < n && is_punct((unsigned char)text[i + 1])) j = i + 1;
        if (is_punct((unsigned char)text[j])) {
            size_t k = j;
            while (k < n && is_punct((unsigned char)text[k])) ++k;
            while (k < n && (text[k] == '\n' || text[k] == '\r')) ++k;
            pieces.push_back(text.substr(i, k - i));
            i = k;
            continue;
        }
        // whitespace run
        size_t k = i;
        while (k < n && is_space((unsigned char)text[k])) ++k;
        pieces.push_back(text.substr(i, k - i));
        i = k;
    }
    return pieces;
}

std::vector<int32_t> Tokenizer::encode_piece(const std::string& piece) const {
    std::string mapped;
    mapped.reserve(piece.size() * 2);
    for (unsigned char c : piece) mapped += byte_map()[c];

    std::vector<std::string> symbols = utf8_chars(mapped);
    while (symbols.size() > 1) {
        int best_rank = INT32_MAX;
        size_t best_index = 0;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto it = merges_.find(symbols[i] + '\x1f' + symbols[i + 1]);
            if (it != merges_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_index = i;
            }
        }
        if (best_rank == INT32_MAX) break;
        symbols[best_index] += symbols[best_index + 1];
        symbols.erase(symbols.begin() + (long)best_index + 1);
    }

    std::vector<int32_t> ids;
    for (const auto& symbol : symbols) {
        auto it = vocab_.find(symbol);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
        } else {
            // byte fallback: <0xXX> tokens are not guaranteed to exist; emit <unk>
            for (const auto& ch : utf8_chars(symbol)) {
                auto jt = vocab_.find(ch);
                if (jt != vocab_.end()) ids.push_back(jt->second);
            }
        }
    }
    return ids;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    for (const auto& piece : pretokenize(text)) {
        auto part = encode_piece(piece);
        ids.insert(ids.end(), part.begin(), part.end());
    }
    return ids;
}

}  // namespace gp
