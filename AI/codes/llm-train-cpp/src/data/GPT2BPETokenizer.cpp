#include "llm/data/GPT2BPETokenizer.hpp"

namespace llm {

GPT2BPETokenizer::GPT2BPETokenizer(const std::string& ranks_path) {
    if (!load_ranks(ranks_path)) {
        throw std::runtime_error("failed to load GPT-2 BPE ranks: " + ranks_path);
    }
}

bool GPT2BPETokenizer::load_ranks(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    ranks_.clear();
    token_by_rank_.clear();
    std::string line;
    while (std::getline(in, line)) {
        auto tab = line.find('\t');
        if (tab == std::string::npos) {
            continue;
        }
        auto bytes = parse_hex(line.substr(0, tab));
        int64_t rank = std::stoll(line.substr(tab + 1));
        ranks_[bytes] = rank;
        token_by_rank_[rank] = bytes;
    }
    return !ranks_.empty();
}

std::vector<int64_t> GPT2BPETokenizer::encode(const std::string& text) const {
    if (!ranks_.empty()) {
        std::vector<int64_t> ids;
        for (const auto& piece : split_gpt2_like(text)) {
            auto piece_ids = bpe_encode_bytes(std::vector<unsigned char>(piece.begin(), piece.end()));
            ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
        }
        return ids;
    }
    std::vector<int64_t> ids;
    ids.reserve(text.size());
    for (unsigned char ch : text) {
        ids.push_back(static_cast<int64_t>(ch));
    }
    return ids;
}

std::string GPT2BPETokenizer::decode(const std::vector<int64_t>& ids) const {
    if (!token_by_rank_.empty()) {
        std::string text;
        for (auto id : ids) {
            auto it = token_by_rank_.find(id);
            if (it != token_by_rank_.end()) {
                for (auto b : it->second) {
                    text.push_back(static_cast<char>(b));
                }
            }
        }
        return text;
    }
    std::string text;
    for (auto id : ids) {
        if (id >= 0 && id < 256) {
            text.push_back(static_cast<char>(id));
        }
    }
    return text;
}

int GPT2BPETokenizer::hex_value(char ch) {
    if (ch >= '0' && ch <= '9') {
        return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
        return 10 + ch - 'a';
    }
    if (ch >= 'A' && ch <= 'F') {
        return 10 + ch - 'A';
    }
    throw std::runtime_error("invalid hex digit");
}

std::vector<unsigned char> GPT2BPETokenizer::parse_hex(const std::string& hex) {
    if (hex.size() % 2 != 0) {
        throw std::runtime_error("invalid hex bytes");
    }
    std::vector<unsigned char> bytes;
    bytes.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        bytes.push_back(static_cast<unsigned char>((hex_value(hex[i]) << 4) | hex_value(hex[i + 1])));
    }
    return bytes;
}

bool GPT2BPETokenizer::is_alpha(unsigned char ch) {
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
}

bool GPT2BPETokenizer::is_digit(unsigned char ch) {
    return ch >= '0' && ch <= '9';
}

bool GPT2BPETokenizer::is_space(unsigned char ch) {
    return ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r';
}

std::vector<std::string> GPT2BPETokenizer::split_gpt2_like(const std::string& text) {
    std::vector<std::string> pieces;
    size_t i = 0;
    while (i < text.size()) {
        unsigned char ch = static_cast<unsigned char>(text[i]);
        size_t start = i;
        bool leading_space = false;
        if (is_space(ch)) {
            leading_space = true;
            ++i;
            if (i >= text.size()) {
                pieces.push_back(text.substr(start, i - start));
                break;
            }
            ch = static_cast<unsigned char>(text[i]);
        }
        if (is_alpha(ch)) {
            ++i;
            while (i < text.size() && is_alpha(static_cast<unsigned char>(text[i]))) {
                ++i;
            }
            pieces.push_back(text.substr(start, i - start));
        } else if (is_digit(ch)) {
            ++i;
            while (i < text.size() && is_digit(static_cast<unsigned char>(text[i]))) {
                ++i;
            }
            pieces.push_back(text.substr(start, i - start));
        } else if (!is_space(ch)) {
            ++i;
            while (i < text.size()) {
                unsigned char next = static_cast<unsigned char>(text[i]);
                if (is_space(next) || is_alpha(next) || is_digit(next)) {
                    break;
                }
                ++i;
            }
            pieces.push_back(text.substr(start, i - start));
        } else if (!leading_space) {
            pieces.push_back(text.substr(start, i - start));
        }
    }
    return pieces;
}

std::vector<int64_t> GPT2BPETokenizer::bpe_encode_bytes(const std::vector<unsigned char>& bytes) const {
    std::vector<std::vector<unsigned char>> parts;
    parts.reserve(bytes.size());
    for (auto b : bytes) {
        parts.push_back({b});
    }
    if (parts.empty()) {
        return {};
    }

    while (parts.size() > 1) {
        int64_t best_rank = std::numeric_limits<int64_t>::max();
        size_t best_index = parts.size();
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            auto merged = parts[i];
            merged.insert(merged.end(), parts[i + 1].begin(), parts[i + 1].end());
            auto it = ranks_.find(merged);
            if (it != ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_index = i;
            }
        }
        if (best_index == parts.size()) {
            break;
        }
        parts[best_index].insert(parts[best_index].end(), parts[best_index + 1].begin(), parts[best_index + 1].end());
        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
    }

    std::vector<int64_t> ids;
    for (const auto& part : parts) {
        auto it = ranks_.find(part);
        if (it == ranks_.end()) {
            for (auto b : part) {
                ids.push_back(static_cast<int64_t>(b));
            }
        } else {
            ids.push_back(it->second);
        }
    }
    return ids;
}

size_t GPT2BPETokenizer::VectorHash::operator()(const std::vector<unsigned char>& v) const {
    size_t h = 1469598103934665603ull;
    for (auto b : v) {
        h ^= static_cast<size_t>(b);
        h *= 1099511628211ull;
    }
    return h;
}

} // namespace llm
