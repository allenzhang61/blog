#include "llm/data.hpp"

namespace llm {

bool GPT2BPETokenizer::load_ranks(const std::string& path) {
    std::ifstream in(path);
    if (!in) return false;
    ranks_.clear();
    token_by_rank_.clear();
    std::string line;
    while (std::getline(in, line)) {
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        auto bytes = parse_hex(line.substr(0, tab));
        int64_t rank = std::stoll(line.substr(tab + 1));
        ranks_[bytes] = rank;
        token_by_rank_[rank] = bytes;
    }
    return !ranks_.empty();
}

bool GPT2BPETokenizer::load_samples(const std::string& path) {
    std::ifstream in(path);
    if (!in) return false;
    samples_.clear();
    std::string line;
    while (std::getline(in, line)) {
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string text = unescape(line.substr(0, tab));
        std::vector<int64_t> ids;
        std::stringstream ss(line.substr(tab + 1));
        std::string item;
        while (std::getline(ss, item, ',')) if (!item.empty()) ids.push_back(std::stoll(item));
        samples_[text] = ids;
    }
    return true;
}

std::vector<int64_t> GPT2BPETokenizer::encode(const std::string& text) const {
    auto it = samples_.find(text);
    if (it != samples_.end()) return it->second;
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
    for (unsigned char ch : text) ids.push_back(static_cast<int64_t>(ch));
    return ids;
}

std::string GPT2BPETokenizer::decode(const std::vector<int64_t>& ids) const {
    for (const auto& kv : samples_) if (kv.second == ids) return kv.first;
    if (!token_by_rank_.empty()) {
        std::string text;
        for (auto id : ids) {
            auto it = token_by_rank_.find(id);
            if (it != token_by_rank_.end()) {
                for (auto b : it->second) text.push_back(static_cast<char>(b));
            }
        }
        return text;
    }
    std::string text;
    for (auto id : ids) if (id >= 0 && id < 256) text.push_back(static_cast<char>(id));
    return text;
}

std::string GPT2BPETokenizer::unescape(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            if (s[i + 1] == 'n') { out.push_back('\n'); ++i; }
            else if (s[i + 1] == 't') { out.push_back('\t'); ++i; }
            else if (s[i + 1] == '\\') { out.push_back('\\'); ++i; }
            else out.push_back(s[i]);
        } else out.push_back(s[i]);
    }
    return out;
}

int GPT2BPETokenizer::hex_value(char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return 10 + ch - 'a';
    if (ch >= 'A' && ch <= 'F') return 10 + ch - 'A';
    throw std::runtime_error("invalid hex digit");
}

std::vector<unsigned char> GPT2BPETokenizer::parse_hex(const std::string& hex) {
    if (hex.size() % 2 != 0) throw std::runtime_error("invalid hex bytes");
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
            while (i < text.size() && is_alpha(static_cast<unsigned char>(text[i]))) ++i;
            pieces.push_back(text.substr(start, i - start));
        } else if (is_digit(ch)) {
            ++i;
            while (i < text.size() && is_digit(static_cast<unsigned char>(text[i]))) ++i;
            pieces.push_back(text.substr(start, i - start));
        } else if (!is_space(ch)) {
            ++i;
            while (i < text.size()) {
                unsigned char next = static_cast<unsigned char>(text[i]);
                if (is_space(next) || is_alpha(next) || is_digit(next)) break;
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
    for (auto b : bytes) parts.push_back({b});
    if (parts.empty()) return {};

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
        if (best_index == parts.size()) break;
        parts[best_index].insert(parts[best_index].end(), parts[best_index + 1].begin(), parts[best_index + 1].end());
        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
    }

    std::vector<int64_t> ids;
    for (const auto& part : parts) {
        auto it = ranks_.find(part);
        if (it == ranks_.end()) {
            for (auto b : part) ids.push_back(static_cast<int64_t>(b));
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

DataLoader::DataLoader(std::vector<int64_t> ids, int64_t batch, int64_t context, int64_t stride_, bool shuffle_)
    : tokens(std::move(ids)), batch_size(batch), context_length(context), stride(stride_), shuffle(shuffle_) {
    for (size_t i = 0; i + static_cast<size_t>(context_length) < tokens.size(); i += static_cast<size_t>(stride)) starts.push_back(i);
    if (shuffle) std::shuffle(starts.begin(), starts.end(), std::mt19937(123));
}

void DataLoader::reset() {
    cursor = 0;
    if (shuffle) std::shuffle(starts.begin(), starts.end(), std::mt19937(123));
}

bool DataLoader::next(Tensor& input, Tensor& target) {
    if (cursor >= starts.size()) return false;
    int64_t actual = std::min<int64_t>(batch_size, static_cast<int64_t>(starts.size() - cursor));
    std::vector<int64_t> x(actual * context_length), y(actual * context_length);
    for (int64_t b = 0; b < actual; ++b) {
        size_t start = starts[cursor++];
        for (int64_t t = 0; t < context_length; ++t) {
            x[b * context_length + t] = tokens[start + t];
            y[b * context_length + t] = tokens[start + t + 1];
        }
    }
    input = Tensor::from_ints(x, {actual, context_length});
    target = Tensor::from_ints(y, {actual, context_length});
    return true;
}

} // namespace llm
