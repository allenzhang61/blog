#pragma once

#include "llm/tensor.hpp"

namespace llm {

class GPT2BPETokenizer {
public:
    bool load_ranks(const std::string& path);
    bool load_samples(const std::string& path);
    std::vector<int64_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int64_t>& ids) const;

private:
    static std::string unescape(const std::string& s);
    static int hex_value(char ch);
    static std::vector<unsigned char> parse_hex(const std::string& hex);
    static bool is_alpha(unsigned char ch);
    static bool is_digit(unsigned char ch);
    static bool is_space(unsigned char ch);
    static std::vector<std::string> split_gpt2_like(const std::string& text);
    std::vector<int64_t> bpe_encode_bytes(const std::vector<unsigned char>& bytes) const;

    struct VectorHash {
        size_t operator()(const std::vector<unsigned char>& v) const;
    };

    std::unordered_map<std::string, std::vector<int64_t>> samples_;
    std::unordered_map<std::vector<unsigned char>, int64_t, VectorHash> ranks_;
    std::unordered_map<int64_t, std::vector<unsigned char>> token_by_rank_;
};

class DataLoader {
public:
    std::vector<int64_t> tokens;
    int64_t batch_size;
    int64_t context_length;
    int64_t stride;
    bool shuffle;
    Device device{};
    size_t cursor{0};
    std::vector<size_t> starts;

    DataLoader(std::vector<int64_t> ids, int64_t batch, int64_t context, int64_t stride_, bool shuffle_, Device device_ = {});
    void reset();
    bool next(Tensor& input, Tensor& target);
};

} // namespace llm
