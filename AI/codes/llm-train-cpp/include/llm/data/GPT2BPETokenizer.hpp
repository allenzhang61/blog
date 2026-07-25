#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm {

// GPT-2 BPE tokenizer。
// 使用已有 GPT-2 BPE rank 文件做 encode/decode，保证 token id 与 GPT-2 体系一致。
class GPT2BPETokenizer {
public:
    // 默认构造：不加载任何 rank，需后续调用 load_ranks。
    GPT2BPETokenizer() = default;

    // 构造时从指定路径加载 BPE rank 数据，加载失败抛出异常。
    explicit GPT2BPETokenizer(const std::string& ranks_path);

    // 加载 BPE merge/rank 数据。
    bool load_ranks(const std::string& path);

    // 将文本编码为 token id 序列。
    std::vector<int64_t> encode(const std::string& text) const;

    // 将 token id 序列解码回文本。
    std::string decode(const std::vector<int64_t>& ids) const;

private:
    // 将一个十六进制字符转换为数值。
    static int hex_value(char ch);

    // 将十六进制字符串解析为字节序列。
    static std::vector<unsigned char> parse_hex(const std::string& hex);

    // 判断 ASCII 字母。
    static bool is_alpha(unsigned char ch);

    // 判断 ASCII 数字。
    static bool is_digit(unsigned char ch);

    // 判断空白字符。
    static bool is_space(unsigned char ch);

    // 按 GPT-2 近似规则先把文本切成若干片段，再交给 BPE 合并。
    static std::vector<std::string> split_gpt2_like(const std::string& text);

    // 对一段字节执行 BPE 合并，返回 token id。
    std::vector<int64_t> bpe_encode_bytes(const std::vector<unsigned char>& bytes) const;

    // vector<unsigned char> 的哈希函数，用于 unordered_map key。
    struct VectorHash {
        size_t operator()(const std::vector<unsigned char>& v) const;
    };

    // BPE token 字节序列 -> rank/id。
    std::unordered_map<std::vector<unsigned char>, int64_t, VectorHash> ranks_;

    // rank/id -> BPE token 字节序列，用于 decode。
    std::unordered_map<int64_t, std::vector<unsigned char>> token_by_rank_;
};

} // namespace llm
