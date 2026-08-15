//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_HFTOKENIZER_H
#define LOCAL_LLM_HFTOKENIZER_H

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "thirdparty/nlohmann/json.hpp"

// 基于 HuggingFace tokenizer.json 的 ByteLevel BPE 分词器。
class HFTokenizer {
public:
    // 从 tokenizer.json 路径加载 vocab / merges / added_tokens。
    explicit HFTokenizer(const std::string &tokenizer_json_path);
    explicit HFTokenizer(const nlohmann::json &tokenizer_json);
    ~HFTokenizer();

    // 不可拷贝（持有 PCRE2 句柄）。
    HFTokenizer(const HFTokenizer &) = delete;
    HFTokenizer &operator=(const HFTokenizer &) = delete;

    // 文本 -> token id 序列。
    std::vector<int> encode(const std::string &text) const;

    // token id 序列 -> 文本（encode 的逆过程）。
    std::string decode(const std::vector<int> &ids) const;

    // 打印基础信息（vocab 大小、merges 数量、特殊 token 等）。
    void debug_dump() const;

private:
    // token 字符串 -> id。
    std::unordered_map<std::string, int> vocab;
    // id -> token 字符串（decode 用）。
    std::unordered_map<int, std::string> id_to_token;
    // BPE 合并规则优先级："A B" 对 -> rank，rank 越小优先级越高。
    std::map<std::pair<std::string, std::string>, int> merge_ranks;

    // 特殊 token（added_tokens）：content -> id。
    std::unordered_map<std::string, int> special_tokens;

    // ByteLevel 的 byte<->unicode 映射表（GPT-2 固定 256 项）。
    std::unordered_map<uint8_t, std::string> byte_to_unicode;
    std::unordered_map<std::string, uint8_t> unicode_to_byte;

    // 预编译的 PCRE2 pattern（GPT-2 Split 正则）。实际类型为 pcre2_code*，
    // 头文件用 void* 持有以避免暴露 pcre2 头。
    void *pre_tokenize_re = nullptr;

    // 将一个 Unicode 码点编码为 UTF-8 追加到 out。
    static void append_utf8(std::string &out, uint32_t cp);
    // 从 UTF-8 字符串按位置解码一个码点，返回码点并把 pos 前移到下一个字符。
    static uint32_t decode_utf8(const std::string &s, size_t &pos);

    // 构建 GPT-2 ByteLevel 的 byte<->unicode 映射。
    void build_byte_unicode_maps();
    // 编译 GPT-2 Split 正则（PCRE2，启用 Unicode 属性）。
    void build_pre_tokenize_regex();

    // NFC 规范化（可先留桩，ASCII/多数场景可直接透传）。
    static std::string nfc_normalize(const std::string &text);

    // 用 PCRE2 跑 GPT-2 Split 正则做 pre-tokenize，切成若干 word。
    std::vector<std::string> pre_tokenize(const std::string &text) const;

    // 对单个 word（已做 byte->unicode 映射）跑 BPE，返回若干 subword。
    std::vector<std::string> bpe(const std::string &token) const;

    void parse(const nlohmann::json &json);
};

#endif //LOCAL_LLM_HFTOKENIZER_H
