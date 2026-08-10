//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEKTOKENIZER_H
#define LOCAL_LLM_DEEPSEEKTOKENIZER_H

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class GgufFile;

// DeepSeek-V2-Lite 分词器：GPT-2 ByteLevel BPE，词表/merges 内嵌在 GGUF
// （tokenizer.ggml.tokens / tokenizer.ggml.merges / tokenizer.ggml.token_type）。
// pre-tokenizer 名义为 deepseek-llm，MVP 复用 GPT-2 Split 正则（对常见文本足够）。
class DeepseekTokenizer {
public:
    explicit DeepseekTokenizer(const GgufFile &gguf);
    ~DeepseekTokenizer();

    DeepseekTokenizer(const DeepseekTokenizer &) = delete;
    DeepseekTokenizer &operator=(const DeepseekTokenizer &) = delete;

    std::vector<int> Encode(const std::string &text) const;
    std::string Decode(const std::vector<int> &ids) const;
    void DebugDump() const;

private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> id_to_token;
    std::map<std::pair<std::string, std::string>, int> merge_ranks;
    std::unordered_map<std::string, int> special_tokens;

    std::unordered_map<uint8_t, std::string> byte_to_unicode;
    std::unordered_map<std::string, uint8_t> unicode_to_byte;

    int bos_token_id_ = -1;
    bool add_bos_ = false;

    void *pre_tokenize_re = nullptr;

    static void append_utf8(std::string &out, uint32_t cp);
    static uint32_t decode_utf8(const std::string &s, size_t &pos);
    void build_byte_unicode_maps();
    void build_pre_tokenize_regex();
    std::vector<std::string> pre_tokenize(const std::string &text) const;
    std::vector<std::string> bpe(const std::string &token) const;
};

#endif // LOCAL_LLM_DEEPSEEKTOKENIZER_H
