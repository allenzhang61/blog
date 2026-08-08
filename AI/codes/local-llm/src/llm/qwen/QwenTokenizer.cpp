//
// Created by zhangyoulun on 8/8/2026.
//

#include "QwenTokenizer.h"

#include "utils/log/Log.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

// PCRE2 需要在包含头文件前指定 code unit 宽度。CMake 会通过编译定义提供该宏，
// 这里加保护以兼容直接用 clang++ 编译的场景。
#ifndef PCRE2_CODE_UNIT_WIDTH
#define PCRE2_CODE_UNIT_WIDTH 8
#endif
#include <pcre2.h>

#include <utf8proc.h>

#include "thirdparty/nlohmann/json.hpp"

// 将一个 Unicode 码点编码为 UTF-8 追加到 out。
void QwenTokenizer::append_utf8(std::string &out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// 从 UTF-8 字符串按位置解码一个码点，返回码点并把 pos 前移到下一个字符。
uint32_t QwenTokenizer::decode_utf8(const std::string &s, size_t &pos) {
    const auto b0 = static_cast<uint8_t>(s[pos]);
    if (b0 < 0x80) {
        pos += 1;
        return b0;
    }
    if ((b0 >> 5) == 0x6 && pos + 1 < s.size() + 1) {
        const auto b1 = static_cast<uint8_t>(s[pos + 1]);
        pos += 2;
        return ((b0 & 0x1F) << 6) | (b1 & 0x3F);
    }
    if ((b0 >> 4) == 0xE) {
        const auto b1 = static_cast<uint8_t>(s[pos + 1]);
        const auto b2 = static_cast<uint8_t>(s[pos + 2]);
        pos += 3;
        return ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
    }
    const auto b1 = static_cast<uint8_t>(s[pos + 1]);
    const auto b2 = static_cast<uint8_t>(s[pos + 2]);
    const auto b3 = static_cast<uint8_t>(s[pos + 3]);
    pos += 4;
    return ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F);
}

// 编译 GPT-2 Split 正则（PCRE2，启用 Unicode 属性 \p{L}/\p{N}/\p{M}）。
void QwenTokenizer::build_pre_tokenize_regex() {
    // 与 tokenizer.json 中 pre_tokenizer.Split 的正则完全一致。
    static const char *kPattern =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    int errcode = 0;
    PCRE2_SIZE erroffset = 0;
    pcre2_code *re = pcre2_compile(
        reinterpret_cast<PCRE2_SPTR>(kPattern), PCRE2_ZERO_TERMINATED,
        PCRE2_UTF | PCRE2_UCP, &errcode, &erroffset, nullptr);
    if (re == nullptr) {
        PCRE2_UCHAR buf[256];
        pcre2_get_error_message(errcode, buf, sizeof(buf));
        throw std::runtime_error(std::string("PCRE2 编译失败：") +
                                 reinterpret_cast<const char *>(buf));
    }
    pre_tokenize_re = re;
}


QwenTokenizer::QwenTokenizer(const std::string &tokenizer_json_path) {
    const std::filesystem::path path(tokenizer_json_path);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("无法打开 tokenizer.json：" + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();

    const nlohmann::json json = nlohmann::json::parse(ss.str());
    const nlohmann::json &model = json.at("model");
    if (model.value("type", std::string {}) != "BPE") {
        throw std::runtime_error("暂只支持 BPE 类型的 tokenizer");
    }

    // 1) vocab：token 字符串 -> id，同时建立 id -> token 反查表。
    for (const auto &[token, id_json] : model.at("vocab").items()) {
        const int id = id_json.get<int>();
        vocab.emplace(token, id);
        id_to_token.emplace(id, token);
    }

    // 2) merges：合并规则按出现顺序即为优先级 rank。
    int rank = 0;
    for (const auto &merge : model.at("merges")) {
        std::string left;
        std::string right;
        if (merge.is_string()) {
            // 形如 "A B"，以第一个空格分隔。
            const std::string pair = merge.get<std::string>();
            const size_t space = pair.find(' ');
            left = pair.substr(0, space);
            right = pair.substr(space + 1);
        } else {
            // 形如 ["A", "B"]。
            left = merge.at(0).get<std::string>();
            right = merge.at(1).get<std::string>();
        }
        merge_ranks.emplace(std::make_pair(left, right), rank);
        ++rank;
    }

    // 3) added_tokens：特殊 token content -> id。
    if (json.contains("added_tokens")) {
        for (const auto &tok : json.at("added_tokens")) {
            const auto content = tok.at("content").get<std::string>();
            const int id = tok.at("id").get<int>();
            special_tokens.emplace(content, id);
            // 特殊 token 的 id 不在 model.vocab 中，这里补进反查表供 Decode 使用。
            id_to_token.emplace(id, content);
        }
    }

    // 4) 构建 ByteLevel 的 byte<->unicode 固定映射。
    build_byte_unicode_maps();

    // 5) 编译 pre-tokenize 用的 PCRE2 正则。
    build_pre_tokenize_regex();
}

QwenTokenizer::~QwenTokenizer() {
    if (pre_tokenize_re != nullptr) {
        pcre2_code_free(static_cast<pcre2_code *>(pre_tokenize_re));
        pre_tokenize_re = nullptr;
    }
}

void QwenTokenizer::build_byte_unicode_maps() {
    // GPT-2 ByteLevel 标准映射：先把可打印区段的字节映射到自身对应的码点，
    // 其余不可打印字节依次映射到 256, 257, ... 的码点。
    std::vector<int> bs;
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
        bs.push_back(b);
    }
    for (int b = 0xA1; b <= 0xAC; ++b) {
        bs.push_back(b);
    }
    for (int b = 0xAE; b <= 0xFF; ++b) {
        bs.push_back(b);
    }

    std::vector<uint32_t> cs(bs.begin(), bs.end());
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(static_cast<uint32_t>(256 + n));
            ++n;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        std::string unicode_char;
        append_utf8(unicode_char, cs[i]);
        const auto byte = static_cast<uint8_t>(bs[i]);
        byte_to_unicode.emplace(byte, unicode_char);
        unicode_to_byte.emplace(unicode_char, byte);
    }
}

std::string QwenTokenizer::nfc_normalize(const std::string &text) {
    // 用 utf8proc 做 NFC 规范化，与 tokenizer.json 中 normalizer=NFC 对齐。
    utf8proc_uint8_t *out = nullptr;
    const utf8proc_ssize_t n = utf8proc_map(
        reinterpret_cast<const utf8proc_uint8_t *>(text.data()),
        static_cast<utf8proc_ssize_t>(text.size()), &out,
        static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
    if (n < 0 || out == nullptr) {
        // 规范化失败（例如非法 UTF-8）时退回原文，保证不崩溃。
        return text;
    }
    std::string result(reinterpret_cast<char *>(out), static_cast<size_t>(n));
    free(out);
    return result;
}

std::vector<std::string> QwenTokenizer::pre_tokenize(const std::string &text) const {
    // 用预编译的 PCRE2 正则（\p{L}/\p{N}/\p{M} 由 PCRE2 的完整 Unicode 属性表支持），
    // 从左到右不重叠地把 text 切成若干 word。behavior=Isolated 等价于“提取每个匹配”。
    std::vector<std::string> pieces;
    if (text.empty()) {
        return pieces;
    }

    auto *re = static_cast<pcre2_code *>(pre_tokenize_re);
    pcre2_match_data *md = pcre2_match_data_create_from_pattern(re, nullptr);
    const auto *subject = reinterpret_cast<PCRE2_SPTR>(text.data());
    const PCRE2_SIZE subject_len = text.size();

    PCRE2_SIZE offset = 0;
    while (offset < subject_len) {
        const int rc = pcre2_match(re, subject, subject_len, offset, 0, md, nullptr);
        if (rc < 0) {
            break; // PCRE2_ERROR_NOMATCH 或其他错误：停止。
        }
        const PCRE2_SIZE *ov = pcre2_get_ovector_pointer(md);
        const PCRE2_SIZE start = ov[0];
        const PCRE2_SIZE end = ov[1];
        if (start < end) {
            pieces.emplace_back(text.substr(start, end - start));
            offset = end;
        } else {
            // 空匹配保护：前移一个字节，避免死循环。
            offset = end + 1;
        }
    }

    pcre2_match_data_free(md);
    return pieces;
}

std::vector<std::string> QwenTokenizer::bpe(const std::string &token) const {
    // token 已是 byte->unicode 映射后的字符串，先拆成“单个 unicode 字符”的序列。
    std::vector<std::string> symbols;
    size_t pos = 0;
    while (pos < token.size()) {
        const size_t start = pos;
        decode_utf8(token, pos);
        symbols.push_back(token.substr(start, pos - start));
    }
    if (symbols.size() < 2) {
        return symbols;
    }

    // 反复合并 rank 最小的相邻对，直到无可合并对。
    while (true) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_index = 0;
        bool found = false;
        for (size_t k = 0; k + 1 < symbols.size(); ++k) {
            const auto it = merge_ranks.find(std::make_pair(symbols[k], symbols[k + 1]));
            if (it != merge_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_index = k;
                found = true;
            }
        }
        if (!found) {
            break;
        }
        symbols[best_index] = symbols[best_index] + symbols[best_index + 1];
        symbols.erase(symbols.begin() + static_cast<long>(best_index) + 1);
    }
    return symbols;
}

std::vector<int> QwenTokenizer::Encode(const std::string &text) const {
    std::vector<int> ids;

    // 1) NFC 规范化。
    const std::string normalized = nfc_normalize(text);

    // 2) 按特殊 token 把文本切成 [普通段, 特殊token, 普通段, ...]。
    //    在每个位置尝试匹配最长的特殊 token。
    size_t i = 0;
    std::string buffer; // 累积当前普通文本段
    auto flush_buffer = [&]() {
        if (buffer.empty()) {
            return;
        }
        // 3) 普通段：pre_tokenize -> byte 映射 -> bpe -> 查 vocab。
        for (const std::string &piece : pre_tokenize(buffer)) {
            std::string mapped;
            for (const char c : piece) {
                mapped += byte_to_unicode.at(static_cast<uint8_t>(c));
            }
            for (const std::string &sub : bpe(mapped)) {
                const auto it = vocab.find(sub);
                if (it == vocab.end()) {
                    throw std::runtime_error("BPE 结果不在 vocab 中：" + sub);
                }
                ids.push_back(it->second);
            }
        }
        buffer.clear();
    };

    while (i < normalized.size()) {
        bool matched = false;
        std::string best_special;
        int best_id = -1;
        for (const auto &[content, id] : special_tokens) {
            if (!content.empty() && normalized.compare(i, content.size(), content) == 0) {
                if (content.size() > best_special.size()) {
                    best_special = content;
                    best_id = id;
                }
            }
        }
        if (!best_special.empty()) {
            flush_buffer();
            ids.push_back(best_id);
            i += best_special.size();
            matched = true;
        }
        if (!matched) {
            buffer.push_back(normalized[i]);
            ++i;
        }
    }
    flush_buffer();

    return ids;
}

std::string QwenTokenizer::Decode(const std::vector<int> &ids) const {
    // 1) id -> token 字符串拼接（特殊 token 也在 id_to_token 里可直接还原）。
    std::string mapped;
    for (const int id : ids) {
        const auto it = id_to_token.find(id);
        if (it == id_to_token.end()) {
            throw std::runtime_error("id 不在 vocab 中：" + std::to_string(id));
        }
        mapped += it->second;
    }

    // 2) unicode -> byte 逐字符还原成原始 UTF-8 字节流。
    std::string out;
    size_t pos = 0;
    while (pos < mapped.size()) {
        const size_t start = pos;
        decode_utf8(mapped, pos);
        const std::string ch = mapped.substr(start, pos - start);
        const auto it = unicode_to_byte.find(ch);
        if (it != unicode_to_byte.end()) {
            out.push_back(static_cast<char>(it->second));
        } else {
            // 特殊 token 等不在 byte 映射表内的字符，原样输出。
            out += ch;
        }
    }
    return out;
}

void QwenTokenizer::DebugDump() const {
    std::ostringstream out;
    out << "QwenTokenizer:\n";
    out << "  vocab_size=" << vocab.size() << "\n";
    out << "  merges=" << merge_ranks.size() << "\n";
    out << "  special_tokens=" << special_tokens.size() << "\n";
    out << "  byte_unicode_map=" << byte_to_unicode.size() << "\n";
    for (const auto &[content, id] : special_tokens) {
        out << "    special: " << content << " -> " << id << "\n";
    }
    Log::debug(out.str());
}
