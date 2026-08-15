//
// Created by zhangyoulun on 9/8/2026.
//

#include "GGUFTokenizer.h"

#include "format/gguf/GgufFile.h"
#include "utils/log/Log.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifndef PCRE2_CODE_UNIT_WIDTH
#define PCRE2_CODE_UNIT_WIDTH 8
#endif
#include <pcre2.h>

void GGUFTokenizer::append_utf8(std::string &out, uint32_t cp) {
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

uint32_t GGUFTokenizer::decode_utf8(const std::string &s, size_t &pos) {
    const auto b0 = static_cast<uint8_t>(s[pos]);
    if (b0 < 0x80) {
        pos += 1;
        return b0;
    }
    if ((b0 >> 5) == 0x6) {
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

void GGUFTokenizer::build_pre_tokenize_regex() {
    static const char *kPattern =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    int errcode = 0;
    PCRE2_SIZE erroffset = 0;
    pcre2_code *re = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(kPattern), PCRE2_ZERO_TERMINATED,
                                   PCRE2_UTF | PCRE2_UCP, &errcode, &erroffset, nullptr);
    if (re == nullptr) {
        PCRE2_UCHAR buf[256];
        pcre2_get_error_message(errcode, buf, sizeof(buf));
        throw std::runtime_error(std::string("PCRE2 编译失败：") +
                                 reinterpret_cast<const char *>(buf));
    }
    pre_tokenize_re = re;
}

GGUFTokenizer::GGUFTokenizer(const GgufFile &gguf) {
    // 1) vocab：tokenizer.ggml.tokens 数组，id = 下标。
    const std::vector<std::string> tokens =
        gguf.metadata<std::vector<std::string>>("tokenizer.ggml.tokens");
    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string &tok = tokens[i];
        vocab.emplace(tok, static_cast<int>(i));
        id_to_token.emplace(static_cast<int>(i), tok);
    }

    // 2) token_type：标记特殊 token（3=CONTROL, 4=USER_DEFINED 视为特殊，字面匹配）。
    if (gguf.contain_metadata("tokenizer.ggml.token_type")) {
        const std::vector<int64_t> token_types =
            gguf.metadata<std::vector<int64_t>>("tokenizer.ggml.token_type");
        for (size_t i = 0; i < token_types.size() && i < tokens.size(); ++i) {
            const int type = static_cast<int>(token_types[i]);
            if (type == 3 || type == 4) {
                special_tokens.emplace(tokens[i], static_cast<int>(i));
            }
        }
    }

    // 3) merges：出现顺序即 rank。
    if (gguf.contain_metadata("tokenizer.ggml.merges")) {
        const std::vector<std::string> merges =
            gguf.metadata<std::vector<std::string>>("tokenizer.ggml.merges");
        int rank = 0;
        for (const std::string &pair : merges) {
            const size_t space = pair.find(' ');
            if (space == std::string::npos) {
                continue;
            }
            merge_ranks.emplace(std::make_pair(pair.substr(0, space), pair.substr(space + 1)), rank);
            ++rank;
        }
    }

    build_byte_unicode_maps();
    build_pre_tokenize_regex();

    // BOS：DeepSeek base 默认 add_bos_token=true。
    if (gguf.contain_metadata("tokenizer.ggml.bos_token_id")) {
        bos_token_id_ = static_cast<int>(gguf.metadata<int64_t>("tokenizer.ggml.bos_token_id"));
    }
    if (gguf.contain_metadata("tokenizer.ggml.add_bos_token")) {
        add_bos_ = gguf.metadata<bool>("tokenizer.ggml.add_bos_token");
    } else {
        add_bos_ = (bos_token_id_ >= 0);
    }
}

GGUFTokenizer::~GGUFTokenizer() {
    if (pre_tokenize_re != nullptr) {
        pcre2_code_free(static_cast<pcre2_code *>(pre_tokenize_re));
        pre_tokenize_re = nullptr;
    }
}

void GGUFTokenizer::build_byte_unicode_maps() {
    std::vector<int> bs;
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);

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

std::vector<std::string> GGUFTokenizer::pre_tokenize(const std::string &text) const {
    std::vector<std::string> pieces;
    if (text.empty()) return pieces;
    auto *re = static_cast<pcre2_code *>(pre_tokenize_re);
    pcre2_match_data *md = pcre2_match_data_create_from_pattern(re, nullptr);
    const auto *subject = reinterpret_cast<PCRE2_SPTR>(text.data());
    const PCRE2_SIZE subject_len = text.size();
    PCRE2_SIZE offset = 0;
    while (offset < subject_len) {
        const int rc = pcre2_match(re, subject, subject_len, offset, 0, md, nullptr);
        if (rc < 0) break;
        const PCRE2_SIZE *ov = pcre2_get_ovector_pointer(md);
        const PCRE2_SIZE start = ov[0];
        const PCRE2_SIZE end = ov[1];
        if (start < end) {
            pieces.emplace_back(text.substr(start, end - start));
            offset = end;
        } else {
            offset = end + 1;
        }
    }
    pcre2_match_data_free(md);
    return pieces;
}

std::vector<std::string> GGUFTokenizer::bpe(const std::string &token) const {
    std::vector<std::string> symbols;
    size_t pos = 0;
    while (pos < token.size()) {
        const size_t start = pos;
        decode_utf8(token, pos);
        symbols.push_back(token.substr(start, pos - start));
    }
    if (symbols.size() < 2) return symbols;
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
        if (!found) break;
        symbols[best_index] = symbols[best_index] + symbols[best_index + 1];
        symbols.erase(symbols.begin() + static_cast<long>(best_index) + 1);
    }
    return symbols;
}

std::vector<int> GGUFTokenizer::encode(const std::string &text) const {
    std::vector<int> ids;
    if (add_bos_ && bos_token_id_ >= 0) {
        ids.push_back(bos_token_id_);
    }
    size_t i = 0;
    std::string buffer;
    auto flush_buffer = [&]() {
        if (buffer.empty()) return;
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
    while (i < text.size()) {
        bool matched = false;
        std::string best_special;
        int best_id = -1;
        for (const auto &[content, id] : special_tokens) {
            if (!content.empty() && text.compare(i, content.size(), content) == 0) {
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
            buffer.push_back(text[i]);
            ++i;
        }
    }
    flush_buffer();
    return ids;
}

std::string GGUFTokenizer::decode(const std::vector<int> &ids) const {
    std::string mapped;
    for (const int id : ids) {
        const auto it = id_to_token.find(id);
        if (it == id_to_token.end()) {
            throw std::runtime_error("id 不在 vocab 中：" + std::to_string(id));
        }
        mapped += it->second;
    }
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
            out += ch;
        }
    }
    return out;
}

void GGUFTokenizer::debug_dump() const {
    std::ostringstream out;
    out << "GGUFTokenizer:\n";
    out << "  vocab_size=" << vocab.size() << "\n";
    out << "  merges=" << merge_ranks.size() << "\n";
    out << "  special_tokens=" << special_tokens.size() << "\n";
    Log::debug(out.str());
}
