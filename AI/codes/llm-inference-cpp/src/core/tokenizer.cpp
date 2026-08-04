#include "tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace llm_inference {

static std::string json_unescape_string(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 >= s.size()) {
            out.push_back(s[i]);
            continue;
        }
        const char e = s[++i];
        switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u':
            // vocab.json for byte-level BPE mostly stores UTF-8 directly; keep unicode escapes readable enough for now.
            if (i + 4 < s.size()) {
                out += "\\u" + s.substr(i + 1, 4);
                i += 4;
            }
            break;
        default:
            out.push_back(e);
        }
    }
    return out;
}

std::unordered_map<int, std::string> load_vocab_reverse(const fs::path & model_dir, double & elapsed) {
    auto start = Clock::now();
    const std::string json = read_text_file(model_dir / "vocab.json");
    std::unordered_map<int, std::string> vocab;
    const std::regex item(R"REGEX("((?:\\.|[^"\\])*)"\s*:\s*([0-9]+))REGEX");
    for (auto it = std::sregex_iterator(json.begin(), json.end(), item); it != std::sregex_iterator(); ++it) {
        vocab.emplace(std::stoi((*it)[2].str()), json_unescape_string((*it)[1].str()));
    }
    elapsed = elapsed_s(start);
    return vocab;
}

static std::unordered_map<uint32_t, uint8_t> bytes_to_unicode_inverse() {
    std::vector<int> bs;
    for (int i = static_cast<int>('!'); i <= static_cast<int>('~'); ++i) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; ++i) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; ++i) bs.push_back(i);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    std::unordered_map<uint32_t, uint8_t> inv;
    for (size_t i = 0; i < bs.size(); ++i) {
        inv[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
    }
    return inv;
}

static bool next_utf8_codepoint(const std::string & s, size_t & i, uint32_t & cp) {
    if (i >= s.size()) {
        return false;
    }
    const unsigned char c = static_cast<unsigned char>(s[i++]);
    if (c < 0x80) {
        cp = c;
        return true;
    }
    int extra = 0;
    cp = 0;
    if ((c & 0xE0) == 0xC0) {
        cp = c & 0x1F;
        extra = 1;
    } else if ((c & 0xF0) == 0xE0) {
        cp = c & 0x0F;
        extra = 2;
    } else if ((c & 0xF8) == 0xF0) {
        cp = c & 0x07;
        extra = 3;
    } else {
        cp = c;
        return true;
    }
    for (int j = 0; j < extra && i < s.size(); ++j) {
        cp = (cp << 6) | (static_cast<unsigned char>(s[i++]) & 0x3F);
    }
    return true;
}

static std::string decode_piece(const std::string & piece) {
    static const auto inv = bytes_to_unicode_inverse();
    std::string bytes;
    size_t i = 0;
    uint32_t cp = 0;
    while (next_utf8_codepoint(piece, i, cp)) {
        auto it = inv.find(cp);
        if (it != inv.end()) {
            bytes.push_back(static_cast<char>(it->second));
        } else if (cp < 128) {
            bytes.push_back(static_cast<char>(cp));
        }
    }
    return bytes;
}

std::string detokenize(const std::vector<int> & ids, const std::unordered_map<int, std::string> & vocab) {
    std::string out;
    for (int id : ids) {
        auto it = vocab.find(id);
        if (it == vocab.end()) {
            out += "<id:" + std::to_string(id) + ">";
            continue;
        }
        const std::string & piece = it->second;
        if (piece.rfind("<|", 0) == 0) {
            continue;
        }
        out += decode_piece(piece);
    }
    return out;
}

std::vector<int> resolve_input_ids(const Args & args) {
    if (!args.input_ids.empty()) {
        return args.input_ids;
    }
    if (args.prompt == DEFAULT_PROMPT && !args.disable_thinking) {
        return DEFAULT_PROMPT_IDS;
    }
    throw std::runtime_error("当前 C++ 版本尚未实现 tokenizer。请使用默认 prompt，或用 --input-ids 传入 token ids。");
}

} // namespace llm_inference
