#include "llm_inference.h"

#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace llm_inference {

std::string read_text_file(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("无法读取文件：" + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int json_int(const std::string & json, const std::string & key, int default_value = 0) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stoi(match[1].str());
    }
    return default_value;
}

float json_float(const std::string & json, const std::string & key, float default_value = 0.0f) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:e-?[0-9]+)?)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stof(match[1].str());
    }
    return default_value;
}

std::vector<std::string> parse_layer_types(const std::string & json) {
    std::vector<std::string> values;
    const std::regex block("\"layer_types\"\\s*:\\s*\\[([^\\]]+)\\]");
    std::smatch match;
    if (!std::regex_search(json, match, block)) {
        return values;
    }
    const std::string body = match[1].str();
    const std::regex item("\"([^\"]+)\"");
    for (auto it = std::sregex_iterator(body.begin(), body.end(), item); it != std::sregex_iterator(); ++it) {
        values.push_back((*it)[1].str());
    }
    return values;
}

ModelConfig load_config(const fs::path & model_dir) {
    const std::string json = read_text_file(model_dir / "config.json");
    ModelConfig config;
    config.hidden_size = json_int(json, "hidden_size");
    config.intermediate_size = json_int(json, "intermediate_size");
    config.num_hidden_layers = json_int(json, "num_hidden_layers");
    config.num_attention_heads = json_int(json, "num_attention_heads");
    config.num_key_value_heads = json_int(json, "num_key_value_heads", config.num_attention_heads);
    config.vocab_size = json_int(json, "vocab_size");
    config.head_dim = json_int(json, "head_dim", config.hidden_size / std::max(config.num_attention_heads, 1));
    config.linear_num_value_heads = json_int(json, "linear_num_value_heads", 32);
    config.linear_num_key_heads = json_int(json, "linear_num_key_heads", 16);
    config.linear_key_head_dim = json_int(json, "linear_key_head_dim", 128);
    config.linear_value_head_dim = json_int(json, "linear_value_head_dim", 128);
    config.linear_conv_kernel_dim = json_int(json, "linear_conv_kernel_dim", 4);
    config.eos_token_id = json_int(json, "eos_token_id", -1);
    config.rms_norm_eps = json_float(json, "rms_norm_eps", 1e-6f);
    config.rope_theta = json_float(json, "rope_theta", 10000000.0f);
    config.partial_rotary_factor = json_float(json, "partial_rotary_factor", 0.25f);
    config.layer_types = parse_layer_types(json);

    if (config.hidden_size <= 0 || config.num_hidden_layers <= 0 || config.num_attention_heads <= 0) {
        throw std::runtime_error("config.json 缺少 Qwen 推理所需字段。");
    }
    if (static_cast<int>(config.layer_types.size()) != config.num_hidden_layers) {
        config.layer_types.clear();
        for (int i = 0; i < config.num_hidden_layers; ++i) {
            config.layer_types.push_back((i + 1) % 4 == 0 ? "full_attention" : "linear_attention");
        }
    }
    return config;
}

} // namespace llm_inference
