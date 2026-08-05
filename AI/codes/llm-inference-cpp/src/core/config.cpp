#include "config.h"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
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

ModelConfig load_config(const fs::path & model_dir) {
    const nlohmann::json json = nlohmann::json::parse(read_text_file(model_dir / "config.json"));
    const nlohmann::json & model_json =
        json.contains("text_config") && json["text_config"].is_object() ? json["text_config"] : json;
    const nlohmann::json & rope_json =
        model_json.contains("rope_parameters") && model_json["rope_parameters"].is_object() ?
            model_json["rope_parameters"] :
            model_json;
    const nlohmann::json empty_vision_json = nlohmann::json::object();
    const nlohmann::json & vision_json =
        json.contains("vision_config") && json["vision_config"].is_object() ? json["vision_config"] : empty_vision_json;

    ModelConfig config;
    config.text.hidden_size = model_json.value("hidden_size", 0);
    config.text.num_hidden_layers = model_json.value("num_hidden_layers", 0);
    config.text.layer_types = model_json.value("layer_types", std::vector<std::string> {});
    config.text.num_attention_heads = model_json.value("num_attention_heads", 0);
    config.text.num_key_value_heads = model_json.value("num_key_value_heads", config.text.num_attention_heads);
    config.text.head_dim = model_json.value("head_dim", 0);
    config.text.linear_conv_kernel_dim = model_json.value("linear_conv_kernel_dim", 4);
    config.text.linear_key_head_dim = model_json.value("linear_key_head_dim", 128);
    config.text.linear_num_key_heads = model_json.value("linear_num_key_heads", 16);
    config.text.linear_num_value_heads = model_json.value("linear_num_value_heads", 32);
    config.text.linear_value_head_dim = model_json.value("linear_value_head_dim", 128);
    config.text.rms_norm_eps = model_json.value("rms_norm_eps", 1e-6f);
    config.text.eos_token_id = model_json.value("eos_token_id", -1);
    config.text.rope_parameters.rope_theta = rope_json.value("rope_theta", 10000000.0f);
    config.text.rope_parameters.partial_rotary_factor = rope_json.value("partial_rotary_factor", 0.25f);
    if (config.text.head_dim <= 0) {
        config.text.head_dim = config.text.hidden_size / std::max(config.text.num_attention_heads, 1);
    }

    config.architectures = json.value("architectures", std::vector<std::string> {});
    config.image_token_id = json.value("image_token_id", -1);
    config.model_type = json.value("model_type", std::string {});
    config.tie_word_embeddings = json.value("tie_word_embeddings", false);
    config.transformers_version = json.value("transformers_version", std::string {});
    config.video_token_id = json.value("video_token_id", -1);
    config.vision_end_token_id = json.value("vision_end_token_id", -1);
    config.vision_start_token_id = json.value("vision_start_token_id", -1);

    config.text.attention_bias = model_json.value("attention_bias", false);
    config.text.attention_dropout = model_json.value("attention_dropout", 0.0f);
    config.text.attn_output_gate = model_json.value("attn_output_gate", false);
    config.text.dtype = model_json.value("dtype", std::string {});
    config.text.full_attention_interval = model_json.value("full_attention_interval", 0);
    config.text.hidden_act = model_json.value("hidden_act", std::string {});
    config.text.initializer_range = model_json.value("initializer_range", 0.0f);
    config.text.intermediate_size = model_json.value("intermediate_size", 0);
    config.text.max_position_embeddings = model_json.value("max_position_embeddings", 0);
    config.text.mlp_only_layers = model_json.value("mlp_only_layers", std::vector<int> {});
    config.text.text_model_type = model_json.value("model_type", std::string {});
    config.text.mtp_num_hidden_layers = model_json.value("mtp_num_hidden_layers", 0);
    config.text.mtp_use_dedicated_embeddings = model_json.value("mtp_use_dedicated_embeddings", false);
    config.text.text_tie_word_embeddings = model_json.value("tie_word_embeddings", false);
    config.text.use_cache = model_json.value("use_cache", true);
    config.text.vocab_size = model_json.value("vocab_size", 0);
    config.text.mamba_ssm_dtype = model_json.value("mamba_ssm_dtype", std::string {});

    config.text.rope_parameters.mrope_interleaved = rope_json.value("mrope_interleaved", false);
    config.text.rope_parameters.mrope_section = rope_json.value("mrope_section", std::vector<int> {});
    config.text.rope_parameters.rope_type = rope_json.value("rope_type", std::string {});

    config.vision.deepstack_visual_indexes = vision_json.value("deepstack_visual_indexes", std::vector<int> {});
    config.vision.depth = vision_json.value("depth", 0);
    config.vision.hidden_act = vision_json.value("hidden_act", std::string {});
    config.vision.hidden_size = vision_json.value("hidden_size", 0);
    config.vision.in_channels = vision_json.value("in_channels", 0);
    config.vision.initializer_range = vision_json.value("initializer_range", 0.0f);
    config.vision.intermediate_size = vision_json.value("intermediate_size", 0);
    config.vision.model_type = vision_json.value("model_type", std::string {});
    config.vision.num_heads = vision_json.value("num_heads", 0);
    config.vision.num_position_embeddings = vision_json.value("num_position_embeddings", 0);
    config.vision.out_hidden_size = vision_json.value("out_hidden_size", 0);
    config.vision.patch_size = vision_json.value("patch_size", 0);
    config.vision.spatial_merge_size = vision_json.value("spatial_merge_size", 0);
    config.vision.temporal_patch_size = vision_json.value("temporal_patch_size", 0);

    if (config.text.hidden_size <= 0 || config.text.num_hidden_layers <= 0 || config.text.num_attention_heads <= 0) {
        throw std::runtime_error("config.json 缺少 Qwen 推理所需字段。");
    }
    if (static_cast<int>(config.text.layer_types.size()) != config.text.num_hidden_layers) {
        config.text.layer_types.clear();
        for (int i = 0; i < config.text.num_hidden_layers; ++i) {
            config.text.layer_types.push_back((i + 1) % 4 == 0 ? "full_attention" : "linear_attention");
        }
    }
    return config;
}

} // namespace llm_inference
