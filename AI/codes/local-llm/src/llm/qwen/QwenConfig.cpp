//
// Created by zhangyoulun on 8/8/2026.
//

#include "QwenConfig.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "thirdparty/nlohmann/json.hpp"
#include "utils/log/Log.h"


QwenConfig::QwenConfig(const std::string &model_dir) {
    const std::filesystem::path path(model_dir);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();

    const nlohmann::json json = nlohmann::json::parse(ss.str());

    const nlohmann::json & model_json = json.at("text_config");
    const nlohmann::json & rope_json = model_json.at("rope_parameters");
    const nlohmann::json & vision_json = json.at("vision_config");

    data.text.hidden_size = model_json.value("hidden_size", 0);
    data.text.num_hidden_layers = model_json.value("num_hidden_layers", 0);
    data.text.layer_types = model_json.value("layer_types", std::vector<std::string> {});
    data.text.num_attention_heads = model_json.value("num_attention_heads", 0);
    data.text.num_key_value_heads = model_json.value("num_key_value_heads", 0);
    data.text.head_dim = model_json.value("head_dim", 0);
    data.text.linear_conv_kernel_dim = model_json.value("linear_conv_kernel_dim", 4);
    data.text.linear_key_head_dim = model_json.value("linear_key_head_dim", 128);
    data.text.linear_num_key_heads = model_json.value("linear_num_key_heads", 16);
    data.text.linear_num_value_heads = model_json.value("linear_num_value_heads", 32);
    data.text.linear_value_head_dim = model_json.value("linear_value_head_dim", 128);
    data.text.rms_norm_eps = model_json.value("rms_norm_eps", 1e-6f);
    data.text.eos_token_id = model_json.value("eos_token_id", -1);
    data.text.rope_parameters.rope_theta = rope_json.value("rope_theta", 10000000.0f);
    data.text.rope_parameters.partial_rotary_factor = rope_json.value("partial_rotary_factor", 0.25f);

    data.architectures = json.value("architectures", std::vector<std::string> {});
    data.image_token_id = json.value("image_token_id", -1);
    data.model_type = json.value("model_type", std::string {});
    data.tie_word_embeddings = json.value("tie_word_embeddings", false);
    data.transformers_version = json.value("transformers_version", std::string {});
    data.video_token_id = json.value("video_token_id", -1);
    data.vision_end_token_id = json.value("vision_end_token_id", -1);
    data.vision_start_token_id = json.value("vision_start_token_id", -1);

    data.text.attention_bias = model_json.value("attention_bias", false);
    data.text.attention_dropout = model_json.value("attention_dropout", 0.0f);
    data.text.attn_output_gate = model_json.value("attn_output_gate", false);
    data.text.dtype = model_json.value("dtype", std::string {});
    data.text.full_attention_interval = model_json.value("full_attention_interval", 0);
    data.text.hidden_act = model_json.value("hidden_act", std::string {});
    data.text.initializer_range = model_json.value("initializer_range", 0.0f);
    data.text.intermediate_size = model_json.value("intermediate_size", 0);
    data.text.max_position_embeddings = model_json.value("max_position_embeddings", 0);
    data.text.mlp_only_layers = model_json.value("mlp_only_layers", std::vector<int> {});
    data.text.model_type = model_json.value("model_type", std::string {});
    data.text.mtp_num_hidden_layers = model_json.value("mtp_num_hidden_layers", 0);
    data.text.mtp_use_dedicated_embeddings = model_json.value("mtp_use_dedicated_embeddings", false);
    data.text.tie_word_embeddings = model_json.value("tie_word_embeddings", false);
    data.text.use_cache = model_json.value("use_cache", true);
    data.text.vocab_size = model_json.value("vocab_size", 0);
    data.text.mamba_ssm_dtype = model_json.value("mamba_ssm_dtype", std::string {});

    data.text.rope_parameters.mrope_interleaved = rope_json.value("mrope_interleaved", false);
    data.text.rope_parameters.mrope_section = rope_json.value("mrope_section", std::vector<int> {});
    data.text.rope_parameters.rope_type = rope_json.value("rope_type", std::string {});

    data.vision.deepstack_visual_indexes = vision_json.value("deepstack_visual_indexes", std::vector<int> {});
    data.vision.depth = vision_json.value("depth", 0);
    data.vision.hidden_act = vision_json.value("hidden_act", std::string {});
    data.vision.hidden_size = vision_json.value("hidden_size", 0);
    data.vision.in_channels = vision_json.value("in_channels", 0);
    data.vision.initializer_range = vision_json.value("initializer_range", 0.0f);
    data.vision.intermediate_size = vision_json.value("intermediate_size", 0);
    data.vision.model_type = vision_json.value("model_type", std::string {});
    data.vision.num_heads = vision_json.value("num_heads", 0);
    data.vision.num_position_embeddings = vision_json.value("num_position_embeddings", 0);
    data.vision.out_hidden_size = vision_json.value("out_hidden_size", 0);
    data.vision.patch_size = vision_json.value("patch_size", 0);
    data.vision.spatial_merge_size = vision_json.value("spatial_merge_size", 0);
    data.vision.temporal_patch_size = vision_json.value("temporal_patch_size", 0);
}

void QwenConfig::DebugDump() {
    nlohmann::json rope_json;
    rope_json["mrope_interleaved"] = data.text.rope_parameters.mrope_interleaved;
    rope_json["mrope_section"] = data.text.rope_parameters.mrope_section;
    rope_json["rope_type"] = data.text.rope_parameters.rope_type;
    rope_json["rope_theta"] = data.text.rope_parameters.rope_theta;
    rope_json["partial_rotary_factor"] = data.text.rope_parameters.partial_rotary_factor;

    nlohmann::json text_json;
    text_json["attention_bias"] = data.text.attention_bias;
    text_json["attention_dropout"] = data.text.attention_dropout;
    text_json["attn_output_gate"] = data.text.attn_output_gate;
    text_json["dtype"] = data.text.dtype;
    text_json["eos_token_id"] = data.text.eos_token_id;
    text_json["full_attention_interval"] = data.text.full_attention_interval;
    text_json["head_dim"] = data.text.head_dim;
    text_json["hidden_act"] = data.text.hidden_act;
    text_json["hidden_size"] = data.text.hidden_size;
    text_json["initializer_range"] = data.text.initializer_range;
    text_json["intermediate_size"] = data.text.intermediate_size;
    text_json["layer_types"] = data.text.layer_types;
    text_json["linear_conv_kernel_dim"] = data.text.linear_conv_kernel_dim;
    text_json["linear_key_head_dim"] = data.text.linear_key_head_dim;
    text_json["linear_num_key_heads"] = data.text.linear_num_key_heads;
    text_json["linear_num_value_heads"] = data.text.linear_num_value_heads;
    text_json["linear_value_head_dim"] = data.text.linear_value_head_dim;
    text_json["max_position_embeddings"] = data.text.max_position_embeddings;
    text_json["mlp_only_layers"] = data.text.mlp_only_layers;
    text_json["model_type"] = data.text.model_type;
    text_json["mtp_num_hidden_layers"] = data.text.mtp_num_hidden_layers;
    text_json["mtp_use_dedicated_embeddings"] = data.text.mtp_use_dedicated_embeddings;
    text_json["num_attention_heads"] = data.text.num_attention_heads;
    text_json["num_hidden_layers"] = data.text.num_hidden_layers;
    text_json["num_key_value_heads"] = data.text.num_key_value_heads;
    text_json["rms_norm_eps"] = data.text.rms_norm_eps;
    text_json["tie_word_embeddings"] = data.text.tie_word_embeddings;
    text_json["use_cache"] = data.text.use_cache;
    text_json["vocab_size"] = data.text.vocab_size;
    text_json["mamba_ssm_dtype"] = data.text.mamba_ssm_dtype;
    text_json["rope_parameters"] = rope_json;

    nlohmann::json vision_json;
    vision_json["deepstack_visual_indexes"] = data.vision.deepstack_visual_indexes;
    vision_json["depth"] = data.vision.depth;
    vision_json["hidden_act"] = data.vision.hidden_act;
    vision_json["hidden_size"] = data.vision.hidden_size;
    vision_json["in_channels"] = data.vision.in_channels;
    vision_json["initializer_range"] = data.vision.initializer_range;
    vision_json["intermediate_size"] = data.vision.intermediate_size;
    vision_json["model_type"] = data.vision.model_type;
    vision_json["num_heads"] = data.vision.num_heads;
    vision_json["num_position_embeddings"] = data.vision.num_position_embeddings;
    vision_json["out_hidden_size"] = data.vision.out_hidden_size;
    vision_json["patch_size"] = data.vision.patch_size;
    vision_json["spatial_merge_size"] = data.vision.spatial_merge_size;
    vision_json["temporal_patch_size"] = data.vision.temporal_patch_size;

    nlohmann::json root;
    root["architectures"] = data.architectures;
    root["image_token_id"] = data.image_token_id;
    root["model_type"] = data.model_type;
    root["text_config"] = text_json;
    root["tie_word_embeddings"] = data.tie_word_embeddings;
    root["transformers_version"] = data.transformers_version;
    root["video_token_id"] = data.video_token_id;
    root["vision_config"] = vision_json;
    root["vision_end_token_id"] = data.vision_end_token_id;
    root["vision_start_token_id"] = data.vision_start_token_id;

    Log::debug("QwenConfig:\n" + root.dump(4));
}
