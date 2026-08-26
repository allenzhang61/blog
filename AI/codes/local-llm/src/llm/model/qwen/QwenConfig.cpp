//
// Created by zhangyoulun on 8/8/2026.
//

#include "QwenConfig.h"

#include <utility>

#include "format/MF.h"
#include "thirdparty/nlohmann/json.hpp"
#include "utils/log/Log.h"

namespace {
int meta_i32(const MF &file, const std::string &key, int default_value) {
    return file.contain_metadata(key) ? static_cast<int>(file.metadata<int64_t>(key)) : default_value;
}

int64_t meta_i64(const MF &file, const std::string &key, int64_t default_value) {
    return file.contain_metadata(key) ? file.metadata<int64_t>(key) : default_value;
}

float meta_f32(const MF &file, const std::string &key, float default_value) {
    return file.contain_metadata(key) ? file.metadata<float>(key) : default_value;
}

bool meta_bool(const MF &file, const std::string &key, bool default_value) {
    return file.contain_metadata(key) ? file.metadata<bool>(key) : default_value;
}

std::string meta_str(const MF &file, const std::string &key, std::string default_value = {}) {
    return file.contain_metadata(key) ? file.metadata<std::string>(key) : std::move(default_value);
}

std::vector<std::string> meta_str_array(const MF &file, const std::string &key) {
    return file.contain_metadata(key) ? file.metadata<std::vector<std::string>>(key) : std::vector<std::string> {};
}

std::vector<int> meta_i32_array(const MF &file, const std::string &key) {
    if (!file.contain_metadata(key)) {
        return {};
    }
    std::vector<int> out;
    for (const int64_t v : file.metadata<std::vector<int64_t>>(key)) {
        out.push_back(static_cast<int>(v));
    }
    return out;
}
} // namespace

QwenConfig::QwenConfig(const MF &mf) {
    data.text.hidden_size = meta_i64(mf, "text_config.hidden_size", 0);
    data.text.num_hidden_layers = meta_i32(mf, "text_config.num_hidden_layers", 0);
    data.text.layer_types = meta_str_array(mf, "text_config.layer_types");
    data.text.num_attention_heads = meta_i32(mf, "text_config.num_attention_heads", 0);
    data.text.num_key_value_heads = meta_i32(mf, "text_config.num_key_value_heads", 0);
    data.text.head_dim = meta_i32(mf, "text_config.head_dim", 0);
    data.text.linear_conv_kernel_dim = meta_i32(mf, "text_config.linear_conv_kernel_dim", 4);
    data.text.linear_key_head_dim = meta_i32(mf, "text_config.linear_key_head_dim", 128);
    data.text.linear_num_key_heads = meta_i32(mf, "text_config.linear_num_key_heads", 16);
    data.text.linear_num_value_heads = meta_i32(mf, "text_config.linear_num_value_heads", 32);
    data.text.linear_value_head_dim = meta_i32(mf, "text_config.linear_value_head_dim", 128);
    data.text.rms_norm_eps = meta_f32(mf, "text_config.rms_norm_eps", 1e-6f);
    data.text.eos_token_id = meta_i32(mf, "text_config.eos_token_id", -1);
    data.text.rope_parameters.rope_theta =
        meta_f32(mf, "text_config.rope_parameters.rope_theta", 10000000.0f);
    data.text.rope_parameters.partial_rotary_factor =
        meta_f32(mf, "text_config.rope_parameters.partial_rotary_factor", 0.25f);

    data.architectures = meta_str_array(mf, "architectures");
    data.image_token_id = meta_i32(mf, "image_token_id", -1);
    data.model_type = meta_str(mf, "model_type");
    data.tie_word_embeddings = meta_bool(mf, "tie_word_embeddings", false);
    data.transformers_version = meta_str(mf, "transformers_version");
    data.video_token_id = meta_i32(mf, "video_token_id", -1);
    data.vision_end_token_id = meta_i32(mf, "vision_end_token_id", -1);
    data.vision_start_token_id = meta_i32(mf, "vision_start_token_id", -1);

    data.text.attention_bias = meta_bool(mf, "text_config.attention_bias", false);
    data.text.attention_dropout = meta_f32(mf, "text_config.attention_dropout", 0.0f);
    data.text.attn_output_gate = meta_bool(mf, "text_config.attn_output_gate", false);
    data.text.dtype = meta_str(mf, "text_config.dtype");
    data.text.full_attention_interval = meta_i32(mf, "text_config.full_attention_interval", 0);
    data.text.hidden_act = meta_str(mf, "text_config.hidden_act");
    data.text.initializer_range = meta_f32(mf, "text_config.initializer_range", 0.0f);
    data.text.intermediate_size = meta_i32(mf, "text_config.intermediate_size", 0);
    data.text.max_position_embeddings = meta_i32(mf, "text_config.max_position_embeddings", 0);
    data.text.mlp_only_layers = meta_i32_array(mf, "text_config.mlp_only_layers");
    data.text.model_type = meta_str(mf, "text_config.model_type");
    data.text.mtp_num_hidden_layers = meta_i32(mf, "text_config.mtp_num_hidden_layers", 0);
    data.text.mtp_use_dedicated_embeddings =
        meta_bool(mf, "text_config.mtp_use_dedicated_embeddings", false);
    data.text.tie_word_embeddings = meta_bool(mf, "text_config.tie_word_embeddings", false);
    data.text.use_cache = meta_bool(mf, "text_config.use_cache", true);
    data.text.vocab_size = meta_i32(mf, "text_config.vocab_size", 0);
    data.text.mamba_ssm_dtype = meta_str(mf, "text_config.mamba_ssm_dtype");

    data.text.rope_parameters.mrope_interleaved =
        meta_bool(mf, "text_config.rope_parameters.mrope_interleaved", false);
    data.text.rope_parameters.mrope_section =
        meta_i32_array(mf, "text_config.rope_parameters.mrope_section");
    data.text.rope_parameters.rope_type =
        meta_str(mf, "text_config.rope_parameters.rope_type");

    data.vision.deepstack_visual_indexes =
        meta_i32_array(mf, "vision_config.deepstack_visual_indexes");
    data.vision.depth = meta_i32(mf, "vision_config.depth", 0);
    data.vision.hidden_act = meta_str(mf, "vision_config.hidden_act");
    data.vision.hidden_size = meta_i64(mf, "vision_config.hidden_size", 0);
    data.vision.in_channels = meta_i32(mf, "vision_config.in_channels", 0);
    data.vision.initializer_range = meta_f32(mf, "vision_config.initializer_range", 0.0f);
    data.vision.intermediate_size = meta_i32(mf, "vision_config.intermediate_size", 0);
    data.vision.model_type = meta_str(mf, "vision_config.model_type");
    data.vision.num_heads = meta_i32(mf, "vision_config.num_heads", 0);
    data.vision.num_position_embeddings = meta_i32(mf, "vision_config.num_position_embeddings", 0);
    data.vision.out_hidden_size = meta_i64(mf, "vision_config.out_hidden_size", 0);
    data.vision.patch_size = meta_i32(mf, "vision_config.patch_size", 0);
    data.vision.spatial_merge_size = meta_i32(mf, "vision_config.spatial_merge_size", 0);
    data.vision.temporal_patch_size = meta_i32(mf, "vision_config.temporal_patch_size", 0);
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
