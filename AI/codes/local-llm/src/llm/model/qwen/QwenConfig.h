//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_QWENCONFIG_H
#define LOCAL_LLM_QWENCONFIG_H
#include <cstdint>
#include <string>
#include <vector>

class MF;

// text_config.rope_parameters 配置。
struct RopeParametersConfig {
    // RoPE theta；推理实际使用。
    float rope_theta = 10000000.0f;
    // RoPE 应用到 head_dim 的比例；推理实际使用。
    float partial_rotary_factor = 0.25f;

    // rope_parameters.mrope_interleaved；当前未使用。
    bool mrope_interleaved = false;
    // rope_parameters.mrope_section；当前未使用。
    std::vector<int> mrope_section;
    // rope_parameters.rope_type；当前未使用。
    std::string rope_type;
};

// text_config 配置。
struct TextConfig {
    // Transformer hidden size；推理实际使用。
    int64_t hidden_size = 0;
    // 总层数；推理实际使用。
    int num_hidden_layers = 0;
    // 每层类型，例如 linear_attention 或 full_attention；推理实际使用。
    std::vector<std::string> layer_types;
    // full attention 的 query heads 数；推理实际使用。
    int num_attention_heads = 0;
    // full attention 的 key/value heads 数；推理实际使用。
    int num_key_value_heads = 0;
    // 单个 attention head 的维度；推理实际使用。
    int head_dim = 0;
    // linear attention 前置 depthwise conv 的 kernel 大小；推理实际使用。
    int linear_conv_kernel_dim = 4;
    // linear attention 的 key head 维度；推理实际使用。
    int linear_key_head_dim = 0;
    // linear attention 的 key heads 数；推理实际使用。
    int linear_num_key_heads = 0;
    // linear attention 的 value heads 数；推理实际使用。
    int linear_num_value_heads = 0;
    // linear attention 的 value head 维度；推理实际使用。
    int linear_value_head_dim = 0;
    // RMSNorm epsilon；推理实际使用。
    float rms_norm_eps = 1e-6f;
    // EOS token id，未配置时为 -1；生成停止条件实际使用。
    int eos_token_id = -1;

    // text_config.rope_parameters；部分 RoPE 参数推理实际使用。
    RopeParametersConfig rope_parameters;

    // text_config.attention_bias；当前未使用。
    bool attention_bias = false;
    // text_config.attention_dropout；当前未使用。
    float attention_dropout = 0.0f;
    // text_config.attn_output_gate；当前未使用。
    bool attn_output_gate = false;
    // text_config.dtype；当前未使用，实际 tensor dtype 来自 safetensors metadata。
    std::string dtype;
    // text_config.full_attention_interval；当前未使用，实际层类型使用 layer_types。
    int full_attention_interval = 0;
    // text_config.hidden_act；当前未使用，当前实现固定使用 SiLU。
    std::string hidden_act;
    // text_config.initializer_range；当前未使用。
    float initializer_range = 0.0f;
    // text_config.intermediate_size；当前未使用，MLP 矩阵维度直接来自权重 shape。
    int intermediate_size = 0;
    // text_config.max_position_embeddings；当前未使用。
    int max_position_embeddings = 0;
    // text_config.mlp_only_layers；当前未使用。
    std::vector<int> mlp_only_layers;
    // text_config.model_type；当前未使用。
    std::string model_type;
    // text_config.mtp_num_hidden_layers；当前未使用。
    int mtp_num_hidden_layers = 0;
    // text_config.mtp_use_dedicated_embeddings；当前未使用。
    bool mtp_use_dedicated_embeddings = false;
    // text_config.tie_word_embeddings；当前未使用。
    bool tie_word_embeddings = false;
    // text_config.use_cache；当前未使用，当前实现总是维护 KV/recurrent cache。
    bool use_cache = true;
    // text_config.vocab_size；当前未使用，embedding/lm_head 维度直接来自权重 shape。
    int vocab_size = 0;
    // text_config.mamba_ssm_dtype；当前未使用。
    std::string mamba_ssm_dtype;
};

// vision_config 配置；当前推理不走视觉分支。
struct VisionConfig {
    // vision_config.deepstack_visual_indexes；当前未使用。
    std::vector<int> deepstack_visual_indexes;
    // vision_config.depth；当前未使用。
    int depth = 0;
    // vision_config.hidden_act；当前未使用。
    std::string hidden_act;
    // vision_config.hidden_size；当前未使用。
    int64_t hidden_size = 0;
    // vision_config.in_channels；当前未使用。
    int in_channels = 0;
    // vision_config.initializer_range；当前未使用。
    float initializer_range = 0.0f;
    // vision_config.intermediate_size；当前未使用。
    int intermediate_size = 0;
    // vision_config.model_type；当前未使用。
    std::string model_type;
    // vision_config.num_heads；当前未使用。
    int num_heads = 0;
    // vision_config.num_position_embeddings；当前未使用。
    int num_position_embeddings = 0;
    // vision_config.out_hidden_size；当前未使用。
    int64_t out_hidden_size = 0;
    // vision_config.patch_size；当前未使用。
    int patch_size = 0;
    // vision_config.spatial_merge_size；当前未使用。
    int spatial_merge_size = 0;
    // vision_config.temporal_patch_size；当前未使用。
    int temporal_patch_size = 0;
};

// config.json 顶层配置。
struct Data {
    // text_config；文本模型推理实际使用。
    TextConfig text;
    // vision_config；当前未使用。
    VisionConfig vision;

    // 顶层 architectures；当前未使用。
    std::vector<std::string> architectures;
    // 顶层 image token id；当前未使用。
    int image_token_id = -1;
    // 顶层 model_type；当前未使用。
    std::string model_type;
    // 顶层 tie_word_embeddings；当前未使用。
    bool tie_word_embeddings = false;
    // 顶层 transformers_version；当前未使用。
    std::string transformers_version;
    // 顶层 video token id；当前未使用。
    int video_token_id = -1;
    // 顶层 vision end token id；当前未使用。
    int vision_end_token_id = -1;
    // 顶层 vision start token id；当前未使用。
    int vision_start_token_id = -1;
};

class QwenConfig {
public:
    explicit QwenConfig(const MF &mf);

    Data data;

    void DebugDump();
};


#endif //LOCAL_LLM_QWENCONFIG_H
