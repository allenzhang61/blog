#pragma once

#include "common.h"

#include <string>
#include <vector>

namespace llm_inference {

// 从 config.json 中抽取出来的模型结构参数。
struct ModelConfig {
    // Transformer hidden size。
    int hidden_size = 0;
    // MLP 中间层维度。
    int intermediate_size = 0;
    // 总层数。
    int num_hidden_layers = 0;
    // full attention 的 query heads 数。
    int num_attention_heads = 0;
    // full attention 的 key/value heads 数。
    int num_key_value_heads = 0;
    // 词表大小。
    int vocab_size = 0;
    // 单个 attention head 的维度。
    int head_dim = 0;
    // linear attention 的 value heads 数。
    int linear_num_value_heads = 0;
    // linear attention 的 key heads 数。
    int linear_num_key_heads = 0;
    // linear attention 的 key head 维度。
    int linear_key_head_dim = 0;
    // linear attention 的 value head 维度。
    int linear_value_head_dim = 0;
    // linear attention 前置 depthwise conv 的 kernel 大小。
    int linear_conv_kernel_dim = 4;
    // EOS token id，未配置时为 -1。
    int eos_token_id = -1;
    // RMSNorm epsilon。
    float rms_norm_eps = 1e-6f;
    // RoPE theta。
    float rope_theta = 10000000.0f;
    // RoPE 应用到 head_dim 的比例。
    float partial_rotary_factor = 0.25f;
    // 每层类型，例如 linear_attention 或 full_attention。
    std::vector<std::string> layer_types;
};

// 以二进制模式读取整个文本文件。
std::string read_text_file(const fs::path & path);

// 从模型目录的 config.json 加载推理所需配置。
ModelConfig load_config(const fs::path & model_dir);

} // namespace llm_inference
