#pragma once

#include "../core/config.h"
#include "../core/safetensors.h"

#include <string>
#include <vector>

namespace llm_inference {

// linear attention 层一次性解析好的权重引用。
struct LinearAttnWeights {
    TensorRef in_proj_qkv;
    TensorRef in_proj_z;
    TensorRef in_proj_b;
    TensorRef in_proj_a;
    TensorRef conv1d;
    TensorRef a_log;
    TensorRef dt_bias;
    TensorRef norm;
    TensorRef out_proj;
};

// full attention 层一次性解析好的权重引用。
struct FullAttnWeights {
    TensorRef q_proj;
    TensorRef k_proj;
    TensorRef v_proj;
    TensorRef q_norm;
    TensorRef k_norm;
    TensorRef o_proj;
};

// MLP 层一次性解析好的权重引用。
struct MlpWeights {
    TensorRef gate;
    TensorRef up;
    TensorRef down;
};

// 单个 transformer 层解析好的权重引用集合。
struct LayerWeights {
    // 层类型："linear_attention" 或 "full_attention"。
    std::string type;
    TensorRef input_norm;
    TensorRef post_norm;
    // 仅 linear_attention 层有效。
    LinearAttnWeights lin;
    // 仅 full_attention 层有效。
    FullAttnWeights full;
    MlpWeights mlp;
};

// 整个模型解析好的权重引用（embed / final norm / 各层）。
struct ModelParams {
    TensorRef embed_tokens;
    TensorRef final_norm;
    std::vector<LayerWeights> layers;
};

// 校验 Qwen3.5 推理路径需要的 tensor 是否齐全。
void validate_qwen_tensors(const ModelWeights & weights, const ModelConfig & config);

// 从 mmap 权重一次性解析出结构化 ModelParams（不拷贝权重数据）。
ModelParams parse_model_params(const ModelWeights & weights, const ModelConfig & config);

} // namespace llm_inference
