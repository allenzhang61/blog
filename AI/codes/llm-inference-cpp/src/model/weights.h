#pragma once

#include "../core/config.h"
#include "../safetensors/safetensors.h"

#include <string>
#include <vector>

namespace llm_inference {

// linear attention 层一次性解析好的权重引用。
struct LinearAttnWeights {
    WeightData in_proj_qkv;
    WeightData in_proj_z;
    WeightData in_proj_b;
    WeightData in_proj_a;
    WeightData conv1d;
    WeightData a_log;
    WeightData dt_bias;
    WeightData norm;
    WeightData out_proj;
};

// full attention 层一次性解析好的权重引用。
struct FullAttnWeights {
    WeightData q_proj;
    WeightData k_proj;
    WeightData v_proj;
    WeightData q_norm;
    WeightData k_norm;
    WeightData o_proj;
};

// MLP 层一次性解析好的权重引用。
struct MlpWeights {
    WeightData gate;
    WeightData up;
    WeightData down;
};

// 单个 transformer 层解析好的权重引用集合。
struct LayerWeights {
    // 层类型："linear_attention" 或 "full_attention"。
    std::string type;
    WeightData input_norm;
    WeightData post_norm;
    // 仅 linear_attention 层有效。
    LinearAttnWeights lin;
    // 仅 full_attention 层有效。
    FullAttnWeights full;
    MlpWeights mlp;
};

// 整个模型解析好的权重引用（embed / final norm / 各层）。
struct ModelParams {
    WeightData embed_tokens;
    WeightData final_norm;
    std::vector<LayerWeights> layers;
};

// 从 mmap 权重一次性解析出结构化 ModelParams（不拷贝权重数据）。
ModelParams parse_model_params(const ModelWeights & weights, const ModelConfig & config);

} // namespace llm_inference
