//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_QWENWEIGHTS_H
#define LOCAL_LLM_QWENWEIGHTS_H
#include <string>
#include <vector>

#include "format/MF.h"

class QwenConfig;

// linear attention 层一次性解析好的权重引用。
struct LinearAttnWeights {
    TensorView in_proj_qkv;
    TensorView in_proj_z;
    TensorView in_proj_b;
    TensorView in_proj_a;
    TensorView conv1d;
    TensorView a_log;
    TensorView dt_bias;
    TensorView norm;
    TensorView out_proj;
};

// full attention 层一次性解析好的权重引用。
struct FullAttnWeights {
    TensorView q_proj;
    TensorView k_proj;
    TensorView v_proj;
    TensorView q_norm;
    TensorView k_norm;
    TensorView o_proj;
};

// MLP 层一次性解析好的权重引用。
struct MlpWeights {
    TensorView gate;
    TensorView up;
    TensorView down;
};

// 单个 transformer 层解析好的权重引用集合。
struct LayerWeights {
    // 层类型："linear_attention" 或 "full_attention"。
    std::string type;
    TensorView input_norm;
    TensorView post_norm;
    // 仅 linear_attention 层有效。
    LinearAttnWeights lin;
    // 仅 full_attention 层有效。
    FullAttnWeights full;
    MlpWeights mlp;
};

// 视觉塔单个 transformer block 的权重引用；当前未使用（纯文本推理不走视觉分支）。
struct VisionBlockWeights {
    TensorView norm1_weight;
    TensorView norm1_bias;
    TensorView norm2_weight;
    TensorView norm2_bias;
    TensorView attn_qkv_weight;
    TensorView attn_qkv_bias;
    TensorView attn_proj_weight;
    TensorView attn_proj_bias;
    TensorView mlp_fc1_weight;
    TensorView mlp_fc1_bias;
    TensorView mlp_fc2_weight;
    TensorView mlp_fc2_bias;
};

// 视觉塔（model.visual.*）解析好的权重引用集合；当前未使用。
struct VisionWeights {
    TensorView patch_embed_proj_weight;
    TensorView patch_embed_proj_bias;
    TensorView pos_embed_weight;
    std::vector<VisionBlockWeights> blocks;
    TensorView merger_norm_weight;
    TensorView merger_norm_bias;
    TensorView merger_fc1_weight;
    TensorView merger_fc1_bias;
    TensorView merger_fc2_weight;
    TensorView merger_fc2_bias;
};

// MTP（多 token 预测）单层权重引用；当前未使用。
struct MtpLayerWeights {
    TensorView input_norm;
    TensorView post_norm;
    TensorView self_attn_q_proj;
    TensorView self_attn_k_proj;
    TensorView self_attn_v_proj;
    TensorView self_attn_o_proj;
    TensorView self_attn_q_norm;
    TensorView self_attn_k_norm;
    TensorView mlp_gate;
    TensorView mlp_up;
    TensorView mlp_down;
};

// MTP（mtp.*）解析好的权重引用集合；当前未使用。
struct MtpWeights {
    TensorView fc_weight;
    TensorView norm_weight;
    TensorView pre_fc_norm_embedding_weight;
    TensorView pre_fc_norm_hidden_weight;
    std::vector<MtpLayerWeights> layers;
};

class QwenWeights {
public:
    QwenWeights(const MF &mf, const QwenConfig &config);

    void DebugDump();

    TensorView embed_tokens;
    TensorView final_norm;
    std::vector<LayerWeights> layers;

    // 视觉塔权重（model.visual.*）；已解析但当前未使用（纯文本推理不走视觉分支）。
    VisionWeights vision;
    // MTP 权重（mtp.*）；已解析但当前未使用（不走多 token 预测路径）。
    MtpWeights mtp;

private:
    // 外部持有的模型文件；QwenWeights 不负责打开/关闭模型文件。
    const MF &mf_;

    // 校验 Qwen3.5 推理路径需要的 tensor 是否齐全。
    void validate_qwen_tensors(int num_hidden_layers, const std::vector<std::string> &layer_types) const;

    // 解析视觉塔（model.visual.*）权重到 this->vision；当前未使用。
    void parse_vision_weights(const QwenConfig &config);

    // 解析 MTP（mtp.*）权重到 this->mtp；当前未使用。
    void parse_mtp_weights(const QwenConfig &config);

    // 将 shape 转为日志/错误信息中使用的可读字符串。
    static std::string shape_to_string(const std::vector<int64_t> &shape);
};


#endif //LOCAL_LLM_QWENWEIGHTS_H
