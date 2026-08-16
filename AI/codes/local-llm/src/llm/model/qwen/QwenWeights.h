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
    MFTensorView in_proj_qkv;
    MFTensorView in_proj_z;
    MFTensorView in_proj_b;
    MFTensorView in_proj_a;
    MFTensorView conv1d;
    MFTensorView a_log;
    MFTensorView dt_bias;
    MFTensorView norm;
    MFTensorView out_proj;
};

// full attention 层一次性解析好的权重引用。
struct FullAttnWeights {
    MFTensorView q_proj;
    MFTensorView k_proj;
    MFTensorView v_proj;
    MFTensorView q_norm;
    MFTensorView k_norm;
    MFTensorView o_proj;
};

// MLP 层一次性解析好的权重引用。
struct MlpWeights {
    MFTensorView gate;
    MFTensorView up;
    MFTensorView down;
};

// 单个 transformer 层解析好的权重引用集合。
struct LayerWeights {
    // 层类型："linear_attention" 或 "full_attention"。
    std::string type;
    MFTensorView input_norm;
    MFTensorView post_norm;
    // 仅 linear_attention 层有效。
    LinearAttnWeights lin;
    // 仅 full_attention 层有效。
    FullAttnWeights full;
    MlpWeights mlp;
};

// 视觉塔单个 transformer block 的权重引用；当前未使用（纯文本推理不走视觉分支）。
struct VisionBlockWeights {
    MFTensorView norm1_weight;
    MFTensorView norm1_bias;
    MFTensorView norm2_weight;
    MFTensorView norm2_bias;
    MFTensorView attn_qkv_weight;
    MFTensorView attn_qkv_bias;
    MFTensorView attn_proj_weight;
    MFTensorView attn_proj_bias;
    MFTensorView mlp_fc1_weight;
    MFTensorView mlp_fc1_bias;
    MFTensorView mlp_fc2_weight;
    MFTensorView mlp_fc2_bias;
};

// 视觉塔（model.visual.*）解析好的权重引用集合；当前未使用。
struct VisionWeights {
    MFTensorView patch_embed_proj_weight;
    MFTensorView patch_embed_proj_bias;
    MFTensorView pos_embed_weight;
    std::vector<VisionBlockWeights> blocks;
    MFTensorView merger_norm_weight;
    MFTensorView merger_norm_bias;
    MFTensorView merger_fc1_weight;
    MFTensorView merger_fc1_bias;
    MFTensorView merger_fc2_weight;
    MFTensorView merger_fc2_bias;
};

// MTP（多 token 预测）单层权重引用；当前未使用。
struct MtpLayerWeights {
    MFTensorView input_norm;
    MFTensorView post_norm;
    MFTensorView self_attn_q_proj;
    MFTensorView self_attn_k_proj;
    MFTensorView self_attn_v_proj;
    MFTensorView self_attn_o_proj;
    MFTensorView self_attn_q_norm;
    MFTensorView self_attn_k_norm;
    MFTensorView mlp_gate;
    MFTensorView mlp_up;
    MFTensorView mlp_down;
};

// MTP（mtp.*）解析好的权重引用集合；当前未使用。
struct MtpWeights {
    MFTensorView fc_weight;
    MFTensorView norm_weight;
    MFTensorView pre_fc_norm_embedding_weight;
    MFTensorView pre_fc_norm_hidden_weight;
    std::vector<MtpLayerWeights> layers;
};

class QwenWeights {
public:
    QwenWeights(const MF &mf, const QwenConfig &config);

    void DebugDump();

    MFTensorView embed_tokens;
    MFTensorView final_norm;
    std::vector<LayerWeights> layers;

    // 视觉塔权重（model.visual.*）；已解析但当前未使用（纯文本推理不走视觉分支）。
    VisionWeights vision;
    // MTP 权重（mtp.*）；已解析但当前未使用（不走多 token 预测路径）。
    MtpWeights mtp;

private:
    // 外部持有的模型文件；QwenWeights 不负责打开/关闭模型文件。
    const MF &mf_;

    // 校验 Qwen3.5 推理路径需要的 tensor 是否齐全，且关键 shape 与 config 一致。
    void validate(const QwenConfig &config) const;

    // 解析视觉塔（model.visual.*）权重到 this->vision；当前未使用。
    void parse_vision_weights(const QwenConfig &config);

    // 解析 MTP（mtp.*）权重到 this->mtp；当前未使用。
    void parse_mtp_weights(const QwenConfig &config);

    // 将 shape 转为日志/错误信息中使用的可读字符串。
    static std::string shape_to_string(const std::vector<int64_t> &shape);
};


#endif //LOCAL_LLM_QWENWEIGHTS_H
