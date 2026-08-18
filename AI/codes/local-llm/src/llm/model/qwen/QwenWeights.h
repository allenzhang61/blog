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
// 记号：hidden=hidden_size；key_total=linear_num_key_heads*linear_key_head_dim；
//       value_total=linear_num_value_heads*linear_value_head_dim；
//       conv_dim=key_total*2+value_total；kernel=linear_conv_kernel_dim。
struct LinearAttnWeights {
    // 本层在 linear attention 层序列中的下标，用于索引 QwenSession::linear_attn_recurrent_states。
    size_t type_index = 0;
    Tensor in_proj_qkv;  // [conv_dim, hidden]
    Tensor in_proj_z;    // [value_total, hidden]
    Tensor in_proj_b;    // [linear_num_value_heads, hidden]
    Tensor in_proj_a;    // [linear_num_value_heads, hidden]
    Tensor conv1d;       // [conv_dim, kernel]（深度可分离，逐通道）
    Tensor a_log;        // [linear_num_value_heads]
    Tensor dt_bias;      // [linear_num_value_heads]
    Tensor norm;         // [linear_value_head_dim]（每 head 的 gated RMSNorm）
    Tensor out_proj;     // [hidden, value_total]
};

// full attention 层一次性解析好的权重引用。
// 记号：hidden=hidden_size；q_total=num_attention_heads*head_dim；
//       kv_total=num_key_value_heads*head_dim。
struct FullAttnWeights {
    // 本层在 full attention 层序列中的下标，用于索引 QwenSession::full_attn_kv_cache。
    size_t type_index = 0;
    Tensor q_proj;  // [q_total*2, hidden]（每 head 交错输出 [q, gate]）
    Tensor k_proj;  // [kv_total, hidden]
    Tensor v_proj;  // [kv_total, hidden]
    Tensor q_norm;  // [head_dim]
    Tensor k_norm;  // [head_dim]
    Tensor o_proj;  // [hidden, q_total]
};

// MLP 层一次性解析好的权重引用。
// 记号：hidden=hidden_size；intermediate=intermediate_size。
struct MlpWeights {
    Tensor gate_proj;  // [intermediate, hidden]
    Tensor up_proj;    // [intermediate, hidden]
    Tensor down_proj;  // [hidden, intermediate]
};

// 单个 transformer 层解析好的权重引用集合。
struct LayerWeights {
    // 层类型："linear_attention" 或 "full_attention"。
    std::string type;
    // 本层在同类型层序列中的下标，用于索引 QwenSession 的 fullAttnKVCaches / linearAttnRecurrentStates。
    size_t type_index = 0;
    Tensor input_layernorm;           // [hidden_size]（注意力前 RMSNorm）
    Tensor post_attention_layernorm;   // [hidden_size]（MLP 前 RMSNorm）
    // 仅 linear_attention 层有效。
    LinearAttnWeights lin;
    // 仅 full_attention 层有效。
    FullAttnWeights full;
    MlpWeights mlp;
};

// 视觉塔单个 transformer block 的权重引用；当前未使用（纯文本推理不走视觉分支）。
struct VisionBlockWeights {
    Tensor norm1_weight;
    Tensor norm1_bias;
    Tensor norm2_weight;
    Tensor norm2_bias;
    Tensor attn_qkv_weight;
    Tensor attn_qkv_bias;
    Tensor attn_proj_weight;
    Tensor attn_proj_bias;
    Tensor mlp_fc1_weight;
    Tensor mlp_fc1_bias;
    Tensor mlp_fc2_weight;
    Tensor mlp_fc2_bias;
};

// 视觉塔（model.visual.*）解析好的权重引用集合；当前未使用。
struct VisionWeights {
    Tensor patch_embed_proj_weight;
    Tensor patch_embed_proj_bias;
    Tensor pos_embed_weight;
    std::vector<VisionBlockWeights> blocks;
    Tensor merger_norm_weight;
    Tensor merger_norm_bias;
    Tensor merger_fc1_weight;
    Tensor merger_fc1_bias;
    Tensor merger_fc2_weight;
    Tensor merger_fc2_bias;
};

// MTP（多 token 预测）单层权重引用；当前未使用。
struct MtpLayerWeights {
    Tensor attn_norm;
    Tensor ffn_norm;
    Tensor self_attn_q_proj;
    Tensor self_attn_k_proj;
    Tensor self_attn_v_proj;
    Tensor self_attn_o_proj;
    Tensor self_attn_q_norm;
    Tensor self_attn_k_norm;
    Tensor mlp_gate;
    Tensor mlp_up;
    Tensor mlp_down;
};

// MTP（mtp.*）解析好的权重引用集合；当前未使用。
struct MtpWeights {
    Tensor fc_weight;
    Tensor norm_weight;
    Tensor pre_fc_norm_embedding_weight;
    Tensor pre_fc_norm_hidden_weight;
    std::vector<MtpLayerWeights> layers;
};

class QwenWeights {
public:
    QwenWeights(const MF &mf, const QwenConfig &config);

    void DebugDump();

    Tensor token_embd;  // [vocab_size, hidden_size]
    Tensor output_norm; // [hidden_size]
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
