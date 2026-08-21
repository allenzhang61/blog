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
    DiskTensor in_proj_qkv;  // [conv_dim, hidden]
    DiskTensor in_proj_z;    // [value_total, hidden]
    DiskTensor in_proj_b;    // [linear_num_value_heads, hidden]
    DiskTensor in_proj_a;    // [linear_num_value_heads, hidden]
    DiskTensor conv1d;       // [conv_dim, kernel]（深度可分离，逐通道）
    DiskTensor a_log;        // [linear_num_value_heads]
    DiskTensor dt_bias;      // [linear_num_value_heads]
    DiskTensor norm;         // [linear_value_head_dim]（每 head 的 gated RMSNorm）
    DiskTensor out_proj;     // [hidden, value_total]
};

// full attention 层一次性解析好的权重引用。
// 记号：hidden=hidden_size；q_total=num_attention_heads*head_dim；
//       kv_total=num_key_value_heads*head_dim。
struct FullAttnWeights {
    // 本层在 full attention 层序列中的下标，用于索引 QwenSession::full_attn_kv_cache。
    size_t type_index = 0;
    DiskTensor q_proj;  // [q_total*2, hidden]（每 head 交错输出 [q, gate]）
    DiskTensor k_proj;  // [kv_total, hidden]
    DiskTensor v_proj;  // [kv_total, hidden]
    DiskTensor q_norm;  // [head_dim]
    DiskTensor k_norm;  // [head_dim]
    DiskTensor o_proj;  // [hidden, q_total]
};

// MLP 层一次性解析好的权重引用。
// 记号：hidden=hidden_size；intermediate=intermediate_size。
struct MlpWeights {
    DiskTensor gate_proj;  // [intermediate, hidden]
    DiskTensor up_proj;    // [intermediate, hidden]
    DiskTensor down_proj;  // [hidden, intermediate]
};

// 单个 transformer 层解析好的权重引用集合。
struct LayerWeights {
    // 层类型："linear_attention" 或 "full_attention"。
    std::string type;
    // 本层在同类型层序列中的下标，用于索引 QwenSession 的 fullAttnKVCaches / linearAttnRecurrentStates。
    size_t type_index = 0;
    DiskTensor input_layernorm;           // [hidden_size]（注意力前 RMSNorm）
    DiskTensor post_attention_layernorm;   // [hidden_size]（MLP 前 RMSNorm）
    // 仅 linear_attention 层有效。
    LinearAttnWeights lin;
    // 仅 full_attention 层有效。
    FullAttnWeights full;
    MlpWeights mlp;
};

// 视觉塔单个 transformer block 的权重引用；当前未使用（纯文本推理不走视觉分支）。
struct VisionBlockWeights {
    DiskTensor norm1_weight;
    DiskTensor norm1_bias;
    DiskTensor norm2_weight;
    DiskTensor norm2_bias;
    DiskTensor attn_qkv_weight;
    DiskTensor attn_qkv_bias;
    DiskTensor attn_proj_weight;
    DiskTensor attn_proj_bias;
    DiskTensor mlp_fc1_weight;
    DiskTensor mlp_fc1_bias;
    DiskTensor mlp_fc2_weight;
    DiskTensor mlp_fc2_bias;
};

// 视觉塔（model.visual.*）解析好的权重引用集合；当前未使用。
struct VisionWeights {
    DiskTensor patch_embed_proj_weight;
    DiskTensor patch_embed_proj_bias;
    DiskTensor pos_embed_weight;
    std::vector<VisionBlockWeights> blocks;
    DiskTensor merger_norm_weight;
    DiskTensor merger_norm_bias;
    DiskTensor merger_fc1_weight;
    DiskTensor merger_fc1_bias;
    DiskTensor merger_fc2_weight;
    DiskTensor merger_fc2_bias;
};

// MTP（多 token 预测）单层权重引用；当前未使用。
struct MtpLayerWeights {
    DiskTensor attn_norm;
    DiskTensor ffn_norm;
    DiskTensor self_attn_q_proj;
    DiskTensor self_attn_k_proj;
    DiskTensor self_attn_v_proj;
    DiskTensor self_attn_o_proj;
    DiskTensor self_attn_q_norm;
    DiskTensor self_attn_k_norm;
    DiskTensor mlp_gate;
    DiskTensor mlp_up;
    DiskTensor mlp_down;
};

// MTP（mtp.*）解析好的权重引用集合；当前未使用。
struct MtpWeights {
    DiskTensor fc_weight;
    DiskTensor norm_weight;
    DiskTensor pre_fc_norm_embedding_weight;
    DiskTensor pre_fc_norm_hidden_weight;
    std::vector<MtpLayerWeights> layers;
};

class QwenWeights {
public:
    QwenWeights(const MF &mf, const QwenConfig &config);

    void DebugDump();

    DiskTensor token_embd;  // [vocab_size, hidden_size]
    DiskTensor output_norm; // [hidden_size]
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
