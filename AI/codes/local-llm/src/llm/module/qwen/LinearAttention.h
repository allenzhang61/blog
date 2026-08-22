//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_LINEARATTENTION_H
#define LOCAL_LLM_LINEARATTENTION_H

#include "llm/module/Module.h"
#include "tensor/CPUTensor.h"
#include "tensor/StorageTensor.h"

struct LinearAttnWeights;
struct TextConfig;
class QwenSession;

// Linear attention 子层（门控 delta / SSM 风格，类 Mamba/GDN），模型主体（24/32 层）：
//   - in_proj_qkv/z/b/a：输入投影，产生混合 QKV、门控 z、状态更新系数 b/a；
//   - conv1d：作用在序列维度的 depthwise 因果卷积（kernel=4）；
//   - a_log / dt_bias：SSM 递归衰减 / 步长参数；
//   - gated delta 递归更新 recurrent state；
//   - norm / out_proj：输出归一化与投影。
// 维护 recurrent state（conv 滑窗 + 递归状态）而非 KV cache，跨 token 存活，
// 来自 QwenSession（按 type_index 取 linear_attn_recurrent_states）。中间量走 session.scratch。
class LinearAttention : public Module {
public:
    LinearAttention(const LinearAttnWeights &weights, const TextConfig &config);

    // prefill：一次处理 tokens 个位置，扫描更新 recurrent state 并算出输出。
    // g_hidden：[tokens, hidden_size]；g_out：[tokens, hidden_size]。
    void prefill(QwenSession &session, const GPUTensor &g_hidden, const GPUTensor &g_out);

    // decode：处理单个新 token，基于已有 recurrent state 递推一步。
    // g_hidden：[1, hidden_size]；g_out：[1, hidden_size]。
    void decode(QwenSession &session, const GPUTensor &g_hidden_f32, const GPUTensor &g_out_f32);

private:
    // Linear attention 权重：
    //   s_in_proj_qkv [conv_dim, hidden_size]
    //   s_in_proj_z [value_total, hidden_size]
    //   s_in_proj_b/s_in_proj_a [linear_num_value_heads, hidden_size]
    //   s_conv1d [conv_dim, linear_conv_kernel_dim]
    //   s_a_log/s_dt_bias [linear_num_value_heads]
    //   s_norm [linear_value_head_dim]
    //   s_out_proj [hidden_size, value_total]
    const LinearAttnWeights &weights_;
    const TextConfig &config_;
    // 本层在 linear attention 层序列中的下标，用于索引 QwenSession::linear_attn_recurrent_states。
    size_t type_index_ = 0;
};


#endif //LOCAL_LLM_LINEARATTENTION_H
