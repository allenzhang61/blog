//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_FULLATTENTION_H
#define LOCAL_LLM_FULLATTENTION_H

#include <cstddef>

#include "Module.h"

struct FullAttnWeights;
struct TextConfig;
struct FullAttnKVCache;
class QwenForwardScratch;
class CudaWeightPool;

// Full attention 子层：分组查询注意力（GQA）+ QK-Norm + partial RoPE + 输出门控。
//   - Q heads = num_attention_heads(16)，KV heads = num_key_value_heads(4)，head_dim=256；
//   - q_norm / k_norm 对 Q、K 做 RMSNorm；
//   - RoPE 仅作用 head_dim 的前 partial_rotary_factor(0.25) 部分；
//   - attn_output_gate=true，输出经门控。
// KV cache 跨 token 存活，来自 QwenSession（按 layer 取 FullAttnKVCache）。
class FullAttention : public Module {
public:
    FullAttention(const FullAttnWeights &weights, const TextConfig &config, CudaWeightPool *pool);

    // prefill：一次处理 tokens 个位置，写满 KV cache 并算出注意力输出。
    // d_hidden：输入隐状态 [tokens, hidden_size]；d_out：注意力输出 [tokens, hidden_size]。
    void prefill(const float *d_hidden, float *d_out, size_t tokens,
                 FullAttnKVCache &kv, QwenForwardScratch &scratch);

    // decode：处理位置 pos 的单个 token，追加写入 KV cache 并算出注意力输出。
    // d_hidden：[1, hidden_size]；d_out：[1, hidden_size]。
    void decode(const float *d_hidden, float *d_out, int pos,
                FullAttnKVCache &kv, QwenForwardScratch &scratch);

private:
    const FullAttnWeights &weights_;
    const TextConfig &config_;
    CudaWeightPool *pool_ = nullptr;
};


#endif //LOCAL_LLM_FULLATTENTION_H
