//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_FULLATTENTION_H
#define LOCAL_LLM_FULLATTENTION_H

#include "llm/module/Module.h"
#include "tensor/Tensor.h"

struct FullAttnWeights;
struct TextConfig;
class QwenSession;

// Full attention 子层：分组查询注意力（GQA）+ QK-Norm + partial RoPE + 输出门控。
//   - Q heads = num_attention_heads(16)，KV heads = num_key_value_heads(4)，head_dim=256；
//   - q_norm / k_norm 对 Q、K 做 RMSNorm；
//   - RoPE 仅作用 head_dim 的前 partial_rotary_factor(0.25) 部分；
//   - attn_output_gate=true，输出经门控。
// KV cache 跨 token 存活，来自 QwenSession（按 type_index 取 full_attn_kv_cache）。中间量走 session.scratch。
class FullAttention : public Module {
public:
    FullAttention(const FullAttnWeights &weights, const TextConfig &config);

    // prefill：一次处理 tokens 个位置，写满 KV cache 并算出注意力输出。
    // hidden：输入隐状态 [tokens, hidden_size]；out：注意力输出 [tokens, hidden_size]。
    void prefill(QwenSession &session, const Tensor &hidden, const Tensor &out);

    // decode：处理位置 pos 的单个 token，追加写入 KV cache 并算出注意力输出。
    // hidden：[1, hidden_size]；out：[1, hidden_size]。
    void decode(QwenSession &session, const Tensor &hidden, const Tensor &out, int pos);

private:
    const FullAttnWeights &weights_;
    const TextConfig &config_;
    // 本层在 full attention 层序列中的下标，用于索引 QwenSession::full_attn_kv_cache。
    size_t type_index_ = 0;
};


#endif //LOCAL_LLM_FULLATTENTION_H
