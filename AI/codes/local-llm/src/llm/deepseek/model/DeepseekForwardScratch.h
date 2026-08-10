//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEKFORWARDSCRATCH_H
#define LOCAL_LLM_DEEPSEEKFORWARDSCRATCH_H

#include "backend/cuda/mem/CudaScratchBuffer.h"

// DeepSeek 前向的临时激活缓冲（grow-only，随 Session 存活，保证并发隔离）。
struct DeepseekForwardScratch {
    // 主隐状态与归一化输出
    CudaScratchBuffer<float> hidden;       // [tokens, hidden]
    CudaScratchBuffer<float> normed;       // [tokens, hidden]
    CudaScratchBuffer<uint16_t> normed_lowp;

    // 反量化临时 f16（权重按需反量化到这里，再做 gemm）
    CudaScratchBuffer<uint16_t> deq_a;     // 通用反量化 buffer A（大权重）
    CudaScratchBuffer<uint16_t> deq_b;     // 通用反量化 buffer B（专家/其它）

    // MLA
    CudaScratchBuffer<float> q;            // [tokens, n_heads*qk_head]
    CudaScratchBuffer<float> kv_a;         // [tokens, kv_lora+qk_rope]
    CudaScratchBuffer<uint16_t> latent_lowp; // 反量化/转低精度用于 kv_b gemm
    CudaScratchBuffer<float> kv_b_out;     // [tokens, n_heads*(qk_nope+v_head)]
    CudaScratchBuffer<float> attn;         // [tokens, n_heads*v_head]
    CudaScratchBuffer<uint16_t> attn_lowp;
    CudaScratchBuffer<float> attn_out;     // [tokens, hidden]

    // FFN / MoE
    CudaScratchBuffer<uint16_t> ffn_in_lowp;
    CudaScratchBuffer<float> gate;         // [tokens, ffn]
    CudaScratchBuffer<float> up;           // [tokens, ffn]
    CudaScratchBuffer<float> act;          // SiLU(gate)*up
    CudaScratchBuffer<uint16_t> act_lowp;
    CudaScratchBuffer<float> ffn_out;      // [tokens, hidden]
    CudaScratchBuffer<float> moe_out;      // 累加 [tokens, hidden]
    CudaScratchBuffer<float> router_logits;// [tokens, n_experts]
    CudaScratchBuffer<int> top_idx;        // [tokens, k]
    CudaScratchBuffer<float> top_w;        // [tokens, k]
    CudaScratchBuffer<float> expert_out;   // 单 token 单专家中间 [hidden]

    // logits
    CudaScratchBuffer<float> logits;       // [vocab]
    CudaScratchBuffer<uint16_t> logits_in_lowp;
    CudaScratchBuffer<float> argmax_vals;
    CudaScratchBuffer<int> argmax_idx;
    CudaScratchBuffer<float> best_val;
    CudaScratchBuffer<int> best_idx;

    size_t total_bytes() const {
        return hidden.bytes() + normed.bytes() + normed_lowp.bytes() + deq_a.bytes() + deq_b.bytes() +
               q.bytes() + kv_a.bytes() + latent_lowp.bytes() + kv_b_out.bytes() + attn.bytes() +
               attn_lowp.bytes() + attn_out.bytes() + ffn_in_lowp.bytes() + gate.bytes() + up.bytes() +
               act.bytes() + act_lowp.bytes() + ffn_out.bytes() + moe_out.bytes() +
               router_logits.bytes() + top_idx.bytes() + top_w.bytes() + expert_out.bytes() +
               logits.bytes() + logits_in_lowp.bytes() + argmax_vals.bytes() + argmax_idx.bytes() +
               best_val.bytes() + best_idx.bytes();
    }
};

#endif // LOCAL_LLM_DEEPSEEKFORWARDSCRATCH_H
