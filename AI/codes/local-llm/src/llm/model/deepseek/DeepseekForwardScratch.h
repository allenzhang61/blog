//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_FORWARD_SCRATCH_H
#define LOCAL_LLM_DEEPSEEK_FORWARD_SCRATCH_H

#include <vector>

#include "backend/cuda/mem/CudaScratchBuffer.h"

// DeepSeek 前向的临时激活缓冲（grow-only，随 Session 存活，保证并发隔离）。
struct DeepseekForwardScratch {
    // 主隐状态与归一化输出
    CudaScratchBuffer<float> hidden;       // [tokens, hidden]
    CudaScratchBuffer<float> normed;       // [tokens, hidden]
    CudaScratchBuffer<uint16_t> normed_lowp;

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

    // host 端采样：logits 拷回主机后交给 Sampler（不计入 device 显存）。
    std::vector<float> h_logits;

    size_t total_bytes() const {
        return hidden.bytes() + normed.bytes() + normed_lowp.bytes() +
               q.bytes() + kv_a.bytes() + latent_lowp.bytes() + kv_b_out.bytes() + attn.bytes() +
               attn_lowp.bytes() + attn_out.bytes() + ffn_in_lowp.bytes() + gate.bytes() + up.bytes() +
               act.bytes() + act_lowp.bytes() + ffn_out.bytes() + moe_out.bytes() +
               router_logits.bytes() + top_idx.bytes() + top_w.bytes() + expert_out.bytes() +
               logits.bytes() + logits_in_lowp.bytes();
    }
};

#endif // LOCAL_LLM_DEEPSEEK_FORWARD_SCRATCH_H
