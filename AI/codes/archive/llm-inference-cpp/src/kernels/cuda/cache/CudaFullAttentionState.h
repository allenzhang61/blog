#pragma once

#include "CudaScratchBuffer.h"

#include <cstdint>

namespace llm_inference {

// full attention 层在 CUDA 路径上的 KV cache 和批量临时 buffer。
class CudaFullAttentionState {
public:
    int n_heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int max_seq_len = 0;
    float * q_and_gate = nullptr;
    float * projection = nullptr;
    float * k = nullptr;
    float * v = nullptr;
    float * q = nullptr;
    float * gate = nullptr;
    float * key_cache = nullptr;
    float * value_cache = nullptr;
    float * attn = nullptr;
    uint16_t * attn_bf16 = nullptr;
    CudaScratchBuffer<float> batch_projection;
    CudaScratchBuffer<float> batch_q;
    CudaScratchBuffer<float> batch_gate;
    CudaScratchBuffer<float> batch_attn;
    CudaScratchBuffer<uint16_t> batch_attn_lowp;
    CudaScratchBuffer<float> batch_k;
    CudaScratchBuffer<float> batch_v;

    // 释放 full attention 的 KV cache 和批量临时 buffer。
    ~CudaFullAttentionState();

    // prefill 结束后释放 batch-only 临时 buffer，保留 decode 需要的 KV cache。
    void release_batch_buffers();

    // 确保 CUDA KV cache 已按指定形状初始化。
    static CudaFullAttentionState * ensure(
        void *& state_handle,
        int n_heads,
        int kv_heads,
        int head_dim,
        int max_seq_len);

    // 释放通过 RunState void* 保存的 CUDA KV cache/state。
    static void destroy(void * state);
};

} // namespace llm_inference
