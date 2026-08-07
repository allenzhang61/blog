#pragma once

#include "CudaScratchBuffer.h"

#include <cstdint>

namespace llm_inference {

// linear attention 层在 CUDA 路径上的跨 token 状态和批量临时 buffer。
class CudaLinearAttentionState {
public:
    int key_heads = 0;
    int value_heads = 0;
    int k_dim = 0;
    int v_dim = 0;
    int kernel = 0;
    float * conv_state = nullptr;
    float * recurrent_state = nullptr;
    float * mixed = nullptr;
    float * projection = nullptr;
    float * z = nullptr;
    float * b = nullptr;
    float * a = nullptr;
    float * conv_out = nullptr;
    float * gated = nullptr;
    uint16_t * gated_bf16 = nullptr;
    CudaScratchBuffer<float> batch_projection;
    CudaScratchBuffer<float> batch_conv_out;
    CudaScratchBuffer<float> batch_gated;
    CudaScratchBuffer<uint16_t> batch_gated_lowp;
    CudaScratchBuffer<float> batch_z;
    CudaScratchBuffer<float> batch_b;
    CudaScratchBuffer<float> batch_a;

    // 释放 linear attention 的 recurrent/conv state 和批量临时 buffer。
    ~CudaLinearAttentionState();

    // prefill 结束后释放 batch-only 临时 buffer，保留 decode 需要的 recurrent state。
    void release_batch_buffers();

    // 确保 CUDA state 已按指定形状初始化。
    static CudaLinearAttentionState * ensure(
        void *& state_handle,
        int key_heads,
        int value_heads,
        int k_dim,
        int v_dim,
        int kernel);

    // 释放通过 RunState void* 保存的 CUDA cache/state。
    static void destroy(void * state);
};

} // namespace llm_inference
