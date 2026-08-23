//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_QWENSESSION_H
#define LOCAL_LLM_QWENSESSION_H
#include <cstddef>
#include <vector>

#include "backend/cuda/mem/SessionBase.h"
#include "tensor/CPUTensor.h"
#include "tensor/GPUTensor.h"
#include "tensor/StorageTensor.h"

class QwenConfig;

// full attention 层的 KV cache：跨 token，撑满整个请求。
// key_cache / value_cache 形状均为 [max_seq_len, num_key_value_heads * head_dim]。
struct FullAttnKVCache {
    GPUTensor g_key_cache_f32;
    GPUTensor g_value_cache_f32;
    // 已写入 KV cache 的 token 数（prefill + 已生成）。
    int seq_len = 0;
};

// linear attention 层的跨 token 状态：撑满整个请求。
struct LinearAttnRecurrentState {
    // depthwise conv 的滑动窗口状态，形状 [conv_dim, kernel]，
    // 其中 conv_dim = 2 * (key_heads * key_head_dim) + value_heads * value_head_dim。
    GPUTensor g_conv_state_f32;
    // 线性注意力的 recurrent 状态，形状 [value_heads, key_head_dim, value_head_dim]。
    GPUTensor g_recurrent_state_f32;
};

// 一次推理请求的资源作用域（per-request RAII scope）：
// 持有跨 token 存活的 KV cache / recurrent state，析构时随成员自动释放显存。
// 前向过程中反复覆盖的临时中间结果（激活 / logits 等）不放这里，交由前向 scratch 管理。
class QwenSession : public SessionBase {
public:
    // 按 config 为每一层分配对应的 KV cache / recurrent state。
    // max_seq_len = inputs.numel() + max_output_tokens。
    QwenSession(const QwenConfig &config, const CPUTensor &c_input, int max_output_tokens);

    // 释放 device 端 pos buffer（d_pos_）。
    ~QwenSession() override;

    // decode 单步的 pos，常驻 device，供 full attention decode kernel 从 device buffer 读取，
    // 使 kernel 参数在步与步之间不变（CUDA Graph capture / replay 前置条件）。
    int *d_pos() const { return d_pos_; }

    // decode 单步的 token id，常驻 device：embedding 从此读取输入 token，argmax 把下一 token 写回此处，
    // 构成 device 端闭环（token 不再每步 H2D/D2H），使整段 decode 可一次 capture、后续 replay。
    int *d_token() const { return d_token_; }

    // 每个 full_attention 层一份 KV cache；顺序与 config.layer_types 中 full 层出现顺序一致。
    std::vector<FullAttnKVCache> full_attn_kv_cache;
    // 每个 linear_attention 层一份 recurrent state；顺序与 config.layer_types 中 linear 层出现顺序一致。
    std::vector<LinearAttnRecurrentState> linear_attn_recurrent_states;

    // === MemoryUsageProvider ===
    // 跨 token 状态字节数：所有 full attention KV cache + linear attention
    // recurrent / conv state 之和（随 max_seq_len 增长）。
    size_t kv_state_bytes() const override;

private:
    // decode 单步 pos 的 device buffer（sizeof(int)），构造时分配、析构时释放。
    int *d_pos_ = nullptr;
    // decode 单步 token id 的 device buffer（sizeof(int)），构造时分配、析构时释放。
    int *d_token_ = nullptr;
};


#endif //LOCAL_LLM_QWENSESSION_H
