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
    GPUTensor g_key_cache;
    GPUTensor g_value_cache;
    // 已写入 KV cache 的 token 数（prefill + 已生成）。
    int seq_len = 0;
};

// linear attention 层的跨 token 状态：撑满整个请求。
struct LinearAttnRecurrentState {
    // depthwise conv 的滑动窗口状态，形状 [conv_dim, kernel]，
    // 其中 conv_dim = 2 * (key_heads * key_head_dim) + value_heads * value_head_dim。
    GPUTensor g_conv_state;
    // 线性注意力的 recurrent 状态，形状 [value_heads, key_head_dim, value_head_dim]。
    GPUTensor g_recurrent_state;
};

// 一次推理请求的资源作用域（per-request RAII scope）：
// 持有跨 token 存活的 KV cache / recurrent state，析构时随成员自动释放显存。
// 前向过程中反复覆盖的临时中间结果（激活 / logits 等）不放这里，交由前向 scratch 管理。
class QwenSession : public SessionBase {
public:
    // 按 config 为每一层分配对应的 KV cache / recurrent state。
    // max_seq_len = inputs.numel() + max_output_tokens。
    QwenSession(const QwenConfig &config, const CPUTensor &c_input, int max_output_tokens);

    // 每个 full_attention 层一份 KV cache；顺序与 config.layer_types 中 full 层出现顺序一致。
    std::vector<FullAttnKVCache> full_attn_kv_cache;
    // 每个 linear_attention 层一份 recurrent state；顺序与 config.layer_types 中 linear 层出现顺序一致。
    std::vector<LinearAttnRecurrentState> linear_attn_recurrent_states;

    // === MemoryUsageProvider ===
    // 跨 token 状态字节数：所有 full attention KV cache + linear attention
    // recurrent / conv state 之和（随 max_seq_len 增长）。
    size_t kv_state_bytes() const override;
};


#endif //LOCAL_LLM_QWENSESSION_H
