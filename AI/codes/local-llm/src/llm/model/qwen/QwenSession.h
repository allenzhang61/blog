//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_QWENSESSION_H
#define LOCAL_LLM_QWENSESSION_H
#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "backend/cuda/mem/SessionBase.h"
#include "backend/cuda/graph/CudaGraph.h"
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
    // max_seq_len = input_ids.size() + max_output_tokens。
    QwenSession(const QwenConfig &config, std::vector<int> h_input_i32, int max_output_tokens);

    ~QwenSession() override = default;

    // decode 单步的 pos，常驻 device，供 full attention decode kernel 从 device buffer 读取，
    // 使 kernel 参数在步与步之间不变（CUDA Graph capture / replay 前置条件）。
    const GPUTensor &d_pos() const { return d_pos_; }

    // decode 单步的 token id，常驻 device：embedding 从此读取输入 token，argmax 把下一 token 写回此处，
    // 构成 device 端闭环（token 不再每步 H2D/D2H），使整段 decode 可一次 capture、后续 replay。
    const GPUTensor &d_token() const { return d_token_; }

    // 每个 full_attention 层一份 KV cache；顺序与 config.layer_types 中 full 层出现顺序一致。
    std::vector<FullAttnKVCache> full_attn_kv_cache;
    // 每个 linear_attention 层一份 recurrent state；顺序与 config.layer_types 中 linear 层出现顺序一致。
    std::vector<LinearAttnRecurrentState> linear_attn_recurrent_states;

    // === CUDA Graph（decode 单步）===
    // graph 绑定本 session 的 scratch / KV / token / pos device buffer，必须随 session 一起释放。
    CudaGraph decode_graph;
    // 本 session 已跑过的贪心 decode 步数：首步走 eager 路径预热 scratch，随后 capture/replay。
    int decode_greedy_steps = 0;

    // === MemoryUsageProvider ===
    // 跨 token 状态字节数：所有 full attention KV cache + linear attention
    // recurrent / conv state 之和（随 max_seq_len 增长）。
    size_t kv_state_bytes() const override;

private:
    // decode 单步 pos 和 token id 的 device tensor，由 GPUTensor 自动管理显存。
    GPUTensor d_pos_;
    GPUTensor d_token_;
};


#endif //LOCAL_LLM_QWENSESSION_H
