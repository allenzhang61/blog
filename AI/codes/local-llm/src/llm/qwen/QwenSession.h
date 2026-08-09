//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_QWENSESSION_H
#define LOCAL_LLM_QWENSESSION_H
#include <vector>

#include "QwenForwardScratch.h"
#include "backend/cuda/mem/CudaWeight.h"
#include "utils/stats/MemoryUsageProvider.h"

class QwenConfig;

// full attention 层的 KV cache：跨 token，撑满整个请求。
// key_cache / value_cache 形状均为 [max_seq_len, num_key_value_heads * head_dim]。
struct FullAttnKVCache {
    CudaWeight key_cache;
    CudaWeight value_cache;
    // 已写入 KV cache 的 token 数（prefill + 已生成）。
    int seq_len = 0;
};

// linear attention 层的跨 token 状态：撑满整个请求。
struct LinearAttnRecurrentState {
    // depthwise conv 的滑动窗口状态，形状 [conv_dim, kernel]，
    // 其中 conv_dim = 2 * (key_heads * key_head_dim) + value_heads * value_head_dim。
    CudaWeight conv_state;
    // 线性注意力的 recurrent 状态，形状 [value_heads, key_head_dim, value_head_dim]。
    CudaWeight recurrent_state;
};

// 一次推理请求的资源作用域（per-request RAII scope）：
// 持有跨 token 存活的 KV cache / recurrent state，析构时随成员自动释放显存。
// 前向过程中反复覆盖的临时中间结果（激活 / logits 等）不放这里，交由前向 scratch 管理。
class QwenSession : public MemoryUsageProvider {
public:
    // 按 config 为每一层分配对应的 KV cache / recurrent state。
    // max_seq_len = inputs.size() + max_output_tokens。
    QwenSession(const QwenConfig &config, const std::vector<int> &inputs, int max_output_tokens);

    // token id 无需在 device 长期保留（attention 靠 KV cache，decode 只依赖上一个 token）；
    // GPU 采样用的 logits scratch 属于前向流程，不放 Session。
    // CudaScratchBuffer<int> d_outputs;

    // 生成的 token id（host），逐步追加，请求结束返回给调用方。
    std::vector<int> h_outputs;

    // 每个 full_attention 层一份 KV cache；顺序与 config.layer_types 中 full 层出现顺序一致。
    std::vector<FullAttnKVCache> fullAttnKVCaches;
    // 每个 linear_attention 层一份 recurrent state；顺序与 config.layer_types 中 linear 层出现顺序一致。
    std::vector<LinearAttnRecurrentState> linearAttnRecurrentStates;

    // 本次请求前向过程的临时激活暂存区（grow-only 复用），随 Session 存活。
    // 放在 Session 内保证并发请求间天然隔离：每个请求独享一份，互不覆盖。
    QwenForwardScratch forwardScratch;

    // 本次请求的最大序列长度（prefill 输入 + 最大生成）。
    int max_seq_len = 0;

    // === MemoryUsageProvider ===
    // 跨 token 状态字节数：所有 full attention KV cache + linear attention
    // recurrent / conv state 之和（随 max_seq_len 增长）。
    size_t kv_state_bytes() const override;
    // 前向临时激活峰值字节数：forwardScratch 各 buffer 之和。
    size_t scratch_bytes() const override { return forwardScratch.total_bytes(); }
};


#endif //LOCAL_LLM_QWENSESSION_H
