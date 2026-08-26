//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_SESSIONBASE_H
#define LOCAL_LLM_SESSIONBASE_H

#include <cstddef>
#include <vector>

#include "backend/cuda/mem/CudaScratch.h"
#include "tensor/CPUScratch.h"
#include "tensor/CPUTensor.h"
#include "utils/stats/MemoryUsageProvider.h"

// 一次推理请求作用域（per-request RAII scope）的通用基类，与具体模型无关。
// 汇集所有模型 Session 的共性：前向 scratch、已生成 token、最大序列长度，
// 以及 MemoryUsageProvider 中与 scratch 相关的上报。
//
// 各模型专属的跨 token 状态（Qwen 的 KV cache / recurrent state、DeepSeek 的
// latent KV cache 等）形态不同，仍留在各自子类里，并各自实现 kv_state_bytes()。
class SessionBase : public MemoryUsageProvider {
public:
    ~SessionBase() override = default;

    // 本次请求前向过程的临时激活暂存区（grow-only 复用），随 Session 存活。
    // 放在 Session 内保证并发请求间天然隔离：每个请求独享一份，互不覆盖。
    CudaScratch cuda_scratch;
    CPUScratch cpu_scratch;

    // tokenizer 编码后的输入 token，数据由 cpu_scratch 持有。
    CPUTensor h_input_i32_;

    // 生成的 token id（host），逐步追加，请求结束返回给调用方；
    // 同时作为重复惩罚所需的历史 token 序列。
    std::vector<int> h_output_i32_;

    void append_output(int token_id) { h_output_i32_.push_back(token_id); }
    int prev_token_id() const { return h_output_i32_.back(); }
    int decode_pos() const { return static_cast<int>(h_input_i32_.numel() + h_output_i32_.size() - 1); }

    // 本次请求的最大序列长度（prefill 输入 + 最大生成）。
    size_t max_seq_len_ = 0;

    // === MemoryUsageProvider ===
    // 前向临时激活峰值字节数：scratch 各 buffer 之和（所有模型一致）。
    size_t scratch_bytes() const override { return cuda_scratch.total_bytes(); }
    // 跨 token 状态字节数由各模型子类实现（KV cache / recurrent state 形态不同）。
    size_t kv_state_bytes() const override = 0;
};

#endif // LOCAL_LLM_SESSIONBASE_H
