//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_QWENFORWARDSCRATCH_H
#define LOCAL_LLM_QWENFORWARDSCRATCH_H

#include <cstdint>
#include <vector>

#include "backend/cuda/mem/CudaScratchBuffer.h"

// 前向过程中反复覆盖的临时激活暂存区（grow-only 复用）。
// 与 QwenSession 里的跨 token 状态（KV cache / recurrent / conv state）和
// CudaWeightPool 的持久权重职责分离：这里只放“算完即可丢弃、下一步 / 下一层会
// 覆盖”的中间结果。尺寸随本次处理的 token 数（prefill 为整段输入，decode 为 1）
// 变化，用 ensure / ensure_bytes 只增不减地复用同一块 device 内存。
//
// 一份足以覆盖整条前向：full attention / linear attention / mlp / norm / 采样
// 各阶段按层、按顺序执行，同一时刻只用到其中一部分字段，彼此不冲突。
// 本对象作为 QwenSession 成员随请求存活，从而保证并发请求间天然隔离
// （每个请求独享一份，互不覆盖）。
class QwenForwardScratch {
public:
    // ================= full attention 层 =================
    // q_total  = num_attention_heads * head_dim
    // kv_total = num_key_value_heads * head_dim

    // q + gate 合并投影输出，元素数 tokens * q_total * 2。
    CudaScratchBuffer<float> full_projection;
    // 拆出的 query，元素数 tokens * q_total。
    CudaScratchBuffer<float> full_q;
    // 拆出的 gate，元素数 tokens * q_total。
    CudaScratchBuffer<float> full_gate;
    // key，元素数 tokens * kv_total。
    CudaScratchBuffer<float> full_k;
    // value，元素数 tokens * kv_total。
    CudaScratchBuffer<float> full_v;
    // 注意力输出（float），元素数 tokens * q_total。
    CudaScratchBuffer<float> full_attn;
    // 注意力输出的低精度（bf16/fp16）版本，元素数 tokens * q_total。
    CudaScratchBuffer<uint16_t> full_attn_lowp;

    // ================= linear attention 层 =================
    // value_total = linear_num_value_heads * linear_value_head_dim
    // conv_dim    = 2 * (linear_num_key_heads * linear_key_head_dim) + value_total

    // in_proj 混合投影输出（送入因果卷积），元素数 tokens * conv_dim。
    CudaScratchBuffer<float> linear_projection;
    // z 门控投影，元素数 tokens * value_total。
    CudaScratchBuffer<float> linear_z;
    // b 系数（每 value head 一个），元素数 tokens * linear_num_value_heads。
    CudaScratchBuffer<float> linear_b;
    // a 系数（每 value head 一个），元素数 tokens * linear_num_value_heads。
    CudaScratchBuffer<float> linear_a;
    // depthwise 因果卷积输出，元素数 tokens * conv_dim。
    CudaScratchBuffer<float> linear_conv_out;
    // 门控后的注意力输出（float），元素数 tokens * value_total。
    CudaScratchBuffer<float> linear_gated;
    // 上面输出的低精度（bf16/fp16）版本，喂给后续 matmul，元素数 tokens * value_total。
    CudaScratchBuffer<uint16_t> linear_gated_lowp;

    // ================= mlp / norm / 残差 / 采样 =================

    // gemm 输入的通用低精度（BF16/F16）缓冲：Qwen 权重为 BF16，cublasGemmEx 要求
    // 激活与权重同 dtype，故投影前把 float 激活转成权重 dtype 存于此。
    // 元素数按当前投影的输入维度（tokens * in_dim）。
    CudaScratchBuffer<uint16_t> input_lowp_buffer;

    // 通用输出向量 / logits buffer。
    CudaScratchBuffer<float> y_buffer;
    // MLP gate projection 临时输出。
    CudaScratchBuffer<float> gate_buffer;
    // MLP up projection 临时输出。
    CudaScratchBuffer<float> up_buffer;
    // MLP SiLU(gate) * up 的 float 临时结果。
    CudaScratchBuffer<float> prod_buffer;
    // MLP product 转成 BF16/F16 后的低精度输入。
    CudaScratchBuffer<uint16_t> prod_lowp_buffer;
    // attention / mixer 子层输出缓存。
    CudaScratchBuffer<float> mixer_buffer;
    // MLP 子层输出缓存。
    CudaScratchBuffer<float> mlp_out_buffer;
    // 完整 transformer layer 输出缓存。
    CudaScratchBuffer<float> layer_out_buffer;
    // token hidden 双缓冲，用于 decode 层间传递。
    CudaScratchBuffer<float> token_hidden_a;
    CudaScratchBuffer<float> token_hidden_b;
    // ---- host 端采样 ----
    // logits 从 device 拷回 host 的暂存（长度 vocab），供 Sampler 做温度/top-k/
    // top-p/重复惩罚。不计入 device 显存统计。
    std::vector<float> h_logits;

    // 所有 grow-only buffer 当前容量（即峰值）字节数之和，供显存统计用。
    size_t total_bytes() const {
        return full_projection.bytes() + full_q.bytes() + full_gate.bytes() + full_k.bytes() +
               full_v.bytes() + full_attn.bytes() + full_attn_lowp.bytes() +
               linear_projection.bytes() + linear_z.bytes() + linear_b.bytes() + linear_a.bytes() +
               linear_conv_out.bytes() + linear_gated.bytes() + linear_gated_lowp.bytes() +
               input_lowp_buffer.bytes() + y_buffer.bytes() + gate_buffer.bytes() +
               up_buffer.bytes() + prod_buffer.bytes() +
               prod_lowp_buffer.bytes() + mixer_buffer.bytes() + mlp_out_buffer.bytes() +
               layer_out_buffer.bytes() + token_hidden_a.bytes() + token_hidden_b.bytes();
    }
};


#endif //LOCAL_LLM_QWENFORWARDSCRATCH_H
