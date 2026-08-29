//
// Created by zhangyoulun on 9/8/2026.
//

#include "FullAttention.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>

#include "llm/model/qwen/QwenWeights.h"
#include "llm/model/qwen/QwenConfig.h"
#include "llm/model/qwen/QwenSession.h"
#include "tensor/GPUTensor.h"
#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "tensor/TensorTool.h"

FullAttention::FullAttention(const FullAttnWeights &weights, const TextConfig &config)
    : weights_(weights), config_(config), type_index_(weights.type_index) {}

void FullAttention::prefill(QwenSession &session, const GPUTensor &g_hidden_f32, const GPUTensor &g_out_f32) {
    const int64_t input_size = g_hidden_f32.rows();
    FullAttnKVCache &kv = session.full_attn_kv_cache[type_index_];
    CudaScratch &scratch = session.cuda_scratch;
    const int n_heads = config_.num_attention_heads;
    const int kv_heads = config_.num_key_value_heads;
    const int head_dim = config_.head_dim;
    const int64_t q_total = static_cast<int64_t>(n_heads) * head_dim;
    const int64_t kv_total = static_cast<int64_t>(kv_heads) * head_dim;
    const float eps = config_.rms_norm_eps;
    const float theta = config_.rope_parameters.rope_theta;
    const float partial = config_.rope_parameters.partial_rotary_factor;
    const int start_pos = kv.seq_len;

    // q_proj 输出 q_total*2（每 head 交错 [q, gate]）；k/v_proj 输出 kv_total。
    GPUTensor g_full_projection_f32 = GPUTensor(
        scratch, scratch_key::kFullProjection,
        {input_size, q_total * 2}, DType::F32);
    GPUTensor g_full_k_f32 = GPUTensor(
        scratch, scratch_key::kFullK,
        {input_size, kv_total}, DType::F32);
    GPUTensor g_full_v_f32 = GPUTensor(
        scratch, scratch_key::kFullV,
        {input_size, kv_total}, DType::F32);
    GPUTensor g_full_q_f32 = GPUTensor(
        scratch, scratch_key::kFullQ,
        {input_size, q_total}, DType::F32);
    GPUTensor g_full_gate_f32 = GPUTensor(
        scratch, scratch_key::kFullGate,
        {input_size, q_total}, DType::F32);
    GPUTensor g_full_attn_f32 = GPUTensor(
        scratch, scratch_key::kFullAttn,
        {input_size, q_total}, DType::F32);
    // q/k/v_proj 共享同一份 hidden 输入且权重同 dtype：只转一次 bf16/f16 复用，
    // 省掉原先每个 GEMM 各自一次 f32->bf16 拷贝（旧路径每层 3 次冗余转换）。
    const void *d_hidden_lowp = TensorTool::prepare_lowp_input(
        g_hidden_f32, weights_.s_q_proj.dtype, scratch, scratch_key::kInputLowp);
    const int64_t rows = g_hidden_f32.rows();
    TensorTool::gemm_lowp(weights_.s_q_proj, d_hidden_lowp, rows, g_full_projection_f32, "fullattn.d_q_proj");
    TensorTool::gemm_lowp(weights_.s_k_proj, d_hidden_lowp, rows, g_full_k_f32, "fullattn.d_k_proj");
    TensorTool::gemm_lowp(weights_.s_v_proj, d_hidden_lowp, rows, g_full_v_f32, "fullattn.d_v_proj");

    TensorTool::full_attention_q_batch(g_full_projection_f32, weights_.s_q_norm, g_full_q_f32, g_full_gate_f32,
                                  n_heads, head_dim, start_pos, theta, partial, eps);
    TensorTool::full_attention_kv_batch(g_full_k_f32, g_full_v_f32, weights_.s_k_norm, kv.g_key_cache_f32, kv.g_value_cache_f32,
                                   kv_heads, head_dim, /*max_seq_len=*/0, start_pos, theta, partial, eps);
    TensorTool::full_attention_attend_batch(g_full_q_f32, g_full_gate_f32, kv.g_key_cache_f32, kv.g_value_cache_f32, g_full_attn_f32,
                                       n_heads, kv_heads, head_dim, /*max_seq_len=*/0, start_pos);

    // o_proj：[g_hidden, q_total] · attn[q_total, tokens] -> [g_hidden, tokens]。
    TensorTool::gemm(weights_.s_o_proj, g_full_attn_f32, g_out_f32, scratch, scratch_key::kFullAttnLowp, "fullattn.d_o_proj");

    kv.seq_len += static_cast<int>(input_size);
    // 不做全设备同步：同流顺序保证依赖，barrier 交给前向末尾的 lm_head 同步 D2H。
    check_cuda(cudaGetLastError(), "FullAttention prefill kernel launch 失败");
}

void FullAttention::decode(QwenSession &session, const GPUTensor &g_hidden_f32, const GPUTensor &g_out_f32, int pos) {
    FullAttnKVCache &kv = session.full_attn_kv_cache[type_index_];
    CudaScratch &scratch = session.cuda_scratch;
    const int n_heads = config_.num_attention_heads;
    const int kv_heads = config_.num_key_value_heads;
    const int head_dim = config_.head_dim;
    const int64_t q_total = static_cast<int64_t>(n_heads) * head_dim;
    const int64_t kv_total = static_cast<int64_t>(kv_heads) * head_dim;
    const float eps = config_.rms_norm_eps;
    const float theta = config_.rope_parameters.rope_theta;
    const float partial = config_.rope_parameters.partial_rotary_factor;

    GPUTensor g_full_projection_f32 = GPUTensor(
        scratch, scratch_key::kFullProjection, {1, q_total * 2}, DType::F32);
    GPUTensor g_full_k_f32 = GPUTensor(
        scratch, scratch_key::kFullK, {1, kv_total}, DType::F32);
    GPUTensor g_full_v_f32 = GPUTensor(
        scratch, scratch_key::kFullV, {1, kv_total}, DType::F32);
    GPUTensor g_full_q_f32 = GPUTensor(
        scratch, scratch_key::kFullQ, {1, q_total}, DType::F32);
    GPUTensor g_full_gate_f32 = GPUTensor(
        scratch, scratch_key::kFullGate, {1, q_total}, DType::F32);
    GPUTensor g_full_attn_f32 = GPUTensor(
        scratch, scratch_key::kFullAttn, {1, q_total}, DType::F32);
    // q/k/v_proj 共享同一份 hidden 输入且权重同 dtype：只转一次 bf16/f16 复用，
    // 省掉原先每个 GEMM 各自一次 f32->bf16 拷贝（旧路径每层 3 次冗余转换）。
    const void *d_hidden_lowp = TensorTool::prepare_lowp_input(
        g_hidden_f32, weights_.s_q_proj.dtype, scratch, scratch_key::kInputLowp);
    const int64_t rows = g_hidden_f32.rows();
    TensorTool::gemm_lowp(weights_.s_q_proj, d_hidden_lowp, rows, g_full_projection_f32, "fullattn.d_q_proj");
    TensorTool::gemm_lowp(weights_.s_k_proj, d_hidden_lowp, rows, g_full_k_f32, "fullattn.d_k_proj");
    TensorTool::gemm_lowp(weights_.s_v_proj, d_hidden_lowp, rows, g_full_v_f32, "fullattn.d_v_proj");

    // decode kernel 从 device buffer 读 pos，使 kernel 参数在步与步之间不变（CUDA Graph 前置条件）。
    // pos 的 device 值由 QwenModel 在每步 graph 外统一写入 session.d_pos()，此处只读地址、不做同步 H2D
    //（若在此处做同步 H2D 会破坏 graph capture）。
    int *d_pos = session.d_pos().data<int>();
    // attend 的 smem 现按 max_seq_len 上限固定，故传入真实 max_seq_len（不再是 0）。
    const int max_seq_len = static_cast<int>(session.max_seq_len_);

    TensorTool::full_attention_q(g_full_projection_f32, weights_.s_q_norm, g_full_q_f32, g_full_gate_f32,
                            n_heads, head_dim, d_pos, theta, partial, eps);
    TensorTool::full_attention_kv(g_full_k_f32, g_full_v_f32, weights_.s_k_norm, kv.g_key_cache_f32, kv.g_value_cache_f32,
                             kv_heads, head_dim, /*max_seq_len=*/0, d_pos, theta, partial, eps);
    TensorTool::full_attention_attend(g_full_q_f32, g_full_gate_f32, kv.g_key_cache_f32, kv.g_value_cache_f32, g_full_attn_f32,
                                 n_heads, kv_heads, head_dim, max_seq_len, d_pos);

    TensorTool::gemm(weights_.s_o_proj, g_full_attn_f32, g_out_f32, scratch, scratch_key::kFullAttnLowp, "fullattn.d_o_proj");

    kv.seq_len = pos + 1;
    // 同 prefill：不做全设备同步，依赖同流顺序 + lm_head 末尾同步 D2H。
    check_cuda(cudaGetLastError(), "FullAttention decode kernel launch 失败");
}
