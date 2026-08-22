//
// Created by zhangyoulun on 21/8/2026.
//

#include "tensor/TensorTool.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"
#include "tensor/GPUTensor.h"

#include <stdexcept>

namespace {

int norm_weight_type_of(DType dtype) {
    if (dtype == DType::BF16) return 0;
    if (dtype == DType::F16) return 1;
    if (dtype == DType::F32) return 2;
    throw std::runtime_error(std::string("RMSNorm 不支持的 norm dtype：") + dtype_name(dtype));
}

const uint16_t *lowp_data(const StorageTensor &s_weight) {
    GPUTensor g_device_weight = s_weight.to_gpu(true);
    return static_cast<const uint16_t *>(g_device_weight.data());
}

const float *f32_data(const StorageTensor &s_weight) {
    GPUTensor g_device_weight = s_weight.to_gpu(true);
    return static_cast<const float *>(g_device_weight.data());
}

} // namespace

void TensorTool::gemm(const StorageTensor &s_weight, const GPUTensor &g_input, const GPUTensor &g_output,
                      CudaScratch &scratch, const std::string &lowp_key, const char *name) {
    GPUTensor g_device_weight = s_weight.to_gpu(true);
    const int out_dim = static_cast<int>(s_weight.shape[0]);
    const int in_dim = static_cast<int>(s_weight.shape[1]);
    const size_t input_size = static_cast<size_t>(g_input.rows());
    uint16_t *d_input_lowp = scratch.ensure<uint16_t>(lowp_key, static_cast<size_t>(g_input.numel()));
    GemmInput gemm_in = prepare_gemm_input(static_cast<float *>(g_input.data()), d_input_lowp,
                                           static_cast<size_t>(g_input.numel()), g_device_weight.dtype, nullptr);
    gemm_weight(global_cuda_weight_pool().handle, g_device_weight, gemm_in.ptr, static_cast<float *>(g_output.data()),
                out_dim, in_dim, input_size, gemm_in.type, name);
}

void TensorTool::embedding_lookup(const StorageTensor &s_table, CPUTensor c_input, const GPUTensor &g_hidden,
                                  CudaScratch &scratch) {
    GPUTensor g_device_input = c_input.to_gpu(scratch, scratch_key::kInput,
                                          "cudaMemcpy embedding token ids 失败");
    GPUTensor g_device_table = s_table.to_gpu(true);
    const int lowp_type = (g_device_table.dtype == DType::F16) ? 1 : 0;
    launch_embedding_lookup(static_cast<int *>(g_device_input.data()), static_cast<float *>(g_hidden.data()),
                            static_cast<const uint16_t *>(g_device_table.data()),
                            static_cast<int>(c_input.numel()), static_cast<int>(s_table.shape[0]),
                            static_cast<int>(s_table.shape[1]), lowp_type, nullptr);
}

void TensorTool::rms_norm(const StorageTensor &s_weight, const GPUTensor &g_input, const GPUTensor &g_output,
                          float eps, bool one_plus) {
    GPUTensor g_device_weight = s_weight.to_gpu(true);
    launch_rms_norm(static_cast<float *>(g_input.data()), static_cast<float *>(g_output.data()), g_device_weight.data(),
                    norm_weight_type_of(g_device_weight.dtype), static_cast<int>(g_input.rows()),
                    static_cast<int>(g_input.cols()), eps, one_plus, nullptr);
}

void TensorTool::add(const GPUTensor &g_a, const GPUTensor &g_b, const GPUTensor &g_out, void *stream) {
    launch_add(static_cast<float *>(g_a.data()), static_cast<float *>(g_b.data()), static_cast<float *>(g_out.data()), static_cast<int>(g_out.numel()), stream);
}

void TensorTool::silu_mul(const GPUTensor &g_gate, const GPUTensor &g_up, const GPUTensor &g_out, void *stream) {
    launch_silu_mul(static_cast<float *>(g_gate.data()), static_cast<float *>(g_up.data()), static_cast<float *>(g_out.data()), static_cast<int>(g_out.numel()), stream);
}

void TensorTool::full_attention_q(const GPUTensor &g_q_and_gate, const StorageTensor &s_q_norm_weight,
                                  const GPUTensor &g_q, const GPUTensor &g_gate, int n_heads, int head_dim, int pos,
                                  float rope_theta, float partial_rotary_factor, float eps,
                                  void *stream) {
    launch_full_attention_q(static_cast<float *>(g_q_and_gate.data()), lowp_data(s_q_norm_weight), static_cast<float *>(g_q.data()), static_cast<float *>(g_gate.data()), n_heads, head_dim, pos,
                            rope_theta, partial_rotary_factor, eps, stream);
}

void TensorTool::full_attention_q_batch(const GPUTensor &g_q_and_gate, const StorageTensor &s_q_norm_weight,
                                        const GPUTensor &g_q, const GPUTensor &g_gate, int n_heads,
                                        int head_dim, int start_pos, float rope_theta,
                                        float partial_rotary_factor, float eps, void *stream) {
    launch_full_attention_q_batch(static_cast<float *>(g_q_and_gate.data()), lowp_data(s_q_norm_weight), static_cast<float *>(g_q.data()), static_cast<float *>(g_gate.data()),
                                  static_cast<int>(g_q_and_gate.rows()), n_heads,
                                  head_dim, start_pos, rope_theta, partial_rotary_factor, eps,
                                  stream);
}

void TensorTool::full_attention_kv(const GPUTensor &g_k_in, const GPUTensor &g_v_in,
                                   const StorageTensor &s_k_norm_weight, const GPUTensor &g_key_cache,
                                   const GPUTensor &g_value_cache, int kv_heads, int head_dim,
                                   int max_seq_len, int pos, float rope_theta,
                                   float partial_rotary_factor, float eps, void *stream) {
    launch_full_attention_kv(static_cast<float *>(g_k_in.data()), static_cast<float *>(g_v_in.data()), lowp_data(s_k_norm_weight),
                             static_cast<float *>(g_key_cache.data()), static_cast<float *>(g_value_cache.data()), kv_heads,
                             head_dim, max_seq_len, pos, rope_theta, partial_rotary_factor,
                             eps, stream);
}

void TensorTool::full_attention_kv_batch(const GPUTensor &g_k_in, const GPUTensor &g_v_in,
                                         const StorageTensor &s_k_norm_weight, const GPUTensor &g_key_cache,
                                         const GPUTensor &g_value_cache, int kv_heads,
                                         int head_dim, int max_seq_len, int start_pos,
                                         float rope_theta, float partial_rotary_factor,
                                         float eps, void *stream) {
    launch_full_attention_kv_batch(static_cast<float *>(g_k_in.data()), static_cast<float *>(g_v_in.data()), lowp_data(s_k_norm_weight),
                                   static_cast<float *>(g_key_cache.data()), static_cast<float *>(g_value_cache.data()),
                                   static_cast<int>(g_k_in.rows()), kv_heads, head_dim, max_seq_len, start_pos,
                                   rope_theta, partial_rotary_factor, eps, stream);
}

void TensorTool::full_attention_attend(const GPUTensor &g_q, const GPUTensor &g_gate,
                                       const GPUTensor &g_key_cache, const GPUTensor &g_value_cache,
                                       const GPUTensor &g_attn, int n_heads, int kv_heads, int head_dim,
                                       int max_seq_len, int pos, void *stream) {
    launch_full_attention_attend(static_cast<float *>(g_q.data()), static_cast<float *>(g_gate.data()), static_cast<float *>(g_key_cache.data()),
                                 static_cast<float *>(g_value_cache.data()), static_cast<float *>(g_attn.data()), n_heads, kv_heads,
                                 head_dim, max_seq_len, pos, stream);
}

void TensorTool::full_attention_attend_batch(const GPUTensor &g_q, const GPUTensor &g_gate,
                                             const GPUTensor &g_key_cache, const GPUTensor &g_value_cache,
                                             const GPUTensor &g_attn, int n_heads, int kv_heads,
                                             int head_dim, int max_seq_len, int start_pos,
                                             void *stream) {
    launch_full_attention_attend_batch(static_cast<float *>(g_q.data()), static_cast<float *>(g_gate.data()), static_cast<float *>(g_key_cache.data()),
                                       static_cast<float *>(g_value_cache.data()), static_cast<float *>(g_attn.data()), static_cast<int>(g_q.rows()),
                                       n_heads, kv_heads, head_dim, max_seq_len, start_pos,
                                       stream);
}

void TensorTool::linear_attention_conv(const GPUTensor &g_mixed, const StorageTensor &s_conv_weight,
                                       const GPUTensor &g_conv_state, const GPUTensor &g_conv_out,
                                       int kernel, void *stream) {
    launch_linear_attention_conv(static_cast<float *>(g_mixed.data()), lowp_data(s_conv_weight), static_cast<float *>(g_conv_state.data()),
                                 static_cast<float *>(g_conv_out.data()), static_cast<int>(g_mixed.cols()),
                                 kernel, stream);
}

void TensorTool::linear_attention_conv_batch(const GPUTensor &g_mixed, const StorageTensor &s_conv_weight,
                                             const GPUTensor &g_conv_state, const GPUTensor &g_conv_out,
                                             int kernel, void *stream) {
    launch_linear_attention_conv_batch(static_cast<float *>(g_mixed.data()), lowp_data(s_conv_weight),
                                       static_cast<float *>(g_conv_state.data()), static_cast<float *>(g_conv_out.data()),
                                       static_cast<int>(g_mixed.rows()), static_cast<int>(g_mixed.cols()),
                                       kernel, stream);
}

void TensorTool::linear_attention_recurrent(const GPUTensor &g_conv_out, const GPUTensor &g_z,
                                            const GPUTensor &g_b, const GPUTensor &g_a,
                                            const StorageTensor &s_a_log, const StorageTensor &s_dt_bias,
                                            const StorageTensor &s_norm_weight, const GPUTensor &g_recurrent_state,
                                            const GPUTensor &g_gated, int key_heads, int value_heads,
                                            int k_dim, int v_dim, float eps, void *stream) {
    launch_linear_attention_recurrent(static_cast<float *>(g_conv_out.data()), static_cast<float *>(g_z.data()), static_cast<float *>(g_b.data()), static_cast<float *>(g_a.data()),
                                      f32_data(s_a_log), lowp_data(s_dt_bias), f32_data(s_norm_weight),
                                      static_cast<float *>(g_recurrent_state.data()), static_cast<float *>(g_gated.data()), key_heads, value_heads,
                                      k_dim, v_dim, eps, stream);
}

void TensorTool::linear_attention_recurrent_batch(const GPUTensor &g_conv_out, const GPUTensor &g_z,
                                                  const GPUTensor &g_b, const GPUTensor &g_a,
                                                  const StorageTensor &s_a_log, const StorageTensor &s_dt_bias,
                                                  const StorageTensor &s_norm_weight,
                                                  const GPUTensor &g_recurrent_state, const GPUTensor &g_gated,
                                                  int key_heads, int value_heads,
                                                  int k_dim, int v_dim, float eps,
                                                  void *stream) {
    launch_linear_attention_recurrent_batch(static_cast<float *>(g_conv_out.data()), static_cast<float *>(g_z.data()), static_cast<float *>(g_b.data()), static_cast<float *>(g_a.data()),
                                            f32_data(s_a_log), lowp_data(s_dt_bias), f32_data(s_norm_weight),
                                            static_cast<float *>(g_recurrent_state.data()), static_cast<float *>(g_gated.data()),
                                            static_cast<int>(g_conv_out.rows()), key_heads,
                                            value_heads, k_dim, v_dim, eps, stream);
}

void TensorTool::mla_kv_a(const GPUTensor &g_kv_a, const StorageTensor &s_kv_a_norm_weight,
                          const GPUTensor &g_kv_cache, int kv_lora, int qk_rope, int max_seq_len,
                          int pos, const GPUTensor &g_inv_freq, float eps, void *stream) {
    launch_mla_kv_a(static_cast<float *>(g_kv_a.data()), f32_data(s_kv_a_norm_weight), static_cast<float *>(g_kv_cache.data()),
                    kv_lora, qk_rope, max_seq_len, pos, static_cast<float *>(g_inv_freq.data()), eps, stream);
}

void TensorTool::mla_kv_a_batch(const GPUTensor &g_kv_a, const StorageTensor &s_kv_a_norm_weight,
                                const GPUTensor &g_kv_cache, int kv_lora, int qk_rope,
                                int max_seq_len, int start_pos, const GPUTensor &g_inv_freq,
                                float eps, void *stream) {
    launch_mla_kv_a_batch(static_cast<float *>(g_kv_a.data()), f32_data(s_kv_a_norm_weight), static_cast<float *>(g_kv_cache.data()),
                          static_cast<int>(g_kv_a.rows()), kv_lora, qk_rope,
                          max_seq_len, start_pos, static_cast<float *>(g_inv_freq.data()), eps, stream);
}

void TensorTool::mla_rope_q(const GPUTensor &g_q, int n_heads, int qk_nope, int qk_rope,
                            int pos, const GPUTensor &g_inv_freq, void *stream) {
    launch_mla_rope_q(static_cast<float *>(g_q.data()), n_heads, qk_nope, qk_rope, pos, static_cast<float *>(g_inv_freq.data()), stream);
}

void TensorTool::mla_rope_q_batch(const GPUTensor &g_q, int n_heads, int qk_nope,
                                  int qk_rope, int start_pos, const GPUTensor &g_inv_freq,
                                  void *stream) {
    launch_mla_rope_q_batch(static_cast<float *>(g_q.data()), static_cast<int>(g_q.rows()), n_heads, qk_nope,
                            qk_rope, start_pos, static_cast<float *>(g_inv_freq.data()), stream);
}

void TensorTool::mla_attend(const GPUTensor &g_q, const GPUTensor &g_kv_b_out, const GPUTensor &g_kv_cache,
                            const GPUTensor &g_attn, int n_heads, int qk_nope, int qk_rope,
                            int v_head, int kv_lora, int max_seq_len, int pos,
                            float softmax_scale, void *stream) {
    launch_mla_attend(static_cast<float *>(g_q.data()), static_cast<float *>(g_kv_b_out.data()), static_cast<float *>(g_kv_cache.data()), static_cast<float *>(g_attn.data()),
                      n_heads, qk_nope, qk_rope,
                      v_head, kv_lora, max_seq_len, pos, softmax_scale, stream);
}

void TensorTool::mla_attend_batch(const GPUTensor &g_q, const GPUTensor &g_kv_b_out,
                                  const GPUTensor &g_kv_cache, const GPUTensor &g_attn,
                                  int n_heads, int qk_nope, int qk_rope, int v_head,
                                  int kv_lora, int max_seq_len, int start_pos,
                                  float softmax_scale, void *stream) {
    launch_mla_attend_batch(static_cast<float *>(g_q.data()), static_cast<float *>(g_kv_b_out.data()), static_cast<float *>(g_kv_cache.data()), static_cast<float *>(g_attn.data()),
                            static_cast<int>(g_q.rows()), n_heads, qk_nope,
                            qk_rope, v_head, kv_lora, max_seq_len, start_pos,
                            softmax_scale, stream);
}

void TensorTool::moe_router_topk(const GPUTensor &g_router_logits, const GPUTensor &g_top_idx, const GPUTensor &g_top_w,
                                 int n_experts, int k, float routed_scaling,
                                 void *stream) {
    launch_moe_router_topk(static_cast<float *>(g_router_logits.data()), static_cast<int *>(g_top_idx.data()), static_cast<float *>(g_top_w.data()),
                           static_cast<int>(g_router_logits.rows()), n_experts, k,
                           routed_scaling, stream);
}

void TensorTool::moe_accumulate(const GPUTensor &g_expert_out, float weight, const GPUTensor &g_out,
                                void *stream) {
    launch_moe_accumulate(static_cast<float *>(g_expert_out.data()), weight, static_cast<float *>(g_out.data()),
                          static_cast<int>(g_out.numel()), stream);
}
