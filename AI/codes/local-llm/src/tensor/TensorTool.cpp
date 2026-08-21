//
// Created by zhangyoulun on 21/8/2026.
//

#include "tensor/TensorTool.h"

#include "backend/cuda/mem/CudaScratch.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"

#include <stdexcept>

namespace {

int norm_weight_type_of(DType dtype) {
    if (dtype == DType::BF16) return 0;
    if (dtype == DType::F16) return 1;
    if (dtype == DType::F32) return 2;
    throw std::runtime_error(std::string("RMSNorm 不支持的 norm dtype：") + dtype_name(dtype));
}

const uint16_t *lowp_data(const Tensor &weight) {
    weight.to_gpu();
    Tensor device_weight = weight.try_dequant();
    return static_cast<const uint16_t *>(device_weight.gpu_data_dequant);
}

const float *f32_data(const Tensor &weight) {
    weight.to_gpu();
    Tensor device_weight = weight.try_dequant();
    return static_cast<const float *>(device_weight.gpu_data_dequant);
}

} // namespace

void TensorTool::gemm(const Tensor &weight, const Tensor &input, const Tensor &output,
                      CudaScratch &scratch, const std::string &lowp_key, const char *name) {
    weight.to_gpu();
    Tensor device_weight = weight.try_dequant();
    const int out_dim = static_cast<int>(weight.shape[0]);
    const int in_dim = static_cast<int>(weight.shape[1]);
    const size_t input_size = static_cast<size_t>(input.rows());
    uint16_t *d_input_lowp = scratch.ensure<uint16_t>(lowp_key, static_cast<size_t>(input.numel()));
    GemmInput gemm_in = prepare_gemm_input(input.gpu_f32(), d_input_lowp,
                                           static_cast<size_t>(input.numel()), device_weight.dtype_dequant, nullptr);
    gemm_weight(global_cuda_weight_pool().handle, device_weight, gemm_in.ptr, output.gpu_f32(),
                out_dim, in_dim, input_size, gemm_in.type, name);
}

void TensorTool::embedding_lookup(const Tensor &table, Tensor input, const Tensor &hidden,
                                  CudaScratch &scratch) {
    input.to_gpu(scratch, scratch_key::kInput, "cudaMemcpy embedding token ids 失败");
    table.to_gpu();
    Tensor device_table = table.try_dequant();
    const int lowp_type = (device_table.dtype_dequant == DType::F16) ? 1 : 0;
    launch_embedding_lookup(input.gpu_i32(), hidden.gpu_f32(),
                            static_cast<const uint16_t *>(device_table.gpu_data_dequant),
                            static_cast<int>(input.numel()), static_cast<int>(table.shape[0]),
                            static_cast<int>(table.shape[1]), lowp_type, nullptr);
}

void TensorTool::rms_norm(const Tensor &weight, const Tensor &input, const Tensor &output,
                          float eps, bool one_plus) {
    weight.to_gpu();
    Tensor device_weight = weight.try_dequant();
    launch_rms_norm(input.gpu_f32(), output.gpu_f32(), device_weight.gpu_data_dequant,
                    norm_weight_type_of(device_weight.dtype_dequant), static_cast<int>(input.rows()),
                    static_cast<int>(input.cols()), eps, one_plus, nullptr);
}

void TensorTool::add(const Tensor &a, const Tensor &b, const Tensor &out, void *stream) {
    launch_add(a.gpu_f32(), b.gpu_f32(), out.gpu_f32(), static_cast<int>(out.numel()), stream);
}

void TensorTool::silu_mul(const Tensor &gate, const Tensor &up, const Tensor &out, void *stream) {
    launch_silu_mul(gate.gpu_f32(), up.gpu_f32(), out.gpu_f32(), static_cast<int>(out.numel()), stream);
}

void TensorTool::full_attention_q(const Tensor &q_and_gate, const Tensor &q_norm_weight,
                                  const Tensor &q, const Tensor &gate, int n_heads, int head_dim, int pos,
                                  float rope_theta, float partial_rotary_factor, float eps,
                                  void *stream) {
    launch_full_attention_q(q_and_gate.gpu_f32(), lowp_data(q_norm_weight), q.gpu_f32(), gate.gpu_f32(), n_heads, head_dim, pos,
                            rope_theta, partial_rotary_factor, eps, stream);
}

void TensorTool::full_attention_q_batch(const Tensor &q_and_gate, const Tensor &q_norm_weight,
                                        const Tensor &q, const Tensor &gate, int n_heads,
                                        int head_dim, int start_pos, float rope_theta,
                                        float partial_rotary_factor, float eps, void *stream) {
    launch_full_attention_q_batch(q_and_gate.gpu_f32(), lowp_data(q_norm_weight), q.gpu_f32(), gate.gpu_f32(),
                                  static_cast<int>(q_and_gate.rows()), n_heads,
                                  head_dim, start_pos, rope_theta, partial_rotary_factor, eps,
                                  stream);
}

void TensorTool::full_attention_kv(const Tensor &k_in, const Tensor &v_in,
                                   const Tensor &k_norm_weight, const Tensor &key_cache,
                                   const Tensor &value_cache, int kv_heads, int head_dim,
                                   int max_seq_len, int pos, float rope_theta,
                                   float partial_rotary_factor, float eps, void *stream) {
    launch_full_attention_kv(k_in.gpu_f32(), v_in.gpu_f32(), lowp_data(k_norm_weight),
                             key_cache.gpu_f32(), value_cache.gpu_f32(), kv_heads,
                             head_dim, max_seq_len, pos, rope_theta, partial_rotary_factor,
                             eps, stream);
}

void TensorTool::full_attention_kv_batch(const Tensor &k_in, const Tensor &v_in,
                                         const Tensor &k_norm_weight, const Tensor &key_cache,
                                         const Tensor &value_cache, int kv_heads,
                                         int head_dim, int max_seq_len, int start_pos,
                                         float rope_theta, float partial_rotary_factor,
                                         float eps, void *stream) {
    launch_full_attention_kv_batch(k_in.gpu_f32(), v_in.gpu_f32(), lowp_data(k_norm_weight),
                                   key_cache.gpu_f32(), value_cache.gpu_f32(),
                                   static_cast<int>(k_in.rows()), kv_heads, head_dim, max_seq_len, start_pos,
                                   rope_theta, partial_rotary_factor, eps, stream);
}

void TensorTool::full_attention_attend(const Tensor &q, const Tensor &gate,
                                       const Tensor &key_cache, const Tensor &value_cache,
                                       const Tensor &attn, int n_heads, int kv_heads, int head_dim,
                                       int max_seq_len, int pos, void *stream) {
    launch_full_attention_attend(q.gpu_f32(), gate.gpu_f32(), key_cache.gpu_f32(),
                                 value_cache.gpu_f32(), attn.gpu_f32(), n_heads, kv_heads,
                                 head_dim, max_seq_len, pos, stream);
}

void TensorTool::full_attention_attend_batch(const Tensor &q, const Tensor &gate,
                                             const Tensor &key_cache, const Tensor &value_cache,
                                             const Tensor &attn, int n_heads, int kv_heads,
                                             int head_dim, int max_seq_len, int start_pos,
                                             void *stream) {
    launch_full_attention_attend_batch(q.gpu_f32(), gate.gpu_f32(), key_cache.gpu_f32(),
                                       value_cache.gpu_f32(), attn.gpu_f32(), static_cast<int>(q.rows()),
                                       n_heads, kv_heads, head_dim, max_seq_len, start_pos,
                                       stream);
}

void TensorTool::linear_attention_conv(const Tensor &mixed, const Tensor &conv_weight,
                                       const Tensor &conv_state, const Tensor &conv_out,
                                       int kernel, void *stream) {
    launch_linear_attention_conv(mixed.gpu_f32(), lowp_data(conv_weight), conv_state.gpu_f32(),
                                 conv_out.gpu_f32(), static_cast<int>(mixed.cols()),
                                 kernel, stream);
}

void TensorTool::linear_attention_conv_batch(const Tensor &mixed, const Tensor &conv_weight,
                                             const Tensor &conv_state, const Tensor &conv_out,
                                             int kernel, void *stream) {
    launch_linear_attention_conv_batch(mixed.gpu_f32(), lowp_data(conv_weight),
                                       conv_state.gpu_f32(), conv_out.gpu_f32(),
                                       static_cast<int>(mixed.rows()), static_cast<int>(mixed.cols()),
                                       kernel, stream);
}

void TensorTool::linear_attention_recurrent(const Tensor &conv_out, const Tensor &z,
                                            const Tensor &b, const Tensor &a,
                                            const Tensor &a_log, const Tensor &dt_bias,
                                            const Tensor &norm_weight, const Tensor &recurrent_state,
                                            const Tensor &gated, int key_heads, int value_heads,
                                            int k_dim, int v_dim, float eps, void *stream) {
    launch_linear_attention_recurrent(conv_out.gpu_f32(), z.gpu_f32(), b.gpu_f32(), a.gpu_f32(),
                                      f32_data(a_log), lowp_data(dt_bias), f32_data(norm_weight),
                                      recurrent_state.gpu_f32(), gated.gpu_f32(), key_heads, value_heads,
                                      k_dim, v_dim, eps, stream);
}

void TensorTool::linear_attention_recurrent_batch(const Tensor &conv_out, const Tensor &z,
                                                  const Tensor &b, const Tensor &a,
                                                  const Tensor &a_log, const Tensor &dt_bias,
                                                  const Tensor &norm_weight,
                                                  const Tensor &recurrent_state, const Tensor &gated,
                                                  int key_heads, int value_heads,
                                                  int k_dim, int v_dim, float eps,
                                                  void *stream) {
    launch_linear_attention_recurrent_batch(conv_out.gpu_f32(), z.gpu_f32(), b.gpu_f32(), a.gpu_f32(),
                                            f32_data(a_log), lowp_data(dt_bias), f32_data(norm_weight),
                                            recurrent_state.gpu_f32(), gated.gpu_f32(),
                                            static_cast<int>(conv_out.rows()), key_heads,
                                            value_heads, k_dim, v_dim, eps, stream);
}

void TensorTool::mla_kv_a(const Tensor &kv_a, const Tensor &kv_a_norm_weight,
                          const Tensor &kv_cache, int kv_lora, int qk_rope, int max_seq_len,
                          int pos, const Tensor &inv_freq, float eps, void *stream) {
    launch_mla_kv_a(kv_a.gpu_f32(), f32_data(kv_a_norm_weight), kv_cache.gpu_f32(),
                    kv_lora, qk_rope, max_seq_len, pos, inv_freq.gpu_f32(), eps, stream);
}

void TensorTool::mla_kv_a_batch(const Tensor &kv_a, const Tensor &kv_a_norm_weight,
                                const Tensor &kv_cache, int kv_lora, int qk_rope,
                                int max_seq_len, int start_pos, const Tensor &inv_freq,
                                float eps, void *stream) {
    launch_mla_kv_a_batch(kv_a.gpu_f32(), f32_data(kv_a_norm_weight), kv_cache.gpu_f32(),
                          static_cast<int>(kv_a.rows()), kv_lora, qk_rope,
                          max_seq_len, start_pos, inv_freq.gpu_f32(), eps, stream);
}

void TensorTool::mla_rope_q(const Tensor &q, int n_heads, int qk_nope, int qk_rope,
                            int pos, const Tensor &inv_freq, void *stream) {
    launch_mla_rope_q(q.gpu_f32(), n_heads, qk_nope, qk_rope, pos, inv_freq.gpu_f32(), stream);
}

void TensorTool::mla_rope_q_batch(const Tensor &q, int n_heads, int qk_nope,
                                  int qk_rope, int start_pos, const Tensor &inv_freq,
                                  void *stream) {
    launch_mla_rope_q_batch(q.gpu_f32(), static_cast<int>(q.rows()), n_heads, qk_nope,
                            qk_rope, start_pos, inv_freq.gpu_f32(), stream);
}

void TensorTool::mla_attend(const Tensor &q, const Tensor &kv_b_out, const Tensor &kv_cache,
                            const Tensor &attn, int n_heads, int qk_nope, int qk_rope,
                            int v_head, int kv_lora, int max_seq_len, int pos,
                            float softmax_scale, void *stream) {
    launch_mla_attend(q.gpu_f32(), kv_b_out.gpu_f32(), kv_cache.gpu_f32(), attn.gpu_f32(),
                      n_heads, qk_nope, qk_rope,
                      v_head, kv_lora, max_seq_len, pos, softmax_scale, stream);
}

void TensorTool::mla_attend_batch(const Tensor &q, const Tensor &kv_b_out,
                                  const Tensor &kv_cache, const Tensor &attn,
                                  int n_heads, int qk_nope, int qk_rope, int v_head,
                                  int kv_lora, int max_seq_len, int start_pos,
                                  float softmax_scale, void *stream) {
    launch_mla_attend_batch(q.gpu_f32(), kv_b_out.gpu_f32(), kv_cache.gpu_f32(), attn.gpu_f32(),
                            static_cast<int>(q.rows()), n_heads, qk_nope,
                            qk_rope, v_head, kv_lora, max_seq_len, start_pos,
                            softmax_scale, stream);
}

void TensorTool::moe_router_topk(const Tensor &router_logits, const Tensor &top_idx, const Tensor &top_w,
                                 int n_experts, int k, float routed_scaling,
                                 void *stream) {
    launch_moe_router_topk(router_logits.gpu_f32(), top_idx.gpu_i32(), top_w.gpu_f32(),
                           static_cast<int>(router_logits.rows()), n_experts, k,
                           routed_scaling, stream);
}

void TensorTool::moe_accumulate(const Tensor &expert_out, float weight, const Tensor &out,
                                void *stream) {
    launch_moe_accumulate(expert_out.gpu_f32(), weight, out.gpu_f32(),
                          static_cast<int>(out.numel()), stream);
}
