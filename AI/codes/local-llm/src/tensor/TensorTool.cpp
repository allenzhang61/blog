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
        throw std::runtime_error(std::string("RMSNorm 不支持的 norm dtype：") + dtype_name(dtype));
    }

    int embedding_weight_type_of(DType dtype) {
        if (dtype == DType::BF16) return 0;
        if (dtype == DType::F16) return 1;
        throw std::runtime_error(std::string("Embedding 不支持的 norm dtype：") + dtype_name(dtype));
    }

    cudaDataType_t cuda_type_of(DType dtype) {
        if (dtype == DType::BF16) return CUDA_R_16BF;
        if (dtype == DType::F16) return CUDA_R_16F;
        if (dtype == DType::F32) return CUDA_R_32F;
        throw std::runtime_error(std::string("gemm 不支持 dtype: ") + dtype_name(dtype));
    }

    //todo 删掉这层封装
    const uint16_t *lowp_data(const StorageTensor &s_weight) {
        GPUTensor g_device_weight = s_weight.to_gpu(true);
        return g_device_weight.data<uint16_t>();
    }

    void validate_mla_q_shape(const GPUTensor &g_q, int input_size, int n_heads,
                              int qk_head, const char *op) {
        if (g_q.ndim() == 3) {
            if (g_q.dim(0) != input_size || g_q.dim(1) != n_heads ||
                g_q.dim(2) != qk_head) {
                throw std::runtime_error(std::string(op) + " q shape 应为 [input_size, n_heads, qk_head]");
            }
            return;
        }
        if (g_q.ndim() == 2) {
            if (g_q.dim(0) != input_size || g_q.dim(1) != n_heads * qk_head) {
                throw std::runtime_error(std::string(op) + " q shape 应为 [input_size, n_heads*qk_head]");
            }
            return;
        }
        throw std::runtime_error(std::string(op) + " q shape 只支持二维或三维");
    }
} // namespace

//w*x=y
void TensorTool::gemm(const StorageTensor &s_weight, const GPUTensor &g_input_f32, const GPUTensor &g_output_f32,
                      CudaScratch &scratch, const std::string &lowp_key, const char *name) {
    GPUTensor g_weight = s_weight.to_gpu(true);
    uint16_t *d_input_lowp = scratch.ensure<uint16_t>(lowp_key, static_cast<size_t>(g_input_f32.numel()));

    const void *d_input = g_input_f32.data<float>();
    cudaDataType_t d_input_type = CUDA_R_32F;
    if (g_weight.dtype == DType::BF16 || g_weight.dtype == DType::F16) {
        if (g_weight.dtype == DType::BF16) {
            launch_f32_to_bf16_copy(g_input_f32.data<float>(), d_input_lowp,
                                    g_input_f32.numel(), nullptr);
        } else {
            launch_f32_to_f16_copy(g_input_f32.data<float>(), d_input_lowp,
                                   g_input_f32.numel(), nullptr);
        }
        d_input = d_input_lowp;
        d_input_type = (g_weight.dtype == DType::BF16) ? CUDA_R_16BF : CUDA_R_16F;
    } else if (g_weight.dtype != DType::F32) {
        throw std::runtime_error(std::string("gemm 不支持 dtype: ") + dtype_name(g_weight.dtype));
    }

    gemm_main(global_cuda_weight_pool().handle, g_weight.data(), d_input, g_output_f32.data<float>(),
              static_cast<int>(s_weight.shape[0]), static_cast<int>(s_weight.shape[1]),
              static_cast<size_t>(g_input_f32.rows()),
              cuda_type_of(g_weight.dtype), d_input_type, g_weight.nbytes,
              name);
}

const void *TensorTool::prepare_lowp_input(const GPUTensor &g_input_f32, DType weight_dtype,
                                           CudaScratch &scratch, const std::string &lowp_key) {
    // 权重是 f32 时无需转换，GEMM 直接吃 f32 输入。
    if (weight_dtype == DType::F32) {
        return g_input_f32.data<float>();
    }
    uint16_t *d_input_lowp = scratch.ensure<uint16_t>(lowp_key, static_cast<size_t>(g_input_f32.numel()));
    if (weight_dtype == DType::BF16) {
        launch_f32_to_bf16_copy(g_input_f32.data<float>(), d_input_lowp, g_input_f32.numel(), nullptr);
    } else if (weight_dtype == DType::F16) {
        launch_f32_to_f16_copy(g_input_f32.data<float>(), d_input_lowp, g_input_f32.numel(), nullptr);
    } else {
        throw std::runtime_error(std::string("prepare_lowp_input 不支持 dtype: ") + dtype_name(weight_dtype));
    }
    return d_input_lowp;
}

void TensorTool::gemm_lowp(const StorageTensor &s_weight, const void *d_input_lowp, int input_rows,
                           const GPUTensor &g_output_f32, const char *name) {
    GPUTensor g_weight = s_weight.to_gpu(true);
    cudaDataType_t d_input_type;
    if (g_weight.dtype == DType::BF16) {
        d_input_type = CUDA_R_16BF;
    } else if (g_weight.dtype == DType::F16) {
        d_input_type = CUDA_R_16F;
    } else if (g_weight.dtype == DType::F32) {
        d_input_type = CUDA_R_32F;
    } else {
        throw std::runtime_error(std::string("gemm_lowp 不支持 dtype: ") + dtype_name(g_weight.dtype));
    }
    gemm_main(global_cuda_weight_pool().handle, g_weight.data(), d_input_lowp, g_output_f32.data<float>(),
              static_cast<int>(s_weight.shape[0]), static_cast<int>(s_weight.shape[1]),
              static_cast<size_t>(input_rows),
              cuda_type_of(g_weight.dtype), d_input_type, g_weight.nbytes,
              name);
}

void TensorTool::embedding_lookup(const StorageTensor &s_table, CPUTensor c_input_i32,
                                  const GPUTensor &g_hidden_f32,
                                  CudaScratch &scratch) {
    GPUTensor g_input_i32 = c_input_i32.to_gpu(scratch, scratch_key::kInput,
                                               "cudaMemcpy embedding token ids 失败");
    GPUTensor g_table_u16 = s_table.to_gpu(true);
    launch_embedding_lookup(g_input_i32.data<int>(), g_hidden_f32.data<float>(),
                            g_table_u16.data<uint16_t>(),
                            static_cast<int>(c_input_i32.numel()), static_cast<int>(s_table.shape[0]),
                            static_cast<int>(s_table.shape[1]), embedding_weight_type_of(s_table.dtype),
                            nullptr);
}

void TensorTool::rms_norm(const StorageTensor &s_weight_u16, const GPUTensor &g_input_f32,
                          const GPUTensor &g_output_f32,
                          float eps, bool one_plus) {
    GPUTensor g_weight_u16 = s_weight_u16.to_gpu(true);
    const int f16_or_bf16 = norm_weight_type_of(g_weight_u16.dtype);
    launch_rms_norm(g_input_f32.data<float>(), g_output_f32.data<float>(), g_weight_u16.data<uint16_t>(),
                    f16_or_bf16, static_cast<int>(g_input_f32.rows()),
                    static_cast<int>(g_input_f32.cols()), eps, one_plus, nullptr);
}

void TensorTool::add(const GPUTensor &g_a_f32, const GPUTensor &g_b_f32, const GPUTensor &g_out_f32, void *stream) {
    launch_add(g_a_f32.data<float>(), g_b_f32.data<float>(), g_out_f32.data<float>(),
               static_cast<int>(g_out_f32.numel()), stream);
}

void TensorTool::silu_mul(const GPUTensor &g_gate_f32, const GPUTensor &g_up_f32, const GPUTensor &g_out_f32, void *stream) {
    launch_silu_mul(g_gate_f32.data<float>(), g_up_f32.data<float>(), g_out_f32.data<float>(), static_cast<int>(g_out_f32.numel()),
                    stream);
}

void TensorTool::full_attention_q(const GPUTensor &g_q_and_gate_f32, const StorageTensor &s_q_norm_weight,
                                  const GPUTensor &g_q_f32, const GPUTensor &g_gate_f32, int n_heads, int head_dim, int pos,
                                  float rope_theta, float partial_rotary_factor, float eps,
                                  void *stream) {
    launch_full_attention_q(g_q_and_gate_f32.data<float>(), lowp_data(s_q_norm_weight), g_q_f32.data<float>(),
                            g_gate_f32.data<float>(), n_heads, head_dim, pos,
                            rope_theta, partial_rotary_factor, eps, stream);
}

void TensorTool::full_attention_q_batch(const GPUTensor &g_q_and_gate_f32, const StorageTensor &s_q_norm_weight_f32,
                                        const GPUTensor &g_q_f32, const GPUTensor &g_gate_f32, int n_heads,
                                        int head_dim, int start_pos, float rope_theta,
                                        float partial_rotary_factor, float eps, void *stream) {
    launch_full_attention_q_batch(g_q_and_gate_f32.data<float>(), lowp_data(s_q_norm_weight_f32), g_q_f32.data<float>(),
                                  g_gate_f32.data<float>(),
                                  static_cast<int>(g_q_and_gate_f32.rows()), n_heads,
                                  head_dim, start_pos, rope_theta, partial_rotary_factor, eps,
                                  stream);
}

void TensorTool::full_attention_kv(const GPUTensor &g_k_in_f32, const GPUTensor &g_v_in_f32,
                                   const StorageTensor &s_k_norm_weight, const GPUTensor &g_key_cache_f32,
                                   const GPUTensor &g_value_cache_f32, int kv_heads, int head_dim,
                                   int max_seq_len, int pos, float rope_theta,
                                   float partial_rotary_factor, float eps, void *stream) {
    launch_full_attention_kv(g_k_in_f32.data<float>(), g_v_in_f32.data<float>(), lowp_data(s_k_norm_weight),
                             g_key_cache_f32.data<float>(), g_value_cache_f32.data<float>(), kv_heads,
                             head_dim, max_seq_len, pos, rope_theta, partial_rotary_factor,
                             eps, stream);
}

void TensorTool::full_attention_kv_batch(const GPUTensor &g_k_in_f32, const GPUTensor &g_v_in_f32,
                                         const StorageTensor &s_k_norm_weight, const GPUTensor &g_key_cache_f32,
                                         const GPUTensor &g_value_cache_f32, int kv_heads,
                                         int head_dim, int max_seq_len, int start_pos,
                                         float rope_theta, float partial_rotary_factor,
                                         float eps, void *stream) {
    launch_full_attention_kv_batch(g_k_in_f32.data<float>(), g_v_in_f32.data<float>(), lowp_data(s_k_norm_weight),
                                   g_key_cache_f32.data<float>(), g_value_cache_f32.data<float>(),
                                   static_cast<int>(g_k_in_f32.rows()), kv_heads, head_dim, max_seq_len, start_pos,
                                   rope_theta, partial_rotary_factor, eps, stream);
}

void TensorTool::full_attention_attend(const GPUTensor &g_q_f32, const GPUTensor &g_gate_f32,
                                       const GPUTensor &g_key_cache_f32, const GPUTensor &g_value_cache_f32,
                                       const GPUTensor &g_attn_f32, int n_heads, int kv_heads, int head_dim,
                                       int max_seq_len, int pos, void *stream) {
    launch_full_attention_attend(g_q_f32.data<float>(), g_gate_f32.data<float>(), g_key_cache_f32.data<float>(),
                                 g_value_cache_f32.data<float>(), g_attn_f32.data<float>(), n_heads, kv_heads,
                                 head_dim, max_seq_len, pos, stream);
}

void TensorTool::full_attention_attend_batch(const GPUTensor &g_q_f32, const GPUTensor &g_gate_f32,
                                             const GPUTensor &g_key_cache_f32, const GPUTensor &g_value_cache_f32,
                                             const GPUTensor &g_attn_f32, int n_heads, int kv_heads,
                                             int head_dim, int max_seq_len, int start_pos,
                                             void *stream) {
    launch_full_attention_attend_batch(g_q_f32.data<float>(), g_gate_f32.data<float>(), g_key_cache_f32.data<float>(),
                                       g_value_cache_f32.data<float>(), g_attn_f32.data<float>(), static_cast<int>(g_q_f32.rows()),
                                       n_heads, kv_heads, head_dim, max_seq_len, start_pos,
                                       stream);
}

void TensorTool::linear_attention_conv(const GPUTensor &g_mixed_f32, const StorageTensor &s_conv_weight,
                                       const GPUTensor &g_conv_state_f32, const GPUTensor &g_conv_out_f32,
                                       int kernel, void *stream) {
    launch_linear_attention_conv(g_mixed_f32.data<float>(), lowp_data(s_conv_weight), g_conv_state_f32.data<float>(),
                                 g_conv_out_f32.data<float>(), static_cast<int>(g_mixed_f32.cols()),
                                 kernel, stream);
}

void TensorTool::linear_attention_conv_batch(const GPUTensor &g_mixed_f32, const StorageTensor &s_conv_weight,
                                             const GPUTensor &g_conv_state_f32, const GPUTensor &g_conv_out_f32,
                                             int kernel, void *stream) {
    launch_linear_attention_conv_batch(g_mixed_f32.data<float>(), lowp_data(s_conv_weight),
                                       g_conv_state_f32.data<float>(), g_conv_out_f32.data<float>(),
                                       static_cast<int>(g_mixed_f32.rows()), static_cast<int>(g_mixed_f32.cols()),
                                       kernel, stream);
}

void TensorTool::linear_attention_recurrent(const GPUTensor &g_conv_out_f32, const GPUTensor &g_z_f32,
                                            const GPUTensor &g_b_f32, const GPUTensor &g_a_f32,
                                            const StorageTensor &s_a_log, const StorageTensor &s_dt_bias,
                                            const StorageTensor &s_norm_weight, const GPUTensor &g_recurrent_state_f32,
                                            const GPUTensor &g_gated_f32, int key_heads, int value_heads,
                                            int k_dim, int v_dim, float eps, void *stream) {
    GPUTensor g_a_log_f32 = s_a_log.to_gpu(true);
    GPUTensor g_norm_weight_f32 = s_norm_weight.to_gpu(true);
    launch_linear_attention_recurrent(g_conv_out_f32.data<float>(), g_z_f32.data<float>(), g_b_f32.data<float>(), g_a_f32.data<float>(),
                                      g_a_log_f32.data<float>(), lowp_data(s_dt_bias), g_norm_weight_f32.data<float>(),
                                      g_recurrent_state_f32.data<float>(), g_gated_f32.data<float>(), key_heads, value_heads,
                                      k_dim, v_dim, eps, stream);
}

void TensorTool::linear_attention_recurrent_batch(const GPUTensor &g_conv_out_f32, const GPUTensor &g_z_f32,
                                                  const GPUTensor &g_b_f32, const GPUTensor &g_a_f32,
                                                  const StorageTensor &s_a_log, const StorageTensor &s_dt_bias,
                                                  const StorageTensor &s_norm_weight,
                                                  const GPUTensor &g_recurrent_state_f32, const GPUTensor &g_gated_f32,
                                                  int key_heads, int value_heads,
                                                  int k_dim, int v_dim, float eps,
                                                  void *stream) {
    GPUTensor g_a_log_f32 = s_a_log.to_gpu(true);
    GPUTensor g_norm_weight_f32 = s_norm_weight.to_gpu(true);
    launch_linear_attention_recurrent_batch(g_conv_out_f32.data<float>(), g_z_f32.data<float>(), g_b_f32.data<float>(),
                                            g_a_f32.data<float>(),
                                            g_a_log_f32.data<float>(), lowp_data(s_dt_bias),
                                            g_norm_weight_f32.data<float>(),
                                            g_recurrent_state_f32.data<float>(), g_gated_f32.data<float>(),
                                            static_cast<int>(g_conv_out_f32.rows()), key_heads,
                                            value_heads, k_dim, v_dim, eps, stream);
}

void TensorTool::mla_kv_a(const GPUTensor &g_kv_a_f32, const StorageTensor &s_kv_a_norm_weight,
                          const GPUTensor &g_kv_cache_f32, //这个是 output
                          int input_size, int kv_lora, int qk_rope,
                          int start_pos, const GPUTensor &g_inv_freq_f32, float eps, void *stream) {
    GPUTensor g_kv_a_norm_weight_f32 = s_kv_a_norm_weight.to_gpu(true);
    launch_mla_kv_a(g_kv_a_f32.data<float>(), g_kv_a_norm_weight_f32.data<float>(),
                    g_kv_cache_f32.data<float>(), input_size, kv_lora, qk_rope,
                    start_pos, g_inv_freq_f32.data<float>(), eps, stream);
}

void TensorTool::mla_rope_q(const GPUTensor &g_q_f32, int input_size, int n_heads, int qk_nope,
                            int qk_rope, int start_pos, const GPUTensor &g_inv_freq_f32,
                            void *stream) {
    validate_mla_q_shape(g_q_f32, input_size, n_heads, qk_nope + qk_rope, "mla_rope_q");
    launch_mla_rope_q(g_q_f32.data<float>(), input_size, n_heads, qk_nope,
                      qk_rope, start_pos, g_inv_freq_f32.data<float>(), stream);
}

void TensorTool::mla_attend(const GPUTensor &g_q_f32, const GPUTensor &g_kv_b_out_f32,
                            const GPUTensor &g_kv_cache_f32, const GPUTensor &g_attn_f32,
                            int input_size, int n_heads, int qk_nope, int qk_rope,
                            int v_head, int kv_lora, int start_pos, float softmax_scale, void *stream) {
    validate_mla_q_shape(g_q_f32, input_size, n_heads, qk_nope + qk_rope, "mla_attend");
    launch_mla_attend_batch(g_q_f32.data<float>(), g_kv_b_out_f32.data<float>(),
                            g_kv_cache_f32.data<float>(), g_attn_f32.data<float>(),
                            input_size, n_heads, qk_nope,
                            qk_rope, v_head, kv_lora, start_pos,
                            softmax_scale, stream);
}

void TensorTool::moe_router_topk(const GPUTensor &g_router_logits_f32, const GPUTensor &g_top_idx_i32,
                                 const GPUTensor &g_top_w_f32,
                                 int n_experts, int k, float routed_scaling,
                                 void *stream) {
    launch_moe_router_topk(g_router_logits_f32.data<float>(), g_top_idx_i32.data<int>(), g_top_w_f32.data<float>(),
                           static_cast<int>(g_router_logits_f32.rows()), n_experts, k,
                           routed_scaling, stream);
}

void TensorTool::moe_accumulate(const GPUTensor &g_expert_out_f32, float weight, const GPUTensor &g_out_f32,
                                void *stream) {
    launch_moe_accumulate(g_expert_out_f32.data<float>(), weight, g_out_f32.data<float>(),
                          static_cast<int>(g_out_f32.numel()), stream);
}
