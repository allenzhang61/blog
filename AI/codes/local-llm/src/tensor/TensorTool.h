//
// Created by zhangyoulun on 21/8/2026.
//

#ifndef LOCAL_LLM_TENSORTOOL_H
#define LOCAL_LLM_TENSORTOOL_H

#include "tensor/CPUTensor.h"
#include "tensor/StorageTensor.h"

#include <cstdint>

class CudaScratch;

class TensorTool {
public:
    // s_weight: [out_dim, in_dim]，g_input/g_output 均为 GPU float 激活视图。
    static void gemm(const StorageTensor &s_weight, const GPUTensor &g_input_f32, const GPUTensor &g_output_f32,
                     CudaScratch &scratch, const std::string &lowp_key, const char *name = "");

    // 把 f32 输入按权重 dtype 转一次低精度并写入 scratch[lowp_key]，返回该 lowp buffer 指针与其 cuda 类型。
    // 供多个共享同一输入的 GEMM 复用，避免对同一份激活重复做 f32->bf16/f16 转换。
    // weight_dtype 必须与后续 gemm_lowp 使用的权重 dtype 一致（决定转换目标精度）。
    static const void *prepare_lowp_input(const GPUTensor &g_input_f32, DType weight_dtype,
                                          CudaScratch &scratch, const std::string &lowp_key);

    // 复用已转换好的低精度输入执行 GEMM：w*x=y。d_input_lowp 由 prepare_lowp_input 得到。
    // 权重 dtype 必须与 prepare_lowp_input 传入的 weight_dtype 一致，否则 GEMM 结果错误。
    static void gemm_lowp(const StorageTensor &s_weight, const void *d_input_lowp, int64_t input_rows,
                          const GPUTensor &g_output_f32, const char *name = "");
    // s_table: embedding s_table [vocab, g_hidden]，g_input 为 GPU token id。
    static void embedding_lookup(const StorageTensor &s_table, CPUTensor c_input_i32, const GPUTensor &g_hidden_f32,
                                 CudaScratch &scratch);
    // decode 单 token 版：token id 从 device buffer d_token 读取（不做 H2D），
    // 使 embedding 可纳入 CUDA Graph 一次 capture、后续 replay。
    static void embedding_lookup_device(const StorageTensor &s_table, const int *d_token, const GPUTensor &g_hidden_f32,
                                        void *stream = nullptr);
    // s_weight: RMSNorm 权重，对 g_input 归一化后写入 g_output。
    static void rms_norm(const StorageTensor &s_weight, const GPUTensor &g_input_f32, const GPUTensor &g_output_f32,
                         float eps, bool one_plus);

    // 融合 add + RMSNorm：g_residual_io = g_x + g_residual（写回残差和），
    // g_norm_out = rmsnorm(g_residual_io) * weight。省一次 kernel launch 与一次显存往返。
    static void add_rms_norm(const StorageTensor &s_weight, const GPUTensor &g_x_f32,
                             const GPUTensor &g_residual_f32, const GPUTensor &g_residual_io_f32,
                             const GPUTensor &g_norm_out_f32, float eps, bool one_plus,
                             void *stream = nullptr);

    static void add(const GPUTensor &g_a_f32, const GPUTensor &g_b_f32, const GPUTensor &g_out_f32, void *stream = nullptr);
    static void silu_mul(const GPUTensor &g_gate_f32, const GPUTensor &g_up_f32, const GPUTensor &g_out_f32, void *stream = nullptr);

    static void full_attention_q(const GPUTensor &g_q_and_gate_f32, const StorageTensor &s_q_norm_weight,
                                 const GPUTensor &g_q_f32, const GPUTensor &g_gate_f32, int n_heads, int head_dim,
                                 const int *pos_dev,
                                 float rope_theta, float partial_rotary_factor, float eps,
                                 void *stream = nullptr);
    static void full_attention_q_batch(const GPUTensor &g_q_and_gate_f32, const StorageTensor &s_q_norm_weight_f32,
                                       const GPUTensor &g_q_f32, const GPUTensor &g_gate_f32, int n_heads, int head_dim,
                                       int start_pos, float rope_theta, float partial_rotary_factor,
                                       float eps, void *stream = nullptr);
    static void full_attention_kv(const GPUTensor &g_k_in_f32, const GPUTensor &g_v_in_f32, const StorageTensor &s_k_norm_weight,
                                  const GPUTensor &g_key_cache_f32, const GPUTensor &g_value_cache_f32, int kv_heads, int head_dim,
                                  int max_seq_len, const int *pos_dev, float rope_theta,
                                  float partial_rotary_factor, float eps, void *stream = nullptr);
    static void full_attention_kv_batch(const GPUTensor &g_k_in_f32, const GPUTensor &g_v_in_f32,
                                        const StorageTensor &s_k_norm_weight, const GPUTensor &g_key_cache_f32,
                                        const GPUTensor &g_value_cache_f32, int kv_heads, int head_dim,
                                        int max_seq_len, int start_pos, float rope_theta,
                                        float partial_rotary_factor, float eps,
                                        void *stream = nullptr);
    static void full_attention_attend(const GPUTensor &g_q_f32, const GPUTensor &g_gate_f32, const GPUTensor &g_key_cache_f32,
                                      const GPUTensor &g_value_cache_f32, const GPUTensor &g_attn_f32, int n_heads,
                                      int kv_heads, int head_dim, int max_seq_len, const int *pos_dev,
                                      void *stream = nullptr);
    static void full_attention_attend_batch(const GPUTensor &g_q_f32, const GPUTensor &g_gate_f32,
                                            const GPUTensor &g_key_cache_f32, const GPUTensor &g_value_cache_f32,
                                            const GPUTensor &g_attn_f32, int n_heads, int kv_heads,
                                            int head_dim, int max_seq_len, int start_pos,
                                            void *stream = nullptr);

    static void linear_attention_conv(const GPUTensor &g_mixed_f32, const StorageTensor &s_conv_weight,
                                      const GPUTensor &g_conv_state_f32, const GPUTensor &g_conv_out_f32,
                                      int kernel, void *stream = nullptr);
    static void linear_attention_conv_batch(const GPUTensor &g_mixed_f32, const StorageTensor &s_conv_weight,
                                            const GPUTensor &g_conv_state_f32, const GPUTensor &g_conv_out_f32,
                                            int kernel, void *stream = nullptr);
    static void linear_attention_recurrent(const GPUTensor &g_conv_out_f32, const GPUTensor &g_z_f32, const GPUTensor &g_b_f32,
                                           const GPUTensor &g_a_f32, const StorageTensor &s_a_log,
                                           const StorageTensor &s_dt_bias, const StorageTensor &s_norm_weight,
                                           const GPUTensor &g_recurrent_state_f32, const GPUTensor &g_gated_f32, int key_heads,
                                           int value_heads, int k_dim, int v_dim, float eps,
                                           void *stream = nullptr);
    static void linear_attention_recurrent_batch(const GPUTensor &g_conv_out_f32, const GPUTensor &g_z_f32,
                                                 const GPUTensor &g_b_f32, const GPUTensor &g_a_f32,
                                                 const StorageTensor &s_a_log, const StorageTensor &s_dt_bias,
                                                 const StorageTensor &s_norm_weight, const GPUTensor &g_recurrent_state_f32,
                                                 const GPUTensor &g_gated_f32, int key_heads,
                                                 int value_heads, int k_dim, int v_dim, float eps,
                                                 void *stream = nullptr);

    // 融合 conv1d→recurrent→读出（单核）。g_recurrent_state 支持 F32 或 BF16（按其 dtype 分派）。
    static void linear_attention_fused(const GPUTensor &g_mixed_f32, const StorageTensor &s_conv_weight,
                                       const GPUTensor &g_conv_state_f32, const GPUTensor &g_z_f32,
                                       const GPUTensor &g_b_f32, const GPUTensor &g_a_f32,
                                       const StorageTensor &s_a_log, const StorageTensor &s_dt_bias,
                                       const StorageTensor &s_norm_weight, const GPUTensor &g_recurrent_state,
                                       const GPUTensor &g_gated_f32, int key_heads, int value_heads,
                                       int k_dim, int v_dim, int kernel, float eps, void *stream = nullptr);
    static void linear_attention_fused_batch(const GPUTensor &g_mixed_f32, const StorageTensor &s_conv_weight,
                                             const GPUTensor &g_conv_state_f32, const GPUTensor &g_z_f32,
                                             const GPUTensor &g_b_f32, const GPUTensor &g_a_f32,
                                             const StorageTensor &s_a_log, const StorageTensor &s_dt_bias,
                                             const StorageTensor &s_norm_weight, const GPUTensor &g_recurrent_state,
                                             const GPUTensor &g_gated_f32, int key_heads, int value_heads,
                                             int k_dim, int v_dim, int kernel, float eps, void *stream = nullptr);

    static void mla_kv_a(const GPUTensor &g_kv_a_f32, const StorageTensor &s_kv_a_norm_weight,
                         const GPUTensor &g_kv_cache_f32, int64_t input_size, int kv_lora, int qk_rope,
                         int start_pos, const GPUTensor &g_inv_freq_f32, float eps, void *stream = nullptr);
    static void mla_rope_q(const GPUTensor &g_q_f32, int64_t input_size, int n_heads, int qk_nope, int qk_rope,
                           int start_pos, const GPUTensor &g_inv_freq_f32, void *stream = nullptr);
    static void mla_attend(const GPUTensor &g_q_f32, const GPUTensor &g_kv_b_out_f32, const GPUTensor &g_kv_cache_f32,
                           const GPUTensor &g_attn_f32, int64_t input_size, int n_heads, int qk_nope, int qk_rope,
                           int v_head, int kv_lora, int start_pos, float softmax_scale, void *stream = nullptr);

    static void moe_router_topk(const GPUTensor &g_router_logits_f32, const GPUTensor &g_top_idx_i32, const GPUTensor &g_top_w_f32,
                                int n_experts, int k, float routed_scaling,
                                void *stream = nullptr);
    static void moe_accumulate(const GPUTensor &g_expert_out_f32, float weight, const GPUTensor &g_out_f32,
                               void *stream = nullptr);

    // 加权累加，权重从 device 读（d_weight 指向 device 端单个 float）：decode 时 top_w 留在 device，
    // 免去每层把权重回读到 host 造成的同步。
    static void moe_accumulate_device(const GPUTensor &g_expert_out_f32, const float *d_weight,
                                      const GPUTensor &g_out_f32, void *stream = nullptr);

    // 对 logits[vocab] 求 argmax，把 token id 写到 device 端 d_out_idx（greedy 用，结果留在 device）。
    static void argmax(const GPUTensor &g_logits_f32, int *d_out_idx, int vocab, void *stream = nullptr);
};

#endif // LOCAL_LLM_TENSORTOOL_H
