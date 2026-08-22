//
// Created by zhangyoulun on 21/8/2026.
//

#ifndef LOCAL_LLM_TENSORTOOL_H
#define LOCAL_LLM_TENSORTOOL_H

#include "tensor/CPUTensor.h"
#include "tensor/StorageTensor.h"

class CudaScratch;

class TensorTool {
public:
    // s_weight: [out_dim, in_dim]，g_input/g_output 均为 GPU float 激活视图。
    static void gemm(const StorageTensor &s_weight, const GPUTensor &g_input_f32, const GPUTensor &g_output_f32,
                     CudaScratch &scratch, const std::string &lowp_key, const char *name = "");
    // s_table: embedding s_table [vocab, g_hidden]，g_input 为 GPU token id。
    static void embedding_lookup(const StorageTensor &s_table, CPUTensor c_input_i32, const GPUTensor &g_hidden_f32,
                                 CudaScratch &scratch);
    // s_weight: RMSNorm 权重，对 g_input 归一化后写入 g_output。
    static void rms_norm(const StorageTensor &s_weight, const GPUTensor &g_input_f32, const GPUTensor &g_output_f32,
                         float eps, bool one_plus);

    static void add(const GPUTensor &g_a_f32, const GPUTensor &g_b_f32, const GPUTensor &g_out_f32, void *stream = nullptr);
    static void silu_mul(const GPUTensor &g_gate_f32, const GPUTensor &g_up_f32, const GPUTensor &g_out_f32, void *stream = nullptr);

    static void full_attention_q(const GPUTensor &g_q_and_gate_f32, const StorageTensor &s_q_norm_weight,
                                 const GPUTensor &g_q_f32, const GPUTensor &g_gate_f32, int n_heads, int head_dim, int pos,
                                 float rope_theta, float partial_rotary_factor, float eps,
                                 void *stream = nullptr);
    static void full_attention_q_batch(const GPUTensor &g_q_and_gate_f32, const StorageTensor &s_q_norm_weight_f32,
                                       const GPUTensor &g_q_f32, const GPUTensor &g_gate_f32, int n_heads, int head_dim,
                                       int start_pos, float rope_theta, float partial_rotary_factor,
                                       float eps, void *stream = nullptr);
    static void full_attention_kv(const GPUTensor &g_k_in_f32, const GPUTensor &g_v_in_f32, const StorageTensor &s_k_norm_weight,
                                  const GPUTensor &g_key_cache_f32, const GPUTensor &g_value_cache_f32, int kv_heads, int head_dim,
                                  int max_seq_len, int pos, float rope_theta,
                                  float partial_rotary_factor, float eps, void *stream = nullptr);
    static void full_attention_kv_batch(const GPUTensor &g_k_in_f32, const GPUTensor &g_v_in_f32,
                                        const StorageTensor &s_k_norm_weight, const GPUTensor &g_key_cache_f32,
                                        const GPUTensor &g_value_cache_f32, int kv_heads, int head_dim,
                                        int max_seq_len, int start_pos, float rope_theta,
                                        float partial_rotary_factor, float eps,
                                        void *stream = nullptr);
    static void full_attention_attend(const GPUTensor &g_q_f32, const GPUTensor &g_gate_f32, const GPUTensor &g_key_cache_f32,
                                      const GPUTensor &g_value_cache_f32, const GPUTensor &g_attn_f32, int n_heads,
                                      int kv_heads, int head_dim, int max_seq_len, int pos,
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

    static void mla_kv_a(const GPUTensor &g_kv_a_f32, const StorageTensor &s_kv_a_norm_weight,
                         const GPUTensor &g_kv_cache_f32, int input_size, int kv_lora, int qk_rope,
                         int start_pos, const GPUTensor &g_inv_freq_f32, float eps, void *stream = nullptr);
    static void mla_rope_q(const GPUTensor &g_q_f32, int input_size, int n_heads, int qk_nope, int qk_rope,
                           int start_pos, const GPUTensor &g_inv_freq_f32, void *stream = nullptr);
    static void mla_attend(const GPUTensor &g_q_f32, const GPUTensor &g_kv_b_out_f32, const GPUTensor &g_kv_cache_f32,
                           const GPUTensor &g_attn_f32, int input_size, int n_heads, int qk_nope, int qk_rope,
                           int v_head, int kv_lora, int start_pos, float softmax_scale, void *stream = nullptr);

    static void moe_router_topk(const GPUTensor &g_router_logits_f32, const GPUTensor &g_top_idx_i32, const GPUTensor &g_top_w_f32,
                                int n_experts, int k, float routed_scaling,
                                void *stream = nullptr);
    static void moe_accumulate(const GPUTensor &g_expert_out_f32, float weight, const GPUTensor &g_out_f32,
                               void *stream = nullptr);
};

#endif // LOCAL_LLM_TENSORTOOL_H
