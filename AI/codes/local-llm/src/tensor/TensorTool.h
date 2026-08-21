//
// Created by zhangyoulun on 21/8/2026.
//

#ifndef LOCAL_LLM_TENSORTOOL_H
#define LOCAL_LLM_TENSORTOOL_H

#include "tensor/CPUTensor.h"
#include "tensor/DiskTensor.h"

class CudaScratch;

class TensorTool {
public:
    // weight: [out_dim, in_dim]，input/output 均为 GPU float 激活视图。
    static void gemm(const DiskTensor &weight, const GPUTensor &input, const GPUTensor &output,
                     CudaScratch &scratch, const std::string &lowp_key, const char *name = "");
    // table: embedding table [vocab, hidden]，input 为 GPU token id。
    static void embedding_lookup(const DiskTensor &table, CPUTensor input, const GPUTensor &hidden,
                                 CudaScratch &scratch);
    // weight: RMSNorm 权重，对 input 归一化后写入 output。
    static void rms_norm(const DiskTensor &weight, const GPUTensor &input, const GPUTensor &output,
                         float eps, bool one_plus);

    static void add(const GPUTensor &a, const GPUTensor &b, const GPUTensor &out, void *stream = nullptr);
    static void silu_mul(const GPUTensor &gate, const GPUTensor &up, const GPUTensor &out, void *stream = nullptr);

    static void full_attention_q(const GPUTensor &q_and_gate, const DiskTensor &q_norm_weight,
                                 const GPUTensor &q, const GPUTensor &gate, int n_heads, int head_dim, int pos,
                                 float rope_theta, float partial_rotary_factor, float eps,
                                 void *stream = nullptr);
    static void full_attention_q_batch(const GPUTensor &q_and_gate, const DiskTensor &q_norm_weight,
                                       const GPUTensor &q, const GPUTensor &gate, int n_heads, int head_dim,
                                       int start_pos, float rope_theta, float partial_rotary_factor,
                                       float eps, void *stream = nullptr);
    static void full_attention_kv(const GPUTensor &k_in, const GPUTensor &v_in, const DiskTensor &k_norm_weight,
                                  const GPUTensor &key_cache, const GPUTensor &value_cache, int kv_heads, int head_dim,
                                  int max_seq_len, int pos, float rope_theta,
                                  float partial_rotary_factor, float eps, void *stream = nullptr);
    static void full_attention_kv_batch(const GPUTensor &k_in, const GPUTensor &v_in,
                                        const DiskTensor &k_norm_weight, const GPUTensor &key_cache,
                                        const GPUTensor &value_cache, int kv_heads, int head_dim,
                                        int max_seq_len, int start_pos, float rope_theta,
                                        float partial_rotary_factor, float eps,
                                        void *stream = nullptr);
    static void full_attention_attend(const GPUTensor &q, const GPUTensor &gate, const GPUTensor &key_cache,
                                      const GPUTensor &value_cache, const GPUTensor &attn, int n_heads,
                                      int kv_heads, int head_dim, int max_seq_len, int pos,
                                      void *stream = nullptr);
    static void full_attention_attend_batch(const GPUTensor &q, const GPUTensor &gate,
                                            const GPUTensor &key_cache, const GPUTensor &value_cache,
                                            const GPUTensor &attn, int n_heads, int kv_heads,
                                            int head_dim, int max_seq_len, int start_pos,
                                            void *stream = nullptr);

    static void linear_attention_conv(const GPUTensor &mixed, const DiskTensor &conv_weight,
                                      const GPUTensor &conv_state, const GPUTensor &conv_out,
                                      int kernel, void *stream = nullptr);
    static void linear_attention_conv_batch(const GPUTensor &mixed, const DiskTensor &conv_weight,
                                            const GPUTensor &conv_state, const GPUTensor &conv_out,
                                            int kernel, void *stream = nullptr);
    static void linear_attention_recurrent(const GPUTensor &conv_out, const GPUTensor &z, const GPUTensor &b,
                                           const GPUTensor &a, const DiskTensor &a_log,
                                           const DiskTensor &dt_bias, const DiskTensor &norm_weight,
                                           const GPUTensor &recurrent_state, const GPUTensor &gated, int key_heads,
                                           int value_heads, int k_dim, int v_dim, float eps,
                                           void *stream = nullptr);
    static void linear_attention_recurrent_batch(const GPUTensor &conv_out, const GPUTensor &z,
                                                 const GPUTensor &b, const GPUTensor &a,
                                                 const DiskTensor &a_log, const DiskTensor &dt_bias,
                                                 const DiskTensor &norm_weight, const GPUTensor &recurrent_state,
                                                 const GPUTensor &gated, int key_heads,
                                                 int value_heads, int k_dim, int v_dim, float eps,
                                                 void *stream = nullptr);

    static void mla_kv_a(const GPUTensor &kv_a, const DiskTensor &kv_a_norm_weight, const GPUTensor &kv_cache,
                         int kv_lora, int qk_rope, int max_seq_len, int pos,
                         const GPUTensor &inv_freq, float eps, void *stream = nullptr);
    static void mla_kv_a_batch(const GPUTensor &kv_a, const DiskTensor &kv_a_norm_weight, const GPUTensor &kv_cache,
                               int kv_lora, int qk_rope, int max_seq_len,
                               int start_pos, const GPUTensor &inv_freq, float eps,
                               void *stream = nullptr);
    static void mla_rope_q(const GPUTensor &q, int n_heads, int qk_nope, int qk_rope, int pos,
                           const GPUTensor &inv_freq, void *stream = nullptr);
    static void mla_rope_q_batch(const GPUTensor &q, int n_heads, int qk_nope, int qk_rope,
                                 int start_pos, const GPUTensor &inv_freq, void *stream = nullptr);
    static void mla_attend(const GPUTensor &q, const GPUTensor &kv_b_out, const GPUTensor &kv_cache,
                           const GPUTensor &attn, int n_heads, int qk_nope, int qk_rope, int v_head,
                           int kv_lora, int max_seq_len, int pos, float softmax_scale,
                           void *stream = nullptr);
    static void mla_attend_batch(const GPUTensor &q, const GPUTensor &kv_b_out, const GPUTensor &kv_cache,
                                 const GPUTensor &attn, int n_heads, int qk_nope, int qk_rope,
                                 int v_head, int kv_lora, int max_seq_len, int start_pos,
                                 float softmax_scale, void *stream = nullptr);

    static void moe_router_topk(const GPUTensor &router_logits, const GPUTensor &top_idx, const GPUTensor &top_w,
                                int n_experts, int k, float routed_scaling,
                                void *stream = nullptr);
    static void moe_accumulate(const GPUTensor &expert_out, float weight, const GPUTensor &out,
                               void *stream = nullptr);
};

#endif // LOCAL_LLM_TENSORTOOL_H
