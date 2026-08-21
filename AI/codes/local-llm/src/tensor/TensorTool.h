//
// Created by zhangyoulun on 21/8/2026.
//

#ifndef LOCAL_LLM_TENSORTOOL_H
#define LOCAL_LLM_TENSORTOOL_H

#include "tensor/Tensor.h"

class CudaScratch;

class TensorTool {
public:
    // weight: [out_dim, in_dim]，input/output 均为 GPU float 激活视图。
    static void gemm(const Tensor &weight, const Tensor &input, const Tensor &output,
                     CudaScratch &scratch, const std::string &lowp_key, const char *name = "");
    // table: embedding table [vocab, hidden]，input 为 GPU token id。
    static void embedding_lookup(const Tensor &table, Tensor input, const Tensor &hidden,
                                 CudaScratch &scratch);
    // weight: RMSNorm 权重，对 input 归一化后写入 output。
    static void rms_norm(const Tensor &weight, const Tensor &input, const Tensor &output,
                         float eps, bool one_plus);

    static void add(const Tensor &a, const Tensor &b, const Tensor &out, void *stream = nullptr);
    static void silu_mul(const Tensor &gate, const Tensor &up, const Tensor &out, void *stream = nullptr);

    static void full_attention_q(const Tensor &q_and_gate, const Tensor &q_norm_weight,
                                 const Tensor &q, const Tensor &gate, int n_heads, int head_dim, int pos,
                                 float rope_theta, float partial_rotary_factor, float eps,
                                 void *stream = nullptr);
    static void full_attention_q_batch(const Tensor &q_and_gate, const Tensor &q_norm_weight,
                                       const Tensor &q, const Tensor &gate, int n_heads, int head_dim,
                                       int start_pos, float rope_theta, float partial_rotary_factor,
                                       float eps, void *stream = nullptr);
    static void full_attention_kv(const Tensor &k_in, const Tensor &v_in, const Tensor &k_norm_weight,
                                  const Tensor &key_cache, const Tensor &value_cache, int kv_heads, int head_dim,
                                  int max_seq_len, int pos, float rope_theta,
                                  float partial_rotary_factor, float eps, void *stream = nullptr);
    static void full_attention_kv_batch(const Tensor &k_in, const Tensor &v_in,
                                        const Tensor &k_norm_weight, const Tensor &key_cache,
                                        const Tensor &value_cache, int kv_heads, int head_dim,
                                        int max_seq_len, int start_pos, float rope_theta,
                                        float partial_rotary_factor, float eps,
                                        void *stream = nullptr);
    static void full_attention_attend(const Tensor &q, const Tensor &gate, const Tensor &key_cache,
                                      const Tensor &value_cache, const Tensor &attn, int n_heads,
                                      int kv_heads, int head_dim, int max_seq_len, int pos,
                                      void *stream = nullptr);
    static void full_attention_attend_batch(const Tensor &q, const Tensor &gate,
                                            const Tensor &key_cache, const Tensor &value_cache,
                                            const Tensor &attn, int n_heads, int kv_heads,
                                            int head_dim, int max_seq_len, int start_pos,
                                            void *stream = nullptr);

    static void linear_attention_conv(const Tensor &mixed, const Tensor &conv_weight,
                                      const Tensor &conv_state, const Tensor &conv_out,
                                      int kernel, void *stream = nullptr);
    static void linear_attention_conv_batch(const Tensor &mixed, const Tensor &conv_weight,
                                            const Tensor &conv_state, const Tensor &conv_out,
                                            int kernel, void *stream = nullptr);
    static void linear_attention_recurrent(const Tensor &conv_out, const Tensor &z, const Tensor &b,
                                           const Tensor &a, const Tensor &a_log,
                                           const Tensor &dt_bias, const Tensor &norm_weight,
                                           const Tensor &recurrent_state, const Tensor &gated, int key_heads,
                                           int value_heads, int k_dim, int v_dim, float eps,
                                           void *stream = nullptr);
    static void linear_attention_recurrent_batch(const Tensor &conv_out, const Tensor &z,
                                                 const Tensor &b, const Tensor &a,
                                                 const Tensor &a_log, const Tensor &dt_bias,
                                                 const Tensor &norm_weight, const Tensor &recurrent_state,
                                                 const Tensor &gated, int key_heads,
                                                 int value_heads, int k_dim, int v_dim, float eps,
                                                 void *stream = nullptr);

    static void mla_kv_a(const Tensor &kv_a, const Tensor &kv_a_norm_weight, const Tensor &kv_cache,
                         int kv_lora, int qk_rope, int max_seq_len, int pos,
                         const Tensor &inv_freq, float eps, void *stream = nullptr);
    static void mla_kv_a_batch(const Tensor &kv_a, const Tensor &kv_a_norm_weight, const Tensor &kv_cache,
                               int kv_lora, int qk_rope, int max_seq_len,
                               int start_pos, const Tensor &inv_freq, float eps,
                               void *stream = nullptr);
    static void mla_rope_q(const Tensor &q, int n_heads, int qk_nope, int qk_rope, int pos,
                           const Tensor &inv_freq, void *stream = nullptr);
    static void mla_rope_q_batch(const Tensor &q, int n_heads, int qk_nope, int qk_rope,
                                 int start_pos, const Tensor &inv_freq, void *stream = nullptr);
    static void mla_attend(const Tensor &q, const Tensor &kv_b_out, const Tensor &kv_cache,
                           const Tensor &attn, int n_heads, int qk_nope, int qk_rope, int v_head,
                           int kv_lora, int max_seq_len, int pos, float softmax_scale,
                           void *stream = nullptr);
    static void mla_attend_batch(const Tensor &q, const Tensor &kv_b_out, const Tensor &kv_cache,
                                 const Tensor &attn, int n_heads, int qk_nope, int qk_rope,
                                 int v_head, int kv_lora, int max_seq_len, int start_pos,
                                 float softmax_scale, void *stream = nullptr);

    static void moe_router_topk(const Tensor &router_logits, const Tensor &top_idx, const Tensor &top_w,
                                int n_experts, int k, float routed_scaling,
                                void *stream = nullptr);
    static void moe_accumulate(const Tensor &expert_out, float weight, const Tensor &out,
                               void *stream = nullptr);
};

#endif // LOCAL_LLM_TENSORTOOL_H
