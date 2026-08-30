//
// Created by zhangyoulun on 15/8/2026.
//

#include "MLA.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaScratch.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekTrace.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "llm/module/common/RMSNorm.h"
#include "tensor/GPUTensor.h"
#include "tensor/TensorTool.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {
    bool use_mla_absorption() {
        const char *env = std::getenv("LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_MLA_ABSORB");
        return env != nullptr && std::atoi(env) > 0;
    }

    bool use_mla_absorption_hybrid() {
        const char *env = std::getenv("LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_MLA_ABSORB_HYBRID");
        return env != nullptr && std::atoi(env) > 0;
    }

    bool debug_mla_absorption_compare() {
        const char *env = std::getenv("LOCAL_LLM_DEBUG_DEEPSEEK_MLA_ABSORB_COMPARE");
        return env != nullptr && std::atoi(env) > 0;
    }

    bool use_deepseek_quant_direct() {
        const char *direct = std::getenv("LOCAL_LLM_DEEPSEEK_QUANT_DIRECT");
        return direct == nullptr || std::atoi(direct) > 0;
    }

    bool kv_b_prefill_uses_q8_raw_sum(DType dtype, int64_t input_size) {
        if (input_size <= 1) return false;
        if (dtype != DType::Q4_K && dtype != DType::Q6_K) return false;
        const char *tiled = std::getenv("LOCAL_LLM_EXPERIMENTAL_DEEPSEEK_PREFILL_TILED_MMQ");
        return use_deepseek_quant_direct() || (tiled != nullptr && std::atoi(tiled) > 0);
    }

    int env_int(const char *name, int fallback) {
        const char *env = std::getenv(name);
        return env ? std::atoi(env) : fallback;
    }

    struct DiffStats {
        double max_abs = 0.0;
        double mean_abs = 0.0;
        double rmse = 0.0;
        int index = -1;
    };

    DiffStats diff_stats(const std::vector<float> &a, const std::vector<float> &b) {
        DiffStats s;
        const size_t n = std::min(a.size(), b.size());
        if (n == 0) return s;
        double sum = 0.0;
        double sum2 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            const double ad = std::abs(d);
            sum += ad;
            sum2 += d * d;
            if (ad > s.max_abs) {
                s.max_abs = ad;
                s.index = static_cast<int>(i);
            }
        }
        s.mean_abs = sum / static_cast<double>(n);
        s.rmse = std::sqrt(sum2 / static_cast<double>(n));
        return s;
    }

    void print_diff(const char *name, const DiffStats &s) {
        std::cerr << "[mla_absorb_compare] " << name
                  << " max_abs=" << s.max_abs
                  << " mean_abs=" << s.mean_abs
                  << " rmse=" << s.rmse
                  << " max_idx=" << s.index << "\n";
    }

    void copy_to_host(const GPUTensor &g, std::vector<float> &h, const std::string &name) {
        h.resize(static_cast<size_t>(g.numel()));
        cuda_memcpy_d2h(h.data(), g.data<float>(), h.size() * sizeof(float), name);
    }

    float f16_bits_to_float(uint16_t h) {
        const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
        uint32_t exp = (h >> 10) & 0x1fu;
        uint32_t mant = h & 0x03ffu;
        uint32_t f;
        if (exp == 0) {
            if (mant == 0) {
                f = sign;
            } else {
                exp = 1;
                while ((mant & 0x0400u) == 0) {
                    mant <<= 1;
                    --exp;
                }
                mant &= 0x03ffu;
                f = sign | ((exp + 112u) << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            f = sign | 0x7f800000u | (mant << 13);
        } else {
            f = sign | ((exp + 112u) << 23) | (mant << 13);
        }
        float out;
        std::memcpy(&out, &f, sizeof(out));
        return out;
    }

    float q8_1_at_host(const uint8_t *row, int idx, int in_dim) {
        if (idx >= in_dim) return 0.0f;
        const int qblk = idx >> 5;
        const int lane = idx & 31;
        const uint8_t *base = row + static_cast<size_t>(qblk) * 36;
        uint16_t h = static_cast<uint16_t>(base[0]) | (static_cast<uint16_t>(base[1]) << 8);
        const float d = f16_bits_to_float(h);
        const int8_t q = reinterpret_cast<const int8_t *>(base + 4)[lane];
        return d * static_cast<float>(q);
    }

    float q8_1_stored_sum_host(const uint8_t *row, int qblk) {
        const uint8_t *base = row + static_cast<size_t>(qblk) * 36;
        const uint16_t h = static_cast<uint16_t>(base[2]) | (static_cast<uint16_t>(base[3]) << 8);
        return f16_bits_to_float(h);
    }

    float q8_1_dequant_sum_host(const uint8_t *row, int qblk, int in_dim) {
        const int start = qblk * 32;
        float sum = 0.0f;
        for (int lane = 0; lane < 32 && start + lane < in_dim; ++lane) {
            sum += q8_1_at_host(row, start + lane, in_dim);
        }
        return sum;
    }

    void debug_compare_mla_absorption(DeepseekSession &session, const DeepseekLayerWeights &lw,
                                      const GPUTensor &g_q_f32, const GPUTensor &g_kv_cache_f32,
                                      const uint8_t *latent_q8_1_cache, size_t latent_q8_1_row_bytes,
                                      const GPUTensor &g_kv_b_cache_f32, const GPUTensor &g_attn_f32,
                                      const GPUTensor &g_attn_out_f32, int layer, int start_pos,
                                      int n_heads, int qk_nope, int qk_rope, int v_head, int kv_lora,
                                      int hidden_size, float softmax_scale) {
        static int emitted = 0;
        const int target_layer = env_int("LOCAL_LLM_DEBUG_DEEPSEEK_MLA_ABSORB_LAYER", 0);
        const int max_reports = env_int("LOCAL_LLM_DEBUG_DEEPSEEK_MLA_ABSORB_REPORTS", 1);
        if (!debug_mla_absorption_compare() || layer != target_layer || emitted >= max_reports) return;
        ++emitted;

        auto &scratch = session.cuda_scratch;
        auto g_q_abs_f32 = GPUTensor(scratch, "mla_cmp_q_abs",
                                     {static_cast<int64_t>(n_heads), static_cast<int64_t>(kv_lora)}, DType::F32);
        auto g_attn_latent_f32 = GPUTensor(scratch, "mla_cmp_attn_latent",
                                           {static_cast<int64_t>(n_heads), static_cast<int64_t>(kv_lora)}, DType::F32);
        const int blocks_per_row = static_cast<int>((latent_q8_1_row_bytes + 35) / 36);
        auto g_q_abs_xsum_delta_f32 = GPUTensor(scratch, "mla_cmp_q_xsum_delta",
                                                {static_cast<int64_t>(n_heads), static_cast<int64_t>(blocks_per_row)},
                                                DType::F32);
        auto g_attn_xsum_delta_f32 = GPUTensor(scratch, "mla_cmp_attn_xsum_delta",
                                               {static_cast<int64_t>(n_heads), static_cast<int64_t>(blocks_per_row)},
                                               DType::F32);
        auto g_attn_scores_f32 = GPUTensor(scratch, "mla_cmp_scores",
                                           {static_cast<int64_t>(n_heads),
                                            static_cast<int64_t>(session.max_seq_len_)}, DType::F32);
        auto g_attn_abs_f32 = GPUTensor(scratch, "mla_cmp_attn",
                                        {1, static_cast<int64_t>(n_heads * v_head)}, DType::F32);
        auto g_attn_out_abs_f32 = GPUTensor(scratch, "mla_cmp_attn_out",
                                            {1, static_cast<int64_t>(hidden_size)}, DType::F32);
        auto g_attn_hybrid_f32 = GPUTensor(scratch, "mla_cmp_attn_hybrid",
                                           {1, static_cast<int64_t>(n_heads * v_head)}, DType::F32);
        auto g_attn_out_hybrid_f32 = GPUTensor(scratch, "mla_cmp_attn_out_hybrid",
                                               {1, static_cast<int64_t>(hidden_size)}, DType::F32);

        if (!TensorTool::mla_absorb_components(*lw.s_attn_kv_b, g_q_f32,
                                               latent_q8_1_cache, latent_q8_1_row_bytes,
                                               g_kv_cache_f32, g_q_abs_f32, g_q_abs_xsum_delta_f32,
                                               g_attn_latent_f32, g_attn_scores_f32, g_attn_abs_f32,
                                               g_attn_xsum_delta_f32,
                                               n_heads, qk_nope, qk_rope, v_head, kv_lora,
                                               session.d_pos().data<int>(),
                                               static_cast<int>(session.max_seq_len_), softmax_scale)) {
            std::cerr << "[mla_absorb_compare] skipped: absorption components unavailable\n";
            return;
        }
        TensorTool::gemm(*lw.s_attn_output, g_attn_abs_f32, g_attn_out_abs_f32, scratch,
                         "mla_cmp_attn_out_lowp", "ds.debug.mla_absorb.attn_output");
        if (TensorTool::mla_absorb_decode_v_cache(*lw.s_attn_kv_b, g_q_f32,
                                                  latent_q8_1_cache, latent_q8_1_row_bytes,
                                                  g_kv_cache_f32, g_kv_b_cache_f32, g_attn_hybrid_f32,
                                                  n_heads, qk_nope, qk_rope, v_head, kv_lora,
                                                  session.d_pos().data<int>(),
                                                  static_cast<int>(session.max_seq_len_), softmax_scale,
                                                  scratch)) {
            TensorTool::gemm(*lw.s_attn_output, g_attn_hybrid_f32, g_attn_out_hybrid_f32, scratch,
                             "mla_cmp_attn_out_hybrid_lowp", "ds.debug.mla_hybrid.attn_output");
        }
        check_cuda(cudaStreamSynchronize(get_current_cuda_stream()), "mla_absorb_compare sync");

        const int seq = start_pos + 1;
        const int qk_head = qk_nope + qk_rope;
        const int kv_total = kv_lora + qk_rope;
        const int kvb_stride = qk_nope + v_head;
        std::vector<float> q, kv_cache, kv_b_cache, q_abs, q_abs_xsum_delta;
        std::vector<float> attn_latent, attn_xsum_delta, attn_exp, attn_abs, attn_hybrid;
        std::vector<float> out_exp, out_abs, out_hybrid;
        std::vector<uint8_t> latent_q8_cache(static_cast<size_t>(seq) * latent_q8_1_row_bytes);
        copy_to_host(g_q_f32, q, "mla cmp q");
        kv_cache.resize(static_cast<size_t>(seq) * kv_total);
        cuda_memcpy_d2h(kv_cache.data(), g_kv_cache_f32.data<float>(), kv_cache.size() * sizeof(float),
                        "mla cmp kv_cache");
        cuda_memcpy_d2h(latent_q8_cache.data(), latent_q8_1_cache, latent_q8_cache.size(),
                        "mla cmp latent_q8_1_cache");
        kv_b_cache.resize(static_cast<size_t>(seq) * n_heads * kvb_stride);
        cuda_memcpy_d2h(kv_b_cache.data(), g_kv_b_cache_f32.data<float>(), kv_b_cache.size() * sizeof(float),
                        "mla cmp kv_b_cache");
        copy_to_host(g_q_abs_f32, q_abs, "mla cmp q_abs");
        copy_to_host(g_q_abs_xsum_delta_f32, q_abs_xsum_delta, "mla cmp q_abs_xsum_delta");
        copy_to_host(g_attn_latent_f32, attn_latent, "mla cmp attn_latent");
        copy_to_host(g_attn_xsum_delta_f32, attn_xsum_delta, "mla cmp attn_xsum_delta");
        copy_to_host(g_attn_f32, attn_exp, "mla cmp attn_exp");
        copy_to_host(g_attn_abs_f32, attn_abs, "mla cmp attn_abs");
        copy_to_host(g_attn_hybrid_f32, attn_hybrid, "mla cmp attn_hybrid");
        copy_to_host(g_attn_out_f32, out_exp, "mla cmp out_exp");
        copy_to_host(g_attn_out_abs_f32, out_abs, "mla cmp out_abs");
        copy_to_host(g_attn_out_hybrid_f32, out_hybrid, "mla cmp out_hybrid");

        std::vector<float> score_exp(static_cast<size_t>(n_heads) * seq);
        std::vector<float> score_abs(static_cast<size_t>(n_heads) * seq);
        std::vector<float> prob_exp(static_cast<size_t>(n_heads) * seq);
        std::vector<float> prob_abs(static_cast<size_t>(n_heads) * seq);
        std::vector<float> attn_latent_cpu(static_cast<size_t>(n_heads) * kv_lora);

        for (int h = 0; h < n_heads; ++h) {
            const float *qh = q.data() + static_cast<size_t>(h) * qk_head;
            const float *q_nope = qh;
            const float *q_rope = qh + qk_nope;
            float max_exp = -std::numeric_limits<float>::infinity();
            float max_abs = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < seq; ++t) {
                const float *k_nope = kv_b_cache.data() + (static_cast<size_t>(t) * n_heads + h) * kvb_stride;
                const float *kv = kv_cache.data() + static_cast<size_t>(t) * kv_total;
                const float *k_rope = kv + kv_lora;
                float se = 0.0f;
                for (int d = 0; d < qk_nope; ++d) se += q_nope[d] * k_nope[d];
                for (int d = 0; d < qk_rope; ++d) se += q_rope[d] * k_rope[d];
                float sa = 0.0f;
                const float *qa = q_abs.data() + static_cast<size_t>(h) * kv_lora;
                const uint8_t *latent_q8 = latent_q8_cache.data() + static_cast<size_t>(t) * latent_q8_1_row_bytes;
                for (int d = 0; d < kv_lora; ++d) sa += qa[d] * q8_1_at_host(latent_q8, d, kv_lora);
                const float *q_delta = q_abs_xsum_delta.data() + static_cast<size_t>(h) * blocks_per_row;
                for (int qblk = 0; qblk < blocks_per_row; ++qblk) {
                    sa += q_delta[qblk] *
                          (q8_1_stored_sum_host(latent_q8, qblk) -
                           q8_1_dequant_sum_host(latent_q8, qblk, kv_lora));
                }
                for (int d = 0; d < qk_rope; ++d) sa += q_rope[d] * k_rope[d];
                se *= softmax_scale;
                sa *= softmax_scale;
                score_exp[static_cast<size_t>(h) * seq + t] = se;
                score_abs[static_cast<size_t>(h) * seq + t] = sa;
                max_exp = std::max(max_exp, se);
                max_abs = std::max(max_abs, sa);
            }
            float denom_exp = 0.0f;
            float denom_abs = 0.0f;
            for (int t = 0; t < seq; ++t) {
                const size_t idx = static_cast<size_t>(h) * seq + t;
                prob_exp[idx] = std::exp(score_exp[idx] - max_exp);
                prob_abs[idx] = std::exp(score_abs[idx] - max_abs);
                denom_exp += prob_exp[idx];
                denom_abs += prob_abs[idx];
            }
            for (int t = 0; t < seq; ++t) {
                prob_exp[static_cast<size_t>(h) * seq + t] /= denom_exp;
                prob_abs[static_cast<size_t>(h) * seq + t] /= denom_abs;
            }
            for (int d = 0; d < kv_lora; ++d) {
                float sum = 0.0f;
                for (int t = 0; t < seq; ++t) {
                    const uint8_t *latent_q8 = latent_q8_cache.data() + static_cast<size_t>(t) * latent_q8_1_row_bytes;
                    sum += prob_abs[static_cast<size_t>(h) * seq + t] *
                           q8_1_at_host(latent_q8, d, kv_lora);
                }
                attn_latent_cpu[static_cast<size_t>(h) * kv_lora + d] = sum;
            }
        }

        std::cerr << "[mla_absorb_compare] layer=" << layer << " pos=" << start_pos
                  << " seq=" << seq << " q_abs_max="
                  << *std::max_element(q_abs.begin(), q_abs.end(), [](float a, float b) {
                         return std::abs(a) < std::abs(b);
                     }) << "\n";
        print_diff("kq_score expanded_vs_absorb", diff_stats(score_exp, score_abs));
        print_diff("softmax expanded_vs_absorb", diff_stats(prob_exp, prob_abs));
        print_diff("attn_latent kernel_vs_cpu", diff_stats(attn_latent, attn_latent_cpu));
        std::cerr << "[mla_absorb_compare] attn_xsum_delta_max="
                  << *std::max_element(attn_xsum_delta.begin(), attn_xsum_delta.end(), [](float a, float b) {
                         return std::abs(a) < std::abs(b);
                     }) << "\n";
        print_diff("attn_ctx expanded_vs_absorb", diff_stats(attn_exp, attn_abs));
        print_diff("attn_out expanded_vs_absorb", diff_stats(out_exp, out_abs));
        print_diff("attn_ctx expanded_vs_hybrid", diff_stats(attn_exp, attn_hybrid));
        print_diff("attn_out expanded_vs_hybrid", diff_stats(out_exp, out_hybrid));
    }
}

MLA::MLA(const DeepseekLayerWeights &weights, const DeepseekConfig &config)
    : config_(config), lw_(weights) {
}

void MLA::forward(DeepseekSession &session, const GPUTensor &g_hidden_f32, int start_pos, bool use_device_pos) {
    const int64_t input_size = g_hidden_f32.rows();
    auto &scratch = session.cuda_scratch;
    const int layer = lw_.layer_index;
    const int64_t hidden_size = config_.hidden_size;
    const int n_heads = config_.num_heads;
    const int qk_nope = config_.qk_nope_head_dim;
    const int qk_rope = config_.qk_rope_head_dim;
    const int qk_head = qk_nope + qk_rope; //config_.qk_head_dim(); // 192
    const int v_head = config_.v_head_dim; // 128
    const int kv_lora = config_.kv_lora_rank; // 512
    const int kv_total = kv_lora + qk_rope; // 576
    const int kvb_out = n_heads * (qk_nope + v_head); // 4096
    const GPUTensor &g_inv_freq_f32 = session.g_inv_freq_f32;

    auto g_normed_f32 = GPUTensor(scratch, scratch_key::kNormed, {input_size, hidden_size}, DType::F32);
    RMSNorm::forward(*lw_.s_attn_norm, g_hidden_f32, g_normed_f32,
                     config_.rms_norm_eps, /*one_plus=*/false);
    deepseek_trace::tensor(session, g_normed_f32, "attn_normed", session.trace_pos, session.trace_layer);

    auto g_q_f32 = GPUTensor(scratch, scratch_key::kQ,
                                  {
                                      input_size,
                                      static_cast<int64_t>(n_heads),
                                      static_cast<int64_t>(qk_nope + qk_rope) //(qk_head)
                                  }, DType::F32);
    TensorTool::gemm(*lw_.s_attn_q, g_normed_f32, g_q_f32, scratch, scratch_key::kNormedLowp, "ds.gemm.d_attn_q");
    if (use_device_pos) {
        TensorTool::mla_rope_q_device_pos(g_q_f32, input_size, n_heads, qk_nope, qk_rope,
                                          session.d_pos().data<int>(), g_inv_freq_f32);
    } else {
        TensorTool::mla_rope_q(g_q_f32, input_size, n_heads, qk_nope, qk_rope, start_pos, g_inv_freq_f32);
    }
    deepseek_trace::tensor(session, g_q_f32, "attn_q_rope", session.trace_pos, session.trace_layer);

    GPUTensor g_kv_a_f32 = GPUTensor(
        scratch, scratch_key::kKvA,
        {input_size, static_cast<int64_t>(kv_total)}, DType::F32);
    TensorTool::gemm(*lw_.s_attn_kv_a_mqa, g_normed_f32, g_kv_a_f32, scratch, scratch_key::kNormedLowp, "ds.gemm.kv_a");
    deepseek_trace::tensor(session, g_kv_a_f32, "kv_a", session.trace_pos, session.trace_layer);

    GPUTensor &g_kv_cache_f32 = session.kv_caches[layer].g_cache_f32;
    if (use_device_pos) {
        TensorTool::mla_kv_a_device_pos(g_kv_a_f32, *lw_.s_attn_kv_a_norm, g_kv_cache_f32, input_size,
                                        kv_lora, qk_rope, session.d_pos().data<int>(), g_inv_freq_f32,
                                        config_.rms_norm_eps);
    } else {
        TensorTool::mla_kv_a(g_kv_a_f32, *lw_.s_attn_kv_a_norm, g_kv_cache_f32, input_size, kv_lora, qk_rope,
                             start_pos, g_inv_freq_f32, config_.rms_norm_eps);
    }
    session.kv_caches[layer].seq_len = start_pos + static_cast<int>(input_size);

    const bool absorb_decode = use_device_pos && input_size == 1 && use_mla_absorption();
    auto &kv_cache = session.kv_caches[layer];
    uint8_t *latent_q8_1_cache = static_cast<uint8_t *>(kv_cache.latent_q8_1_cache.ptr);
    const size_t latent_q8_1_row_bytes = kv_cache.latent_q8_1_row_bytes;
    GPUTensor &g_kv_b_cache_f32 = kv_cache.g_kv_b_cache_f32;
    if (!absorb_decode) {
        GPUTensor g_latent_f32;
        GPUTensor g_kv_b_new_f32;
        if (use_device_pos) {
            g_latent_f32 = GPUTensor(
                scratch, scratch_key::kAttn,
                {1, static_cast<int64_t>(kv_lora)}, DType::F32);
            TensorTool::mla_gather_latent_device_pos(g_kv_cache_f32, g_latent_f32, kv_lora, qk_rope,
                                                     session.d_pos().data<int>());
            g_kv_b_new_f32 = GPUTensor(
                scratch, scratch_key::kKvBOut,
                {1, static_cast<int64_t>(kvb_out)}, DType::F32);
        } else if (input_size == 1) {
            const size_t kvb_row_bytes = static_cast<size_t>(kvb_out) * sizeof(float);
            g_kv_b_new_f32 = GPUTensor(
                g_kv_b_cache_f32, static_cast<size_t>(start_pos) * kvb_row_bytes,
                {input_size, static_cast<int64_t>(kvb_out)});
            // decode: kv_b 必须使用 mla_kv_a 写入 cache 后的 normalized latent。
            g_latent_f32 = GPUTensor(
                g_kv_cache_f32, static_cast<size_t>(start_pos) * kv_total * sizeof(float),
                {1, static_cast<int64_t>(kv_lora)});
        } else {
            const size_t kvb_row_bytes = static_cast<size_t>(kvb_out) * sizeof(float);
            g_kv_b_new_f32 = GPUTensor(
                g_kv_b_cache_f32, static_cast<size_t>(start_pos) * kvb_row_bytes,
                {input_size, static_cast<int64_t>(kvb_out)});
            // prefill/batch: kv_lora 与 qk_rope 交错存入 kv_a，先压紧本批 token 的 latent 段。
            g_latent_f32 = GPUTensor(
                scratch, scratch_key::kAttn,
                {input_size, static_cast<int64_t>(kv_lora)}, DType::F32);
            const float *src = g_kv_cache_f32.data<float>() + static_cast<size_t>(start_pos) * kv_total;
            cuda_memcpy2d_d2d(g_latent_f32.data(), kv_lora * sizeof(float), src,
                              kv_total * sizeof(float), kv_lora * sizeof(float), static_cast<size_t>(input_size),
                              "ds.gather.latent.new");
        }
        deepseek_trace::tensor(session, g_latent_f32, "kv_latent", session.trace_pos, session.trace_layer);
        if (use_device_pos) {
            TensorTool::mla_store_latent_q8_1_device_pos(g_kv_cache_f32, latent_q8_1_cache, kv_lora, qk_rope,
                                                         latent_q8_1_row_bytes, session.d_pos().data<int>());
        } else {
            const bool raw_sum = kv_b_prefill_uses_q8_raw_sum(lw_.s_attn_kv_b->dtype, input_size);
            TensorTool::mla_store_latent_q8_1(g_latent_f32, latent_q8_1_cache, kv_lora,
                                              latent_q8_1_row_bytes, start_pos, raw_sum);
        }
        TensorTool::gemm(*lw_.s_attn_kv_b, g_latent_f32, g_kv_b_new_f32, scratch, scratch_key::kLatentLowp,
                         "ds.gemm.kv_b");
        if (use_device_pos) {
            TensorTool::mla_store_kv_b_device_pos(g_kv_b_new_f32, g_kv_b_cache_f32, kvb_out,
                                                  session.d_pos().data<int>());
        }
        deepseek_trace::tensor(session, g_kv_b_new_f32, "kv_b_out", session.trace_pos, session.trace_layer);
    } else {
        TensorTool::mla_store_latent_q8_1_device_pos(g_kv_cache_f32, latent_q8_1_cache, kv_lora, qk_rope,
                                                     latent_q8_1_row_bytes, session.d_pos().data<int>());
    }

    GPUTensor g_attn_f32 = GPUTensor(
        scratch, scratch_key::kAttn,
        {input_size, static_cast<int64_t>(n_heads * v_head)}, DType::F32);
    if (absorb_decode && use_mla_absorption_hybrid() &&
        TensorTool::mla_absorb_decode_v_cache(*lw_.s_attn_kv_b, g_q_f32, latent_q8_1_cache, latent_q8_1_row_bytes,
                                              g_kv_cache_f32, g_kv_b_cache_f32, g_attn_f32,
                                              n_heads, qk_nope, qk_rope, v_head, kv_lora,
                                              session.d_pos().data<int>(), static_cast<int>(session.max_seq_len_),
                                              session.attn_softmax_scale, scratch)) {
        // Experimental hybrid absorption path: K uses latent absorption, V reuses expanded cache.
    } else if (absorb_decode &&
        TensorTool::mla_absorb_decode(*lw_.s_attn_kv_b, g_q_f32, latent_q8_1_cache, latent_q8_1_row_bytes,
                                      g_kv_cache_f32, g_attn_f32,
                                      n_heads, qk_nope, qk_rope, v_head, kv_lora, session.d_pos().data<int>(),
                                      static_cast<int>(session.max_seq_len_), session.attn_softmax_scale,
                                      scratch)) {
        // Experimental full absorption path: K/V both use latent absorption.
    } else if (use_device_pos) {
        TensorTool::mla_attend_device_pos(g_q_f32, g_kv_b_cache_f32, g_kv_cache_f32, g_attn_f32, input_size,
                                          n_heads, qk_nope, qk_rope, v_head, kv_lora, session.d_pos().data<int>(),
                                          static_cast<int>(session.max_seq_len_), session.attn_softmax_scale);
    } else {
        TensorTool::mla_attend(g_q_f32, g_kv_b_cache_f32, g_kv_cache_f32, g_attn_f32, input_size, n_heads,
                               qk_nope, qk_rope, v_head, kv_lora, start_pos, session.attn_softmax_scale);
    }
    deepseek_trace::tensor(session, g_attn_f32, "attn_ctx", session.trace_pos, session.trace_layer);

    GPUTensor g_attn_out_f32 = GPUTensor(scratch, scratch_key::kAttnOut, {input_size, hidden_size}, DType::F32);
    TensorTool::gemm(*lw_.s_attn_output, g_attn_f32, g_attn_out_f32, scratch, scratch_key::kAttnLowp,
                     "ds.gemm.d_attn_output");
    deepseek_trace::tensor(session, g_attn_out_f32, "attn_out", session.trace_pos, session.trace_layer);
    if (!absorb_decode && use_device_pos && input_size == 1) {
        debug_compare_mla_absorption(session, lw_, g_q_f32, g_kv_cache_f32,
                                     latent_q8_1_cache, latent_q8_1_row_bytes, g_kv_b_cache_f32, g_attn_f32,
                                     g_attn_out_f32, layer, start_pos, n_heads, qk_nope, qk_rope, v_head,
                                     kv_lora, static_cast<int>(hidden_size), session.attn_softmax_scale);
    }
    TensorTool::add(g_hidden_f32, g_attn_out_f32, g_hidden_f32);
}
