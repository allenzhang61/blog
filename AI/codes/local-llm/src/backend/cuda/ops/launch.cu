//
// Created by zhangyoulun on 9/8/2026.
//

#include "kernel.cuh"
#include "kernel_internal.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "utils/stats/ScopedTimer.h"

// ================= launch 封装 =================

void launch_add(const float *a, const float *b, float *out, int n, void *stream) {
    ScopedGpuTimer timer("add", as_stream(stream));
    add_kernel<<<grid_for(n), kBlock, 0, as_stream(stream)>>>(a, b, out, n);
}

void launch_bf16_gemv(const uint16_t *weight, const uint16_t *x, float *y,
                      int out_dim, int in_dim, void *stream) {
    ScopedGpuTimer timer("bf16_gemv", as_stream(stream));
    // 每个 warp 算一个输出行 => 需要 out_dim 个 warp。
    constexpr int warps_per_block = kBlock / 32;
    const int blocks = (out_dim + warps_per_block - 1) / warps_per_block;
    bf16_gemv_kernel<<<blocks, kBlock, 0, as_stream(stream)>>>(weight, x, y, out_dim, in_dim);
}

template <int QUANT_TYPE>
void launch_quant_gemv_typed(const uint8_t *weight, size_t row_bytes, const float *x,
                             float *y, int out_dim, int in_dim, int m,
                             bool f16_operands, cudaStream_t stream) {
    constexpr int warps_per_block = kBlock / 32;
    const int blocks = (out_dim + warps_per_block - 1) / warps_per_block;
    if (f16_operands) {
        quant_gemv_kernel<QUANT_TYPE, true><<<blocks, kBlock, 0, stream>>>(
            weight, row_bytes, x, y, out_dim, in_dim, m);
    } else {
        quant_gemv_kernel<QUANT_TYPE, false><<<blocks, kBlock, 0, stream>>>(
            weight, row_bytes, x, y, out_dim, in_dim, m);
    }
}

void launch_quant_gemv(DType quant_type, const uint8_t *weight, size_t row_bytes, const float *x,
                       float *y, int out_dim, int in_dim, int m, bool f16_operands, void *stream) {
    const cudaStream_t s = as_stream(stream);
    switch (quant_type) {
        case DType::Q4_K:
            launch_quant_gemv_typed<12>(weight, row_bytes, x, y, out_dim, in_dim, m, f16_operands, s);
            break;
        case DType::Q6_K:
            launch_quant_gemv_typed<14>(weight, row_bytes, x, y, out_dim, in_dim, m, f16_operands, s);
            break;
        case DType::Q5_0:
            launch_quant_gemv_typed<6>(weight, row_bytes, x, y, out_dim, in_dim, m, f16_operands, s);
            break;
        case DType::Q8_0:
            launch_quant_gemv_typed<8>(weight, row_bytes, x, y, out_dim, in_dim, m, f16_operands, s);
            break;
        default:
            break;
    }
}

template <int QUANT_TYPE>
void launch_quant_matmul_typed(const uint8_t *weight, size_t row_bytes, const float *x,
                               float *y, int out_dim, int in_dim, int m,
                               bool f16_operands, cudaStream_t stream) {
    constexpr int warps_per_block = kBlock / 32;
    const int total = m * out_dim;
    const int blocks = (total + warps_per_block - 1) / warps_per_block;
    if (f16_operands) {
        quant_matmul_kernel<QUANT_TYPE, true><<<blocks, kBlock, 0, stream>>>(
            weight, row_bytes, x, y, out_dim, in_dim, m);
    } else {
        quant_matmul_kernel<QUANT_TYPE, false><<<blocks, kBlock, 0, stream>>>(
            weight, row_bytes, x, y, out_dim, in_dim, m);
    }
}

void launch_quant_matmul(DType quant_type, const uint8_t *weight, size_t row_bytes, const float *x,
                         float *y, int out_dim, int in_dim, int m, bool f16_operands, void *stream) {
    const cudaStream_t s = as_stream(stream);
    switch (quant_type) {
        case DType::Q4_K:
            launch_quant_matmul_typed<12>(weight, row_bytes, x, y, out_dim, in_dim, m, f16_operands, s);
            break;
        case DType::Q6_K:
            launch_quant_matmul_typed<14>(weight, row_bytes, x, y, out_dim, in_dim, m, f16_operands, s);
            break;
        case DType::Q5_0:
            launch_quant_matmul_typed<6>(weight, row_bytes, x, y, out_dim, in_dim, m, f16_operands, s);
            break;
        case DType::Q8_0:
            launch_quant_matmul_typed<8>(weight, row_bytes, x, y, out_dim, in_dim, m, f16_operands, s);
            break;
        default:
            break;
    }
}

template <int QUANT_TYPE>
void launch_quant_swiglu_typed(const uint8_t *gate_weight, const uint8_t *up_weight,
                               size_t gate_row_bytes, size_t up_row_bytes,
                               const float *x, float *act, int ffn_dim, int in_dim,
                               bool f16_operands, cudaStream_t stream) {
    constexpr int warps_per_block = kBlock / 32;
    const int blocks = (ffn_dim + warps_per_block - 1) / warps_per_block;
    if (f16_operands) {
        quant_swiglu_kernel<QUANT_TYPE, true><<<blocks, kBlock, 0, stream>>>(
            gate_weight, up_weight, gate_row_bytes, up_row_bytes, x, act, ffn_dim, in_dim);
    } else {
        quant_swiglu_kernel<QUANT_TYPE, false><<<blocks, kBlock, 0, stream>>>(
            gate_weight, up_weight, gate_row_bytes, up_row_bytes, x, act, ffn_dim, in_dim);
    }
}

void launch_quant_swiglu(DType quant_type, const uint8_t *gate_weight, const uint8_t *up_weight,
                         size_t gate_row_bytes, size_t up_row_bytes, const float *x, float *act,
                         int ffn_dim, int in_dim, bool f16_operands, void *stream) {
    const cudaStream_t s = as_stream(stream);
    switch (quant_type) {
        case DType::Q4_K:
            launch_quant_swiglu_typed<12>(gate_weight, up_weight, gate_row_bytes, up_row_bytes,
                                          x, act, ffn_dim, in_dim, f16_operands, s);
            break;
        case DType::Q6_K:
            launch_quant_swiglu_typed<14>(gate_weight, up_weight, gate_row_bytes, up_row_bytes,
                                          x, act, ffn_dim, in_dim, f16_operands, s);
            break;
        case DType::Q5_0:
            launch_quant_swiglu_typed<6>(gate_weight, up_weight, gate_row_bytes, up_row_bytes,
                                         x, act, ffn_dim, in_dim, f16_operands, s);
            break;
        case DType::Q8_0:
            launch_quant_swiglu_typed<8>(gate_weight, up_weight, gate_row_bytes, up_row_bytes,
                                         x, act, ffn_dim, in_dim, f16_operands, s);
            break;
        default:
            break;
    }
}

template <int QUANT_TYPE>
void launch_quant_swiglu_indexed_typed(const uint8_t *gate_weight, const uint8_t *up_weight,
                                       size_t gate_expert_bytes, size_t up_expert_bytes,
                                       size_t gate_row_bytes, size_t up_row_bytes, const float *x,
                                       const int *expert_ids, float *act, int k, int ffn_dim,
                                       int in_dim, bool f16_operands, cudaStream_t stream) {
    constexpr int warps_per_block = kBlock / 32;
    const int total = k * ffn_dim;
    const int blocks = (total + warps_per_block - 1) / warps_per_block;
    if (f16_operands) {
        quant_swiglu_indexed_kernel<QUANT_TYPE, true><<<blocks, kBlock, 0, stream>>>(
            gate_weight, up_weight, gate_expert_bytes, up_expert_bytes, gate_row_bytes, up_row_bytes,
            x, expert_ids, act, k, ffn_dim, in_dim);
    } else {
        quant_swiglu_indexed_kernel<QUANT_TYPE, false><<<blocks, kBlock, 0, stream>>>(
            gate_weight, up_weight, gate_expert_bytes, up_expert_bytes, gate_row_bytes, up_row_bytes,
            x, expert_ids, act, k, ffn_dim, in_dim);
    }
}

void launch_quant_swiglu_indexed(DType quant_type, const uint8_t *gate_weight, const uint8_t *up_weight,
                                 size_t gate_expert_bytes, size_t up_expert_bytes,
                                 size_t gate_row_bytes, size_t up_row_bytes, const float *x,
                                 const int *expert_ids, float *act, int k, int ffn_dim,
                                 int in_dim, bool f16_operands, void *stream) {
    const cudaStream_t s = as_stream(stream);
    switch (quant_type) {
        case DType::Q4_K:
            launch_quant_swiglu_indexed_typed<12>(gate_weight, up_weight, gate_expert_bytes, up_expert_bytes,
                                                  gate_row_bytes, up_row_bytes, x, expert_ids, act,
                                                  k, ffn_dim, in_dim, f16_operands, s);
            break;
        case DType::Q6_K:
            launch_quant_swiglu_indexed_typed<14>(gate_weight, up_weight, gate_expert_bytes, up_expert_bytes,
                                                  gate_row_bytes, up_row_bytes, x, expert_ids, act,
                                                  k, ffn_dim, in_dim, f16_operands, s);
            break;
        default:
            break;
    }
}

size_t q8_1_row_bytes(int in_dim) {
    const int blocks_per_row = (in_dim + 31) / 32;
    return static_cast<size_t>(blocks_per_row) * 36;
}

size_t q8_1_mmq_row_bytes(int in_dim) {
    const int groups_per_row = (in_dim + 127) / 128;
    return static_cast<size_t>(groups_per_row) * 144;
}

void launch_quantize_q8_1(const float *x, uint8_t *x_q8_1, int in_dim, int m, void *stream,
                          bool store_raw_sum) {
    const int blocks_per_row = (in_dim + 31) / 32;
    const int blocks = m * blocks_per_row;
    quantize_q8_1_kernel<<<blocks, 32, 0, as_stream(stream)>>>(
        x, x_q8_1, in_dim, m, blocks_per_row, store_raw_sum);
}

void launch_quantize_q8_1_mmq(const float *x, uint8_t *x_q8_1, int in_dim, int m, void *stream,
                              bool store_raw_sum) {
    const int groups_per_row = (in_dim + 127) / 128;
    const int blocks = m * groups_per_row;
    quantize_q8_1_mmq_kernel<<<blocks, 128, 0, as_stream(stream)>>>(
        x, x_q8_1, in_dim, m, groups_per_row, store_raw_sum);
}

template <int QUANT_TYPE>
void launch_quant_gemv_q8_1_typed(const uint8_t *weight, size_t row_bytes, const uint8_t *x_q8_1,
                                  float *y, int out_dim, int in_dim, int blocks_per_row, cudaStream_t stream) {
    constexpr int warps_per_block = kBlock / 32;
    const int blocks = (out_dim + warps_per_block - 1) / warps_per_block;
    quant_gemv_q8_1_kernel<QUANT_TYPE><<<blocks, kBlock, 0, stream>>>(
        weight, row_bytes, x_q8_1, y, out_dim, in_dim, blocks_per_row);
}

void launch_quant_gemv_q8_1(DType quant_type, const uint8_t *weight, size_t row_bytes,
                            const uint8_t *x_q8_1, float *y,
                            int out_dim, int in_dim, void *stream) {
    const cudaStream_t s = as_stream(stream);
    const int blocks_per_row = (in_dim + 31) / 32;
    switch (quant_type) {
        case DType::Q4_K:
            launch_quant_gemv_q8_1_typed<12>(weight, row_bytes, x_q8_1, y, out_dim, in_dim, blocks_per_row, s);
            break;
        case DType::Q6_K:
            launch_quant_gemv_q8_1_typed<14>(weight, row_bytes, x_q8_1, y, out_dim, in_dim, blocks_per_row, s);
            break;
        case DType::Q5_0:
            launch_quant_gemv_q8_1_typed<6>(weight, row_bytes, x_q8_1, y, out_dim, in_dim, blocks_per_row, s);
            break;
        case DType::Q8_0:
            launch_quant_gemv_q8_1_typed<8>(weight, row_bytes, x_q8_1, y, out_dim, in_dim, blocks_per_row, s);
            break;
        default:
            break;
    }
}

template <int QUANT_TYPE>
void launch_quant_down_q8_1_indexed_accum_typed(const uint8_t *down_weight, size_t down_expert_bytes,
                                                size_t down_row_bytes, const uint8_t *act_q8_1,
                                                const int *expert_ids, const float *route_weights,
                                                float *out, int k, int hidden_size, int ffn_dim,
                                                int blocks_per_row, cudaStream_t stream) {
    constexpr int warps_per_block = kBlock / 32;
    const int total = k * hidden_size;
    const int blocks = (total + warps_per_block - 1) / warps_per_block;
    quant_down_q8_1_indexed_accum_kernel<QUANT_TYPE><<<blocks, kBlock, 0, stream>>>(
        down_weight, down_expert_bytes, down_row_bytes, act_q8_1, expert_ids, route_weights,
        out, k, hidden_size, ffn_dim, blocks_per_row);
}

void launch_quant_down_q8_1_indexed_accum(DType quant_type, const uint8_t *down_weight,
                                          size_t down_expert_bytes, size_t down_row_bytes,
                                          const uint8_t *act_q8_1, const int *expert_ids,
                                          const float *route_weights, float *out,
                                          int k, int hidden_size, int ffn_dim, void *stream) {
    const cudaStream_t s = as_stream(stream);
    const int blocks_per_row = static_cast<int>(q8_1_row_bytes(ffn_dim) / 36);
    switch (quant_type) {
        case DType::Q4_K:
            launch_quant_down_q8_1_indexed_accum_typed<12>(down_weight, down_expert_bytes, down_row_bytes,
                                                           act_q8_1, expert_ids, route_weights, out,
                                                           k, hidden_size, ffn_dim, blocks_per_row, s);
            break;
        case DType::Q6_K:
            launch_quant_down_q8_1_indexed_accum_typed<14>(down_weight, down_expert_bytes, down_row_bytes,
                                                           act_q8_1, expert_ids, route_weights, out,
                                                           k, hidden_size, ffn_dim, blocks_per_row, s);
            break;
        case DType::Q5_0:
            launch_quant_down_q8_1_indexed_accum_typed<6>(down_weight, down_expert_bytes, down_row_bytes,
                                                          act_q8_1, expert_ids, route_weights, out,
                                                          k, hidden_size, ffn_dim, blocks_per_row, s);
            break;
        case DType::Q8_0:
            launch_quant_down_q8_1_indexed_accum_typed<8>(down_weight, down_expert_bytes, down_row_bytes,
                                                          act_q8_1, expert_ids, route_weights, out,
                                                          k, hidden_size, ffn_dim, blocks_per_row, s);
            break;
        default:
            break;
    }
}

template <int QUANT_TYPE>
void launch_quant_matmul_q8_1_typed(const uint8_t *weight, size_t row_bytes,
                                    const uint8_t *x_q8_1, float *y,
                                    int out_dim, int in_dim, int m,
                                    cudaStream_t stream) {
    constexpr int warps_per_block = kBlock / 32;
    const int blocks_per_row = static_cast<int>(q8_1_row_bytes(in_dim) / 36);
    const int total = m * out_dim;
    const int blocks = (total + warps_per_block - 1) / warps_per_block;
    quant_matmul_q8_1_kernel<QUANT_TYPE><<<blocks, kBlock, 0, stream>>>(
        weight, row_bytes, x_q8_1, y, out_dim, in_dim, m, blocks_per_row);
}

void launch_quant_matmul_q8_1(DType quant_type, const uint8_t *weight, size_t row_bytes,
                              const uint8_t *x_q8_1, float *y,
                              int out_dim, int in_dim, int m, void *stream) {
    const cudaStream_t s = as_stream(stream);
    switch (quant_type) {
        case DType::Q4_K:
            launch_quant_matmul_q8_1_typed<12>(weight, row_bytes, x_q8_1, y, out_dim, in_dim, m, s);
            break;
        case DType::Q6_K:
            launch_quant_matmul_q8_1_typed<14>(weight, row_bytes, x_q8_1, y, out_dim, in_dim, m, s);
            break;
        case DType::Q5_0:
            launch_quant_matmul_q8_1_typed<6>(weight, row_bytes, x_q8_1, y, out_dim, in_dim, m, s);
            break;
        case DType::Q8_0:
            launch_quant_matmul_q8_1_typed<8>(weight, row_bytes, x_q8_1, y, out_dim, in_dim, m, s);
            break;
        default:
            break;
    }
}

template <int QUANT_TYPE>
void launch_quant_matmul_q8_1_mmq_typed(const uint8_t *weight, size_t row_bytes,
                                        const uint8_t *x_q8_1, float *y,
                                        int out_dim, int in_dim, int m,
                                        cudaStream_t stream) {
    constexpr int out_tile = 128;
    constexpr int row_tile = 8;
    const int groups_per_row = static_cast<int>(q8_1_mmq_row_bytes(in_dim) / 144);
    const dim3 grid((out_dim + out_tile - 1) / out_tile, (m + row_tile - 1) / row_tile);
    const dim3 block(32, row_tile);
    quant_matmul_q8_1_mmq_kernel<QUANT_TYPE><<<grid, block, 0, stream>>>(
        weight, row_bytes, x_q8_1, y, out_dim, in_dim, m, groups_per_row);
}

void launch_quant_matmul_q8_1_mmq(DType quant_type, const uint8_t *weight, size_t row_bytes,
                                  const uint8_t *x_q8_1, float *y,
                                  int out_dim, int in_dim, int m, void *stream) {
    const cudaStream_t s = as_stream(stream);
    switch (quant_type) {
        case DType::Q4_K:
            launch_quant_matmul_q8_1_mmq_typed<12>(weight, row_bytes, x_q8_1, y, out_dim, in_dim, m, s);
            break;
        case DType::Q6_K:
            launch_quant_matmul_q8_1_mmq_typed<14>(weight, row_bytes, x_q8_1, y, out_dim, in_dim, m, s);
            break;
        default:
            break;
    }
}

void launch_quant_embedding(DType quant_type, const int *input, float *output, const uint8_t *table,
                            size_t row_bytes, int hidden_size, int input_size, void *stream) {
    cudaStream_t s = as_stream(stream);
    switch (quant_type) {
        case DType::Q4_K:
            quant_embedding_kernel<12><<<input_size, kBlock, 0, s>>>(input, output, table, row_bytes, hidden_size, input_size);
            break;
        case DType::Q6_K:
            quant_embedding_kernel<14><<<input_size, kBlock, 0, s>>>(input, output, table, row_bytes, hidden_size, input_size);
            break;
        case DType::Q5_0:
            quant_embedding_kernel<6><<<input_size, kBlock, 0, s>>>(input, output, table, row_bytes, hidden_size, input_size);
            break;
        case DType::Q8_0:
            quant_embedding_kernel<8><<<input_size, kBlock, 0, s>>>(input, output, table, row_bytes, hidden_size, input_size);
            break;
        default:
            break;
    }
}

void launch_silu_mul(const float *gate, const float *up, float *out, int n, void *stream) {
    ScopedGpuTimer timer("silu_mul", as_stream(stream));
    silu_mul_kernel<<<grid_for(n), kBlock, 0, as_stream(stream)>>>(gate, up, out, n);
}

void launch_embedding_lookup(const int *input, float *output, const uint16_t *table,
                             int input_size, int vocab_size, int hidden_size, int weight_type, void *stream) {
    ScopedGpuTimer timer("embedding_lookup", as_stream(stream));
    embedding_lookup_kernel<<<input_size, kBlock, 0, as_stream(stream)>>>(
        input, output, table, vocab_size, hidden_size, weight_type);
}

/*
rows的值：
- layer 内 prefill norm：rows = input_size
- layer 内 decode norm：rows = 1
- final output norm：一直是 rows = 1
- Deepseek 的 decode 也是把单 token 包成 {1, hidden_size}，所以也是 1
后面如果支持真正 batch，比如 shape 变成 {batch, seq, hidden}，那当前 rows() 会返回 batch * seq，就不只 input_size / 1 了。
    但现在这套实现是单 batch，所以可以按 input_size 或 1 理解。
 *
 */
void launch_rms_norm(const float *input, float *output, const uint16_t *weight, int weight_type,
                     int rows, int hidden_size, float eps, bool one_plus, void *stream) {
    ScopedGpuTimer timer("rms_norm", as_stream(stream));
    rms_norm_kernel<<<rows, kBlock, kBlock * sizeof(float), as_stream(stream)>>>(
        input, output, weight, weight_type, hidden_size, eps, one_plus);
}

// 融合 add + RMSNorm：out_residual = x + residual；out_norm = rmsnorm(out_residual) * weight。
void launch_add_rms_norm(const float *x, const float *residual, float *out_residual, float *out_norm,
                         const uint16_t *weight, int weight_type, int rows, int hidden_size,
                         float eps, bool one_plus, void *stream) {
    ScopedGpuTimer timer("add_rms_norm", as_stream(stream));
    add_rms_norm_kernel<<<rows, kBlock, kBlock * sizeof(float), as_stream(stream)>>>(
        x, residual, out_residual, out_norm, weight, weight_type, hidden_size, eps, one_plus);
}

// ---- full attention ----

void launch_full_attention_q(const float *q_and_gate, const uint16_t *q_norm_weight,
                             float *q, float *gate, int n_heads, int head_dim, const int *pos_dev,
                             float rope_theta, float partial_rotary_factor, float eps, void *stream) {
    ScopedGpuTimer timer("full_attention_q", as_stream(stream));
    full_attention_q_kernel<<<n_heads, kBlock, 0, as_stream(stream)>>>(
        q_and_gate, q_norm_weight, q, gate, n_heads, head_dim, pos_dev,
        rope_theta, partial_rotary_factor, eps);
}

void launch_full_attention_q_batch(const float *q_and_gate, const uint16_t *q_norm_weight,
                                   float *q, float *gate, int tokens, int n_heads, int head_dim,
                                   int start_pos, float rope_theta, float partial_rotary_factor,
                                   float eps, void *stream) {
    ScopedGpuTimer timer("full_attention_q_batch", as_stream(stream));
    full_attention_q_batch_kernel<<<tokens * n_heads, kBlock, 0, as_stream(stream)>>>(
        q_and_gate, q_norm_weight, q, gate, tokens, n_heads, head_dim, start_pos,
        rope_theta, partial_rotary_factor, eps);
}

void launch_full_attention_kv(const float *k_in, const float *v_in, const uint16_t *k_norm_weight,
                              void *key_cache, void *value_cache, bool kv_bf16, int kv_heads, int head_dim,
                              int max_seq_len, const int *pos_dev, float rope_theta,
                              float partial_rotary_factor, float eps, void *stream) {
    ScopedGpuTimer timer("full_attention_kv", as_stream(stream));
    if (kv_bf16) {
        full_attention_kv_kernel<__nv_bfloat16><<<kv_heads, kBlock, 0, as_stream(stream)>>>(
            k_in, v_in, k_norm_weight, static_cast<__nv_bfloat16 *>(key_cache),
            static_cast<__nv_bfloat16 *>(value_cache), kv_heads, head_dim, max_seq_len, pos_dev,
            rope_theta, partial_rotary_factor, eps);
    } else {
        full_attention_kv_kernel<float><<<kv_heads, kBlock, 0, as_stream(stream)>>>(
            k_in, v_in, k_norm_weight, static_cast<float *>(key_cache),
            static_cast<float *>(value_cache), kv_heads, head_dim, max_seq_len, pos_dev,
            rope_theta, partial_rotary_factor, eps);
    }
}

void launch_full_attention_kv_batch(const float *k_in, const float *v_in, const uint16_t *k_norm_weight,
                                    void *key_cache, void *value_cache, bool kv_bf16, int tokens, int kv_heads,
                                    int head_dim, int max_seq_len, int start_pos, float rope_theta,
                                    float partial_rotary_factor, float eps, void *stream) {
    ScopedGpuTimer timer("full_attention_kv_batch", as_stream(stream));
    if (kv_bf16) {
        full_attention_kv_batch_kernel<__nv_bfloat16><<<tokens * kv_heads, kBlock, 0, as_stream(stream)>>>(
            k_in, v_in, k_norm_weight, static_cast<__nv_bfloat16 *>(key_cache),
            static_cast<__nv_bfloat16 *>(value_cache), tokens, kv_heads, head_dim, max_seq_len,
            start_pos, rope_theta, partial_rotary_factor, eps);
    } else {
        full_attention_kv_batch_kernel<float><<<tokens * kv_heads, kBlock, 0, as_stream(stream)>>>(
            k_in, v_in, k_norm_weight, static_cast<float *>(key_cache),
            static_cast<float *>(value_cache), tokens, kv_heads, head_dim, max_seq_len,
            start_pos, rope_theta, partial_rotary_factor, eps);
    }
}

void launch_full_attention_attend(const float *q, const float *gate, const void *key_cache,
                                  const void *value_cache, bool kv_bf16, float *attn, int n_heads, int kv_heads,
                                  int head_dim, int max_seq_len, const int *pos_dev, void *stream) {
    ScopedGpuTimer timer("full_attention_attend", as_stream(stream));
    // flash（online softmax）版：smem 只需 s_p[kBlock] + s_acc[head_dim] + s_red[kBlock]，
    // 与序列长度无关，天然满足 CUDA Graph 一次 capture / 多次 replay 的 smem 恒定要求。
    size_t smem = (static_cast<size_t>(2 * kBlock) + head_dim) * sizeof(float);
    if (kv_bf16) {
        full_attention_attend_kernel<__nv_bfloat16><<<n_heads, kBlock, smem, as_stream(stream)>>>(
            q, gate, static_cast<const __nv_bfloat16 *>(key_cache),
            static_cast<const __nv_bfloat16 *>(value_cache), attn, n_heads, kv_heads, head_dim, max_seq_len, pos_dev);
    } else {
        full_attention_attend_kernel<float><<<n_heads, kBlock, smem, as_stream(stream)>>>(
            q, gate, static_cast<const float *>(key_cache),
            static_cast<const float *>(value_cache), attn, n_heads, kv_heads, head_dim, max_seq_len, pos_dev);
    }
}

void launch_full_attention_attend_batch(const float *q, const float *gate, const void *key_cache,
                                        const void *value_cache, bool kv_bf16, float *attn, int tokens, int n_heads,
                                        int kv_heads, int head_dim, int max_seq_len, int start_pos,
                                        void *stream) {
    ScopedGpuTimer timer("full_attention_attend_batch", as_stream(stream));
    // flash（online softmax）版：smem 只需 s_p[kBlock] + s_acc[head_dim] + s_red[kBlock]，与序列长度无关。
    size_t smem = (static_cast<size_t>(2 * kBlock) + head_dim) * sizeof(float);
    if (kv_bf16) {
        full_attention_attend_batch_kernel<__nv_bfloat16><<<tokens * n_heads, kBlock, smem, as_stream(stream)>>>(
            q, gate, static_cast<const __nv_bfloat16 *>(key_cache),
            static_cast<const __nv_bfloat16 *>(value_cache), attn, tokens, n_heads, kv_heads, head_dim, max_seq_len, start_pos);
    } else {
        full_attention_attend_batch_kernel<float><<<tokens * n_heads, kBlock, smem, as_stream(stream)>>>(
            q, gate, static_cast<const float *>(key_cache),
            static_cast<const float *>(value_cache), attn, tokens, n_heads, kv_heads, head_dim, max_seq_len, start_pos);
    }
}

// ---- linear attention ----

void launch_linear_attention_conv(const float *mixed, const uint16_t *conv_weight, float *conv_state,
                                  float *conv_out, int conv_dim, int kernel, void *stream) {
    ScopedGpuTimer timer("linear_attention_conv", as_stream(stream));
    linear_attention_conv_kernel<<<grid_for(conv_dim), kBlock, 0, as_stream(stream)>>>(
        mixed, conv_weight, conv_state, conv_out, conv_dim, kernel);
}

void launch_linear_attention_conv_batch(const float *mixed, const uint16_t *conv_weight,
                                        float *conv_state, float *conv_out, int tokens, int conv_dim,
                                        int kernel, void *stream) {
    ScopedGpuTimer timer("linear_attention_conv_batch", as_stream(stream));
    linear_attention_conv_batch_kernel<<<grid_for(conv_dim), kBlock, 0, as_stream(stream)>>>(
        mixed, conv_weight, conv_state, conv_out, tokens, conv_dim, kernel);
}

void launch_linear_attention_recurrent(const float *conv_out, const float *z, const float *b,
                                       const float *a, const float *a_log, const uint16_t *dt_bias,
                                       const float *norm_weight, void *recurrent_state, bool state_bf16,
                                       float *gated, int key_heads, int value_heads, int k_dim,
                                       int v_dim, float eps, void *stream) {
    ScopedGpuTimer timer("linear_attention_recurrent", as_stream(stream));
    size_t smem = (static_cast<size_t>(2 * k_dim + 2 * v_dim) + 2 * kLinearRecurBlock) * sizeof(float);
    if (state_bf16) {
        linear_attention_recurrent_kernel<__nv_bfloat16><<<value_heads, kLinearRecurBlock, smem, as_stream(stream)>>>(
            conv_out, z, b, a, a_log, dt_bias, norm_weight,
            static_cast<__nv_bfloat16 *>(recurrent_state), gated,
            key_heads, value_heads, k_dim, v_dim, eps);
    } else {
        linear_attention_recurrent_kernel<float><<<value_heads, kLinearRecurBlock, smem, as_stream(stream)>>>(
            conv_out, z, b, a, a_log, dt_bias, norm_weight,
            static_cast<float *>(recurrent_state), gated,
            key_heads, value_heads, k_dim, v_dim, eps);
    }
}

void launch_linear_attention_recurrent_batch(const float *conv_out, const float *z, const float *b,
                                             const float *a, const float *a_log, const uint16_t *dt_bias,
                                             const float *norm_weight, float *recurrent_state, float *gated,
                                             int tokens, int key_heads, int value_heads, int k_dim, int v_dim,
                                             float eps, void *stream) {
    ScopedGpuTimer timer("linear_attention_recurrent_batch", as_stream(stream));
    size_t smem = (static_cast<size_t>(2 * k_dim + 2 * v_dim) + 2 * kLinearRecurBlock) * sizeof(float);
    linear_attention_recurrent_batch_kernel<<<value_heads, kLinearRecurBlock, smem, as_stream(stream)>>>(
        conv_out, z, b, a, a_log, dt_bias, norm_weight, recurrent_state, gated, tokens,
        key_heads, value_heads, k_dim, v_dim, eps);
}

// 融合 kernel launch（conv1d→recurrent→读出 单核）。grid=key_heads，按 state_bf16 分派模板实例。
void launch_linear_attention_fused(const float *mixed, const uint16_t *conv_weight, float *conv_state,
                                   const float *z, const float *b, const float *a, const float *a_log,
                                   const uint16_t *dt_bias, const float *norm_weight,
                                   void *recurrent_state, bool state_bf16, float *gated,
                                   int key_heads, int value_heads, int k_dim, int v_dim,
                                   int kernel, float eps, void *stream) {
    ScopedGpuTimer timer("linear_attention_fused", as_stream(stream));
    size_t smem = (static_cast<size_t>(4 * k_dim + 3 * v_dim) + 2 * kLinearRecurBlock) * sizeof(float);
    if (state_bf16) {
        linear_attention_fused_kernel<__nv_bfloat16><<<key_heads, kLinearRecurBlock, smem, as_stream(stream)>>>(
            mixed, conv_weight, conv_state, z, b, a, a_log, dt_bias, norm_weight,
            static_cast<__nv_bfloat16 *>(recurrent_state), gated,
            key_heads, value_heads, k_dim, v_dim, kernel, eps);
    } else {
        linear_attention_fused_kernel<float><<<key_heads, kLinearRecurBlock, smem, as_stream(stream)>>>(
            mixed, conv_weight, conv_state, z, b, a, a_log, dt_bias, norm_weight,
            static_cast<float *>(recurrent_state), gated,
            key_heads, value_heads, k_dim, v_dim, kernel, eps);
    }
}

void launch_linear_attention_fused_batch(const float *mixed, const uint16_t *conv_weight, float *conv_state,
                                         const float *z, const float *b, const float *a, const float *a_log,
                                         const uint16_t *dt_bias, const float *norm_weight,
                                         void *recurrent_state, bool state_bf16, float *gated, int tokens,
                                         int key_heads, int value_heads, int k_dim, int v_dim,
                                         int kernel, float eps, void *stream) {
    ScopedGpuTimer timer("linear_attention_fused_batch", as_stream(stream));
    size_t smem = (static_cast<size_t>(4 * k_dim + 3 * v_dim) + 2 * kLinearRecurBlock) * sizeof(float);
    if (state_bf16) {
        linear_attention_fused_batch_kernel<__nv_bfloat16><<<key_heads, kLinearRecurBlock, smem, as_stream(stream)>>>(
            mixed, conv_weight, conv_state, z, b, a, a_log, dt_bias, norm_weight,
            static_cast<__nv_bfloat16 *>(recurrent_state), gated, tokens,
            key_heads, value_heads, k_dim, v_dim, kernel, eps);
    } else {
        linear_attention_fused_batch_kernel<float><<<key_heads, kLinearRecurBlock, smem, as_stream(stream)>>>(
            mixed, conv_weight, conv_state, z, b, a, a_log, dt_bias, norm_weight,
            static_cast<float *>(recurrent_state), gated, tokens,
            key_heads, value_heads, k_dim, v_dim, kernel, eps);
    }
}

void launch_dequantize_q4k_to_f16(const uint8_t *src, uint16_t *out, int64_t num_elements, void *stream) {
    ScopedGpuTimer timer("dequantize_q4k_to_f16", as_stream(stream));
    const int64_t nblocks = num_elements / 256; // 调用方保证 num_elements % 256 == 0
    dequantize_q4k_to_f16_kernel<<<static_cast<unsigned int>(nblocks), 128, 0, as_stream(stream)>>>(
        src, out, nblocks);
}

void launch_dequantize_q80_to_f16(const uint8_t *src, uint16_t *out, int64_t num_elements, void *stream) {
    ScopedGpuTimer timer("dequantize_q80_to_f16", as_stream(stream));
    const int64_t nblocks = num_elements / 32;
    dequantize_q80_to_f16_kernel<<<static_cast<unsigned int>(nblocks), 32, 0, as_stream(stream)>>>(
        src, out, nblocks);
}

void launch_dequantize_q50_to_f16(const uint8_t *src, uint16_t *out, int64_t num_elements, void *stream) {
    ScopedGpuTimer timer("dequantize_q50_to_f16", as_stream(stream));
    const int64_t nblocks = num_elements / 32;
    dequantize_q50_to_f16_kernel<<<static_cast<unsigned int>(nblocks), 16, 0, as_stream(stream)>>>(
        src, out, nblocks);
}

void launch_dequantize_q6k_to_f16(const uint8_t *src, uint16_t *out, int64_t num_elements, void *stream) {
    ScopedGpuTimer timer("dequantize_q6k_to_f16", as_stream(stream));
    const int64_t nblocks = num_elements / 256;
    dequantize_q6k_to_f16_kernel<<<static_cast<unsigned int>(nblocks), 32, 0, as_stream(stream)>>>(
        src, out, nblocks);
}

void launch_f32_to_f16_copy(const float *src, uint16_t *out, int64_t num_elements, void *stream) {
    ScopedGpuTimer timer("f32_to_f16_copy", as_stream(stream));
    f32_to_f16_copy_kernel<<<grid_for(static_cast<int>(num_elements)), kBlock, 0, as_stream(stream)>>>(
        src, out, num_elements);
}

void launch_f32_to_bf16_copy(const float *src, uint16_t *out, int64_t num_elements, void *stream) {
    ScopedGpuTimer timer("f32_to_bf16_copy", as_stream(stream));
    f32_to_bf16_copy_kernel<<<grid_for(static_cast<int>(num_elements)), kBlock, 0, as_stream(stream)>>>(
        src, out, num_elements);
}

// ---- MLA ----

void launch_mla_kv_a(const float *kv_a, const float *kv_a_norm_weight, float *output_kv_cache,
                     int input_size, int kv_lora, int qk_rope, int start_pos,
                     const float *inv_freq, float eps, void *stream) {
    ScopedGpuTimer timer("mla_kv_a", as_stream(stream));
    size_t smem = (static_cast<size_t>(kBlock) + kv_lora + qk_rope) * sizeof(float);
    mla_kv_a_kernel<<<input_size, kBlock, smem, as_stream(stream)>>>(
        kv_a, kv_a_norm_weight, output_kv_cache, input_size, kv_lora, qk_rope, start_pos, inv_freq, eps);
}

void launch_mla_rope_q(float *q, int input_size, int n_heads, int qk_nope, int qk_rope,
                       int start_pos, const float *inv_freq, void *stream) {
    ScopedGpuTimer timer("mla_rope_q", as_stream(stream));
    mla_rope_q_kernel<<<input_size * n_heads, kBlock, 0, as_stream(stream)>>>(
        q, input_size, n_heads, qk_nope, qk_rope, start_pos, inv_freq);
}

void launch_mla_attend_batch(const float *q, const float *kv_b_out, const float *kv_cache, float *attn,
                             int input_size, int n_heads, int qk_nope, int qk_rope, int v_head, int kv_lora,
                             int start_pos, float softmax_scale, void *stream) {
    ScopedGpuTimer timer("mla_attend_batch", as_stream(stream));
    size_t max_pos = static_cast<size_t>(start_pos + input_size - 1);
    size_t smem = (max_pos + 1 + kBlock) * sizeof(float);
    mla_attend_batch_kernel<<<input_size * n_heads, kBlock, smem, as_stream(stream)>>>(
        q, kv_b_out, kv_cache, attn, input_size, n_heads, qk_nope, qk_rope, v_head, kv_lora,
        start_pos, softmax_scale);
}

// ---- MoE ----

void launch_moe_router_topk(const float *router_logits, int *top_idx, float *top_w,
                            int tokens, int n_experts, int k, float routed_scaling, void *stream) {
    ScopedGpuTimer timer("moe_router_topk", as_stream(stream));
    moe_router_topk_kernel<<<tokens, 32, 0, as_stream(stream)>>>(
        router_logits, top_idx, top_w, tokens, n_experts, k, routed_scaling);
}

void launch_moe_accumulate(const float *expert_out, float weight, float *out, int n, void *stream) {
    ScopedGpuTimer timer("moe_accumulate", as_stream(stream));
    moe_accumulate_kernel<<<grid_for(n), kBlock, 0, as_stream(stream)>>>(expert_out, weight, out, n);
}

void launch_moe_accumulate_device(const float *expert_out, const float *weight, float *out, int n,
                                  void *stream) {
    ScopedGpuTimer timer("moe_accumulate_device", as_stream(stream));
    moe_accumulate_device_kernel<<<grid_for(n), kBlock, 0, as_stream(stream)>>>(expert_out, weight, out, n);
}

// ---- argmax ----

// 两阶段 argmax 的局部结果缓冲：阶段1 每个 block 输出一对 (val, idx)，
// 阶段2 单 block 归约。按最大 block 数一次性懒分配、进程内复用（decode 每步都调用）。
namespace {
constexpr int kArgmaxMaxBlocks = 512; // 上限：15w 词表 / 256 ≈ 586，取 512 足够摊到多 SM
float *g_argmax_partial_vals = nullptr;
int *g_argmax_partial_idxs = nullptr;

void ensure_argmax_partial_buffers() {
    if (g_argmax_partial_vals == nullptr) {
        g_argmax_partial_vals = static_cast<float *>(
            cuda_malloc_device(sizeof(float) * kArgmaxMaxBlocks, "argmax partial vals"));
        g_argmax_partial_idxs = static_cast<int *>(
            cuda_malloc_device(sizeof(int) * kArgmaxMaxBlocks, "argmax partial idxs"));
    }
}
} // namespace

void launch_argmax(const float *logits, int n, int *out_idx, void *stream) {
    ScopedGpuTimer timer("argmax", as_stream(stream));
    cudaStream_t s = as_stream(stream);
    int blocks = grid_for(n);
    if (blocks > kArgmaxMaxBlocks) blocks = kArgmaxMaxBlocks;
    ensure_argmax_partial_buffers();
    // 阶段1：多 block 并行求局部 argmax，摊平 15w 词表的串行扫描。
    argmax_partial_kernel<<<blocks, kBlock, 0, s>>>(logits, n, g_argmax_partial_vals,
                                                    g_argmax_partial_idxs);
    // 阶段2：单 block 归约 blocks 个局部结果得到全局 argmax。
    argmax_final_kernel<<<1, kBlock, 0, s>>>(g_argmax_partial_vals, g_argmax_partial_idxs,
                                             blocks, out_idx);
}
