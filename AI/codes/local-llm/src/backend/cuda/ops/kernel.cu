//
// Created by zhangyoulun on 9/8/2026.
//

#include "kernel.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cfloat>

#include "utils/stats/ScopedTimer.h"

namespace {

constexpr int kBlock = 256;
constexpr int kBlockConst = 256; // 与 kBlock 一致，供静态 shared 数组使用
constexpr int kMaxHeadDim = 256; // Qwen head_dim = 256

inline cudaStream_t as_stream(void *stream) {
    return static_cast<cudaStream_t>(stream);
}

// 把一个低精度元素（bf16 或 f16）读成 float。lowp_type: 0=bf16, 1=f16。
__device__ inline float lowp_to_float(uint16_t v, int lowp_type) {
    if (lowp_type == 1) {
        __half h = __ushort_as_half(v);
        return __half2float(h);
    }
    __nv_bfloat16 b = *reinterpret_cast<const __nv_bfloat16 *>(&v);
    return __bfloat162float(b);
}

// ---- 精度转换 ----

__global__ void float_to_bf16_kernel(const float *input, uint16_t *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __nv_bfloat16 b = __float2bfloat16(input[i]);
    output[i] = *reinterpret_cast<const uint16_t *>(&b);
}

__global__ void float_to_f16_kernel(const float *input, uint16_t *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half h = __float2half(input[i]);
    output[i] = __half_as_ushort(h);
}

// ---- 逐元素 ----

__global__ void add_kernel(const float *a, const float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}

__global__ void silu_mul_kernel(const float *gate, const float *up, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float silu = g / (1.0f + __expf(-g));
    out[i] = silu * up[i];
}

// ---- Embedding ----

// 每个 block 负责一个 token 的一行拷贝。
__global__ void embedding_lookup_kernel(const uint16_t *table, const int *token_ids,
                                        float *output, int vocab, int hidden, int lowp_type) {
    int token = blockIdx.x;
    int id = token_ids[token];
    if (id < 0 || id >= vocab) id = 0;
    const uint16_t *row = table + static_cast<size_t>(id) * hidden;
    float *dst = output + static_cast<size_t>(token) * hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        dst[j] = lowp_to_float(row[j], lowp_type);
    }
}

// ---- RMSNorm ----
// 每个 block 处理一行；block 内先归约平方和，再逐元素缩放。

template <typename OutT>
__global__ void rms_norm_kernel(const float *input, const void *weight, int weight_type,
                                OutT *output, int hidden, float eps, bool one_plus, int lowp_type) {
    int row = blockIdx.x;
    const float *x = input + static_cast<size_t>(row) * hidden;

    extern __shared__ float sdata[];
    float local = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = x[j];
        local += v * v;
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / hidden + eps);

    OutT *dst = output + static_cast<size_t>(row) * hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float w;
        if (weight_type == 2) {
            w = static_cast<const float *>(weight)[j];
        } else {
            w = lowp_to_float(static_cast<const uint16_t *>(weight)[j], weight_type);
        }
        if (one_plus) w += 1.0f;
        float y = x[j] * inv_rms * w;
        if constexpr (sizeof(OutT) == sizeof(float)) {
            dst[j] = y;
        } else if (lowp_type == 1) {
            __half h = __float2half(y);
            dst[j] = __half_as_ushort(h);
        } else {
            __nv_bfloat16 b = __float2bfloat16(y);
            dst[j] = *reinterpret_cast<const uint16_t *>(&b);
        }
    }
}

// ---- 采样 ----
// 阶段一：每个 block 求局部 argmax，写入 block_values / block_indices。
__global__ void argmax_block_kernel(const float *logits, int vocab,
                                    float *block_values, int *block_indices) {
    extern __shared__ char smem[];
    float *sval = reinterpret_cast<float *>(smem);
    int *sidx = reinterpret_cast<int *>(sval + blockDim.x);

    float best = -FLT_MAX;
    int best_i = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vocab; i += gridDim.x * blockDim.x) {
        float v = logits[i];
        if (v > best) { best = v; best_i = i; }
    }
    sval[threadIdx.x] = best;
    sidx[threadIdx.x] = best_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sval[threadIdx.x + s] > sval[threadIdx.x]) {
                sval[threadIdx.x] = sval[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_values[blockIdx.x] = sval[0];
        block_indices[blockIdx.x] = sidx[0];
    }
}

// 阶段二：单 block 归约所有 block 的局部结果，得全局 argmax。
__global__ void argmax_final_kernel(const float *block_values, const int *block_indices,
                                    int num_blocks, float *best_value, int *best_index) {
    float best = -FLT_MAX;
    int best_i = 0;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        if (block_values[i] > best) { best = block_values[i]; best_i = block_indices[i]; }
    }
    extern __shared__ char smem[];
    float *sval = reinterpret_cast<float *>(smem);
    int *sidx = reinterpret_cast<int *>(sval + blockDim.x);
    sval[threadIdx.x] = best;
    sidx[threadIdx.x] = best_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sval[threadIdx.x + s] > sval[threadIdx.x]) {
                sval[threadIdx.x] = sval[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *best_value = sval[0];
        *best_index = sidx[0];
    }
}

// ---- full attention ----

// query/gate 归一化 + RoPE。每个 block 处理 (tok, head)。src 索引已按 tok 偏移。
__device__ inline void full_attn_q_head(const float *q_and_gate, const uint16_t *q_norm_weight,
                                        float *q, float *gate, int head_dim, int pos,
                                        float rope_theta, float partial_rotary_factor, float eps,
                                        size_t src, size_t dst, float *partial) {
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float v = q_and_gate[src + d];
        q[dst + d] = v;
        gate[dst + d] = q_and_gate[src + head_dim + d];
        ss += v * v;
    }
    partial[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float w = 1.0f + lowp_to_float(q_norm_weight[d], 0);
        q[dst + d] *= scale * w;
    }
    __syncthreads();
    const int rotary_dim = static_cast<int>(head_dim * partial_rotary_factor);
    const int half = rotary_dim / 2;
    for (int i = tid; i < half; i += blockDim.x) {
        const float inv_freq = 1.0f / powf(rope_theta, static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
        const float angle = static_cast<float>(pos) * inv_freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float x1 = q[dst + i];
        const float x2 = q[dst + i + half];
        q[dst + i] = x1 * c - x2 * s;
        q[dst + i + half] = x2 * c + x1 * s;
    }
}

__global__ void full_attention_q_kernel(const float *q_and_gate, const uint16_t *q_norm_weight,
                                        float *q, float *gate, int n_heads, int head_dim, int pos,
                                        float rope_theta, float partial_rotary_factor, float eps) {
    __shared__ float partial[kBlockConst];
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const size_t src = static_cast<size_t>(h) * head_dim * 2;
    const size_t dst = static_cast<size_t>(h) * head_dim;
    full_attn_q_head(q_and_gate, q_norm_weight, q, gate, head_dim, pos,
                     rope_theta, partial_rotary_factor, eps, src, dst, partial);
}

__global__ void full_attention_q_batch_kernel(const float *q_and_gate, const uint16_t *q_norm_weight,
                                              float *q, float *gate, int tokens, int n_heads,
                                              int head_dim, int start_pos, float rope_theta,
                                              float partial_rotary_factor, float eps) {
    __shared__ float partial[kBlockConst];
    const int block = blockIdx.x;
    const int tok = block / n_heads;
    const int h = block - tok * n_heads;
    if (tok >= tokens) return;
    const int q_total = n_heads * head_dim;
    const size_t src = static_cast<size_t>(tok) * q_total * 2 + static_cast<size_t>(h) * head_dim * 2;
    const size_t dst = static_cast<size_t>(tok) * q_total + static_cast<size_t>(h) * head_dim;
    full_attn_q_head(q_and_gate, q_norm_weight, q, gate, head_dim, start_pos + tok,
                     rope_theta, partial_rotary_factor, eps, src, dst, partial);
}

__device__ inline void full_attn_kv_head(const float *k_in, const float *v_in,
                                         const uint16_t *k_norm_weight, float *key_cache,
                                         float *value_cache, int kv_heads, int head_dim, int pos,
                                         float rope_theta, float partial_rotary_factor, float eps,
                                         size_t base, float *partial, float *k_local) {
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float v = k_in[base + d];
        k_local[d] = v;
        ss += v * v;
    }
    partial[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float w = 1.0f + lowp_to_float(k_norm_weight[d], 0);
        k_local[d] *= scale * w;
    }
    __syncthreads();
    const int rotary_dim = static_cast<int>(head_dim * partial_rotary_factor);
    const int half = rotary_dim / 2;
    for (int i = tid; i < half; i += blockDim.x) {
        const float inv_freq = 1.0f / powf(rope_theta, static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
        const float angle = static_cast<float>(pos) * inv_freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float x1 = k_local[i];
        const float x2 = k_local[i + half];
        k_local[i] = x1 * c - x2 * s;
        k_local[i + half] = x2 * c + x1 * s;
    }
    __syncthreads();
    const int h = static_cast<int>(base / head_dim) % kv_heads;
    const size_t cache_base = (static_cast<size_t>(pos) * kv_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        key_cache[cache_base + d] = k_local[d];
        value_cache[cache_base + d] = v_in[base + d];
    }
}

__global__ void full_attention_kv_kernel(const float *k_in, const float *v_in,
                                         const uint16_t *k_norm_weight, float *key_cache,
                                         float *value_cache, int kv_heads, int head_dim,
                                         int max_seq_len, int pos, float rope_theta,
                                         float partial_rotary_factor, float eps) {
    __shared__ float partial[kBlockConst];
    __shared__ float k_local[kMaxHeadDim];
    const int h = blockIdx.x;
    if (h >= kv_heads) return;
    const size_t base = static_cast<size_t>(h) * head_dim;
    full_attn_kv_head(k_in, v_in, k_norm_weight, key_cache, value_cache, kv_heads, head_dim, pos,
                      rope_theta, partial_rotary_factor, eps, base, partial, k_local);
    (void) max_seq_len;
}

__global__ void full_attention_kv_batch_kernel(const float *k_in, const float *v_in,
                                               const uint16_t *k_norm_weight, float *key_cache,
                                               float *value_cache, int tokens, int kv_heads,
                                               int head_dim, int max_seq_len, int start_pos,
                                               float rope_theta, float partial_rotary_factor, float eps) {
    __shared__ float partial[kBlockConst];
    __shared__ float k_local[kMaxHeadDim];
    const int block = blockIdx.x;
    const int tok = block / kv_heads;
    const int h = block - tok * kv_heads;
    if (tok >= tokens) return;
    const int kv_total = kv_heads * head_dim;
    const size_t base = static_cast<size_t>(tok) * kv_total + static_cast<size_t>(h) * head_dim;
    full_attn_kv_head(k_in, v_in, k_norm_weight, key_cache, value_cache, kv_heads, head_dim,
                      start_pos + tok, rope_theta, partial_rotary_factor, eps, base, partial, k_local);
    (void) max_seq_len;
}

// causal attention + 输出门控。每个 block 处理一个 (tok, head)，pos 为该 token 绝对位置。
__device__ inline void full_attn_attend_head(const float *q, const float *gate,
                                             const float *key_cache, const float *value_cache,
                                             float *attn, int n_heads, int kv_heads, int head_dim,
                                             int pos, size_t q_off, size_t out_off,
                                             float *scores, float *partial) {
    const int tid = threadIdx.x;
    const int kv_group = n_heads / kv_heads;
    const int h = static_cast<int>(q_off / head_dim) % n_heads;
    const int kh = h / kv_group;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float *qh = q + q_off;

    float local_max = -INFINITY;
    for (int t = tid; t <= pos; t += blockDim.x) {
        const float *key = key_cache + (static_cast<size_t>(t) * kv_heads + kh) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * key[d];
        const float score = dot * scale;
        scores[t] = score;
        local_max = fmaxf(local_max, score);
    }
    partial[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] = fmaxf(partial[tid], partial[tid + s]);
        __syncthreads();
    }
    const float max_score = partial[0];
    float local_denom = 0.0f;
    for (int t = tid; t <= pos; t += blockDim.x) {
        const float e = __expf(scores[t] - max_score);
        scores[t] = e;
        local_denom += e;
    }
    partial[tid] = local_denom;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    const float denom = partial[0];
    float *out = attn + out_off;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            const float prob = scores[t] / denom;
            const float *value = value_cache + (static_cast<size_t>(t) * kv_heads + kh) * head_dim;
            sum += prob * value[d];
        }
        const float g = gate[q_off + d];
        out[d] = sum * (1.0f / (1.0f + __expf(-g)));
    }
}

__global__ void full_attention_attend_kernel(const float *q, const float *gate,
                                             const float *key_cache, const float *value_cache,
                                             float *attn, int n_heads, int kv_heads, int head_dim,
                                             int max_seq_len, int pos) {
    extern __shared__ float shared[];
    float *scores = shared;
    float *partial = scores + pos + 1;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const size_t off = static_cast<size_t>(h) * head_dim;
    full_attn_attend_head(q, gate, key_cache, value_cache, attn, n_heads, kv_heads, head_dim,
                          pos, off, off, scores, partial);
    (void) max_seq_len;
}

__global__ void full_attention_attend_batch_kernel(const float *q, const float *gate,
                                                   const float *key_cache, const float *value_cache,
                                                   float *attn, int tokens, int n_heads, int kv_heads,
                                                   int head_dim, int max_seq_len, int start_pos) {
    extern __shared__ float shared[];
    const int block = blockIdx.x;
    const int tok = block / n_heads;
    const int h = block - tok * n_heads;
    if (tok >= tokens) return;
    const int pos = start_pos + tok;
    float *scores = shared;
    float *partial = scores + pos + 1;
    const int q_total = n_heads * head_dim;
    const size_t off = static_cast<size_t>(tok) * q_total + static_cast<size_t>(h) * head_dim;
    full_attn_attend_head(q, gate, key_cache, value_cache, attn, n_heads, kv_heads, head_dim,
                          pos, off, off, scores, partial);
    (void) max_seq_len;
}

// ---- linear attention ----

__global__ void linear_attention_conv_kernel(const float *mixed, const uint16_t *conv_weight,
                                             float *conv_state, float *conv_out, int conv_dim, int kernel) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= conv_dim) return;
    float *row = conv_state + static_cast<size_t>(d) * kernel;
    for (int i = 0; i < kernel - 1; ++i) row[i] = row[i + 1];
    row[kernel - 1] = mixed[d];
    float sum = 0.0f;
    const uint16_t *w = conv_weight + static_cast<size_t>(d) * kernel;
    for (int i = 0; i < kernel; ++i) sum += lowp_to_float(w[i], 0) * row[i];
    conv_out[d] = sum / (1.0f + __expf(-sum));
}

__global__ void linear_attention_conv_batch_kernel(const float *mixed, const uint16_t *conv_weight,
                                                   float *conv_state, float *conv_out,
                                                   int tokens, int conv_dim, int kernel) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= conv_dim) return;
    float *row = conv_state + static_cast<size_t>(d) * kernel;
    const uint16_t *w = conv_weight + static_cast<size_t>(d) * kernel;
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < kernel - 1; ++i) row[i] = row[i + 1];
        row[kernel - 1] = mixed[static_cast<size_t>(t) * conv_dim + d];
        float sum = 0.0f;
        for (int i = 0; i < kernel; ++i) sum += lowp_to_float(w[i], 0) * row[i];
        conv_out[static_cast<size_t>(t) * conv_dim + d] = sum / (1.0f + __expf(-sum));
    }
}

// gated delta 递归单步。shared: q[k_dim] k[k_dim] delta[v_dim] core[v_dim] partial[2*blockDim]。
__device__ inline void linear_attn_recurrent_step(const float *conv_out, const float *z,
                                                  const float *b, const float *a, const float *a_log,
                                                  const uint16_t *dt_bias, const float *norm_weight,
                                                  float *rec, float *gated_row, int vh, int kh,
                                                  int key_total, int k_dim, int v_dim, float eps,
                                                  float *q, float *k, float *delta, float *core,
                                                  float *partial) {
    const int tid = threadIdx.x;
    const float *query_base = conv_out;
    const float *key_base = conv_out + key_total;
    const float *value_base = conv_out + key_total * 2;

    float ss_q = 0.0f, ss_k = 0.0f;
    for (int i = tid; i < k_dim; i += blockDim.x) {
        const float qv = query_base[kh * k_dim + i];
        const float kv = key_base[kh * k_dim + i];
        q[i] = qv;
        k[i] = kv;
        ss_q += qv * qv;
        ss_k += kv * kv;
    }
    partial[tid] = ss_q;
    partial[blockDim.x + tid] = ss_k;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
            partial[blockDim.x + tid] += partial[blockDim.x + tid + s];
        }
        __syncthreads();
    }
    const float q_scale = rsqrtf(partial[0] + 1e-6f) / sqrtf(static_cast<float>(k_dim));
    const float k_scale = rsqrtf(partial[blockDim.x] + 1e-6f);
    for (int i = tid; i < k_dim; i += blockDim.x) {
        q[i] *= q_scale;
        k[i] *= k_scale;
    }
    __syncthreads();

    const float beta = 1.0f / (1.0f + __expf(-b[vh]));
    const float dt = a[vh] + lowp_to_float(dt_bias[vh], 0);
    const float softplus_dt = dt > 20.0f ? dt : (dt < -20.0f ? __expf(dt) : log1pf(__expf(dt)));
    const float g = -__expf(a_log[vh]) * softplus_dt;
    const float decay = __expf(g);

    const int state_size = k_dim * v_dim;
    for (int i = tid; i < state_size; i += blockDim.x) rec[i] *= decay;
    __syncthreads();

    for (int vd = tid; vd < v_dim; vd += blockDim.x) {
        float kv_mem = 0.0f;
        for (int kd = 0; kd < k_dim; ++kd) kv_mem += rec[static_cast<size_t>(kd) * v_dim + vd] * k[kd];
        const float value = value_base[static_cast<size_t>(vh) * v_dim + vd];
        delta[vd] = (value - kv_mem) * beta;
    }
    __syncthreads();
    for (int i = tid; i < state_size; i += blockDim.x) {
        const int kd = i / v_dim;
        const int vd = i - kd * v_dim;
        rec[i] += k[kd] * delta[vd];
    }
    __syncthreads();
    for (int vd = tid; vd < v_dim; vd += blockDim.x) {
        float sum = 0.0f;
        for (int kd = 0; kd < k_dim; ++kd) sum += rec[static_cast<size_t>(kd) * v_dim + vd] * q[kd];
        core[vd] = sum;
    }
    __syncthreads();
    float ss = 0.0f;
    for (int vd = tid; vd < v_dim; vd += blockDim.x) ss += core[vd] * core[vd];
    partial[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / static_cast<float>(v_dim) + eps);
    for (int vd = tid; vd < v_dim; vd += blockDim.x) {
        const float gate = z[static_cast<size_t>(vh) * v_dim + vd];
        const float silu_gate = gate / (1.0f + __expf(-gate));
        gated_row[static_cast<size_t>(vh) * v_dim + vd] =
            norm_weight[vd] * core[vd] * norm_scale * silu_gate;
    }
}

__global__ void linear_attention_recurrent_kernel(const float *conv_out, const float *z,
                                                  const float *b, const float *a, const float *a_log,
                                                  const uint16_t *dt_bias, const float *norm_weight,
                                                  float *recurrent_state, float *gated,
                                                  int key_heads, int value_heads, int k_dim, int v_dim,
                                                  float eps) {
    extern __shared__ float shared[];
    float *q = shared;
    float *k = q + k_dim;
    float *delta = k + k_dim;
    float *core = delta + v_dim;
    float *partial = core + v_dim;
    const int vh = blockIdx.x;
    const int repeat = value_heads / key_heads;
    const int kh = vh / repeat;
    const int key_total = key_heads * k_dim;
    float *rec = recurrent_state + static_cast<size_t>(vh) * k_dim * v_dim;
    linear_attn_recurrent_step(conv_out, z, b, a, a_log, dt_bias, norm_weight, rec, gated, vh, kh,
                               key_total, k_dim, v_dim, eps, q, k, delta, core, partial);
}

__global__ void linear_attention_recurrent_batch_kernel(const float *conv_out, const float *z,
                                                        const float *b, const float *a, const float *a_log,
                                                        const uint16_t *dt_bias, const float *norm_weight,
                                                        float *recurrent_state, float *gated, int tokens,
                                                        int key_heads, int value_heads, int k_dim, int v_dim,
                                                        float eps) {
    extern __shared__ float shared[];
    float *q = shared;
    float *k = q + k_dim;
    float *delta = k + k_dim;
    float *core = delta + v_dim;
    float *partial = core + v_dim;
    const int vh = blockIdx.x;
    const int repeat = value_heads / key_heads;
    const int kh = vh / repeat;
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    float *rec = recurrent_state + static_cast<size_t>(vh) * k_dim * v_dim;
    for (int tok = 0; tok < tokens; ++tok) {
        const float *token_conv = conv_out + static_cast<size_t>(tok) * conv_dim;
        const float *z_base = z + static_cast<size_t>(tok) * value_total;
        const float *b_base = b + static_cast<size_t>(tok) * value_heads;
        const float *a_base = a + static_cast<size_t>(tok) * value_heads;
        float *gated_row = gated + static_cast<size_t>(tok) * value_total;
        linear_attn_recurrent_step(token_conv, z_base, b_base, a_base, a_log, dt_bias, norm_weight,
                                   rec, gated_row, vh, kh, key_total, k_dim, v_dim, eps,
                                   q, k, delta, core, partial);
        __syncthreads();
    }
}

__global__ void float_to_lowp_kernel(const float *input, uint16_t *output, int n, int lowp_type) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (lowp_type == 1) {
        __half h = __float2half(input[i]);
        output[i] = __half_as_ushort(h);
    } else {
        __nv_bfloat16 b = __float2bfloat16(input[i]);
        output[i] = *reinterpret_cast<const uint16_t *>(&b);
    }
}

inline int grid_for(int n) { return (n + kBlock - 1) / kBlock; }

} // namespace

// ================= launch 封装 =================

void launch_float_to_bf16(const float *input, uint16_t *output, int n, void *stream) {
    ScopedGpuTimer timer("float_to_bf16", as_stream(stream));
    float_to_bf16_kernel<<<grid_for(n), kBlock, 0, as_stream(stream)>>>(input, output, n);
}

void launch_float_to_f16(const float *input, uint16_t *output, int n, void *stream) {
    ScopedGpuTimer timer("float_to_f16", as_stream(stream));
    float_to_f16_kernel<<<grid_for(n), kBlock, 0, as_stream(stream)>>>(input, output, n);
}

void launch_add(const float *a, const float *b, float *out, int n, void *stream) {
    ScopedGpuTimer timer("residual_add", as_stream(stream));
    add_kernel<<<grid_for(n), kBlock, 0, as_stream(stream)>>>(a, b, out, n);
}

void launch_silu_mul(const float *gate, const float *up, float *out, int n, void *stream) {
    ScopedGpuTimer timer("mlp.silu_mul", as_stream(stream));
    silu_mul_kernel<<<grid_for(n), kBlock, 0, as_stream(stream)>>>(gate, up, out, n);
}

void launch_embedding_lookup(const uint16_t *table, const int *token_ids, float *output,
                             int tokens, int vocab, int hidden, int lowp_type, void *stream) {
    ScopedGpuTimer timer("embedding", as_stream(stream));
    embedding_lookup_kernel<<<tokens, kBlock, 0, as_stream(stream)>>>(
        table, token_ids, output, vocab, hidden, lowp_type);
}

void launch_rms_norm(const float *input, const void *weight, int weight_type, float *output,
                     int rows, int hidden, float eps, bool one_plus, void *stream) {
    ScopedGpuTimer timer("rms_norm", as_stream(stream));
    rms_norm_kernel<float><<<rows, kBlock, kBlock * sizeof(float), as_stream(stream)>>>(
        input, weight, weight_type, output, hidden, eps, one_plus, 0);
}

void launch_rms_norm_to_lowp(const float *input, const void *weight, int weight_type, uint16_t *output,
                             int rows, int hidden, float eps, bool one_plus,
                             int lowp_type, void *stream) {
    ScopedGpuTimer timer("rms_norm_to_lowp", as_stream(stream));
    rms_norm_kernel<uint16_t><<<rows, kBlock, kBlock * sizeof(float), as_stream(stream)>>>(
        input, weight, weight_type, output, hidden, eps, one_plus, lowp_type);
}

void launch_argmax(const float *logits, int vocab,
                   float *block_values, int *block_indices,
                   float *best_value, int *best_index, void *stream) {
    ScopedGpuTimer timer("argmax", as_stream(stream));
    cudaStream_t s = as_stream(stream);
    int blocks = grid_for(vocab);
    if (blocks > 1024) blocks = 1024; // 上限，避免 block_values 过大
    size_t smem = kBlock * (sizeof(float) + sizeof(int));
    argmax_block_kernel<<<blocks, kBlock, smem, s>>>(logits, vocab, block_values, block_indices);
    argmax_final_kernel<<<1, kBlock, smem, s>>>(block_values, block_indices, blocks, best_value, best_index);
}

// ---- full attention ----

void launch_full_attention_q(const float *q_and_gate, const uint16_t *q_norm_weight,
                             float *q, float *gate, int n_heads, int head_dim, int pos,
                             float rope_theta, float partial_rotary_factor, float eps, void *stream) {
    ScopedGpuTimer timer("fullattn.rope_q", as_stream(stream));
    full_attention_q_kernel<<<n_heads, kBlock, 0, as_stream(stream)>>>(
        q_and_gate, q_norm_weight, q, gate, n_heads, head_dim, pos,
        rope_theta, partial_rotary_factor, eps);
}

void launch_full_attention_q_batch(const float *q_and_gate, const uint16_t *q_norm_weight,
                                   float *q, float *gate, int tokens, int n_heads, int head_dim,
                                   int start_pos, float rope_theta, float partial_rotary_factor,
                                   float eps, void *stream) {
    ScopedGpuTimer timer("fullattn.rope_q", as_stream(stream));
    full_attention_q_batch_kernel<<<tokens * n_heads, kBlock, 0, as_stream(stream)>>>(
        q_and_gate, q_norm_weight, q, gate, tokens, n_heads, head_dim, start_pos,
        rope_theta, partial_rotary_factor, eps);
}

void launch_full_attention_kv(const float *k_in, const float *v_in, const uint16_t *k_norm_weight,
                              float *key_cache, float *value_cache, int kv_heads, int head_dim,
                              int max_seq_len, int pos, float rope_theta, float partial_rotary_factor,
                              float eps, void *stream) {
    ScopedGpuTimer timer("fullattn.rope_kv", as_stream(stream));
    full_attention_kv_kernel<<<kv_heads, kBlock, 0, as_stream(stream)>>>(
        k_in, v_in, k_norm_weight, key_cache, value_cache, kv_heads, head_dim, max_seq_len, pos,
        rope_theta, partial_rotary_factor, eps);
}

void launch_full_attention_kv_batch(const float *k_in, const float *v_in, const uint16_t *k_norm_weight,
                                    float *key_cache, float *value_cache, int tokens, int kv_heads,
                                    int head_dim, int max_seq_len, int start_pos, float rope_theta,
                                    float partial_rotary_factor, float eps, void *stream) {
    ScopedGpuTimer timer("fullattn.rope_kv", as_stream(stream));
    full_attention_kv_batch_kernel<<<tokens * kv_heads, kBlock, 0, as_stream(stream)>>>(
        k_in, v_in, k_norm_weight, key_cache, value_cache, tokens, kv_heads, head_dim, max_seq_len,
        start_pos, rope_theta, partial_rotary_factor, eps);
}

void launch_full_attention_attend(const float *q, const float *gate, const float *key_cache,
                                  const float *value_cache, float *attn, int n_heads, int kv_heads,
                                  int head_dim, int max_seq_len, int pos, void *stream) {
    ScopedGpuTimer timer("fullattn.attend", as_stream(stream));
    size_t smem = (static_cast<size_t>(pos + 1) + kBlock) * sizeof(float);
    full_attention_attend_kernel<<<n_heads, kBlock, smem, as_stream(stream)>>>(
        q, gate, key_cache, value_cache, attn, n_heads, kv_heads, head_dim, max_seq_len, pos);
}

void launch_full_attention_attend_batch(const float *q, const float *gate, const float *key_cache,
                                        const float *value_cache, float *attn, int tokens, int n_heads,
                                        int kv_heads, int head_dim, int max_seq_len, int start_pos,
                                        void *stream) {
    ScopedGpuTimer timer("fullattn.attend", as_stream(stream));
    // shared 大小需覆盖本段最大位置的 scores。
    size_t max_pos = static_cast<size_t>(start_pos + tokens - 1);
    size_t smem = (max_pos + 1 + kBlock) * sizeof(float);
    full_attention_attend_batch_kernel<<<tokens * n_heads, kBlock, smem, as_stream(stream)>>>(
        q, gate, key_cache, value_cache, attn, tokens, n_heads, kv_heads, head_dim, max_seq_len, start_pos);
}

// ---- linear attention ----

void launch_linear_attention_conv(const float *mixed, const uint16_t *conv_weight, float *conv_state,
                                  float *conv_out, int conv_dim, int kernel, void *stream) {
    ScopedGpuTimer timer("linattn.conv", as_stream(stream));
    linear_attention_conv_kernel<<<grid_for(conv_dim), kBlock, 0, as_stream(stream)>>>(
        mixed, conv_weight, conv_state, conv_out, conv_dim, kernel);
}

void launch_linear_attention_conv_batch(const float *mixed, const uint16_t *conv_weight,
                                        float *conv_state, float *conv_out, int tokens, int conv_dim,
                                        int kernel, void *stream) {
    ScopedGpuTimer timer("linattn.conv", as_stream(stream));
    linear_attention_conv_batch_kernel<<<grid_for(conv_dim), kBlock, 0, as_stream(stream)>>>(
        mixed, conv_weight, conv_state, conv_out, tokens, conv_dim, kernel);
}

void launch_linear_attention_recurrent(const float *conv_out, const float *z, const float *b,
                                       const float *a, const float *a_log, const uint16_t *dt_bias,
                                       const float *norm_weight, float *recurrent_state, float *gated,
                                       int key_heads, int value_heads, int k_dim, int v_dim,
                                       float eps, void *stream) {
    ScopedGpuTimer timer("linattn.recurrent", as_stream(stream));
    size_t smem = (static_cast<size_t>(2 * k_dim + 2 * v_dim) + 2 * kBlock) * sizeof(float);
    linear_attention_recurrent_kernel<<<value_heads, kBlock, smem, as_stream(stream)>>>(
        conv_out, z, b, a, a_log, dt_bias, norm_weight, recurrent_state, gated,
        key_heads, value_heads, k_dim, v_dim, eps);
}

void launch_linear_attention_recurrent_batch(const float *conv_out, const float *z, const float *b,
                                             const float *a, const float *a_log, const uint16_t *dt_bias,
                                             const float *norm_weight, float *recurrent_state, float *gated,
                                             int tokens, int key_heads, int value_heads, int k_dim, int v_dim,
                                             float eps, void *stream) {
    ScopedGpuTimer timer("linattn.recurrent", as_stream(stream));
    size_t smem = (static_cast<size_t>(2 * k_dim + 2 * v_dim) + 2 * kBlock) * sizeof(float);
    linear_attention_recurrent_batch_kernel<<<value_heads, kBlock, smem, as_stream(stream)>>>(
        conv_out, z, b, a, a_log, dt_bias, norm_weight, recurrent_state, gated, tokens,
        key_heads, value_heads, k_dim, v_dim, eps);
}

void launch_float_to_lowp(const float *input, uint16_t *output, int n, int lowp_type, void *stream) {
    ScopedGpuTimer timer("float_to_lowp", as_stream(stream));
    float_to_lowp_kernel<<<grid_for(n), kBlock, 0, as_stream(stream)>>>(input, output, n, lowp_type);
}
