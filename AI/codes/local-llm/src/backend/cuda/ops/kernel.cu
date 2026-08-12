//
// Created by zhangyoulun on 9/8/2026.
//

#include "kernel.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

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

// ---- Q4_K 反量化 ----
// 从 12 字节 scales 解出第 j 个 sub-block 的 6-bit scale/min（与 llama.cpp get_scale_min_k4 一致）。
__device__ inline void q4k_scale_min(int j, const uint8_t *scales, uint8_t &sc, uint8_t &m) {
    if (j < 4) {
        sc = scales[j] & 63;
        m = scales[j + 4] & 63;
    } else {
        sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

// 每个 CUDA block 处理一个 Q4_K super-block（256 元素）。128 线程，每线程出 2 个元素。
// super-block 布局：d(f16) dmin(f16) scales[12] qs[128]。
__global__ void dequantize_q4k_to_f16_kernel(const uint8_t *src, uint16_t *out, int64_t nblocks) {
    const int64_t blk = blockIdx.x;
    if (blk >= nblocks) return;

    const uint8_t *base = src + blk * 144;               // 144 = sizeof(BlockQ4K)
    const uint16_t d_h = *reinterpret_cast<const uint16_t *>(base);
    const uint16_t dmin_h = *reinterpret_cast<const uint16_t *>(base + 2);
    const float d = __half2float(__ushort_as_half(d_h));
    const float dmin = __half2float(__ushort_as_half(dmin_h));
    const uint8_t *scales = base + 4;
    const uint8_t *qs = base + 16;                        // 4 + 12

    // 128 线程覆盖 128 字节 qs；线程 t 处理 qs[t] 的低/高 4-bit（分属相邻 32 元素的两 sub-block 对）。
    const int t = threadIdx.x; // 0..127
    if (t >= 128) return;

    // qs 每 32 字节服务一对 sub-block（j, j+1）：低 4-bit -> sub-block j，高 4-bit -> sub-block j+1。
    const int pair = t / 32;          // 0..3 -> sub-block 对 (2*pair, 2*pair+1)
    const int inner = t % 32;         // sub-block 内元素下标 0..31
    const int j_lo = 2 * pair;
    const int j_hi = 2 * pair + 1;

    uint8_t sc_lo, m_lo, sc_hi, m_hi;
    q4k_scale_min(j_lo, scales, sc_lo, m_lo);
    q4k_scale_min(j_hi, scales, sc_hi, m_hi);

    const uint8_t q = qs[t];
    const float v_lo = d * sc_lo * (q & 0x0F) - dmin * m_lo;
    const float v_hi = d * sc_hi * (q >> 4) - dmin * m_hi;

    // 输出布局：sub-block j_lo 占 [j_lo*32, j_lo*32+32)，j_hi 紧随其后。
    uint16_t *out_blk = out + blk * 256;
    out_blk[j_lo * 32 + inner] = __half_as_ushort(__float2half(v_lo));
    out_blk[j_hi * 32 + inner] = __half_as_ushort(__float2half(v_hi));
}

// ---- Q8_0 反量化 ----
// block=32 元素，布局：f16 d + int8 qs[32] = 34 字节。y = d * qs[i]。
__global__ void dequantize_q80_to_f16_kernel(const uint8_t *src, uint16_t *out, int64_t nblocks) {
    const int64_t blk = blockIdx.x;
    if (blk >= nblocks) return;
    const uint8_t *base = src + blk * 34;
    const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(base)));
    const int8_t *qs = reinterpret_cast<const int8_t *>(base + 2);
    const int t = threadIdx.x;
    if (t >= 32) return;
    const float y = d * static_cast<float>(qs[t]);
    out[blk * 32 + t] = __half_as_ushort(__float2half(y));
}

// ---- Q5_0 反量化 ----
// block=32 元素，布局：f16 d + uint8 qh[4] + uint8 qs[16] = 22 字节。
// 5-bit：低 4-bit 在 qs，高 1-bit 在 qh（32-bit）。x=(q&0x1F)-16；y=d*x。
// j∈[0,16): q0=(qs[j]&0xF)|(((qh>>j)&1)<<4) -> 元素 j；q1=(qs[j]>>4)|(((qh>>(j+16))&1)<<4) -> 元素 j+16。
__global__ void dequantize_q50_to_f16_kernel(const uint8_t *src, uint16_t *out, int64_t nblocks) {
    const int64_t blk = blockIdx.x;
    if (blk >= nblocks) return;
    const uint8_t *base = src + blk * 22;
    const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(base)));
    uint32_t qh;
    // qh 4 字节 little-endian
    qh = base[2] | (base[3] << 8) | (base[4] << 16) | (static_cast<uint32_t>(base[5]) << 24);
    const uint8_t *qs = base + 6;
    const int j = threadIdx.x;
    if (j >= 16) return;
    const int xh0 = ((qh >> j) & 1) << 4;
    const int xh1 = ((qh >> (j + 16)) & 1) << 4;
    const int q0 = (qs[j] & 0x0F) | xh0;
    const int q1 = (qs[j] >> 4) | xh1;
    const float y0 = d * static_cast<float>(q0 - 16);
    const float y1 = d * static_cast<float>(q1 - 16);
    uint16_t *out_blk = out + blk * 32;
    out_blk[j] = __half_as_ushort(__float2half(y0));
    out_blk[j + 16] = __half_as_ushort(__float2half(y1));
}

// ---- Q6_K 反量化 ----
// super-block=256 元素，布局：uint8 ql[128] + uint8 qh[64] + int8 scales[16] + f16 d = 210 字节。
// 与 llama.cpp dequantize_row_q6_K 一致：256 元素分 2 个 half（每 128），每 half 内 4 组 32。
__global__ void dequantize_q6k_to_f16_kernel(const uint8_t *src, uint16_t *out, int64_t nblocks) {
    const int64_t blk = blockIdx.x;
    if (blk >= nblocks) return;
    const uint8_t *base = src + blk * 210;
    const uint8_t *ql = base;
    const uint8_t *qh = base + 128;
    const int8_t *scales = reinterpret_cast<const int8_t *>(base + 192);
    const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(base + 208)));

    // 每线程处理一个 l∈[0,32)，覆盖两个 half（n=0,128 起点）。参照 llama.cpp 内层循环。
    const int l = threadIdx.x;
    if (l >= 32) return;
    uint16_t *y = out + blk * 256;
    // 两个 128-元素 half（n = 0 与 128）
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int n = half * 128;
        const int is = l / 16;
        const uint8_t *qlp = ql + (n / 2);
        const uint8_t *qhp = qh + (n / 4);
        const int8_t *sc = scales + (n / 16);
        const int8_t q1 = static_cast<int8_t>((qlp[l +  0] & 0xF) | (((qhp[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = static_cast<int8_t>((qlp[l + 32] & 0xF) | (((qhp[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = static_cast<int8_t>((qlp[l +  0] >> 4)  | (((qhp[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = static_cast<int8_t>((qlp[l + 32] >> 4)  | (((qhp[l] >> 6) & 3) << 4)) - 32;
        y[n + l +  0] = __half_as_ushort(__float2half(d * sc[is + 0] * static_cast<float>(q1)));
        y[n + l + 32] = __half_as_ushort(__float2half(d * sc[is + 2] * static_cast<float>(q2)));
        y[n + l + 64] = __half_as_ushort(__float2half(d * sc[is + 4] * static_cast<float>(q3)));
        y[n + l + 96] = __half_as_ushort(__float2half(d * sc[is + 6] * static_cast<float>(q4)));
    }
}

// ---- F32 -> f16 直转（GGUF F32 权重上传时用）----
__global__ void f32_to_f16_copy_kernel(const float *src, uint16_t *out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __half_as_ushort(__float2half(src[i]));
}

// ================= MLA（多头潜在注意力）=================
// 解耦 RoPE：对 rope 段做 GPT-NeoX 风格旋转（对 (i, i+half) 配对）。
// inv_freq[half] 为预计算好的频率（含 YARN 缩放），host 侧一次算好。
__device__ inline void mla_rope_inplace(float *vec, int rope_dim, int pos, const float *inv_freq) {
    const int half = rope_dim / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float angle = static_cast<float>(pos) * inv_freq[i];
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float x1 = vec[i];
        const float x2 = vec[i + half];
        vec[i] = x1 * c - x2 * s;
        vec[i + half] = x2 * c + x1 * s;
    }
}

// kv_a 处理：latent 段 RMSNorm（无 weight one_plus，标准 RMSNorm），k_rope 段 RoPE，
// 写入 kv_cache 布局 [max_seq_len, kv_lora + qk_rope]。每个 block 处理一个 (tok)。
__device__ inline void mla_kv_a_row(const float *kv_a, const float *kv_a_norm_weight,
                                    float *kv_cache, int kv_lora, int qk_rope, int pos,
                                    const float *inv_freq, float eps, float *partial, float *cache_row) {
    const int tid = threadIdx.x;
    // latent RMSNorm
    float ss = 0.0f;
    for (int d = tid; d < kv_lora; d += blockDim.x) {
        const float v = kv_a[d];
        ss += v * v;
    }
    partial[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / static_cast<float>(kv_lora) + eps);
    for (int d = tid; d < kv_lora; d += blockDim.x) {
        cache_row[d] = kv_a[d] * scale * kv_a_norm_weight[d];
    }
    // k_rope 段先拷贝
    for (int d = tid; d < qk_rope; d += blockDim.x) {
        cache_row[kv_lora + d] = kv_a[kv_lora + d];
    }
    __syncthreads();
    // 对 k_rope 段做 RoPE
    mla_rope_inplace(cache_row + kv_lora, qk_rope, pos, inv_freq);
    __syncthreads();
    const int total = kv_lora + qk_rope;
    for (int d = tid; d < total; d += blockDim.x) {
        kv_cache[static_cast<size_t>(pos) * total + d] = cache_row[d];
    }
}

__global__ void mla_kv_a_kernel(const float *kv_a, const float *kv_a_norm_weight, float *kv_cache,
                                int kv_lora, int qk_rope, int pos, const float *inv_freq, float eps) {
    extern __shared__ float shared[];
    float *partial = shared;
    float *cache_row = partial + blockDim.x;
    mla_kv_a_row(kv_a, kv_a_norm_weight, kv_cache, kv_lora, qk_rope, pos, inv_freq, eps,
                 partial, cache_row);
}

__global__ void mla_kv_a_batch_kernel(const float *kv_a, const float *kv_a_norm_weight, float *kv_cache,
                                      int tokens, int kv_lora, int qk_rope, int start_pos,
                                      const float *inv_freq, float eps) {
    extern __shared__ float shared[];
    float *partial = shared;
    float *cache_row = partial + blockDim.x;
    const int tok = blockIdx.x;
    if (tok >= tokens) return;
    const int total = kv_lora + qk_rope;
    mla_kv_a_row(kv_a + static_cast<size_t>(tok) * total, kv_a_norm_weight, kv_cache,
                 kv_lora, qk_rope, start_pos + tok, inv_freq, eps, partial, cache_row);
}

// q 的 rope 段旋转：q 布局 [n_heads, qk_nope + qk_rope]，只旋转后 qk_rope 维。每 block 处理一个 (tok,head)。
__global__ void mla_rope_q_kernel(float *q, int n_heads, int qk_nope, int qk_rope, int pos,
                                  const float *inv_freq) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int qk_head = qk_nope + qk_rope;
    float *qh = q + static_cast<size_t>(h) * qk_head + qk_nope;
    mla_rope_inplace(qh, qk_rope, pos, inv_freq);
}

__global__ void mla_rope_q_batch_kernel(float *q, int tokens, int n_heads, int qk_nope, int qk_rope,
                                        int start_pos, const float *inv_freq) {
    const int block = blockIdx.x;
    const int tok = block / n_heads;
    const int h = block - tok * n_heads;
    if (tok >= tokens) return;
    const int qk_head = qk_nope + qk_rope;
    float *qh = q + (static_cast<size_t>(tok) * n_heads + h) * qk_head + qk_nope;
    mla_rope_inplace(qh, qk_rope, start_pos + tok, inv_freq);
}

// MLA attend：每 block 处理一个 (tok, head)，pos 为该 token 绝对位置。
//   q 布局 [n_heads, qk_nope+qk_rope]；q_nope=前 qk_nope，q_rope=后 qk_rope。
//   kv_b_out 布局 [seq, n_heads*(qk_nope+v_head)]：每 (t,head) 的 k_nope[qk_nope] 后接 v[v_head]。
//   k_rope 从 kv_cache[t] 的 [kv_lora, kv_lora+qk_rope) 段取（所有 head 共享）。
//   score = (q_nope·k_nope + q_rope·k_rope) * softmax_scale，causal softmax，加权 v。
//   softmax_scale = 1/sqrt(qk_head) * mscale^2（YARN）。
__device__ inline void mla_attend_head(const float *q, const float *kv_b_out, const float *kv_cache,
                                       float *attn, int n_heads, int qk_nope, int qk_rope, int v_head,
                                       int kv_lora, int pos, float softmax_scale, size_t q_off, size_t out_off,
                                       float *scores, float *partial) {
    const int tid = threadIdx.x;
    const int qk_head = qk_nope + qk_rope;
    const int kvb_stride = qk_nope + v_head;
    const int total_kv = kv_lora + qk_rope;
    const int h = static_cast<int>(q_off / qk_head) % n_heads;
    const float scale = softmax_scale;
    const float *qh = q + q_off;
    const float *q_nope = qh;
    const float *q_rope = qh + qk_nope;

    float local_max = -INFINITY;
    for (int t = tid; t <= pos; t += blockDim.x) {
        const float *k_nope = kv_b_out + (static_cast<size_t>(t) * n_heads + h) * kvb_stride;
        const float *k_rope = kv_cache + static_cast<size_t>(t) * total_kv + kv_lora;
        float dot = 0.0f;
        for (int d = 0; d < qk_nope; ++d) dot += q_nope[d] * k_nope[d];
        for (int d = 0; d < qk_rope; ++d) dot += q_rope[d] * k_rope[d];
        const float sc = dot * scale;
        scores[t] = sc;
        local_max = fmaxf(local_max, sc);
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
    for (int d = tid; d < v_head; d += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            const float prob = scores[t] / denom;
            const float *v = kv_b_out + (static_cast<size_t>(t) * n_heads + h) * kvb_stride + qk_nope;
            sum += prob * v[d];
        }
        out[d] = sum;
    }
}

__global__ void mla_attend_kernel(const float *q, const float *kv_b_out, const float *kv_cache,
                                  float *attn, int n_heads, int qk_nope, int qk_rope, int v_head,
                                  int kv_lora, int pos, float softmax_scale) {
    extern __shared__ float shared[];
    float *scores = shared;
    float *partial = scores + pos + 1;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int qk_head = qk_nope + qk_rope;
    const size_t q_off = static_cast<size_t>(h) * qk_head;
    const size_t out_off = static_cast<size_t>(h) * v_head;
    mla_attend_head(q, kv_b_out, kv_cache, attn, n_heads, qk_nope, qk_rope, v_head, kv_lora,
                    pos, softmax_scale, q_off, out_off, scores, partial);
}

__global__ void mla_attend_batch_kernel(const float *q, const float *kv_b_out, const float *kv_cache,
                                        float *attn, int tokens, int n_heads, int qk_nope, int qk_rope,
                                        int v_head, int kv_lora, int start_pos, float softmax_scale) {
    extern __shared__ float shared[];
    const int block = blockIdx.x;
    const int tok = block / n_heads;
    const int h = block - tok * n_heads;
    if (tok >= tokens) return;
    const int pos = start_pos + tok;
    float *scores = shared;
    float *partial = scores + pos + 1;
    const int qk_head = qk_nope + qk_rope;
    const size_t q_off = (static_cast<size_t>(tok) * n_heads + h) * qk_head;
    const size_t out_off = (static_cast<size_t>(tok) * n_heads + h) * v_head;
    mla_attend_head(q, kv_b_out, kv_cache, attn, n_heads, qk_nope, qk_rope, v_head, kv_lora,
                    pos, softmax_scale, q_off, out_off, scores, partial);
}

// ================= MoE 路由 =================
// 每个 block 处理一个 token：对 router_logits 做 sigmoid? DeepSeek-V2 用 softmax over all experts，
// 再 top-k，然后对被选 k 个权重再归一化并乘 routed_scaling。此处遵循 V2-Lite（scoring_func=softmax）。
__global__ void moe_router_topk_kernel(const float *router_logits, int *top_idx, float *top_w,
                                       int tokens, int n_experts, int k, float routed_scaling) {
    const int tok = blockIdx.x;
    if (tok >= tokens) return;
    if (threadIdx.x != 0) return; // 单线程处理（n_experts=64, k=6，规模小）
    const float *logits = router_logits + static_cast<size_t>(tok) * n_experts;

    // softmax over all experts（数值稳定）
    float maxv = -INFINITY;
    for (int e = 0; e < n_experts; ++e) maxv = fmaxf(maxv, logits[e]);
    float denom = 0.0f;
    for (int e = 0; e < n_experts; ++e) denom += __expf(logits[e] - maxv);

    // top-k 选择（在 softmax 概率上选，等价于在 logits 上选）。
    // V2-Lite: norm_topk_prob=false，不对被选权重再归一化，只乘 routed_scaling_factor(=1.0)。
    int *idx_out = top_idx + static_cast<size_t>(tok) * k;
    float *w_out = top_w + static_cast<size_t>(tok) * k;
    bool used[64]; // n_experts <= 64
    for (int e = 0; e < n_experts; ++e) used[e] = false;
    for (int r = 0; r < k; ++r) {
        float best = -INFINITY;
        int best_e = -1;
        for (int e = 0; e < n_experts; ++e) {
            if (used[e]) continue;
            if (logits[e] > best) { best = logits[e]; best_e = e; }
        }
        used[best_e] = true;
        const float prob = __expf(logits[best_e] - maxv) / denom;
        idx_out[r] = best_e;
        w_out[r] = prob * routed_scaling;
    }
}

// 专家输出加权累加：out += weight * expert_out。
__global__ void moe_accumulate_kernel(const float *expert_out, float weight, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] += weight * expert_out[i];
}

} // namespace

// ================= launch 封装 =================

void launch_add(const float *a, const float *b, float *out, int n, void *stream) {
    ScopedGpuTimer timer("add", as_stream(stream));
    add_kernel<<<grid_for(n), kBlock, 0, as_stream(stream)>>>(a, b, out, n);
}

void launch_silu_mul(const float *gate, const float *up, float *out, int n, void *stream) {
    ScopedGpuTimer timer("silu_mul", as_stream(stream));
    silu_mul_kernel<<<grid_for(n), kBlock, 0, as_stream(stream)>>>(gate, up, out, n);
}

void launch_embedding_lookup(const uint16_t *table, const int *token_ids, float *output,
                             int tokens, int vocab, int hidden, int lowp_type, void *stream) {
    ScopedGpuTimer timer("embedding_lookup", as_stream(stream));
    embedding_lookup_kernel<<<tokens, kBlock, 0, as_stream(stream)>>>(
        table, token_ids, output, vocab, hidden, lowp_type);
}

void launch_rms_norm(const float *input, const void *weight, int weight_type, float *output,
                     int rows, int hidden, float eps, bool one_plus, void *stream) {
    ScopedGpuTimer timer("rms_norm", as_stream(stream));
    rms_norm_kernel<float><<<rows, kBlock, kBlock * sizeof(float), as_stream(stream)>>>(
        input, weight, weight_type, output, hidden, eps, one_plus, 0);
}

// ---- full attention ----

void launch_full_attention_q(const float *q_and_gate, const uint16_t *q_norm_weight,
                             float *q, float *gate, int n_heads, int head_dim, int pos,
                             float rope_theta, float partial_rotary_factor, float eps, void *stream) {
    ScopedGpuTimer timer("full_attention_q", as_stream(stream));
    full_attention_q_kernel<<<n_heads, kBlock, 0, as_stream(stream)>>>(
        q_and_gate, q_norm_weight, q, gate, n_heads, head_dim, pos,
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
                              float *key_cache, float *value_cache, int kv_heads, int head_dim,
                              int max_seq_len, int pos, float rope_theta, float partial_rotary_factor,
                              float eps, void *stream) {
    ScopedGpuTimer timer("full_attention_kv", as_stream(stream));
    full_attention_kv_kernel<<<kv_heads, kBlock, 0, as_stream(stream)>>>(
        k_in, v_in, k_norm_weight, key_cache, value_cache, kv_heads, head_dim, max_seq_len, pos,
        rope_theta, partial_rotary_factor, eps);
}

void launch_full_attention_kv_batch(const float *k_in, const float *v_in, const uint16_t *k_norm_weight,
                                    float *key_cache, float *value_cache, int tokens, int kv_heads,
                                    int head_dim, int max_seq_len, int start_pos, float rope_theta,
                                    float partial_rotary_factor, float eps, void *stream) {
    ScopedGpuTimer timer("full_attention_kv_batch", as_stream(stream));
    full_attention_kv_batch_kernel<<<tokens * kv_heads, kBlock, 0, as_stream(stream)>>>(
        k_in, v_in, k_norm_weight, key_cache, value_cache, tokens, kv_heads, head_dim, max_seq_len,
        start_pos, rope_theta, partial_rotary_factor, eps);
}

void launch_full_attention_attend(const float *q, const float *gate, const float *key_cache,
                                  const float *value_cache, float *attn, int n_heads, int kv_heads,
                                  int head_dim, int max_seq_len, int pos, void *stream) {
    ScopedGpuTimer timer("full_attention_attend", as_stream(stream));
    size_t smem = (static_cast<size_t>(pos + 1) + kBlock) * sizeof(float);
    full_attention_attend_kernel<<<n_heads, kBlock, smem, as_stream(stream)>>>(
        q, gate, key_cache, value_cache, attn, n_heads, kv_heads, head_dim, max_seq_len, pos);
}

void launch_full_attention_attend_batch(const float *q, const float *gate, const float *key_cache,
                                        const float *value_cache, float *attn, int tokens, int n_heads,
                                        int kv_heads, int head_dim, int max_seq_len, int start_pos,
                                        void *stream) {
    ScopedGpuTimer timer("full_attention_attend_batch", as_stream(stream));
    // shared 大小需覆盖本段最大位置的 scores。
    size_t max_pos = static_cast<size_t>(start_pos + tokens - 1);
    size_t smem = (max_pos + 1 + kBlock) * sizeof(float);
    full_attention_attend_batch_kernel<<<tokens * n_heads, kBlock, smem, as_stream(stream)>>>(
        q, gate, key_cache, value_cache, attn, tokens, n_heads, kv_heads, head_dim, max_seq_len, start_pos);
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
                                       const float *norm_weight, float *recurrent_state, float *gated,
                                       int key_heads, int value_heads, int k_dim, int v_dim,
                                       float eps, void *stream) {
    ScopedGpuTimer timer("linear_attention_recurrent", as_stream(stream));
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
    ScopedGpuTimer timer("linear_attention_recurrent_batch", as_stream(stream));
    size_t smem = (static_cast<size_t>(2 * k_dim + 2 * v_dim) + 2 * kBlock) * sizeof(float);
    linear_attention_recurrent_batch_kernel<<<value_heads, kBlock, smem, as_stream(stream)>>>(
        conv_out, z, b, a, a_log, dt_bias, norm_weight, recurrent_state, gated, tokens,
        key_heads, value_heads, k_dim, v_dim, eps);
}

void launch_float_to_lowp(const float *input, uint16_t *output, int n, int lowp_type, void *stream) {
    ScopedGpuTimer timer("float_to_lowp", as_stream(stream));
    float_to_lowp_kernel<<<grid_for(n), kBlock, 0, as_stream(stream)>>>(input, output, n, lowp_type);
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

// ---- MLA ----

void launch_mla_kv_a(const float *kv_a, const float *kv_a_norm_weight, float *kv_cache,
                     int kv_lora, int qk_rope, int max_seq_len, int pos,
                     const float *inv_freq, float eps, void *stream) {
    ScopedGpuTimer timer("mla_kv_a", as_stream(stream));
    (void) max_seq_len;
    size_t smem = (static_cast<size_t>(kBlock) + kv_lora + qk_rope) * sizeof(float);
    mla_kv_a_kernel<<<1, kBlock, smem, as_stream(stream)>>>(
        kv_a, kv_a_norm_weight, kv_cache, kv_lora, qk_rope, pos, inv_freq, eps);
}

void launch_mla_kv_a_batch(const float *kv_a, const float *kv_a_norm_weight, float *kv_cache,
                           int tokens, int kv_lora, int qk_rope, int max_seq_len, int start_pos,
                           const float *inv_freq, float eps, void *stream) {
    ScopedGpuTimer timer("mla_kv_a_batch", as_stream(stream));
    (void) max_seq_len;
    size_t smem = (static_cast<size_t>(kBlock) + kv_lora + qk_rope) * sizeof(float);
    mla_kv_a_batch_kernel<<<tokens, kBlock, smem, as_stream(stream)>>>(
        kv_a, kv_a_norm_weight, kv_cache, tokens, kv_lora, qk_rope, start_pos, inv_freq, eps);
}

void launch_mla_rope_q(float *q, int n_heads, int qk_nope, int qk_rope, int pos,
                       const float *inv_freq, void *stream) {
    ScopedGpuTimer timer("mla_rope_q", as_stream(stream));
    mla_rope_q_kernel<<<n_heads, kBlock, 0, as_stream(stream)>>>(
        q, n_heads, qk_nope, qk_rope, pos, inv_freq);
}

void launch_mla_rope_q_batch(float *q, int tokens, int n_heads, int qk_nope, int qk_rope,
                             int start_pos, const float *inv_freq, void *stream) {
    ScopedGpuTimer timer("mla_rope_q_batch", as_stream(stream));
    mla_rope_q_batch_kernel<<<tokens * n_heads, kBlock, 0, as_stream(stream)>>>(
        q, tokens, n_heads, qk_nope, qk_rope, start_pos, inv_freq);
}

void launch_mla_attend(const float *q, const float *kv_b_out, const float *kv_cache, float *attn,
                       int n_heads, int qk_nope, int qk_rope, int v_head, int kv_lora,
                       int max_seq_len, int pos, float softmax_scale, void *stream) {
    ScopedGpuTimer timer("mla_attend", as_stream(stream));
    (void) max_seq_len;
    size_t smem = (static_cast<size_t>(pos + 1) + kBlock) * sizeof(float);
    mla_attend_kernel<<<n_heads, kBlock, smem, as_stream(stream)>>>(
        q, kv_b_out, kv_cache, attn, n_heads, qk_nope, qk_rope, v_head, kv_lora, pos, softmax_scale);
}

void launch_mla_attend_batch(const float *q, const float *kv_b_out, const float *kv_cache, float *attn,
                             int tokens, int n_heads, int qk_nope, int qk_rope, int v_head, int kv_lora,
                             int max_seq_len, int start_pos, float softmax_scale, void *stream) {
    ScopedGpuTimer timer("mla_attend_batch", as_stream(stream));
    (void) max_seq_len;
    size_t max_pos = static_cast<size_t>(start_pos + tokens - 1);
    size_t smem = (max_pos + 1 + kBlock) * sizeof(float);
    mla_attend_batch_kernel<<<tokens * n_heads, kBlock, smem, as_stream(stream)>>>(
        q, kv_b_out, kv_cache, attn, tokens, n_heads, qk_nope, qk_rope, v_head, kv_lora,
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
