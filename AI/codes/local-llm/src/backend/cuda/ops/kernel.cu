//
// Created by zhangyoulun on 9/8/2026.
//

#include "kernel.cuh"
#include "kernel_internal.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

constexpr int kBlockConst = 256; // 与 kBlock 一致，供静态 shared 数组使用
constexpr int kMaxHeadDim = 256; // Qwen head_dim = 256

// 把一个低精度元素（bf16 或 f16）读成 float。lowp_type: 0=bf16, 1=f16。
__device__ inline float f16_or_bf16_to_float(uint16_t v, int f16_or_bf16) {
    if (f16_or_bf16 == 1) {
        __half h = __ushort_as_half(v);
        return __half2float(h);
    }
    __nv_bfloat16 b = *reinterpret_cast<const __nv_bfloat16 *>(&v);
    return __bfloat162float(b);
}

// 读第 j 个 norm 权重（gamma）成 float。weight_type: 0=bf16, 1=f16, 2=f32。
// f32 时 weight 实为 float* 的位模式（DeepSeek GGUF 的 *_norm.weight 为 F32），
// 按 j 取对应 float；f16/bf16 时按 uint16 元素取。
__device__ inline float norm_weight_to_float(const uint16_t *weight, int j, int weight_type) {
    if (weight_type == 2) {
        return reinterpret_cast<const float *>(weight)[j];
    }
    return f16_or_bf16_to_float(weight[j], weight_type);
}

// 持久状态（recurrent state / KV cache）可选 fp32 或 bf16 存储：kernel 内一律 float 计算，
// 只在读写显存时做精度转换。定义放在文件前部，供 full attention / linear attention 共用。
template <typename StateT> __device__ inline float state_to_float(StateT v);
template <> __device__ inline float state_to_float<float>(float v) { return v; }
template <> __device__ inline float state_to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename StateT> __device__ inline StateT float_to_state(float v);
template <> __device__ inline float float_to_state<float>(float v) { return v; }
template <> __device__ inline __nv_bfloat16 float_to_state<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ---- 逐元素 ----

// 手写 bf16 GEMV：Y[out_dim] = W[out_dim, in_dim] · X[in_dim]（decode 单 token 的投影专用）。
// W 行主序 bf16（每行 in_dim 个连续元素），X bf16（in_dim），Y f32。
// 每个 warp 负责一个输出行：warp 内 32 lane 沿 in_dim 分工累加点积，再 warp 归约。
// 用 __nv_bfloat162 一次读 2 个元素提高带宽利用（in_dim 为偶数，Qwen 全部满足）。
// 注：曾尝试把 x 协作缓存进 shared 复用，但在本模型 in_dim（2560/9216）下 x 本就常驻 L2，
//     额外的 __syncthreads + shared 写入反而更慢（实测 81→77 tok/s），故保持直读全局。
// 针对 M=1（GEMV、访存瓶颈）场景，避免 cuBLAS 走 tensor-core GEMM tile 在 M=1 时的算力浪费。
__global__ void bf16_gemv_kernel(const uint16_t *weight, const uint16_t *x, float *y,
                                 int out_dim, int in_dim) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= out_dim) return;

    const __nv_bfloat162 *w2 = reinterpret_cast<const __nv_bfloat162 *>(
        weight + static_cast<size_t>(warp_id) * in_dim);
    const __nv_bfloat162 *x2 = reinterpret_cast<const __nv_bfloat162 *>(x);
    const int n2 = in_dim >> 1;  // in_dim/2 个 bf162

    float acc = 0.0f;
    for (int k = lane; k < n2; k += 32) {
        float2 wv = __bfloat1622float2(w2[k]);
        float2 xv = __bfloat1622float2(x2[k]);
        acc += wv.x * xv.x + wv.y * xv.y;
    }
    // warp 内归约
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) y[warp_id] = acc;
}

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
__global__ void embedding_lookup_kernel(const int *input, float *output, const uint16_t *table,
                                        int vocab_size, int hidden_size, int weight_type) {
    int i = blockIdx.x;
    int id = input[i];
    if (id < 0 || id >= vocab_size) id = 0;
    const uint16_t *row = table + static_cast<size_t>(id) * hidden_size;
    float *dst = output + static_cast<size_t>(i) * hidden_size;
    for (int j = threadIdx.x; j < hidden_size; j += blockDim.x) {
        dst[j] = f16_or_bf16_to_float(row[j], weight_type);
    }
}

// ---- RMSNorm ----
// 每个 block 处理一行；block 内先归约平方和，再逐元素缩放。
// y = x / sqrt(mean(x^2 + eps)) * weight
__global__ void rms_norm_kernel(const float *input, float *output, const uint16_t *weight,
                                int weight_type, int hidden_size, float eps, bool one_plus) {
    int row = blockIdx.x; //每个block处理一行，block内多个线程协作处理这一行的hidden个元素
    const float *x = input + static_cast<size_t>(row) * hidden_size; // 指向当前行起始地址

    extern __shared__ float sdata[]; // sdata长度是blockDim.x，即block里的线程数
    // 求平方和，每个线程算一部分
    // 线程按 blockDim.x 步长跨步遍历(grid-stride 风格),每个线程累加自己负责元素的平方,得到局部和 local 。
    float local = 0.0f;
    for (int j = threadIdx.x; j < hidden_size; j += blockDim.x) {
        float v = x[j];
        local += v * v;
    }
    // block内归约求总平方和
    sdata[threadIdx.x] = local; // 各线程把local写入共享内存sdata
    __syncthreads();
    // 经典树形归约parallel reduction：每轮把后半段加到前半段，循环折半
    //  把 sdata[0..blockDim.x-1] 里所有元素求和,最终结果存到 sdata[0]
    // 复杂度 :串行相加要 O(n) 步,树形归约只要 O(log n) 步
    // 前提约束：blockDim.x 必须是 2 的幂
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        // if 控制只有前半段线程参与归约
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    // sdata[0] / hidden_size 即 mean(x²),加 eps 防止除零, rsqrtf 是 1/sqrt(...) 的快速倒数平方根
    float inv_rms = rsqrtf(sdata[0] / hidden_size + eps);

    float *dst = output + static_cast<size_t>(row) * hidden_size;
    for (int j = threadIdx.x; j < hidden_size; j += blockDim.x) {
        float w = norm_weight_to_float(weight, j, weight_type);
        if (one_plus) w += 1.0f;
        float y = x[j] * inv_rms * w;
        dst[j] = y;
    }
}

// ---- 融合 add + RMSNorm ----
// out_residual = x + residual（写回残差累加结果）；out_norm = rmsnorm(out_residual) * weight。
// 把 DecoderLayer 里相邻的 add + rms_norm 融成一个 kernel，省一次 launch 与一次显存往返。
// 每个 block 处理一行；block 内先加法+归约平方和，再逐元素缩放。
__global__ void add_rms_norm_kernel(const float *x, const float *residual, float *out_residual,
                                    float *out_norm, const uint16_t *weight, int weight_type,
                                    int hidden_size, float eps, bool one_plus) {
    int row = blockIdx.x;
    const float *xr = x + static_cast<size_t>(row) * hidden_size;
    const float *rr = residual + static_cast<size_t>(row) * hidden_size;
    float *sr = out_residual + static_cast<size_t>(row) * hidden_size;

    extern __shared__ float sdata[];
    float local = 0.0f;
    // 先做残差加法并写回，同时累加平方和（融合，避免二次读显存）。
    for (int j = threadIdx.x; j < hidden_size; j += blockDim.x) {
        float v = xr[j] + rr[j];
        sr[j] = v;
        local += v * v;
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / hidden_size + eps);

    float *dst = out_norm + static_cast<size_t>(row) * hidden_size;
    for (int j = threadIdx.x; j < hidden_size; j += blockDim.x) {
        float w = norm_weight_to_float(weight, j, weight_type);
        if (one_plus) w += 1.0f;
        dst[j] = sr[j] * inv_rms * w;
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
        const float w = 1.0f + f16_or_bf16_to_float(q_norm_weight[d], 0);
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
                                        float *q, float *gate, int n_heads, int head_dim,
                                        const int *pos_dev,
                                        float rope_theta, float partial_rotary_factor, float eps) {
    const int pos = *pos_dev;
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

template <typename KvT>
__device__ inline void full_attn_kv_head(const float *k_in, const float *v_in,
                                         const uint16_t *k_norm_weight, KvT *key_cache,
                                         KvT *value_cache, int kv_heads, int head_dim, int pos,
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
        const float w = 1.0f + f16_or_bf16_to_float(k_norm_weight[d], 0);
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
        key_cache[cache_base + d] = float_to_state<KvT>(k_local[d]);
        value_cache[cache_base + d] = float_to_state<KvT>(v_in[base + d]);
    }
}

template <typename KvT>
__global__ void full_attention_kv_kernel(const float *k_in, const float *v_in,
                                         const uint16_t *k_norm_weight, KvT *key_cache,
                                         KvT *value_cache, int kv_heads, int head_dim,
                                         int max_seq_len, const int *pos_dev, float rope_theta,
                                         float partial_rotary_factor, float eps) {
    const int pos = *pos_dev;
    __shared__ float partial[kBlockConst];
    __shared__ float k_local[kMaxHeadDim];
    const int h = blockIdx.x;
    if (h >= kv_heads) return;
    const size_t base = static_cast<size_t>(h) * head_dim;
    full_attn_kv_head<KvT>(k_in, v_in, k_norm_weight, key_cache, value_cache, kv_heads, head_dim, pos,
                      rope_theta, partial_rotary_factor, eps, base, partial, k_local);
    (void) max_seq_len;
}
template __global__ void full_attention_kv_kernel<float>(
    const float *, const float *, const uint16_t *, float *, float *, int, int, int,
    const int *, float, float, float);
template __global__ void full_attention_kv_kernel<__nv_bfloat16>(
    const float *, const float *, const uint16_t *, __nv_bfloat16 *, __nv_bfloat16 *, int, int, int,
    const int *, float, float, float);

template <typename KvT>
__global__ void full_attention_kv_batch_kernel(const float *k_in, const float *v_in,
                                               const uint16_t *k_norm_weight, KvT *key_cache,
                                               KvT *value_cache, int tokens, int kv_heads,
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
    full_attn_kv_head<KvT>(k_in, v_in, k_norm_weight, key_cache, value_cache, kv_heads, head_dim,
                      start_pos + tok, rope_theta, partial_rotary_factor, eps, base, partial, k_local);
    (void) max_seq_len;
}
template __global__ void full_attention_kv_batch_kernel<float>(
    const float *, const float *, const uint16_t *, float *, float *, int, int, int, int, int,
    float, float, float);
template __global__ void full_attention_kv_batch_kernel<__nv_bfloat16>(
    const float *, const float *, const uint16_t *, __nv_bfloat16 *, __nv_bfloat16 *, int, int, int, int, int,
    float, float, float);

// causal attention + 输出门控。每个 block 处理一个 (tok, head)，pos 为该 token 绝对位置。
// FlashAttention 式实现：按 tile（每 tile = blockDim 个 key）遍历 KV，用 online softmax
// 增量维护 running max(m) / running sum(l) / 输出累加(acc)，不再 materialize 整行 scores。
// shared 需求为 O(blockDim + head_dim)，与序列长度无关（利于长上下文与 CUDA Graph 稳定 smem）。
// 数学上与传统三趟 softmax 完全等价，数值上用 online rescale 保证稳定。
//   shared 布局：s_p[blockDim]（tile 内 score/概率）| s_acc[head_dim]（输出累加）| s_red[blockDim]（归约）
template <typename KvT>
__device__ inline void full_attn_attend_head(const float *q, const float *gate,
                                             const KvT *key_cache, const KvT *value_cache,
                                             float *attn, int n_heads, int kv_heads, int head_dim,
                                             int pos, size_t q_off, size_t out_off,
                                             float *s_p, float *s_acc, float *s_red) {
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kv_group = n_heads / kv_heads;
    const int h = static_cast<int>(q_off / head_dim) % n_heads;
    const int kh = h / kv_group;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float *qh = q + q_off;

    // running 状态（每个 block 一份，放在 shared 的标量位）：m=running max, l=running sum。
    __shared__ float s_m;
    __shared__ float s_l;
    if (tid == 0) {
        s_m = -INFINITY;
        s_l = 0.0f;
    }
    for (int d = tid; d < head_dim; d += nthreads) s_acc[d] = 0.0f;
    __syncthreads();

    const int n_keys = pos + 1;
    for (int tile = 0; tile < n_keys; tile += nthreads) {
        const int j = tile + tid;  // 本线程负责的 key 位置
        // 1) 每线程算一个 key 的 score = q·k * scale（越界置 -inf）。
        float score = -INFINITY;
        if (j < n_keys) {
            const KvT *key = key_cache + (static_cast<size_t>(j) * kv_heads + kh) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += qh[d] * state_to_float<KvT>(key[d]);
            score = dot * scale;
        }
        s_p[tid] = score;
        // 2) 归约求本 tile 的最大 score。
        s_red[tid] = score;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) s_red[tid] = fmaxf(s_red[tid], s_red[tid + s]);
            __syncthreads();
        }
        const float tile_max = s_red[0];
        __syncthreads();
        // 3) online softmax：更新全局 running max，得到缩放系数。
        const float m_old = s_m;
        const float m_new = fmaxf(m_old, tile_max);
        const float correction = __expf(m_old - m_new);  // m_old=-inf 时为 0，安全
        // 4) 本线程把自己的 score 转成 exp(score - m_new)，存回 s_p。
        const float p = (j < n_keys) ? __expf(s_p[tid] - m_new) : 0.0f;
        s_p[tid] = p;
        // 5) 归约求本 tile 的概率和。
        s_red[tid] = p;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) s_red[tid] += s_red[tid + s];
            __syncthreads();
        }
        const float tile_sum = s_red[0];
        // 6) 缩放旧的 acc/l 并并入本 tile 贡献（acc 按 d 维分工，每线程负责若干 d）。
        const int tile_len = min(nthreads, n_keys - tile);
        for (int d = tid; d < head_dim; d += nthreads) {
            float a = s_acc[d] * correction;
            for (int u = 0; u < tile_len; ++u) {
                const int jj = tile + u;
                const KvT *value = value_cache + (static_cast<size_t>(jj) * kv_heads + kh) * head_dim;
                a += s_p[u] * state_to_float<KvT>(value[d]);
            }
            s_acc[d] = a;
        }
        if (tid == 0) {
            s_l = s_l * correction + tile_sum;
            s_m = m_new;
        }
        __syncthreads();
    }

    // 7) 归一化 + 输出门控。
    const float denom = s_l;
    float *out = attn + out_off;
    for (int d = tid; d < head_dim; d += nthreads) {
        const float g = gate[q_off + d];
        out[d] = (s_acc[d] / denom) * (1.0f / (1.0f + __expf(-g)));
    }
}

template <typename KvT>
__global__ void full_attention_attend_kernel(const float *q, const float *gate,
                                             const KvT *key_cache, const KvT *value_cache,
                                             float *attn, int n_heads, int kv_heads, int head_dim,
                                             int max_seq_len, const int *pos_dev) {
    const int pos = *pos_dev;
    extern __shared__ float shared[];
    // flash 布局（与 seqlen 无关）：s_p[blockDim] | s_acc[head_dim] | s_red[blockDim]
    float *s_p = shared;
    float *s_acc = s_p + blockDim.x;
    float *s_red = s_acc + head_dim;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const size_t off = static_cast<size_t>(h) * head_dim;
    full_attn_attend_head<KvT>(q, gate, key_cache, value_cache, attn, n_heads, kv_heads, head_dim,
                          pos, off, off, s_p, s_acc, s_red);
    (void) max_seq_len;
}
template __global__ void full_attention_attend_kernel<float>(
    const float *, const float *, const float *, const float *, float *, int, int, int, int, const int *);
template __global__ void full_attention_attend_kernel<__nv_bfloat16>(
    const float *, const float *, const __nv_bfloat16 *, const __nv_bfloat16 *, float *, int, int, int, int, const int *);

template <typename KvT>
__global__ void full_attention_attend_batch_kernel(const float *q, const float *gate,
                                                   const KvT *key_cache, const KvT *value_cache,
                                                   float *attn, int tokens, int n_heads, int kv_heads,
                                                   int head_dim, int max_seq_len, int start_pos) {
    extern __shared__ float shared[];
    const int block = blockIdx.x;
    const int tok = block / n_heads;
    const int h = block - tok * n_heads;
    if (tok >= tokens) return;
    const int pos = start_pos + tok;
    // flash 布局（与 seqlen 无关）：s_p[blockDim] | s_acc[head_dim] | s_red[blockDim]
    float *s_p = shared;
    float *s_acc = s_p + blockDim.x;
    float *s_red = s_acc + head_dim;
    const int q_total = n_heads * head_dim;
    const size_t off = static_cast<size_t>(tok) * q_total + static_cast<size_t>(h) * head_dim;
    full_attn_attend_head<KvT>(q, gate, key_cache, value_cache, attn, n_heads, kv_heads, head_dim,
                          pos, off, off, s_p, s_acc, s_red);
    (void) max_seq_len;
}
template __global__ void full_attention_attend_batch_kernel<float>(
    const float *, const float *, const float *, const float *, float *, int, int, int, int, int, int);
template __global__ void full_attention_attend_batch_kernel<__nv_bfloat16>(
    const float *, const float *, const __nv_bfloat16 *, const __nv_bfloat16 *, float *, int, int, int, int, int, int);

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
    for (int i = 0; i < kernel; ++i) sum += f16_or_bf16_to_float(w[i], 0) * row[i];
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
        for (int i = 0; i < kernel; ++i) sum += f16_or_bf16_to_float(w[i], 0) * row[i];
        conv_out[static_cast<size_t>(t) * conv_dim + d] = sum / (1.0f + __expf(-sum));
    }
}

// gated delta 递归单步。shared: q[k_dim] k[k_dim] delta[v_dim] core[v_dim] partial[2*blockDim]。
template <typename StateT>
__device__ inline void linear_attn_recurrent_step(const float *conv_out, const float *z,
                                                  const float *b, const float *a, const float *a_log,
                                                  const uint16_t *dt_bias, const float *norm_weight,
                                                  StateT *rec, float *gated_row, int vh, int kh,
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
    const float dt = a[vh] + f16_or_bf16_to_float(dt_bias[vh], 0);
    const float softplus_dt = dt > 20.0f ? dt : (dt < -20.0f ? __expf(dt) : log1pf(__expf(dt)));
    const float g = -__expf(a_log[vh]) * softplus_dt;
    const float decay = __expf(g);

    const int state_size = k_dim * v_dim;
    for (int i = tid; i < state_size; i += blockDim.x) rec[i] = float_to_state<StateT>(state_to_float<StateT>(rec[i]) * decay);
    __syncthreads();

    for (int vd = tid; vd < v_dim; vd += blockDim.x) {
        float kv_mem = 0.0f;
        for (int kd = 0; kd < k_dim; ++kd) kv_mem += state_to_float<StateT>(rec[static_cast<size_t>(kd) * v_dim + vd]) * k[kd];
        const float value = value_base[static_cast<size_t>(vh) * v_dim + vd];
        delta[vd] = (value - kv_mem) * beta;
    }
    __syncthreads();
    for (int i = tid; i < state_size; i += blockDim.x) {
        const int kd = i / v_dim;
        const int vd = i - kd * v_dim;
        rec[i] = float_to_state<StateT>(state_to_float<StateT>(rec[i]) + k[kd] * delta[vd]);
    }
    __syncthreads();
    for (int vd = tid; vd < v_dim; vd += blockDim.x) {
        float sum = 0.0f;
        for (int kd = 0; kd < k_dim; ++kd) sum += state_to_float<StateT>(rec[static_cast<size_t>(kd) * v_dim + vd]) * q[kd];
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

template <typename StateT>
__global__ void linear_attention_recurrent_kernel(const float *conv_out, const float *z,
                                                  const float *b, const float *a, const float *a_log,
                                                  const uint16_t *dt_bias, const float *norm_weight,
                                                  StateT *recurrent_state, float *gated,
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
    StateT *rec = recurrent_state + static_cast<size_t>(vh) * k_dim * v_dim;
    linear_attn_recurrent_step<StateT>(conv_out, z, b, a, a_log, dt_bias, norm_weight, rec, gated, vh, kh,
                               key_total, k_dim, v_dim, eps, q, k, delta, core, partial);
}
// 显式实例化：decode 用 bf16 state（与 fused kernel 一致），float 版备用。
template __global__ void linear_attention_recurrent_kernel<float>(
    const float *, const float *, const float *, const float *, const float *, const uint16_t *,
    const float *, float *, float *, int, int, int, int, float);
template __global__ void linear_attention_recurrent_kernel<__nv_bfloat16>(
    const float *, const float *, const float *, const float *, const float *, const uint16_t *,
    const float *, __nv_bfloat16 *, float *, int, int, int, int, float);

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
        linear_attn_recurrent_step<float>(token_conv, z_base, b_base, a_base, a_log, dt_bias, norm_weight,
                                   rec, gated_row, vh, kh, key_total, k_dim, v_dim, eps,
                                   q, k, delta, core, partial);
        __syncthreads();
    }
}

// ---- linear attention 融合 kernel（conv1d + gated delta 递归 + 读出，单核） ----
//
// 分块方式：每个 CUDA block 处理 1 个 key_head 及其 repeat 个 value_head。
// 这样 q/k 的 conv 通道（每个 key_head 独有）只被本 block 移位写回 conv_state，
// value 的 conv 通道（每个 value_head 独有）也只在本 block 内处理，彻底避开 conv_state 竞态。
// conv 结果直接留在 shared，不再落显存的 conv_out，省去一次 launch + 一次全量往返。
//
// recurrent_state 用模板参数 StateT 支持 fp32 或 bf16 存储；kernel 内一律用 float 计算，
// 只在读写持久状态时做精度转换。
//
// shared 布局（每 block）：
//   qk_conv[2*k_dim]   -- 本 key_head 的 q、k 卷积输出（各 k_dim）
//   v_conv[v_dim]      -- 当前 value_head 的 value 卷积输出
//   q[k_dim] k[k_dim] delta[v_dim] core[v_dim] partial[2*blockDim]
// 注：state_to_float / float_to_state 已在文件前部（recurrent step 前）定义。

// 单个 value_head 的融合递归步；conv 已在 shared（qk_conv / v_conv）。
template <typename StateT>
__device__ inline void linear_attn_fused_step(const float *qk_conv, const float *v_conv,
                                              const float *z, const float *b, const float *a,
                                              const float *a_log, const uint16_t *dt_bias,
                                              const float *norm_weight, StateT *rec, float *gated_row,
                                              int vh, int k_dim, int v_dim, float eps,
                                              float *q, float *k, float *delta, float *core,
                                              float *partial) {
    const int tid = threadIdx.x;
    float ss_q = 0.0f, ss_k = 0.0f;
    for (int i = tid; i < k_dim; i += blockDim.x) {
        const float qv = qk_conv[i];
        const float kv = qk_conv[k_dim + i];
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
    const float dt = a[vh] + f16_or_bf16_to_float(dt_bias[vh], 0);
    const float softplus_dt = dt > 20.0f ? dt : (dt < -20.0f ? __expf(dt) : log1pf(__expf(dt)));
    const float g = -__expf(a_log[vh]) * softplus_dt;
    const float decay = __expf(g);

    const int state_size = k_dim * v_dim;
    for (int i = tid; i < state_size; i += blockDim.x) rec[i] = float_to_state<StateT>(state_to_float<StateT>(rec[i]) * decay);
    __syncthreads();

    for (int vd = tid; vd < v_dim; vd += blockDim.x) {
        float kv_mem = 0.0f;
        for (int kd = 0; kd < k_dim; ++kd) kv_mem += state_to_float<StateT>(rec[static_cast<size_t>(kd) * v_dim + vd]) * k[kd];
        const float value = v_conv[vd];
        delta[vd] = (value - kv_mem) * beta;
    }
    __syncthreads();
    for (int i = tid; i < state_size; i += blockDim.x) {
        const int kd = i / v_dim;
        const int vd = i - kd * v_dim;
        rec[i] = float_to_state<StateT>(state_to_float<StateT>(rec[i]) + k[kd] * delta[vd]);
    }
    __syncthreads();
    for (int vd = tid; vd < v_dim; vd += blockDim.x) {
        float sum = 0.0f;
        for (int kd = 0; kd < k_dim; ++kd) sum += state_to_float<StateT>(rec[static_cast<size_t>(kd) * v_dim + vd]) * q[kd];
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

// 本 block（1 个 key_head）把 q/k 卷积输出算进 shared，并移位写回 conv_state（q/k 通道本 block 独占）。
// 只处理 q、k（各 k_dim）；value 通道由 fused_conv_v 单独处理（每 value_head 独有）。
__device__ inline void linear_attn_fused_conv_qk(const float *mixed_row, const uint16_t *conv_weight,
                                                 float *conv_state, float *qk_conv,
                                                 int kh, int k_dim, int key_total, int kernel) {
    const int tid = threadIdx.x;
    for (int idx = tid; idx < 2 * k_dim; idx += blockDim.x) {
        int global_d;
        int dst_i;
        if (idx < k_dim) {                 // q
            global_d = kh * k_dim + idx;
            dst_i = idx;
        } else {                           // k
            const int j = idx - k_dim;
            global_d = key_total + kh * k_dim + j;
            dst_i = k_dim + j;
        }
        float *row = conv_state + static_cast<size_t>(global_d) * kernel;
        for (int i = 0; i < kernel - 1; ++i) row[i] = row[i + 1];
        row[kernel - 1] = mixed_row[global_d];
        float sum = 0.0f;
        const uint16_t *w = conv_weight + static_cast<size_t>(global_d) * kernel;
        for (int i = 0; i < kernel; ++i) sum += f16_or_bf16_to_float(w[i], 0) * row[i];
        qk_conv[dst_i] = sum / (1.0f + __expf(-sum));
    }
}

// 处理当前 value_head 的 value 卷积通道（该 value_head 独有），移位写回 conv_state。
__device__ inline void linear_attn_fused_conv_v(const float *mixed_row, const uint16_t *conv_weight,
                                                float *conv_state, float *v_conv,
                                                int vh, int v_dim, int key_total, int kernel) {
    const int tid = threadIdx.x;
    for (int j = tid; j < v_dim; j += blockDim.x) {
        const int global_d = key_total * 2 + vh * v_dim + j;
        float *row = conv_state + static_cast<size_t>(global_d) * kernel;
        for (int i = 0; i < kernel - 1; ++i) row[i] = row[i + 1];
        row[kernel - 1] = mixed_row[global_d];
        float sum = 0.0f;
        const uint16_t *w = conv_weight + static_cast<size_t>(global_d) * kernel;
        for (int i = 0; i < kernel; ++i) sum += f16_or_bf16_to_float(w[i], 0) * row[i];
        v_conv[j] = sum / (1.0f + __expf(-sum));
    }
}

// decode 单 token 融合 kernel。grid=key_heads，每 block 处理该 key_head 的 repeat 个 value_head。
// q/k conv 每 token 只移位/算一次；value conv 各 value_head 一次；recurrent 各 value_head 一次。
// 注：q/k conv 会移位写回 conv_state（副作用），必须每 key_head 每 token 恰好执行一次，
// 因此这里保持 grid=key_heads、由本 block 内串行处理其 repeat 个 value_head。
template <typename StateT>
__global__ void linear_attention_fused_kernel(const float *mixed, const uint16_t *conv_weight,
                                              float *conv_state, const float *z, const float *b,
                                              const float *a, const float *a_log,
                                              const uint16_t *dt_bias, const float *norm_weight,
                                              StateT *recurrent_state, float *gated,
                                              int key_heads, int value_heads, int k_dim, int v_dim,
                                              int kernel, float eps) {
    extern __shared__ float shared[];
    const int kh = blockIdx.x;
    const int repeat = value_heads / key_heads;
    const int key_total = key_heads * k_dim;
    float *qk_conv = shared;               // 2*k_dim
    float *v_conv = qk_conv + 2 * k_dim;   // v_dim
    float *q = v_conv + v_dim;             // k_dim
    float *k = q + k_dim;                  // k_dim
    float *delta = k + k_dim;              // v_dim
    float *core = delta + v_dim;           // v_dim
    float *partial = core + v_dim;         // 2*blockDim
    linear_attn_fused_conv_qk(mixed, conv_weight, conv_state, qk_conv, kh, k_dim, key_total, kernel);
    __syncthreads();
    for (int r = 0; r < repeat; ++r) {
        const int vh = kh * repeat + r;
        linear_attn_fused_conv_v(mixed, conv_weight, conv_state, v_conv, vh, v_dim, key_total, kernel);
        __syncthreads();
        StateT *rec = recurrent_state + static_cast<size_t>(vh) * k_dim * v_dim;
        linear_attn_fused_step<StateT>(qk_conv, v_conv, z, b, a, a_log, dt_bias, norm_weight,
                                       rec, gated, vh, k_dim, v_dim, eps,
                                       q, k, delta, core, partial);
        __syncthreads();
    }
}

// prefill 批量融合 kernel。grid=key_heads，每 block 顺序处理所有 token（递归状态跨 token 累积）。
template <typename StateT>
__global__ void linear_attention_fused_batch_kernel(const float *mixed, const uint16_t *conv_weight,
                                                    float *conv_state, const float *z, const float *b,
                                                    const float *a, const float *a_log,
                                                    const uint16_t *dt_bias, const float *norm_weight,
                                                    StateT *recurrent_state, float *gated, int tokens,
                                                    int key_heads, int value_heads, int k_dim, int v_dim,
                                                    int kernel, float eps) {
    extern __shared__ float shared[];
    const int kh = blockIdx.x;
    const int repeat = value_heads / key_heads;
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    float *qk_conv = shared;
    float *v_conv = qk_conv + 2 * k_dim;
    float *q = v_conv + v_dim;
    float *k = q + k_dim;
    float *delta = k + k_dim;
    float *core = delta + v_dim;
    float *partial = core + v_dim;
    for (int tok = 0; tok < tokens; ++tok) {
        const float *mixed_row = mixed + static_cast<size_t>(tok) * conv_dim;
        const float *z_row = z + static_cast<size_t>(tok) * value_total;
        const float *b_row = b + static_cast<size_t>(tok) * value_heads;
        const float *a_row = a + static_cast<size_t>(tok) * value_heads;
        float *gated_row = gated + static_cast<size_t>(tok) * value_total;
        linear_attn_fused_conv_qk(mixed_row, conv_weight, conv_state, qk_conv, kh, k_dim, key_total, kernel);
        __syncthreads();
        for (int r = 0; r < repeat; ++r) {
            const int vh = kh * repeat + r;
            linear_attn_fused_conv_v(mixed_row, conv_weight, conv_state, v_conv, vh, v_dim, key_total, kernel);
            __syncthreads();
            StateT *rec = recurrent_state + static_cast<size_t>(vh) * k_dim * v_dim;
            linear_attn_fused_step<StateT>(qk_conv, v_conv, z_row, b_row, a_row, a_log, dt_bias, norm_weight,
                                           rec, gated_row, vh, k_dim, v_dim, eps,
                                           q, k, delta, core, partial);
            __syncthreads();
        }
    }
}

// 显式实例化（fp32 与 bf16 状态）供 launch.cu 使用。
template __global__ void linear_attention_fused_kernel<float>(
    const float *, const uint16_t *, float *, const float *, const float *, const float *,
    const float *, const uint16_t *, const float *, float *, float *,
    int, int, int, int, int, float);
template __global__ void linear_attention_fused_kernel<__nv_bfloat16>(
    const float *, const uint16_t *, float *, const float *, const float *, const float *,
    const float *, const uint16_t *, const float *, __nv_bfloat16 *, float *,
    int, int, int, int, int, float);
template __global__ void linear_attention_fused_batch_kernel<float>(
    const float *, const uint16_t *, float *, const float *, const float *, const float *,
    const float *, const uint16_t *, const float *, float *, float *, int,
    int, int, int, int, int, float);
template __global__ void linear_attention_fused_batch_kernel<__nv_bfloat16>(
    const float *, const uint16_t *, float *, const float *, const float *, const float *,
    const float *, const uint16_t *, const float *, __nv_bfloat16 *, float *, int,
    int, int, int, int, int, float);

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

    const uint8_t *base = src + blk * 144; // 144 = sizeof(BlockQ4K)
    const uint16_t d_h = *reinterpret_cast<const uint16_t *>(base);
    const uint16_t dmin_h = *reinterpret_cast<const uint16_t *>(base + 2);
    const float d = __half2float(__ushort_as_half(d_h));
    const float dmin = __half2float(__ushort_as_half(dmin_h));
    const uint8_t *scales = base + 4;
    const uint8_t *qs = base + 16; // 4 + 12

    // 128 线程覆盖 128 字节 qs；线程 t 处理 qs[t] 的低/高 4-bit（分属相邻 32 元素的两 sub-block 对）。
    const int t = threadIdx.x; // 0..127
    if (t >= 128) return;

    // qs 每 32 字节服务一对 sub-block（j, j+1）：低 4-bit -> sub-block j，高 4-bit -> sub-block j+1。
    const int pair = t / 32; // 0..3 -> sub-block 对 (2*pair, 2*pair+1)
    const int inner = t % 32; // sub-block 内元素下标 0..31
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
        const int8_t q1 = static_cast<int8_t>((qlp[l + 0] & 0xF) | (((qhp[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = static_cast<int8_t>((qlp[l + 32] & 0xF) | (((qhp[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = static_cast<int8_t>((qlp[l + 0] >> 4) | (((qhp[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = static_cast<int8_t>((qlp[l + 32] >> 4) | (((qhp[l] >> 6) & 3) << 4)) - 32;
        y[n + l + 0] = __half_as_ushort(__float2half(d * sc[is + 0] * static_cast<float>(q1)));
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

__global__ void f32_to_bf16_copy_kernel(const float *src, uint16_t *out, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __nv_bfloat16 b = __float2bfloat16(src[i]);
    out[i] = *reinterpret_cast<const uint16_t *>(&b);
}

// ================= 量化直算 GEMV（decode 单 token，权重量化常驻，不展开 F16）=================
// 逐 dtype 提供「给定行内线性下标 idx，返回该权重元素的反量化 float 值」的 device 函数。
// 每个 warp 负责一个输出行（out_dim 行），warp 内 32 lane 沿 in_dim 分工点积后归约。
// 与对应 dequantize_* kernel 的块布局严格一致（Q4_K/Q6_K/Q5_0/Q8_0），保证数值等价。

// Q4_K：super-block 256 元素 / 144 字节；布局 d(f16) dmin(f16) scales[12] qs[128]。
__device__ inline float q4k_at(const uint8_t *row_base, int idx) {
    const int64_t blk = idx >> 8;          // idx / 256
    const int in_blk = idx & 255;          // 0..255
    const uint8_t *base = row_base + blk * 144;
    const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(base)));
    const float dmin = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(base + 2)));
    const uint8_t *scales = base + 4;
    const uint8_t *qs = base + 16;
    const int j = in_blk >> 5;             // sub-block 0..7
    const int inner = in_blk & 31;         // 0..31
    // qs 每 32 字节服务一对 sub-block(2p,2p+1)：低4bit->2p, 高4bit->2p+1。
    const int pair = j >> 1;
    const uint8_t q = qs[pair * 32 + inner];
    uint8_t sc, m;
    q4k_scale_min(j, scales, sc, m);
    const int nib = (j & 1) ? (q >> 4) : (q & 0x0F);
    return d * sc * nib - dmin * m;
}

// Q6_K：super-block 256 元素 / 210 字节；布局 ql[128] qh[64] scales[16](int8) d(f16)。
__device__ inline float q6k_at(const uint8_t *row_base, int idx) {
    const int64_t blk = idx >> 8;
    const int in_blk = idx & 255;
    const uint8_t *base = row_base + blk * 210;
    const uint8_t *ql = base;
    const uint8_t *qh = base + 128;
    const int8_t *scales = reinterpret_cast<const int8_t *>(base + 192);
    const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(base + 208)));
    // 还原 dequantize_q6k 的映射：half=in_blk/128, n=half*128, l=(in_blk%128)%32, grp=(in_blk%128)/32。
    const int half = in_blk >> 7;          // 0 或 1
    const int n = half * 128;
    const int within = in_blk & 127;       // 0..127
    const int grp = within >> 5;           // 0..3
    const int l = within & 31;             // 0..31
    const int is = l / 16;
    const uint8_t *qlp = ql + (n / 2);
    const uint8_t *qhp = qh + (n / 4);
    const int8_t *sc = scales + (n / 16);
    int8_t q;
    int scale_idx;
    if (grp == 0) {
        q = static_cast<int8_t>((qlp[l + 0] & 0xF) | (((qhp[l] >> 0) & 3) << 4)) - 32;
        scale_idx = is + 0;
    } else if (grp == 1) {
        q = static_cast<int8_t>((qlp[l + 32] & 0xF) | (((qhp[l] >> 2) & 3) << 4)) - 32;
        scale_idx = is + 2;
    } else if (grp == 2) {
        q = static_cast<int8_t>((qlp[l + 0] >> 4) | (((qhp[l] >> 4) & 3) << 4)) - 32;
        scale_idx = is + 4;
    } else {
        q = static_cast<int8_t>((qlp[l + 32] >> 4) | (((qhp[l] >> 6) & 3) << 4)) - 32;
        scale_idx = is + 6;
    }
    return d * sc[scale_idx] * static_cast<float>(q);
}

// Q5_0：block 32 元素 / 22 字节；布局 d(f16) qh[4] qs[16]。元素 j∈[0,16)->低半，j+16->高半。
__device__ inline float q50_at(const uint8_t *row_base, int idx) {
    const int64_t blk = idx >> 5;          // idx / 32
    const int in_blk = idx & 31;           // 0..31
    const uint8_t *base = row_base + blk * 22;
    const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(base)));
    const uint32_t qh = base[2] | (base[3] << 8) | (base[4] << 16) | (static_cast<uint32_t>(base[5]) << 24);
    const uint8_t *qs = base + 6;
    int q;
    if (in_blk < 16) {
        const int j = in_blk;
        q = (qs[j] & 0x0F) | (((qh >> j) & 1) << 4);
    } else {
        const int j = in_blk - 16;
        q = (qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
    }
    return d * static_cast<float>(q - 16);
}

// Q8_0：block 32 元素 / 34 字节；布局 d(f16) qs[32](int8)。
__device__ inline float q80_at(const uint8_t *row_base, int idx) {
    const int64_t blk = idx >> 5;
    const int in_blk = idx & 31;
    const uint8_t *base = row_base + blk * 34;
    const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(base)));
    const int8_t *qs = reinterpret_cast<const int8_t *>(base + 2);
    return d * static_cast<float>(qs[in_blk]);
}

// 量化直算 GEMM：Y[M,out_dim] = X[M,in_dim] · W[out_dim,in_dim]^T，权重量化常驻、on-the-fly 反量化。
// quant_type：12=Q4_K 14=Q6_K 6=Q5_0 8=Q8_0。row_bytes 为每行量化字节数（out_dim 行等长）。
// X/Y 均为 row-major（x[m*in_dim+k], y[m*out_dim+o]），与 gemm_main 的列主序 [dim,tokens] 一致。
// 每个 warp 负责一个输出行 o，内层循环 M 个 token；权重块按需解量化（M 小，重解成本远低于 F16 展开）。
template <bool F16_OPERANDS>
__device__ inline float quant_gemv_operand(float v) {
    if constexpr (F16_OPERANDS) {
        return __half2float(__float2half(v));
    }
    return v;
}

template <int QUANT_TYPE, bool F16_OPERANDS>
__global__ void quant_gemv_kernel(const uint8_t *weight, size_t row_bytes, const float *x,
                                  float *y, int out_dim, int in_dim, int m) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= out_dim) return;
    const uint8_t *row_base = weight + static_cast<size_t>(warp_id) * row_bytes;
    for (int row = 0; row < m; ++row) {
        const float *xr = x + static_cast<size_t>(row) * in_dim;
        float acc = 0.0f;
        for (int k = lane; k < in_dim; k += 32) {
            float w;
            if (QUANT_TYPE == 12) w = q4k_at(row_base, k);
            else if (QUANT_TYPE == 14) w = q6k_at(row_base, k);
            else if (QUANT_TYPE == 6) w = q50_at(row_base, k);
            else w = q80_at(row_base, k);
            w = quant_gemv_operand<F16_OPERANDS>(w);
            const float xv = quant_gemv_operand<F16_OPERANDS>(xr[k]);
            acc = fmaf(w, xv, acc);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) y[static_cast<size_t>(row) * out_dim + warp_id] = acc;
    }
}

template __global__ void quant_gemv_kernel<12, false>(const uint8_t *, size_t, const float *, float *, int, int, int);
template __global__ void quant_gemv_kernel<14, false>(const uint8_t *, size_t, const float *, float *, int, int, int);
template __global__ void quant_gemv_kernel<6, false>(const uint8_t *, size_t, const float *, float *, int, int, int);
template __global__ void quant_gemv_kernel<8, false>(const uint8_t *, size_t, const float *, float *, int, int, int);
template __global__ void quant_gemv_kernel<12, true>(const uint8_t *, size_t, const float *, float *, int, int, int);
template __global__ void quant_gemv_kernel<14, true>(const uint8_t *, size_t, const float *, float *, int, int, int);
template __global__ void quant_gemv_kernel<6, true>(const uint8_t *, size_t, const float *, float *, int, int, int);
template __global__ void quant_gemv_kernel<8, true>(const uint8_t *, size_t, const float *, float *, int, int, int);

// ---- llama.cpp-style Q8_1 activation + quant GEMV（实验路径）----
// Q8_1 activation block: f16 d + f16 sum + int8 qs[32] = 36 bytes.
// d = max(abs(x)) / 127, qs[i] = round(x[i] / d), sum = d * sum(qs)。
__global__ void quantize_q8_1_kernel(const float *x, uint8_t *x_q8_1, int in_dim, int m, int blocks_per_row) {
    const int block = blockIdx.x;
    const int row = block / blocks_per_row;
    const int qblk = block - row * blocks_per_row;
    if (row >= m) return;

    const int lane = threadIdx.x;
    const int k = qblk * 32 + lane;
    const float v = (lane < 32 && k < in_dim) ? x[static_cast<size_t>(row) * in_dim + k] : 0.0f;

    float amax = fabsf(v);
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, offset));
    }
    amax = __shfl_sync(0xffffffff, amax, 0);
    const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;

    int q = __float2int_rn(v * id);
    q = max(-127, min(127, q));

    int qsum = q;
    for (int offset = 16; offset > 0; offset >>= 1) {
        qsum += __shfl_down_sync(0xffffffff, qsum, offset);
    }

    uint8_t *dst = x_q8_1 + static_cast<size_t>(block) * 36;
    if (lane == 0) {
        *reinterpret_cast<uint16_t *>(dst) = __half_as_ushort(__float2half(d));
        *reinterpret_cast<uint16_t *>(dst + 2) = __half_as_ushort(__float2half(d * static_cast<float>(qsum)));
    }
    if (lane < 32) {
        reinterpret_cast<int8_t *>(dst + 4)[lane] = static_cast<int8_t>(q);
    }
}

template <int QUANT_TYPE>
__global__ void quant_gemv_q8_1_kernel(const uint8_t *weight, size_t row_bytes, const uint8_t *x_q8_1,
                                       float *y, int out_dim, int in_dim, int blocks_per_row) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= out_dim) return;

    const uint8_t *row_base = weight + static_cast<size_t>(warp_id) * row_bytes;
    float acc = 0.0f;
    for (int qblk = 0; qblk < blocks_per_row; ++qblk) {
        const int k = qblk * 32 + lane;
        const uint8_t *x_blk = x_q8_1 + static_cast<size_t>(qblk) * 36;
        const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(x_blk)));
        const float xsum = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(x_blk + 2)));
        const int8_t qx = (k < in_dim) ? reinterpret_cast<const int8_t *>(x_blk + 4)[lane] : 0;

        if (QUANT_TYPE == 12) {
            // Q4_K: w = d_w * scale * q - dmin_w * min.
            // Q8_1 already stores sum(x) for this 32-wide block, so the min term can be
            // accumulated once per block instead of per element:
            // dot = d_w * scale * d_x * sum(q * qx) - dmin_w * min * sum(x).
            const int super = qblk >> 3;
            const int j = qblk & 7;
            const uint8_t *base = row_base + static_cast<size_t>(super) * 144;
            const float dw = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(base)));
            const float dmin = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(base + 2)));
            const uint8_t *scales = base + 4;
            const uint8_t *qs = base + 16;
            const int pair = j >> 1;
            const uint8_t packed = qs[pair * 32 + lane];
            const int qw = (j & 1) ? (packed >> 4) : (packed & 0x0F);

            int qdot = qw * static_cast<int>(qx);
            for (int offset = 16; offset > 0; offset >>= 1) {
                qdot += __shfl_down_sync(0xffffffff, qdot, offset);
            }
            if (lane == 0) {
                uint8_t sc, m;
                q4k_scale_min(j, scales, sc, m);
                acc = fmaf(dw * static_cast<float>(sc) * d, static_cast<float>(qdot), acc);
                acc -= dmin * static_cast<float>(m) * xsum;
            }
            continue;
        }
        if (QUANT_TYPE == 14) {
            // Q6_K: w = d_w * scale * q. Each 32-wide block has two 16-element
            // scale groups, so reduce the two dot sums separately.
            const int super = qblk >> 3;
            const int sub = qblk & 7;
            const int half = sub >> 2;
            const int grp = sub & 3;
            const int n = half * 128;
            const uint8_t *base = row_base + static_cast<size_t>(super) * 210;
            const uint8_t *ql = base;
            const uint8_t *qh = base + 128;
            const int8_t *scales = reinterpret_cast<const int8_t *>(base + 192);
            const float dw = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t *>(base + 208)));
            const uint8_t *qlp = ql + (n / 2);
            const uint8_t *qhp = qh + (n / 4);
            const int8_t *sc = scales + (n / 16);

            int qw;
            if (grp == 0) {
                qw = static_cast<int8_t>((qlp[lane + 0] & 0xF) | (((qhp[lane] >> 0) & 3) << 4)) - 32;
            } else if (grp == 1) {
                qw = static_cast<int8_t>((qlp[lane + 32] & 0xF) | (((qhp[lane] >> 2) & 3) << 4)) - 32;
            } else if (grp == 2) {
                qw = static_cast<int8_t>((qlp[lane + 0] >> 4) | (((qhp[lane] >> 4) & 3) << 4)) - 32;
            } else {
                qw = static_cast<int8_t>((qlp[lane + 32] >> 4) | (((qhp[lane] >> 6) & 3) << 4)) - 32;
            }

            const int qprod = qw * static_cast<int>(qx);
            int qdot0 = lane < 16 ? qprod : 0;
            int qdot1 = lane >= 16 ? qprod : 0;
            for (int offset = 16; offset > 0; offset >>= 1) {
                qdot0 += __shfl_down_sync(0xffffffff, qdot0, offset);
                qdot1 += __shfl_down_sync(0xffffffff, qdot1, offset);
            }
            if (lane == 0) {
                const float s0 = static_cast<float>(sc[grp * 2 + 0]);
                const float s1 = static_cast<float>(sc[grp * 2 + 1]);
                acc = fmaf(dw * d, s0 * static_cast<float>(qdot0) + s1 * static_cast<float>(qdot1), acc);
            }
            continue;
        }
        const float xv = d * static_cast<float>(qx);
        float w = 0.0f;
        if (k < in_dim) {
            if (QUANT_TYPE == 6) w = q50_at(row_base, k);
            else w = q80_at(row_base, k);
        }
        acc = fmaf(w, xv, acc);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) y[warp_id] = acc;
}

template __global__ void quant_gemv_q8_1_kernel<12>(const uint8_t *, size_t, const uint8_t *, float *, int, int, int);
template __global__ void quant_gemv_q8_1_kernel<14>(const uint8_t *, size_t, const uint8_t *, float *, int, int, int);
template __global__ void quant_gemv_q8_1_kernel<6>(const uint8_t *, size_t, const uint8_t *, float *, int, int, int);
template __global__ void quant_gemv_q8_1_kernel<8>(const uint8_t *, size_t, const uint8_t *, float *, int, int, int);

// 量化直算 Embedding 查表：table 量化常驻（每 token 一行 hidden 个元素、row_bytes 字节），
// 按 token id 只反量化命中的那一行到 f32，避免把整张 [vocab,hidden] 表展开成 F16。
template <int QUANT_TYPE>
__global__ void quant_embedding_kernel(const int *input, float *output, const uint8_t *table,
                                       size_t row_bytes, int hidden_size, int input_size) {
    const int token = blockIdx.x;                // 每个 block 负责一个输入 token
    if (token >= input_size) return;
    const int id = input[token];
    const uint8_t *row_base = table + static_cast<size_t>(id) * row_bytes;
    float *out = output + static_cast<size_t>(token) * hidden_size;
    for (int k = threadIdx.x; k < hidden_size; k += blockDim.x) {
        float w;
        if (QUANT_TYPE == 12) w = q4k_at(row_base, k);
        else if (QUANT_TYPE == 14) w = q6k_at(row_base, k);
        else if (QUANT_TYPE == 6) w = q50_at(row_base, k);
        else w = q80_at(row_base, k);
        out[k] = w;
    }
}

template __global__ void quant_embedding_kernel<12>(const int *, float *, const uint8_t *, size_t, int, int);
template __global__ void quant_embedding_kernel<14>(const int *, float *, const uint8_t *, size_t, int, int);
template __global__ void quant_embedding_kernel<6>(const int *, float *, const uint8_t *, size_t, int, int);
template __global__ void quant_embedding_kernel<8>(const int *, float *, const uint8_t *, size_t, int, int);

// ================= MLA（多头潜在注意力）=================
// 解耦 RoPE：对 rope 段做 GGML_ROPE_TYPE_NORM 风格旋转（相邻对 (2i, 2i+1) 配对）。
// inv_freq[half] 为预计算好的频率（含 YARN 缩放），host 侧一次算好。
__device__ inline void mla_rope_inplace(float *vec, int rope_dim, int pos, const float *inv_freq) {
    // DeepSeek-V2 GGUF 采用 GGML_ROPE_TYPE_NORM（相邻对 interleaved），
    // 即旋转 (vec[2i], vec[2i+1])，pair i 使用 inv_freq[i]，而非 NeoX 的半分。
    const int half = rope_dim / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float angle = static_cast<float>(pos) * inv_freq[i];
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float x0 = vec[2 * i];
        const float x1 = vec[2 * i + 1];
        vec[2 * i] = x0 * c - x1 * s;
        vec[2 * i + 1] = x0 * s + x1 * c;
    }
}

// kv_a 处理：latent 段 RMSNorm（无 weight one_plus，标准 RMSNorm），k_rope 段 RoPE，
// 写入 kv_cache 布局 [max_seq_len, kv_lora + qk_rope]。每个 block 处理一个 (tok)。
__global__ void mla_kv_a_kernel(const float *kv_a, const float *kv_a_norm_weight, float *output_kv_cache,
                                int input_size, int kv_lora, int qk_rope, int start_pos,
                                const float *inv_freq, float eps) {
    extern __shared__ float shared[];
    float *partial = shared;
    float *cache_row = partial + blockDim.x; //+kBlock
    const int tid = threadIdx.x;
    const int input_index = blockIdx.x;
    if (input_index >= input_size) return;
    const int pos = start_pos + input_index;
    const int total = kv_lora + qk_rope;
    const float *kv_a_row = kv_a + static_cast<size_t>(input_index) * total;

    // latent RMSNorm
    float ss = 0.0f;
    for (int d = tid; d < kv_lora; d += blockDim.x) {
        const float v = kv_a_row[d];
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
        cache_row[d] = kv_a_row[d] * scale * kv_a_norm_weight[d];
    }
    // k_rope 段先拷贝
    for (int d = tid; d < qk_rope; d += blockDim.x) {
        cache_row[kv_lora + d] = kv_a_row[kv_lora + d];
    }
    __syncthreads();
    // 对 k_rope 段做 RoPE
    mla_rope_inplace(cache_row + kv_lora, qk_rope, pos, inv_freq);
    __syncthreads();
    for (int d = tid; d < total; d += blockDim.x) {
        output_kv_cache[static_cast<size_t>(pos) * total + d] = cache_row[d];
    }
}

// q 的 rope 段旋转：q 布局 [input_size, n_heads, qk_nope + qk_rope]，只旋转后 qk_rope 维。
// 每 block 处理一个 (tok, head)；单 token decode 是 input_size=1 的特例。
__global__ void mla_rope_q_kernel(float *q, int input_size, int n_heads, int qk_nope,
                                  int qk_rope, int start_pos, const float *inv_freq) {
    const int block_index = blockIdx.x; //block 数量：input_size*n_heads
    //block_index = input_index * n_heads + head_index
    const int input_index = block_index / n_heads;
    const int head_index = block_index - input_index * n_heads;
    if (input_index >= input_size) return;
    float *qh = q
                + static_cast<size_t>(input_index) * n_heads * (qk_nope + qk_rope)
                + head_index * (qk_nope + qk_rope)
                + qk_nope; //跳过前一半的 qk_nope，只看后一半的 qk_rope
    mla_rope_inplace(qh, qk_rope, start_pos + input_index, inv_freq);
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
            if (logits[e] > best) {
                best = logits[e];
                best_e = e;
            }
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

// 加权累加（权重从 device 读）：out += (*weight) * expert_out。decode 时 top_w 留在 device，
// 避免每层把权重回读到 host 造成同步。
__global__ void moe_accumulate_device_kernel(const float *expert_out, const float *weight, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] += (*weight) * expert_out[i];
}

// ---- argmax ----

// block 内共享内存归约：并列时保留 index 较小者（与 CPU Sampler::argmax 严格大于才更新一致）。
__device__ inline void argmax_block_reduce(float *vals, int *idxs, int tid) {
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (vals[tid + s] > vals[tid] ||
                (vals[tid + s] == vals[tid] && idxs[tid + s] < idxs[tid])) {
                vals[tid] = vals[tid + s];
                idxs[tid] = idxs[tid + s];
            }
        }
        __syncthreads();
    }
}

// 阶段1：多 block 各求 logits 的局部 argmax，写到 partial_vals/partial_idxs[blockIdx.x]。
// 每个 block 用 grid-stride 覆盖 logits 的一段，把 15w 词表的串行扫描摊到多个 SM 上并行。
__global__ void argmax_partial_kernel(const float *logits, int n, float *partial_vals,
                                      int *partial_idxs) {
    __shared__ float vals[kBlockConst];
    __shared__ int idxs[kBlockConst];
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    float local_max = -INFINITY;
    int local_idx = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        const float v = logits[i];
        if (v > local_max || (v == local_max && i < local_idx)) {
            local_max = v;
            local_idx = i;
        }
    }
    vals[tid] = local_max;
    idxs[tid] = local_idx;
    __syncthreads();
    argmax_block_reduce(vals, idxs, tid);
    if (tid == 0) {
        partial_vals[blockIdx.x] = vals[0];
        partial_idxs[blockIdx.x] = idxs[0];
    }
}

// 阶段2：单 block 归约 num_partials 个局部结果，得到全局 argmax，写 out_idx[0]。
__global__ void argmax_final_kernel(const float *partial_vals, const int *partial_idxs,
                                    int num_partials, int *out_idx) {
    __shared__ float vals[kBlockConst];
    __shared__ int idxs[kBlockConst];
    const int tid = threadIdx.x;
    float local_max = -INFINITY;
    int local_idx = 0;
    for (int i = tid; i < num_partials; i += blockDim.x) {
        const float v = partial_vals[i];
        const int gi = partial_idxs[i];
        if (v > local_max || (v == local_max && gi < local_idx)) {
            local_max = v;
            local_idx = gi;
        }
    }
    vals[tid] = local_max;
    idxs[tid] = local_idx;
    __syncthreads();
    argmax_block_reduce(vals, idxs, tid);
    if (tid == 0) out_idx[0] = idxs[0];
}
