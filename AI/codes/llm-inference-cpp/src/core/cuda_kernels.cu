#include "cuda_kernels.h"

#include <cuda_runtime.h>

namespace llm_inference {
namespace {

__device__ uint16_t float_to_bf16_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    return static_cast<uint16_t>(bits >> 16);
}

__device__ float bf16_bits_to_float(uint16_t value) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

__global__ void silu_mul_kernel(const float * gate, const float * up, float * out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

__global__ void float_to_bf16_kernel(const float * input, uint16_t * output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    output[i] = float_to_bf16_bits(input[i]);
}

__global__ void rms_norm_to_bf16_kernel(
    const float * input,
    const uint16_t * weight,
    uint16_t * output,
    int n,
    float eps,
    bool one_plus) {
    __shared__ float partial[256];
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = input[i];
        ss += v * v;
    }
    partial[tid] = ss;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    const float scale = rsqrtf(partial[0] / static_cast<float>(n) + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        const float w = bf16_bits_to_float(weight[i]);
        const float factor = one_plus ? (1.0f + w) : w;
        output[i] = float_to_bf16_bits(input[i] * scale * factor);
    }
}

__global__ void linear_attention_conv_kernel(
    const float * mixed,
    const uint16_t * conv_weight,
    float * conv_state,
    float * conv_out,
    int conv_dim,
    int kernel) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= conv_dim) {
        return;
    }
    float * row = conv_state + static_cast<size_t>(d) * kernel;
    for (int i = 0; i < kernel - 1; ++i) {
        row[i] = row[i + 1];
    }
    row[kernel - 1] = mixed[d];

    float sum = 0.0f;
    const uint16_t * w = conv_weight + static_cast<size_t>(d) * kernel;
    for (int i = 0; i < kernel; ++i) {
        sum += bf16_bits_to_float(w[i]) * row[i];
    }
    conv_out[d] = sum / (1.0f + expf(-sum));
}

__global__ void linear_attention_recurrent_kernel(
    const float * conv_out,
    const float * z,
    const float * b,
    const float * a,
    const float * a_log,
    const uint16_t * dt_bias,
    const float * norm_weight,
    float * recurrent_state,
    float * gated,
    int key_heads,
    int value_heads,
    int k_dim,
    int v_dim,
    float eps) {
    extern __shared__ float shared[];
    float * q = shared;
    float * k = q + k_dim;
    float * delta = k + k_dim;
    float * core = delta + v_dim;
    float * partial = core + v_dim;

    const int vh = blockIdx.x;
    const int tid = threadIdx.x;
    const int repeat = value_heads / key_heads;
    const int kh = vh / repeat;
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const float * query_base = conv_out;
    const float * key_base = conv_out + key_total;
    const float * value_base = conv_out + key_total * 2;
    float * rec = recurrent_state + static_cast<size_t>(vh) * k_dim * v_dim;

    float ss_q = 0.0f;
    float ss_k = 0.0f;
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

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
            partial[blockDim.x + tid] += partial[blockDim.x + tid + stride];
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

    const float beta = 1.0f / (1.0f + expf(-b[vh]));
    const float dt = a[vh] + bf16_bits_to_float(dt_bias[vh]);
    const float softplus_dt = dt > 20.0f ? dt : (dt < -20.0f ? expf(dt) : log1pf(expf(dt)));
    const float g = -expf(a_log[vh]) * softplus_dt;
    const float decay = expf(g);

    const int state_size = k_dim * v_dim;
    for (int i = tid; i < state_size; i += blockDim.x) {
        rec[i] *= decay;
    }
    __syncthreads();

    for (int vd = tid; vd < v_dim; vd += blockDim.x) {
        float kv_mem = 0.0f;
        for (int kd = 0; kd < k_dim; ++kd) {
            kv_mem += rec[static_cast<size_t>(kd) * v_dim + vd] * k[kd];
        }
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
        for (int kd = 0; kd < k_dim; ++kd) {
            sum += rec[static_cast<size_t>(kd) * v_dim + vd] * q[kd];
        }
        core[vd] = sum;
    }
    __syncthreads();

    float ss = 0.0f;
    for (int vd = tid; vd < v_dim; vd += blockDim.x) {
        ss += core[vd] * core[vd];
    }
    partial[tid] = ss;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    const float norm_scale = rsqrtf(partial[0] / static_cast<float>(v_dim) + eps);
    for (int vd = tid; vd < v_dim; vd += blockDim.x) {
        const float gate = z[static_cast<size_t>(vh) * v_dim + vd];
        const float silu_gate = gate / (1.0f + expf(-gate));
        gated[static_cast<size_t>(vh) * v_dim + vd] = norm_weight[vd] * core[vd] * norm_scale * silu_gate;
    }

    (void)value_total;
}

__global__ void full_attention_q_kernel(
    const float * q_and_gate,
    const uint16_t * q_norm_weight,
    float * q,
    float * gate,
    int n_heads,
    int head_dim,
    int pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps) {
    __shared__ float partial[256];
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    if (h >= n_heads) {
        return;
    }
    const int src = h * head_dim * 2;
    const int dst = h * head_dim;
    float ss = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float v = q_and_gate[src + d];
        q[dst + d] = v;
        gate[dst + d] = q_and_gate[src + head_dim + d];
        ss += v * v;
    }
    partial[tid] = ss;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float w = 1.0f + bf16_bits_to_float(q_norm_weight[d]);
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

__global__ void full_attention_kv_kernel(
    const float * k_in,
    const float * v_in,
    const uint16_t * k_norm_weight,
    float * key_cache,
    float * value_cache,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps) {
    __shared__ float partial[256];
    __shared__ float k_local[256];
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    if (h >= kv_heads) {
        return;
    }
    const int base = h * head_dim;
    float ss = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float v = k_in[base + d];
        k_local[d] = v;
        ss += v * v;
    }
    partial[tid] = ss;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float w = 1.0f + bf16_bits_to_float(k_norm_weight[d]);
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

    const size_t cache_base = (static_cast<size_t>(pos) * kv_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        key_cache[cache_base + d] = k_local[d];
        value_cache[cache_base + d] = v_in[base + d];
    }
    (void)max_seq_len;
}

__global__ void full_attention_attend_kernel(
    const float * q,
    const float * gate,
    const float * key_cache,
    const float * value_cache,
    float * attn,
    int n_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int pos) {
    extern __shared__ float shared[];
    float * scores = shared;
    float * partial = scores + pos + 1;
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int kv_group = n_heads / kv_heads;
    const int kh = h / kv_group;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float * qh = q + h * head_dim;

    float local_max = -INFINITY;
    for (int t = tid; t <= pos; t += blockDim.x) {
        const float * key = key_cache + (static_cast<size_t>(t) * kv_heads + kh) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += qh[d] * key[d];
        }
        const float score = dot * scale;
        scores[t] = score;
        local_max = fmaxf(local_max, score);
    }
    partial[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        }
        __syncthreads();
    }
    const float max_score = partial[0];

    float local_denom = 0.0f;
    for (int t = tid; t <= pos; t += blockDim.x) {
        const float e = expf(scores[t] - max_score);
        scores[t] = e;
        local_denom += e;
    }
    partial[tid] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float denom = partial[0];

    float * out = attn + h * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            const float prob = scores[t] / denom;
            const float * value = value_cache + (static_cast<size_t>(t) * kv_heads + kh) * head_dim;
            sum += prob * value[d];
        }
        const float g = gate[h * head_dim + d];
        out[d] = sum * (1.0f / (1.0f + expf(-g)));
    }
    (void)max_seq_len;
}

__global__ void argmax_blocks_kernel(
    const float * values,
    int n,
    float * block_values,
    int * block_indices) {
    __shared__ float s_values[256];
    __shared__ int s_indices[256];
    const int tid = threadIdx.x;
    const int global = blockIdx.x * blockDim.x + tid;
    float best = -INFINITY;
    int best_id = 0;
    if (global < n) {
        best = values[global];
        best_id = global;
    }
    s_values[tid] = best;
    s_indices[tid] = best_id;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = s_values[tid + stride];
            const int other_id = s_indices[tid + stride];
            if (other > s_values[tid] || (other == s_values[tid] && other_id < s_indices[tid])) {
                s_values[tid] = other;
                s_indices[tid] = other_id;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_values[blockIdx.x] = s_values[0];
        block_indices[blockIdx.x] = s_indices[0];
    }
}

__global__ void argmax_final_kernel(
    const float * block_values,
    const int * block_indices,
    float * best_value,
    int * best_index,
    int blocks) {
    __shared__ float s_values[1024];
    __shared__ int s_indices[1024];
    const int tid = threadIdx.x;
    float best = -INFINITY;
    int best_id = 0;
    for (int i = tid; i < blocks; i += blockDim.x) {
        const float value = block_values[i];
        const int id = block_indices[i];
        if (value > best || (value == best && id < best_id)) {
            best = value;
            best_id = id;
        }
    }
    s_values[tid] = best;
    s_indices[tid] = best_id;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = s_values[tid + stride];
            const int other_id = s_indices[tid + stride];
            if (other > s_values[tid] || (other == s_values[tid] && other_id < s_indices[tid])) {
                s_values[tid] = other;
                s_indices[tid] = other_id;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        *best_value = s_values[0];
        *best_index = s_indices[0];
    }
}

} // namespace

void launch_silu_mul(const float * gate, const float * up, float * out, int n, void * stream) {
    const int block = 256;
    const int grid = (n + block - 1) / block;
    silu_mul_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(gate, up, out, n);
}

void launch_float_to_bf16(const float * input, uint16_t * output, int n, void * stream) {
    const int block = 256;
    const int grid = (n + block - 1) / block;
    float_to_bf16_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(input, output, n);
}

void launch_rms_norm_to_bf16(
    const float * input,
    const uint16_t * weight,
    uint16_t * output,
    int n,
    float eps,
    bool one_plus,
    void * stream) {
    rms_norm_to_bf16_kernel<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(input, weight, output, n, eps, one_plus);
}

void launch_linear_attention_conv(
    const float * mixed,
    const uint16_t * conv_weight,
    float * conv_state,
    float * conv_out,
    int conv_dim,
    int kernel,
    void * stream) {
    const int block = 256;
    const int grid = (conv_dim + block - 1) / block;
    linear_attention_conv_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        mixed,
        conv_weight,
        conv_state,
        conv_out,
        conv_dim,
        kernel);
}

void launch_linear_attention_recurrent(
    const float * conv_out,
    const float * z,
    const float * b,
    const float * a,
    const float * a_log,
    const uint16_t * dt_bias,
    const float * norm_weight,
    float * recurrent_state,
    float * gated,
    int key_heads,
    int value_heads,
    int k_dim,
    int v_dim,
    float eps,
    void * stream) {
    const int block = 256;
    const size_t shared_bytes =
        static_cast<size_t>(k_dim * 2 + v_dim * 2 + block * 2) * sizeof(float);
    linear_attention_recurrent_kernel<<<value_heads, block, shared_bytes, static_cast<cudaStream_t>(stream)>>>(
        conv_out,
        z,
        b,
        a,
        a_log,
        dt_bias,
        norm_weight,
        recurrent_state,
        gated,
        key_heads,
        value_heads,
        k_dim,
        v_dim,
        eps);
}

void launch_full_attention_q(
    const float * q_and_gate,
    const uint16_t * q_norm_weight,
    float * q,
    float * gate,
    int n_heads,
    int head_dim,
    int pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps,
    void * stream) {
    full_attention_q_kernel<<<n_heads, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        q_and_gate,
        q_norm_weight,
        q,
        gate,
        n_heads,
        head_dim,
        pos,
        rope_theta,
        partial_rotary_factor,
        eps);
}

void launch_full_attention_kv(
    const float * k_in,
    const float * v_in,
    const uint16_t * k_norm_weight,
    float * key_cache,
    float * value_cache,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    float rope_theta,
    float partial_rotary_factor,
    float eps,
    void * stream) {
    full_attention_kv_kernel<<<kv_heads, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        k_in,
        v_in,
        k_norm_weight,
        key_cache,
        value_cache,
        kv_heads,
        head_dim,
        max_seq_len,
        pos,
        rope_theta,
        partial_rotary_factor,
        eps);
}

void launch_full_attention_attend(
    const float * q,
    const float * gate,
    const float * key_cache,
    const float * value_cache,
    float * attn,
    int n_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    int pos,
    void * stream) {
    const size_t shared_bytes = static_cast<size_t>(pos + 1 + 256) * sizeof(float);
    full_attention_attend_kernel<<<n_heads, 256, shared_bytes, static_cast<cudaStream_t>(stream)>>>(
        q,
        gate,
        key_cache,
        value_cache,
        attn,
        n_heads,
        kv_heads,
        head_dim,
        max_seq_len,
        pos);
}

void launch_argmax_float(
    const float * values,
    int n,
    float * block_values,
    int * block_indices,
    float * best_value,
    int * best_index,
    int blocks,
    void * stream) {
    argmax_blocks_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        values,
        n,
        block_values,
        block_indices);
    argmax_final_kernel<<<1, 1024, 0, static_cast<cudaStream_t>(stream)>>>(
        block_values,
        block_indices,
        best_value,
        best_index,
        blocks);
}

} // namespace llm_inference
