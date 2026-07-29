#include <metal_stdlib>
using namespace metal;

struct ScalarParams {
    float scalar;
};

struct MatmulParams {
    uint m;
    uint k;
    uint n;
};

struct BatchMatmulParams {
    uint batches;
    uint heads;
    uint m;
    uint k;
    uint n;
};

struct SoftmaxParams {
    uint rows;
    uint width;
};

struct LayerNormParams {
    uint rows;
    uint width;
    float eps;
};

struct EmbeddingParams {
    uint count;
    uint dim;
};

struct CrossEntropyParams {
    uint rows;
    uint vocab;
};

struct GradParams {
    uint target_size;
    uint count;
    float scale;
};

struct TransposeParams {
    uint rank;
    uint count;
    uint shape[4];
    uint out_shape[4];
    uint in_strides[4];
    uint out_strides[4];
};

struct AdamWParams {
    uint count;
    float lr;
    float weight_decay;
    float beta1;
    float beta2;
    float eps;
    float bias_correction1;
    float bias_correction2;
};

struct CausalMaskParams {
    uint batches;
    uint heads;
    uint sequence_length;
    float mask_value;
};

struct ElementwiseGradParams {
    uint count;
    uint has_a;
    uint has_b;
};

kernel void add_kernel(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant uint& b_size [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[(b_size == 1) ? 0 : id % b_size];
}

kernel void mul_kernel(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * b[id];
}

kernel void div_kernel(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    out[id] = a[id] / b[id];
}

kernel void mul_scalar_kernel(device const float* a [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant ScalarParams& params [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * params.scalar;
}

kernel void matmul_kernel(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          constant MatmulParams& params [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= params.m || col >= params.n) {
        return;
    }
    float acc = 0.0;
    for (uint p = 0; p < params.k; ++p) {
        acc += a[row * params.k + p] * b[p * params.n + col];
    }
    out[row * params.n + col] = acc;
}

kernel void batch_matmul_kernel(device const float* a [[buffer(0)]],
                                device const float* b [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant BatchMatmulParams& params [[buffer(3)]],
                                uint id [[thread_position_in_grid]]) {
    uint total = params.batches * params.heads * params.m * params.n;
    if (id >= total) {
        return;
    }
    uint col = id % params.n;
    uint row = (id / params.n) % params.m;
    uint head = (id / (params.n * params.m)) % params.heads;
    uint batch = id / (params.n * params.m * params.heads);
    float acc = 0.0;
    for (uint p = 0; p < params.k; ++p) {
        uint ai = ((batch * params.heads + head) * params.m + row) * params.k + p;
        uint bi = ((batch * params.heads + head) * params.k + p) * params.n + col;
        acc += a[ai] * b[bi];
    }
    out[id] = acc;
}

kernel void softmax_kernel(device const float* x [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant SoftmaxParams& params [[buffer(2)]],
                           uint row [[thread_position_in_grid]]) {
    if (row >= params.rows) {
        return;
    }
    uint base = row * params.width;
    float mx = -INFINITY;
    for (uint c = 0; c < params.width; ++c) {
        mx = max(mx, x[base + c]);
    }
    float denom = 0.0;
    for (uint c = 0; c < params.width; ++c) {
        denom += exp(x[base + c] - mx);
    }
    for (uint c = 0; c < params.width; ++c) {
        out[base + c] = exp(x[base + c] - mx) / denom;
    }
}

kernel void layernorm_kernel(device const float* x [[buffer(0)]],
                             device const float* scale [[buffer(1)]],
                             device const float* shift [[buffer(2)]],
                             device float* out [[buffer(3)]],
                             constant LayerNormParams& params [[buffer(4)]],
                             uint row [[thread_position_in_grid]]) {
    if (row >= params.rows) {
        return;
    }
    uint base = row * params.width;
    float mean = 0.0;
    for (uint c = 0; c < params.width; ++c) {
        mean += x[base + c];
    }
    mean /= float(params.width);
    float var = 0.0;
    for (uint c = 0; c < params.width; ++c) {
        float z = x[base + c] - mean;
        var += z * z;
    }
    float inv = rsqrt(var / float(params.width) + params.eps);
    for (uint c = 0; c < params.width; ++c) {
        float xhat = (x[base + c] - mean) * inv;
        out[base + c] = xhat * scale[c] + shift[c];
    }
}

kernel void embedding_kernel(device const float* ids [[buffer(0)]],
                             device const float* weight [[buffer(1)]],
                             device float* out [[buffer(2)]],
                             constant EmbeddingParams& params [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    uint total = params.count * params.dim;
    if (id >= total) {
        return;
    }
    uint token_index = id / params.dim;
    uint d = id % params.dim;
    uint token = uint(ids[token_index]);
    out[id] = weight[token * params.dim + d];
}

kernel void cross_entropy_row_loss_kernel(device const float* logits [[buffer(0)]],
                                          device const float* targets [[buffer(1)]],
                                          device float* row_losses [[buffer(2)]],
                                          constant CrossEntropyParams& params [[buffer(3)]],
                                          uint row [[thread_position_in_grid]]) {
    if (row >= params.rows) {
        return;
    }
    uint base = row * params.vocab;
    float mx = -INFINITY;
    for (uint v = 0; v < params.vocab; ++v) {
        mx = max(mx, logits[base + v]);
    }
    float denom = 0.0;
    for (uint v = 0; v < params.vocab; ++v) {
        denom += exp(logits[base + v] - mx);
    }
    uint target = uint(targets[row]);
    float p = exp(logits[base + target] - mx) / denom;
    row_losses[row] = -log(max(p, 1.0e-12f));
}

kernel void gelu_kernel(device const float* x [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    float v = x[id];
    float u = 0.7978845608f * (v + 0.044715f * v * v * v);
    out[id] = 0.5f * v * (1.0f + tanh(u));
}

kernel void neg_kernel(device const float* a [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    out[id] = -a[id];
}

kernel void pow_kernel(device const float* a [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant ScalarParams& params [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    out[id] = pow(a[id], params.scalar);
}

kernel void copy_kernel(device const float* a [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = a[id];
}

kernel void log_kernel(device const float* a [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    out[id] = log(max(a[id], 1.0e-12f));
}

// 用 host 端预计算的 index 表做任意维度重排（供 transpose 使用）。
kernel void gather_kernel(device const float* a [[buffer(0)]],
                          device const uint* index [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    out[id] = a[index[id]];
}

kernel void transpose_kernel(device const float* a [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant TransposeParams& params [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= params.count) {
        return;
    }
    uint rem = id;
    uint in_flat = 0;
    for (uint d = 0; d < params.rank; ++d) {
        uint idx = rem / params.out_strides[d];
        rem %= params.out_strides[d];
        in_flat += idx * params.in_strides[d];
    }
    out[id] = a[in_flat];
}

// 单线程全量规约求和，count 通过 b_size buffer 传入。
kernel void sum_kernel(device const float* a [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    if (id != 0) {
        return;
    }
    float acc = 0.0;
    for (uint i = 0; i < count; ++i) {
        acc += a[i];
    }
    out[0] = acc;
}

// 单线程全量规约求最大值。
kernel void max_kernel(device const float* a [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    if (id != 0) {
        return;
    }
    float mx = -INFINITY;
    for (uint i = 0; i < count; ++i) {
        mx = max(mx, a[i]);
    }
    out[0] = mx;
}

kernel void fill_kernel(device float* out [[buffer(0)]],
                        constant ScalarParams& params [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = params.scalar;
}

kernel void log_softmax_kernel(device const float* x [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               constant SoftmaxParams& params [[buffer(2)]],
                               uint row [[thread_position_in_grid]]) {
    if (row >= params.rows) {
        return;
    }
    uint base = row * params.width;
    float mx = -INFINITY;
    for (uint c = 0; c < params.width; ++c) {
        mx = max(mx, x[base + c]);
    }
    float denom = 0.0;
    for (uint c = 0; c < params.width; ++c) {
        denom += exp(x[base + c] - mx);
    }
    float log_denom = log(max(denom, 1.0e-12f));
    for (uint c = 0; c < params.width; ++c) {
        out[base + c] = x[base + c] - mx - log_denom;
    }
}

kernel void cross_entropy_loss_kernel(device const float* logits [[buffer(0)]],
                                      device const float* targets [[buffer(1)]],
                                      device float* loss [[buffer(2)]],
                                      constant CrossEntropyParams& params [[buffer(3)]],
                                      uint id [[thread_position_in_grid]]) {
    if (id != 0) {
        return;
    }
    float acc = 0.0;
    for (uint row = 0; row < params.rows; ++row) {
        uint base = row * params.vocab;
        float mx = -INFINITY;
        for (uint v = 0; v < params.vocab; ++v) {
            mx = max(mx, logits[base + v]);
        }
        float denom = 0.0;
        for (uint v = 0; v < params.vocab; ++v) {
            denom += exp(logits[base + v] - mx);
        }
        uint target = uint(targets[row]);
        float p = exp(logits[base + target] - mx) / denom;
        acc += -log(max(p, 1.0e-12f));
    }
    loss[0] = acc / float(params.rows);
}

kernel void add_grad_kernel(device atomic_float* target_grad [[buffer(0)]],
                            device const float* out_grad [[buffer(1)]],
                            constant GradParams& params [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= params.count) {
        return;
    }
    uint target = (params.target_size == 0) ? id : (id % params.target_size);
    atomic_fetch_add_explicit(&target_grad[target], out_grad[id] * params.scale, memory_order_relaxed);
}

kernel void adamw_update_kernel(device float* param [[buffer(0)]],
                                device const float* grad [[buffer(1)]],
                                device float* m [[buffer(2)]],
                                device float* v [[buffer(3)]],
                                constant AdamWParams& params [[buffer(4)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= params.count) {
        return;
    }
    float g = grad[id];
    m[id] = params.beta1 * m[id] + (1.0f - params.beta1) * g;
    v[id] = params.beta2 * v[id] + (1.0f - params.beta2) * g * g;
    float m_hat = m[id] / params.bias_correction1;
    float v_hat = v[id] / params.bias_correction2;
    param[id] -= params.lr * params.weight_decay * param[id];
    param[id] -= params.lr * m_hat / (sqrt(v_hat) + params.eps);
}

kernel void mul_grad_kernel(device float* a_grad [[buffer(0)]],
                            device float* b_grad [[buffer(1)]],
                            device const float* a [[buffer(2)]],
                            device const float* b [[buffer(3)]],
                            device const float* out_grad [[buffer(4)]],
                            constant ElementwiseGradParams& params [[buffer(5)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= params.count) {
        return;
    }
    float g = out_grad[id];
    if (params.has_a != 0) {
        a_grad[id] += b[id] * g;
    }
    if (params.has_b != 0) {
        b_grad[id] += a[id] * g;
    }
}

kernel void div_grad_kernel(device float* a_grad [[buffer(0)]],
                            device float* b_grad [[buffer(1)]],
                            device const float* a [[buffer(2)]],
                            device const float* b [[buffer(3)]],
                            device const float* out_grad [[buffer(4)]],
                            constant ElementwiseGradParams& params [[buffer(5)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= params.count) {
        return;
    }
    float g = out_grad[id];
    if (params.has_a != 0) {
        a_grad[id] += g / b[id];
    }
    if (params.has_b != 0) {
        b_grad[id] += -g * a[id] / (b[id] * b[id]);
    }
}

kernel void mul_scalar_grad_kernel(device float* a_grad [[buffer(0)]],
                                   device const float* out_grad [[buffer(1)]],
                                   constant GradParams& params [[buffer(2)]],
                                   uint id [[thread_position_in_grid]]) {
    if (id >= params.count) {
        return;
    }
    a_grad[id] += out_grad[id] * params.scale;
}

kernel void pow_grad_kernel(device float* a_grad [[buffer(0)]],
                            device const float* a [[buffer(1)]],
                            device const float* out_grad [[buffer(2)]],
                            constant GradParams& params [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= params.count) {
        return;
    }
    a_grad[id] += params.scale * pow(a[id], params.scale - 1.0f) * out_grad[id];
}

kernel void reduce_grad_kernel(device float* a_grad [[buffer(0)]],
                               device const float* out_grad [[buffer(1)]],
                               constant GradParams& params [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= params.count) {
        return;
    }
    a_grad[id] += out_grad[0] * params.scale;
}

kernel void scatter_add_grad_kernel(device atomic_float* a_grad [[buffer(0)]],
                                    device const float* out_grad [[buffer(1)]],
                                    device const uint* index [[buffer(2)]],
                                    constant uint& count [[buffer(3)]],
                                    uint id [[thread_position_in_grid]]) {
    if (id >= count) {
        return;
    }
    atomic_fetch_add_explicit(&a_grad[index[id]], out_grad[id], memory_order_relaxed);
}

kernel void transpose_add_grad_kernel(device atomic_float* target_grad [[buffer(0)]],
                                      device const float* out_grad [[buffer(1)]],
                                      constant TransposeParams& params [[buffer(2)]],
                                      uint id [[thread_position_in_grid]]) {
    if (id >= params.count) {
        return;
    }
    uint rem = id;
    uint in_flat = 0;
    for (uint d = 0; d < params.rank; ++d) {
        uint idx = rem / params.out_strides[d];
        rem %= params.out_strides[d];
        in_flat += idx * params.in_strides[d];
    }
    atomic_fetch_add_explicit(&target_grad[in_flat], out_grad[id], memory_order_relaxed);
}

kernel void matmul_grad_a_kernel(device float* a_grad [[buffer(0)]],
                                 device const float* b [[buffer(1)]],
                                 device const float* out_grad [[buffer(2)]],
                                 constant MatmulParams& params [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    uint p = gid.x;
    uint i = gid.y;
    if (i >= params.m || p >= params.k) {
        return;
    }
    float acc = 0.0;
    for (uint j = 0; j < params.n; ++j) {
        acc += out_grad[i * params.n + j] * b[p * params.n + j];
    }
    a_grad[i * params.k + p] += acc;
}

kernel void matmul_grad_b_kernel(device float* b_grad [[buffer(0)]],
                                 device const float* a [[buffer(1)]],
                                 device const float* out_grad [[buffer(2)]],
                                 constant MatmulParams& params [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint p = gid.y;
    if (p >= params.k || j >= params.n) {
        return;
    }
    float acc = 0.0;
    for (uint i = 0; i < params.m; ++i) {
        acc += a[i * params.k + p] * out_grad[i * params.n + j];
    }
    b_grad[p * params.n + j] += acc;
}

kernel void batch_matmul_grad_a_kernel(device float* a_grad [[buffer(0)]],
                                       device const float* b [[buffer(1)]],
                                       device const float* out_grad [[buffer(2)]],
                                       constant BatchMatmulParams& params [[buffer(3)]],
                                       uint id [[thread_position_in_grid]]) {
    uint total = params.batches * params.heads * params.m * params.k;
    if (id >= total) {
        return;
    }
    uint p = id % params.k;
    uint i = (id / params.k) % params.m;
    uint head = (id / (params.k * params.m)) % params.heads;
    uint batch = id / (params.k * params.m * params.heads);
    float acc = 0.0;
    for (uint j = 0; j < params.n; ++j) {
        uint bi = ((batch * params.heads + head) * params.k + p) * params.n + j;
        uint oi = ((batch * params.heads + head) * params.m + i) * params.n + j;
        acc += out_grad[oi] * b[bi];
    }
    a_grad[id] += acc;
}

kernel void batch_matmul_grad_b_kernel(device float* b_grad [[buffer(0)]],
                                       device const float* a [[buffer(1)]],
                                       device const float* out_grad [[buffer(2)]],
                                       constant BatchMatmulParams& params [[buffer(3)]],
                                       uint id [[thread_position_in_grid]]) {
    uint total = params.batches * params.heads * params.k * params.n;
    if (id >= total) {
        return;
    }
    uint j = id % params.n;
    uint p = (id / params.n) % params.k;
    uint head = (id / (params.n * params.k)) % params.heads;
    uint batch = id / (params.n * params.k * params.heads);
    float acc = 0.0;
    for (uint i = 0; i < params.m; ++i) {
        uint ai = ((batch * params.heads + head) * params.m + i) * params.k + p;
        uint oi = ((batch * params.heads + head) * params.m + i) * params.n + j;
        acc += a[ai] * out_grad[oi];
    }
    b_grad[id] += acc;
}

kernel void softmax_grad_kernel(device float* a_grad [[buffer(0)]],
                                device const float* out [[buffer(1)]],
                                device const float* out_grad [[buffer(2)]],
                                constant SoftmaxParams& params [[buffer(3)]],
                                uint row [[thread_position_in_grid]]) {
    if (row >= params.rows) {
        return;
    }
    uint base = row * params.width;
    float dot = 0.0;
    for (uint c = 0; c < params.width; ++c) {
        dot += out_grad[base + c] * out[base + c];
    }
    for (uint c = 0; c < params.width; ++c) {
        a_grad[base + c] += out[base + c] * (out_grad[base + c] - dot);
    }
}

kernel void cross_entropy_grad_kernel(device float* logits_grad [[buffer(0)]],
                                      device const float* logits [[buffer(1)]],
                                      device const float* targets [[buffer(2)]],
                                      device const float* out_grad [[buffer(3)]],
                                      constant CrossEntropyParams& params [[buffer(4)]],
                                      uint id [[thread_position_in_grid]]) {
    uint total = params.rows * params.vocab;
    if (id >= total) {
        return;
    }
    uint row = id / params.vocab;
    uint v = id % params.vocab;
    uint base = row * params.vocab;
    float mx = -INFINITY;
    for (uint c = 0; c < params.vocab; ++c) {
        mx = max(mx, logits[base + c]);
    }
    float denom = 0.0;
    for (uint c = 0; c < params.vocab; ++c) {
        denom += exp(logits[base + c] - mx);
    }
    float g = exp(logits[id] - mx) / denom;
    if (v == uint(targets[row])) {
        g -= 1.0f;
    }
    logits_grad[id] += out_grad[0] * g / float(params.rows);
}

kernel void cross_entropy_grad_rows_kernel(device float* logits_grad [[buffer(0)]],
                                           device const float* logits [[buffer(1)]],
                                           device const float* targets [[buffer(2)]],
                                           device const float* out_grad [[buffer(3)]],
                                           constant CrossEntropyParams& params [[buffer(4)]],
                                           uint row [[thread_position_in_grid]]) {
    if (row >= params.rows) {
        return;
    }
    uint base = row * params.vocab;
    float mx = -INFINITY;
    for (uint v = 0; v < params.vocab; ++v) {
        mx = max(mx, logits[base + v]);
    }
    float denom = 0.0;
    for (uint v = 0; v < params.vocab; ++v) {
        denom += exp(logits[base + v] - mx);
    }
    uint target = uint(targets[row]);
    float scale = out_grad[0] / float(params.rows);
    for (uint v = 0; v < params.vocab; ++v) {
        float g = exp(logits[base + v] - mx) / denom;
        if (v == target) {
            g -= 1.0f;
        }
        logits_grad[base + v] += scale * g;
    }
}

kernel void embedding_grad_kernel(device atomic_float* weight_grad [[buffer(0)]],
                                  device const float* ids [[buffer(1)]],
                                  device const float* out_grad [[buffer(2)]],
                                  constant EmbeddingParams& params [[buffer(3)]],
                                  uint id [[thread_position_in_grid]]) {
    uint total = params.count * params.dim;
    if (id >= total) {
        return;
    }
    uint token_index = id / params.dim;
    uint d = id % params.dim;
    uint token = uint(ids[token_index]);
    atomic_fetch_add_explicit(&weight_grad[token * params.dim + d], out_grad[id], memory_order_relaxed);
}

kernel void layernorm_grad_x_kernel(device float* x_grad [[buffer(0)]],
                                    device const float* x [[buffer(1)]],
                                    device const float* scale [[buffer(2)]],
                                    device const float* out_grad [[buffer(3)]],
                                    constant LayerNormParams& params [[buffer(4)]],
                                    uint id [[thread_position_in_grid]]) {
    uint total = params.rows * params.width;
    if (id >= total) {
        return;
    }
    uint row = id / params.width;
    uint c = id % params.width;
    uint base = row * params.width;
    float mean = 0.0;
    for (uint i = 0; i < params.width; ++i) {
        mean += x[base + i];
    }
    mean /= float(params.width);
    float var = 0.0;
    for (uint i = 0; i < params.width; ++i) {
        float z = x[base + i] - mean;
        var += z * z;
    }
    float inv = rsqrt(var / float(params.width) + params.eps);
    float sum_dxhat = 0.0;
    float sum_dxhat_xhat = 0.0;
    for (uint i = 0; i < params.width; ++i) {
        float xhat = (x[base + i] - mean) * inv;
        float dxhat = out_grad[base + i] * scale[i];
        sum_dxhat += dxhat;
        sum_dxhat_xhat += dxhat * xhat;
    }
    float xhat = (x[id] - mean) * inv;
    float dxhat = out_grad[id] * scale[c];
    float g = (float(params.width) * dxhat - sum_dxhat - xhat * sum_dxhat_xhat) * inv / float(params.width);
    x_grad[id] += g;
}

kernel void layernorm_grad_scale_shift_kernel(device atomic_float* scale_grad [[buffer(0)]],
                                              device atomic_float* shift_grad [[buffer(1)]],
                                              device const float* x [[buffer(2)]],
                                              device const float* out_grad [[buffer(3)]],
                                              constant LayerNormParams& params [[buffer(4)]],
                                              constant ElementwiseGradParams& flags [[buffer(5)]],
                                              uint id [[thread_position_in_grid]]) {
    uint total = params.rows * params.width;
    if (id >= total) {
        return;
    }
    uint row = id / params.width;
    uint c = id % params.width;
    uint base = row * params.width;
    float mean = 0.0;
    for (uint i = 0; i < params.width; ++i) {
        mean += x[base + i];
    }
    mean /= float(params.width);
    float var = 0.0;
    for (uint i = 0; i < params.width; ++i) {
        float z = x[base + i] - mean;
        var += z * z;
    }
    float inv = rsqrt(var / float(params.width) + params.eps);
    float xhat = (x[id] - mean) * inv;
    if (flags.has_a != 0) {
        atomic_fetch_add_explicit(&scale_grad[c], out_grad[id] * xhat, memory_order_relaxed);
    }
    if (flags.has_b != 0) {
        atomic_fetch_add_explicit(&shift_grad[c], out_grad[id], memory_order_relaxed);
    }
}

kernel void gelu_grad_kernel(device float* x_grad [[buffer(0)]],
                             device const float* x [[buffer(1)]],
                             device const float* out_grad [[buffer(2)]],
                             constant uint& count [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= count) {
        return;
    }
    float v = x[id];
    float u = 0.7978845608f * (v + 0.044715f * v * v * v);
    float th = tanh(u);
    float du = 0.7978845608f * (1.0f + 3.0f * 0.044715f * v * v);
    float g = 0.5f * (1.0f + th) + 0.5f * v * (1.0f - th * th) * du;
    x_grad[id] += out_grad[id] * g;
}

kernel void causal_mask_kernel(device const float* scores [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               constant CausalMaskParams& params [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    uint total = params.batches * params.heads * params.sequence_length * params.sequence_length;
    if (id >= total) {
        return;
    }
    uint j = id % params.sequence_length;
    uint i = (id / params.sequence_length) % params.sequence_length;
    out[id] = (j > i) ? params.mask_value : scores[id];
}

kernel void causal_mask_grad_kernel(device float* scores_grad [[buffer(0)]],
                                    device const float* out_grad [[buffer(1)]],
                                    constant CausalMaskParams& params [[buffer(2)]],
                                    uint id [[thread_position_in_grid]]) {
    uint total = params.batches * params.heads * params.sequence_length * params.sequence_length;
    if (id >= total) {
        return;
    }
    uint j = id % params.sequence_length;
    uint i = (id / params.sequence_length) % params.sequence_length;
    if (j <= i) {
        scores_grad[id] += out_grad[id];
    }
}
