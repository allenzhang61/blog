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
