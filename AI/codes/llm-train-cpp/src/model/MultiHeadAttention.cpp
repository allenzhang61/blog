#include "llm/model/MultiHeadAttention.hpp"

#include "llm/ops.hpp"

namespace llm {

MultiHeadAttention::MultiHeadAttention(int64_t d_in, int64_t d_out_, int64_t context, int64_t heads,
                                       bool qkv_bias, Device device)
    : d_out(d_out_), num_heads(heads), head_dim(d_out_ / heads), context_length(context),
      W_query(d_in, d_out_, qkv_bias, device), W_key(d_in, d_out_, qkv_bias, device),
      W_value(d_in, d_out_, qkv_bias, device), out_proj(d_out_, d_out_, true, device) {
    if (d_out % num_heads != 0) {
        throw std::runtime_error("d_out must be divisible by num_heads");
    }
}

Tensor MultiHeadAttention::forward(const Tensor& x) {
    int64_t B = x.shape()[0], T = x.shape()[1];
    Tensor q = ops::transpose(ops::reshape(W_query.forward(x), {B, T, num_heads, head_dim}), 1, 2);
    Tensor k = ops::transpose(ops::reshape(W_key.forward(x), {B, T, num_heads, head_dim}), 1, 2);
    Tensor v = ops::transpose(ops::reshape(W_value.forward(x), {B, T, num_heads, head_dim}), 1, 2);
    Tensor kt = ops::transpose(k, 2, 3);
    Tensor scores = ops::batch_matmul(q, kt);
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    scores = ops::mul_scalar(scores, scale);
    for (int64_t b = 0; b < B; ++b)
        for (int64_t h = 0; h < num_heads; ++h)
            for (int64_t i = 0; i < T; ++i)
                for (int64_t j = i + 1; j < T; ++j)
                    scores.data()[((b * num_heads + h) * T + i) * T + j] = -1e9;
    Tensor weights = ops::softmax(scores, -1);
    Tensor ctx = ops::batch_matmul(weights, v);
    ctx = ops::reshape(ops::transpose(ctx, 1, 2), {B, T, d_out});
    return out_proj.forward(ctx);
}

std::vector<Tensor*> MultiHeadAttention::parameters() {
    auto p = W_query.parameters();
    auto append = [&](std::vector<Tensor*> q) {
        p.insert(p.end(), q.begin(), q.end());
    };
    append(W_key.parameters());
    append(W_value.parameters());
    append(out_proj.parameters());
    return p;
}

} // namespace llm
