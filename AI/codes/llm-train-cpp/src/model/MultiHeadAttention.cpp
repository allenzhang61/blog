#include "llm/model/MultiHeadAttention.hpp"

#include "llm/ops.hpp"

#include <stdexcept>

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

// 符号：B=batch, T=序列长度, C=d_in(=emb_dim), d_out=输出维度, H=num_heads, Dh=head_dim(=d_out/H)
// x: (B, T, C) -> 返回: (B, T, d_out)
Tensor MultiHeadAttention::forward(const Tensor& x) {
    int64_t B = x.shape()[0], T = x.shape()[1];
    // 线性投影 (B,T,C)->(B,T,d_out)，拆成多头 (B,T,H,Dh)，再转置成 (B,H,T,Dh)
    Tensor q = ops::transpose(ops::reshape(W_query.forward(x), {B, T, num_heads, head_dim}), 1, 2);
    Tensor k = ops::transpose(ops::reshape(W_key.forward(x), {B, T, num_heads, head_dim}), 1, 2);
    Tensor v = ops::transpose(ops::reshape(W_value.forward(x), {B, T, num_heads, head_dim}), 1, 2);
    // k 转置最后两维：(B,H,T,Dh) -> (B,H,Dh,T)
    Tensor kt = ops::transpose(k, 2, 3);
    // 注意力分数 (B,H,T,Dh) @ (B,H,Dh,T) -> (B,H,T,T)
    Tensor scores = ops::batch_matmul(q, kt);
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    scores = ops::mul_scalar(scores, scale);          // (B,H,T,T) 按 1/sqrt(Dh) 缩放
    scores = ops::causal_mask(scores, T);             // (B,H,T,T) 上三角掩码
    Tensor weights = ops::softmax(scores, -1);        // (B,H,T,T) 末维归一化
    // 加权求和 (B,H,T,T) @ (B,H,T,Dh) -> (B,H,T,Dh)
    Tensor ctx = ops::batch_matmul(weights, v);
    // 合并多头：(B,H,T,Dh) -> (B,T,H,Dh) -> (B,T,d_out)
    ctx = ops::reshape(ops::transpose(ctx, 1, 2), {B, T, d_out});
    return out_proj.forward(ctx);                     // (B,T,d_out) -> (B,T,d_out)
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
