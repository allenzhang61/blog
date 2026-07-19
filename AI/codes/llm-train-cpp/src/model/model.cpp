#include "llm/model.hpp"

namespace llm {

std::vector<Tensor*> Module::parameters() { return {}; }

void Module::zero_grad() {
    for (auto* p : parameters()) p->zero_grad();
}

Linear::Linear(int64_t in_features, int64_t out_features, bool bias_enabled, Device device)
    : weight(Tensor::randn({in_features, out_features}, 0.02, device, true)),
      bias(Tensor::zeros({out_features}, device, true)),
      use_bias(bias_enabled) {}

Tensor Linear::forward(const Tensor& x) {
    int64_t in_features = weight.shape()[0];
    int64_t out_features = weight.shape()[1];
    std::vector<int64_t> out_shape = x.shape();
    if (out_shape.empty() || out_shape.back() != in_features) throw std::runtime_error("Linear input shape mismatch");
    int64_t rows = x.numel() / in_features;
    Tensor flat = ops::reshape(x, {rows, in_features});
    Tensor y = ops::matmul(flat, weight);
    if (use_bias) y = ops::add(y, bias);
    out_shape.back() = out_features;
    return ops::reshape(y, out_shape);
}

std::vector<Tensor*> Linear::parameters() {
    if (use_bias) return {&weight, &bias};
    return {&weight};
}

Embedding::Embedding(int64_t num_embeddings, int64_t embedding_dim, Device device)
    : weight(Tensor::randn({num_embeddings, embedding_dim}, 0.02, device, true)) {}

Tensor Embedding::forward(const Tensor& ids) { return ops::embedding(ids, weight); }
std::vector<Tensor*> Embedding::parameters() { return {&weight}; }

LayerNorm::LayerNorm(int64_t emb_dim, Device device)
    : scale(Tensor::ones({emb_dim}, device, true)), shift(Tensor::zeros({emb_dim}, device, true)) {}

Tensor LayerNorm::forward(const Tensor& x) { return ops::layernorm(x, scale, shift, eps); }
std::vector<Tensor*> LayerNorm::parameters() { return {&scale, &shift}; }

Tensor GELU::forward(const Tensor& x) { return ops::gelu(x); }

MultiHeadAttention::MultiHeadAttention(int64_t d_in, int64_t d_out_, int64_t context, int64_t heads, bool qkv_bias, Device device)
    : d_out(d_out_), num_heads(heads), head_dim(d_out_ / heads), context_length(context),
      W_query(d_in, d_out_, qkv_bias, device), W_key(d_in, d_out_, qkv_bias, device),
      W_value(d_in, d_out_, qkv_bias, device), out_proj(d_out_, d_out_, true, device) {
    if (d_out % num_heads != 0) throw std::runtime_error("d_out must be divisible by num_heads");
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
    auto append = [&](std::vector<Tensor*> q) { p.insert(p.end(), q.begin(), q.end()); };
    append(W_key.parameters()); append(W_value.parameters()); append(out_proj.parameters());
    return p;
}

FeedForward::FeedForward(const GPTConfig& cfg) : fc1(cfg.emb_dim, 4 * cfg.emb_dim, true, cfg.device), fc2(4 * cfg.emb_dim, cfg.emb_dim, true, cfg.device) {}
Tensor FeedForward::forward(const Tensor& x) { return fc2.forward(gelu.forward(fc1.forward(x))); }
std::vector<Tensor*> FeedForward::parameters() {
    auto p = fc1.parameters();
    auto p2 = fc2.parameters();
    p.insert(p.end(), p2.begin(), p2.end());
    return p;
}

TransformerBlock::TransformerBlock(const GPTConfig& cfg)
    : att(cfg.emb_dim, cfg.emb_dim, cfg.context_length, cfg.n_heads, cfg.qkv_bias, cfg.device),
      ff(cfg), norm1(cfg.emb_dim, cfg.device), norm2(cfg.emb_dim, cfg.device) {}

Tensor TransformerBlock::forward(const Tensor& x) {
    Tensor y = att.forward(norm1.forward(x));
    Tensor a = ops::add(x, y);
    Tensor z = ff.forward(norm2.forward(a));
    return ops::add(a, z);
}

std::vector<Tensor*> TransformerBlock::parameters() {
    auto p = att.parameters();
    auto append = [&](std::vector<Tensor*> q) { p.insert(p.end(), q.begin(), q.end()); };
    append(ff.parameters()); append(norm1.parameters()); append(norm2.parameters());
    return p;
}

GPTModel::GPTModel(GPTConfig cfg_)
    : cfg(cfg_), tok_emb(cfg.vocab_size, cfg.emb_dim, cfg.device), pos_emb(cfg.context_length, cfg.emb_dim, cfg.device),
      final_norm(cfg.emb_dim, cfg.device), out_head(cfg.emb_dim, cfg.vocab_size, false, cfg.device) {
    for (int64_t i = 0; i < cfg.n_layers; ++i) blocks.emplace_back(cfg);
}

Tensor GPTModel::forward(const Tensor& ids) {
    int64_t B = ids.shape()[0], T = ids.shape()[1];
    std::vector<int64_t> pos(T);
    for (int64_t i = 0; i < T; ++i) pos[i] = i;
    Tensor pos_ids = Tensor::from_ints(pos, {T}, cfg.device);
    Tensor x = ops::add(tok_emb.forward(ids), pos_emb.forward(pos_ids));
    for (auto& block : blocks) x = block.forward(x);
    x = final_norm.forward(x);
    return out_head.forward(x);
}

std::vector<Tensor*> GPTModel::parameters() {
    auto p = tok_emb.parameters();
    auto append = [&](std::vector<Tensor*> q) { p.insert(p.end(), q.begin(), q.end()); };
    append(pos_emb.parameters());
    for (auto& b : blocks) append(b.parameters());
    append(final_norm.parameters());
    append(out_head.parameters());
    return p;
}

} // namespace llm
