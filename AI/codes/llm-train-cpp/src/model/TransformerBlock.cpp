#include "llm/model/TransformerBlock.hpp"

#include "llm/ops.hpp"

namespace llm {

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
    auto append = [&](std::vector<Tensor*> q) {
        p.insert(p.end(), q.begin(), q.end());
    };
    append(ff.parameters());
    append(norm1.parameters());
    append(norm2.parameters());
    return p;
}

} // namespace llm
