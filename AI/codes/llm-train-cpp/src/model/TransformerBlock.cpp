#include "llm/model/TransformerBlock.hpp"

#include "llm/ops.hpp"

namespace llm {

TransformerBlock::TransformerBlock(const GPTConfig& cfg)
    : att(cfg.emb_dim, cfg.emb_dim, cfg.context_length, cfg.n_heads, cfg.qkv_bias, cfg.device),
      ff(cfg), norm1(cfg.emb_dim, cfg.device), norm2(cfg.emb_dim, cfg.device) {}

// 带残差连接的 Transformer 块：x: (B, T, C) -> 返回: (B, T, C)（C=emb_dim）
Tensor TransformerBlock::forward(const Tensor& x) {
    Tensor y = att.forward(norm1.forward(x)); // norm1+注意力，(B,T,C)->(B,T,C)
    Tensor a = ops::add(x, y);                 // 残差相加，(B,T,C)
    Tensor z = ff.forward(norm2.forward(a));   // norm2+前馈，(B,T,C)->(B,T,C)
    return ops::add(a, z);                     // 残差相加，(B,T,C)
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
