#include "llm/model/GPTModel.hpp"

#include "llm/ops.hpp"

namespace llm {

GPTModel::GPTModel(GPTConfig cfg_)
    : cfg(cfg_), tok_emb(cfg.vocab_size, cfg.emb_dim, cfg.device), pos_emb(cfg.context_length, cfg.emb_dim, cfg.device),
      final_norm(cfg.emb_dim, cfg.device), out_head(cfg.emb_dim, cfg.vocab_size, false, cfg.device) {
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        blocks.emplace_back(cfg);
    }
}

Tensor GPTModel::forward(const Tensor& ids) {
    int64_t T = ids.shape()[1];
    std::vector<int64_t> pos(T);
    for (int64_t i = 0; i < T; ++i) {
        pos[i] = i;
    }
    Tensor pos_ids = Tensor::from_ints(pos, {T}, cfg.device);
    Tensor x = ops::add(tok_emb.forward(ids), pos_emb.forward(pos_ids));
    for (auto& block : blocks) {
        x = block.forward(x);
    }
    x = final_norm.forward(x);
    return out_head.forward(x);
}

std::vector<Tensor*> GPTModel::parameters() {
    auto p = tok_emb.parameters();
    auto append = [&](std::vector<Tensor*> q) {
        p.insert(p.end(), q.begin(), q.end());
    };
    append(pos_emb.parameters());
    for (auto& b : blocks) {
        append(b.parameters());
    }
    append(final_norm.parameters());
    append(out_head.parameters());
    return p;
}

} // namespace llm
