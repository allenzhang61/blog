#include "llm/model/FeedForward.hpp"

namespace llm {

FeedForward::FeedForward(const GPTConfig& cfg)
    : fc1(cfg.emb_dim, 4 * cfg.emb_dim, true, cfg.device),
      fc2(4 * cfg.emb_dim, cfg.emb_dim, true, cfg.device) {
}

Tensor FeedForward::forward(const Tensor& x) {
    Tensor hidden = fc1.forward(x);
    Tensor activated = gelu.forward(hidden);
    return fc2.forward(activated);
}

std::vector<Tensor*> FeedForward::parameters() {
    auto p = fc1.parameters();
    auto p2 = fc2.parameters();
    p.insert(p.end(), p2.begin(), p2.end());
    return p;
}

} // namespace llm
