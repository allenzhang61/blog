#include "llm/model/FeedForward.hpp"

namespace llm {

FeedForward::FeedForward(const GPTConfig& cfg)
    : fc1(cfg.emb_dim, 4 * cfg.emb_dim, true, cfg.device),
      fc2(4 * cfg.emb_dim, cfg.emb_dim, true, cfg.device) {
}

// 位置前馈网络，先升维再降回：x: (B, T, C) -> 返回: (B, T, C)
// B=batch size（一次处理多少条序列），T=序列长度（每条多少个 token），C=emb_dim（每个 token 的特征维度）
Tensor FeedForward::forward(const Tensor& x) {
    Tensor hidden = fc1.forward(x);          // (B,T,C) -> (B,T,4C)
    Tensor activated = gelu.forward(hidden); // (B,T,4C) 逐元素激活，形状不变
    return fc2.forward(activated);           // (B,T,4C) -> (B,T,C)
}

std::vector<Tensor*> FeedForward::parameters() {
    auto p = fc1.parameters();
    auto p2 = fc2.parameters();
    p.insert(p.end(), p2.begin(), p2.end());
    return p;
}

} // namespace llm
