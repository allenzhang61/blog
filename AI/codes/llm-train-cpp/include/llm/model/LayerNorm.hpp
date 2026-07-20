#pragma once

#include "llm/model/Module.hpp"

namespace llm {

// LayerNorm 层。
// 对每个 token 的 embedding 维度做归一化，稳定训练。
class LayerNorm : public Module {
public:
    // 缩放参数 gamma。
    Tensor scale;

    // 平移参数 beta。
    Tensor shift;

    // 防止除零的小常数。
    double eps{1e-5};

    // emb_dim 是要归一化的最后一维大小。
    explicit LayerNorm(int64_t emb_dim, Device device = {});

    // 输入和输出形状相同。
    Tensor forward(const Tensor& x);

    // 返回 scale 和 shift。
    std::vector<Tensor*> parameters() override;
};

} // namespace llm
