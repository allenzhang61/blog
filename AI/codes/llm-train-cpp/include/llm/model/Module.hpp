#pragma once

#include "llm/tensor.hpp"

namespace llm {

// 神经网络模块基类。
// 所有可训练模块都通过 parameters() 暴露自己的参数张量。
class Module {
public:
    virtual ~Module() = default;

    // 返回当前模块直接持有的可训练参数。
    // 复合模块会在实现中汇总子模块参数。
    virtual std::vector<Tensor*> parameters();

    // 将模块内所有参数的梯度清零。
    void zero_grad();
};

} // namespace llm
