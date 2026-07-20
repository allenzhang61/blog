#pragma once

#include "llm/model/Module.hpp"

namespace llm {

// GELU 激活函数模块。
// GPT 的 FeedForward 中常用 GELU 代替 ReLU。
class GELU : public Module {
public:
    // 对输入逐元素应用 GELU，输出形状与输入一致。
    Tensor forward(const Tensor& x);
};

} // namespace llm
