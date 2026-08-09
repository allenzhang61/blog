#pragma once

namespace llm_inference {

// 推理模块基类，只表达模型结构层级，不负责训练参数聚合。
class Module {
public:
    virtual ~Module() = default;

    // 返回模块名，方便调试和阅读模型组成。
    virtual const char * name() const = 0;
};

} // namespace llm_inference

