//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_MODULE_H
#define LOCAL_LLM_MODULE_H

// 所有 LLM 前向 Module 的抽象基类（概念上类比 PyTorch 的 nn.Module）。
//
// 与 PyTorch 的关键区别（本项目为手写 CUDA 推理）：
//  1) Module 不拥有权重：权重是 mmap 的 Tensor 引用，device 副本由 Tensor 内部
//     惰性上传并缓存；Module 只持 Tensor 引用。
//  2) Module 不持有临时激活：前向中反复覆盖的中间 buffer 由模型自己的 Scratch
//     统一管理，作为参数传入 forward 或经 Session 访问。
//  3) Module 不持有跨 token 状态：KV cache / recurrent state 在模型自己的 Session
//     里，按 layer_index 取用。
//
// 因此 Module 自身是无 per-request 状态的纯计算单元，天然并发安全——
// 所有随请求变化的状态都在传入的 Session / Scratch 中，每请求一份。
//
// 前向区分两条路径：
//  - prefill：一次处理整段输入（tokens 个），批量 kernel；
//  - decode ：处理单个新 token（1 个），单步 kernel，依赖已有 KV / state。
class Module {
public:
    virtual ~Module() = default;
};


#endif //LOCAL_LLM_MODULE_H
