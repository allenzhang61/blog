//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_BASEMODEL_H
#define LOCAL_LLM_BASEMODEL_H

#include <string>
#include <vector>

class CudaWeightPool;
class MemoryUsageProvider;

// 模型无关的推理接口：把推理主循环（main）与性能采集设施真正依赖的最小能力
// 抽象出来，使其无需依赖任何具体模型类型（QwenModel / DeepseekModel 等）。
//
// 为什么只抽象这一组接口：
//   - main 的主循环只需要「编码 -> prefill -> 逐步 decode -> 解码 -> 停止条件」，
//     以及供采集器读取的「权重池 + 本次请求显存用量」。
//   - 各模型的 Config / Weights / Session 形态差异极大（如 Qwen 的 KV+recurrent
//     state 与 Deepseek 的 latent KV cache、MoE 专家权重），强行统一成基类会引入
//     大量空实现与向下转型，得不偿失。故这些留在各模型内部，只经本接口暴露必需能力。
//
// 生命周期约定：具体实现内部持有并管理一次请求的 Session。prefill() 会为一次新的
// 生成开启内部 Session（喂入整段 prompt），随后 decode() 复用该 Session 逐 token 生成。
class BaseModel {
public:
    virtual ~BaseModel() = default;

    // 模型名（如 "qwen" / "deepseek"），用于日志与报告文件区分。
    virtual const char *name() const = 0;

    // 生成终止 token id（来自各模型的 config）；decode 产出该 id 时应停止。
    virtual int eos_token_id() const = 0;

    // 文本 -> token id 序列。
    virtual std::vector<int> encode(const std::string &text) const = 0;

    // token id 序列 -> 文本（encode 的逆过程）。
    virtual std::string decode_text(const std::vector<int> &ids) const = 0;

    // prefill：为一次新生成开启内部 Session，喂入整段 prompt token，返回首个生成 token id。
    virtual int prefill(const std::vector<int> &input_ids) = 0;

    // decode：喂入上一个 token（位置 pos），返回下一个 token id。复用 prefill 建立的 Session。
    virtual int decode(int prev_token_id, int pos) = 0;

    // 把一个已确定的生成 token 记入当前 Session（供 attention 上下文与最终解码使用）。
    virtual void append_output(int token_id) = 0;
    // 当前 Session 已生成的 token id 序列（prefill 之后有效）。
    virtual const std::vector<int> &outputs() const = 0;

    // === 供性能采集设施使用（与具体 Session/Weights 形态解耦）===
    // 本次请求的显存用量（跨 token 状态 + 临时激活），由内部 Session 实现该接口。
    virtual const MemoryUsageProvider &memory_usage() const = 0;
    // 持久权重池（跨请求常驻），供 MemoryReporter / WeightLoadTracker 采集。
    virtual CudaWeightPool &weight_pool() = 0;
};

#endif // LOCAL_LLM_BASEMODEL_H
