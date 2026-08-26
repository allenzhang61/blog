//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_BASEMODEL_H
#define LOCAL_LLM_BASEMODEL_H

#include <string>
#include <vector>

class MemoryUsageProvider;
class SessionBase;

// 模型无关的推理接口：把推理主循环（main）与性能采集设施真正依赖的最小能力
// 抽象出来，使其无需依赖任何具体模型类型（QwenModel / DeepseekModel 等）。
//
// 为什么只抽象这一组接口：
//   - main 的主循环只需要「创建 session -> prefill -> 逐步 decode -> 解码 -> 停止条件」，
//     以及供采集器读取的「本次请求显存用量」。
//   - 各模型的 Config / Weights / Session 形态差异极大（如 Qwen 的 KV+recurrent
//     state 与 Deepseek 的 latent KV cache、MoE 专家权重），强行统一成基类会引入
//     大量空实现与向下转型，得不偿失。故这些留在各模型内部，只经本接口暴露必需能力。
//
// 生命周期约定：调用方通过 create_session() 创建一次请求的 Session，并负责在请求结束
// 后 delete；prefill() / decode() / output() / memory_usage() 都显式接收该 Session。
class BaseModel {
public:
    virtual ~BaseModel() = default;

    // 模型名（如 "qwen" / "deepseek"），用于日志与报告文件区分。
    virtual const char *name() const = 0;

    // 生成终止 token id（来自各模型的 config）；decode 产出该 id 时应停止。
    virtual int eos_token_id() const = 0;

    // 文本 -> token id 序列。
    virtual std::vector<int> encode_text(const std::string &text) const = 0;

    // token id 序列 -> 文本（encode 的逆过程）。
    virtual std::string decode_text(const std::vector<int> &ids) const = 0;

    // 为一次新生成创建 Session，并在 Session 中保存编码后的输入 token。
    // 返回值归调用方所有，请求结束后必须 delete。
    virtual SessionBase *create_session(const std::string &text) = 0;

    // prefill：喂入 Session 中保存的整段输入 token，返回首个生成 token id。
    virtual int prefill(SessionBase &session) = 0;

    // decode：喂入 Session 中最后一个已生成 token，返回下一个 token id。复用传入的 Session。
    virtual int decode(SessionBase &session) = 0;

    // 解码 Session 已生成 token，返回最终文本。
    virtual std::string output(const SessionBase &session) const = 0;

    // === 供性能采集设施使用（与具体 Session/Weights 形态解耦）===
    // 本次请求的显存用量（跨 token 状态 + 临时激活），由传入 Session 实现该接口。
    virtual const MemoryUsageProvider &memory_usage(const SessionBase &session) const = 0;
};

#endif // LOCAL_LLM_BASEMODEL_H
