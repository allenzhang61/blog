//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_QWENMODEL_H
#define LOCAL_LLM_QWENMODEL_H

#include <memory>
#include <string>
#include <vector>

#include "llm/module/Module.h"
#include "llm/module/common/Embedding.h"
#include "llm/module/qwen/DecoderLayer.h"
#include "llm/module/common/LMHead.h"
#include "llm/module/common/RMSNorm.h"

#include "format/MF.h"
#include "llm/model/BaseModel.h"
#include "llm/model/qwen/QwenConfig.h"
#include "llm/model/qwen/QwenWeights.h"
#include "tensor/CPUTensor.h"
#include "utils/sampling/Sampler.h"

class QwenSession;

// Qwen 文本塔顶层，对应整个 text_config：
//   embed_tokens -> 32 × DecoderLayer -> final_norm -> lm_head。
// 负责串起各子 Module，协调 prefill / decode 两条路径。
//
// 作为 BaseModel 的实现，QwenModel 自持模型级对象：
//   config / weights / tokenizer。
// Module 本身无 per-request 状态；跨 token 状态与临时激活分别在 QwenSession / QwenSession::scratch。
class QwenModel : public Module, public BaseModel {
public:
    // 从已打开的模型文件加载 config / weights / tokenizer，并建立各子 Module。
    // max_output_tokens 用于每次 prefill 时按需分配 session 的 KV cache 容量。
    // sampling 为采样配置（默认贪心）。
    QwenModel(std::unique_ptr<MF> mf, int max_output_tokens, const SamplingConfig &sampling);
    ~QwenModel() override;

    // === BaseModel ===
    const char *name() const override { return "qwen"; }
    int eos_token_id() const override { return config_.data.text.eos_token_id; }
    std::vector<int> encode_text(const std::string &text) const override { return mf_->tokenizer_encode(text); }
    std::string decode_text(const std::vector<int> &ids) const override { return mf_->tokenizer_decode(ids); }

    SessionBase *create_session(const std::string &text) override;
    // prefill：在传入 session 中喂入整段输入 token，跑完各层，返回首个生成 token id。
    int prefill(SessionBase &session) override;
    // decode：喂入 session 中最后一个已生成 token，返回下一个 token id，复用传入的 session。
    int decode(SessionBase &session_base) override;
    std::string output(const SessionBase &session) const override;

    const MemoryUsageProvider &memory_usage(const SessionBase &session) const override;

private:
    // 把 embedding→32 层→final_norm→lm_head GEMM→GPU argmax 这一段纯 GPU 计算录进 graph_，
    // token/pos 均从 session 的 device buffer 读写，构成可 replay 的闭环。
    void record_decode_graph(QwenSession &session);

    // 贪心 decode 单步的 eager 版（不 capture）：与 graph 内计算等价，token 从 d_token 读、
    // GPU argmax 结果写回 d_token。用于首步预热 scratch，使随后 capture 不触发 cudaMalloc。
    void eager_decode_greedy_device(QwenSession &session);

    std::unique_ptr<MF> mf_;
    QwenConfig config_;
    QwenWeights weights_;
    int max_output_tokens_ = 0;
    Sampler sampler_;

    Embedding embedding_;
    std::vector<DecoderLayer> layers_;
    LMHead lm_head_;
};


#endif //LOCAL_LLM_QWENMODEL_H
