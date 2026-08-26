//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEKMODEL_H
#define LOCAL_LLM_DEEPSEEKMODEL_H

#include "llm/model/BaseModel.h"
#include "format/MF.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/module/common/Embedding.h"
#include "llm/module/deepseek/MLA.h"
#include "llm/module/deepseek/MLP.h"
#include "llm/module/common/LMHead.h"
#include "llm/module/common/RMSNorm.h"
#include "tensor/CPUTensor.h"
#include "utils/sampling/Sampler.h"

#include <memory>
#include <string>
#include <vector>

// DeepSeek-V2-Lite（MLA + DeepSeekMoE，Q4_K 量化，GGUF 权重）推理模型，实现 BaseModel。
class DeepseekModel : public BaseModel {
public:
    DeepseekModel(std::unique_ptr<MF> mf, int max_output_tokens, const SamplingConfig &sampling);
    ~DeepseekModel() override;

    const char *name() const override { return "deepseek"; }
    int eos_token_id() const override { return config_.eos_token_id; }
    std::vector<int> encode_text(const std::string &text) const override { return mf_->tokenizer_encode(text); }
    std::string decode_text(const std::vector<int> &ids) const override { return mf_->tokenizer_decode(ids); }
    SessionBase *create_session(const std::string &text) override;
    int prefill(SessionBase &session) override;
    int decode(SessionBase &session) override;
    std::string output(const SessionBase &session) const override;
    const MemoryUsageProvider &memory_usage(const SessionBase &session) const override;

private:
    int forward_session(DeepseekSession &session, const CPUTensor &c_input_i32, int start_pos);

    std::unique_ptr<MF> mf_;
    DeepseekConfig config_;
    DeepseekWeights weights_;
    int max_output_tokens_ = 0;
    Sampler sampler_;
    Embedding embedding_;
    std::vector<MLA> mla_layers_;
    std::vector<MLP> mlp_layers_;
    LMHead lm_head_;
};

#endif // LOCAL_LLM_DEEPSEEKMODEL_H
