//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_FORWARD_H
#define LOCAL_LLM_DEEPSEEK_FORWARD_H

#include "llm/module/Module.h"

#include <vector>

class CudaWeightPool;
class DeepseekConfig;
class DeepseekSession;
class DeepseekWeights;
class MLA;
class MLP;
class Sampler;

// DeepSeek 整体前向：embedding -> layers -> final norm -> lm head -> sample。
class Forward : public Module {
public:
    Forward(const DeepseekConfig &config, const DeepseekWeights &weights, CudaWeightPool *pool,
            MLA *mla, MLP *mlp, Sampler *sampler);

    int forward(DeepseekSession &session, const std::vector<int> &token_ids, int start_pos);

private:
    const DeepseekConfig &config_;
    const DeepseekWeights &weights_;
    CudaWeightPool *pool_ = nullptr;
    MLA *mla_ = nullptr;
    MLP *mlp_ = nullptr;
    Sampler *sampler_ = nullptr;
};

#endif // LOCAL_LLM_DEEPSEEK_FORWARD_H
