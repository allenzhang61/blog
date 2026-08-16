//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_MOE_H
#define LOCAL_LLM_DEEPSEEK_MOE_H

#include "llm/module/Module.h"
#include "llm/module/deepseek/MoERouter.h"
#include "llm/module/deepseek/RoutedExperts.h"
#include "llm/module/deepseek/SharedExperts.h"

class CudaWeightPool;
class DeepseekConfig;
class DeepseekSession;
class DeepseekWeights;

namespace deepseek {
class RMSNorm;
}

class MoE : public Module {
public:
    MoE(const DeepseekConfig &config, const DeepseekWeights &weights, CudaWeightPool *pool,
        deepseek::RMSNorm *rms_norm);

    void forward(DeepseekSession &session, int layer, float *d_hidden, int tokens);

private:
    const DeepseekConfig &config_;
    const DeepseekWeights &weights_;
    CudaWeightPool *pool_ = nullptr;
    deepseek::RMSNorm *rms_norm_ = nullptr;
    MoERouter router_;
    RoutedExperts routed_experts_;
    SharedExperts shared_experts_;
};

#endif // LOCAL_LLM_DEEPSEEK_MOE_H
