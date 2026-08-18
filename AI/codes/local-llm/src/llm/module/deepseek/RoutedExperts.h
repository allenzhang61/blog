//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_ROUTED_EXPERTS_H
#define LOCAL_LLM_DEEPSEEK_ROUTED_EXPERTS_H

#include "llm/module/Module.h"
#include "llm/module/deepseek/MoERouter.h"

class CudaWeightPool;
class DeepseekConfig;
class DeepseekSession;
struct DeepseekLayerWeights;

class RoutedExperts : public Module {
public:
    RoutedExperts(const DeepseekLayerWeights &weights, const DeepseekConfig &config, CudaWeightPool *pool);

    void forward(DeepseekSession &session, const float *d_normed, const MoERoute &route,
                 int input_size, float *d_moe);

private:
    const DeepseekConfig &config_;
    const DeepseekLayerWeights &lw_;
    CudaWeightPool *pool_ = nullptr;
};

#endif // LOCAL_LLM_DEEPSEEK_ROUTED_EXPERTS_H
