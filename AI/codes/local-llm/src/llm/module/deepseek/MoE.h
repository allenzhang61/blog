//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_MOE_H
#define LOCAL_LLM_DEEPSEEK_MOE_H

#include "llm/module/Module.h"
#include "tensor/Tensor.h"
#include "llm/module/deepseek/MoERouter.h"
#include "llm/module/deepseek/RoutedExperts.h"
#include "llm/module/deepseek/SharedExperts.h"

class DeepseekConfig;
class DeepseekSession;
struct DeepseekLayerWeights;

class MoE : public Module {
public:
    MoE(const DeepseekLayerWeights &weights, const DeepseekConfig &config);

    void forward(DeepseekSession &session, const Tensor &hidden);

private:
    const DeepseekConfig &config_;
    const DeepseekLayerWeights &weights_;
    MoERouter router_;
    RoutedExperts routed_experts_;
    SharedExperts shared_experts_;
};

#endif // LOCAL_LLM_DEEPSEEK_MOE_H
