//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_MOE_ROUTER_H
#define LOCAL_LLM_DEEPSEEK_MOE_ROUTER_H

#include "llm/module/Module.h"
#include "tensor/CPUTensor.h"
#include "tensor/DiskTensor.h"

#include <vector>

class DeepseekConfig;
class DeepseekSession;
struct DeepseekLayerWeights;

struct MoERoute {
    std::vector<int> expert_ids;
    std::vector<float> weights;
};

class MoERouter : public Module {
public:
    MoERouter(const DeepseekLayerWeights &weights, const DeepseekConfig &config);

    MoERoute forward(DeepseekSession &session, const GPUTensor &normed);

private:
    const DeepseekConfig &config_;
    const DeepseekLayerWeights &lw_;
};

#endif // LOCAL_LLM_DEEPSEEK_MOE_ROUTER_H
