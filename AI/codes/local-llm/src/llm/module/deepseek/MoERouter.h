//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_MOE_ROUTER_H
#define LOCAL_LLM_DEEPSEEK_MOE_ROUTER_H

#include "llm/module/Module.h"

#include <vector>

class CudaWeightPool;
class DeepseekConfig;
class DeepseekSession;
struct DeepseekLayerWeights;

struct MoERoute {
    std::vector<int> expert_ids;
    std::vector<float> weights;
};

class MoERouter : public Module {
public:
    MoERouter(const DeepseekLayerWeights &weights, const DeepseekConfig &config, CudaWeightPool *pool);

    MoERoute forward(DeepseekSession &session, const float *d_normed, int input_size);

private:
    const DeepseekConfig &config_;
    const DeepseekLayerWeights &lw_;
    CudaWeightPool *pool_ = nullptr;
};

#endif // LOCAL_LLM_DEEPSEEK_MOE_ROUTER_H
