//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_SHARED_EXPERTS_H
#define LOCAL_LLM_DEEPSEEK_SHARED_EXPERTS_H

#include "llm/module/Module.h"

class CudaWeightPool;
class DeepseekConfig;
class DeepseekSession;
struct DeepseekLayerWeights;

class SharedExperts : public Module {
public:
    SharedExperts(const DeepseekConfig &config, CudaWeightPool *pool);

    void forward(DeepseekSession &session, const DeepseekLayerWeights &weights,
                 const float *d_normed, int input_size, float *d_moe);

private:
    const DeepseekConfig &config_;
    CudaWeightPool *pool_ = nullptr;
};

#endif // LOCAL_LLM_DEEPSEEK_SHARED_EXPERTS_H
