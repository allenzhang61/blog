//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_MLA_H
#define LOCAL_LLM_DEEPSEEK_MLA_H

#include "llm/module/Module.h"

class CudaWeightPool;
class DeepseekConfig;
class DeepseekSession;
struct DeepseekLayerWeights;

// MLA attention 子层（每层一个实例，持有该层的权重引用）。
class MLA : public Module {
public:
    MLA(const DeepseekLayerWeights &weights, const DeepseekConfig &config, CudaWeightPool *pool);

    void forward(DeepseekSession &session, float *d_hidden, int input_size, int start_pos);

private:
    const DeepseekConfig &config_;
    const DeepseekLayerWeights &lw_;
    CudaWeightPool *pool_ = nullptr;
};

#endif // LOCAL_LLM_DEEPSEEK_MLA_H
