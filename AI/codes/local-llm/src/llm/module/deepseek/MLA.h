//
// Created by zhangyoulun on 15/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_MLA_H
#define LOCAL_LLM_DEEPSEEK_MLA_H

#include "llm/module/Module.h"

class CudaWeightPool;
class DeepseekConfig;
class DeepseekSession;
class DeepseekWeights;

namespace deepseek {
class RMSNorm;
}

// MLA attention 子层。
class MLA : public Module {
public:
    MLA(const DeepseekConfig &config, const DeepseekWeights &weights, CudaWeightPool *pool,
        deepseek::RMSNorm *rms_norm);

    void forward(DeepseekSession &session, int layer, float *d_hidden, int input_size, int start_pos);

private:
    const DeepseekConfig &config_;
    const DeepseekWeights &weights_;
    CudaWeightPool *pool_ = nullptr;
    deepseek::RMSNorm *rms_norm_ = nullptr;
};

#endif // LOCAL_LLM_DEEPSEEK_MLA_H
