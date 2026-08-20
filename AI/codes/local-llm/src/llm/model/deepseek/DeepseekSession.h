//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_SESSION_H
#define LOCAL_LLM_DEEPSEEK_SESSION_H

#include "backend/cuda/mem/CudaWeight.h"
#include "backend/cuda/mem/SessionBase.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "tensor/Tensor.h"

#include <vector>

// 每层 latent KV cache：布局 [max_seq_len, kv_lora + qk_rope]，float。
struct LatentKVCache {
    CudaWeight cache;
    int seq_len = 0;
};

// 一次推理请求作用域：持有 latent KV cache（每层一份）、YARN inv_freq（device 常量）、
// 前向 scratch 与已生成 token。prefill 时重建，decode 时复用。
class DeepseekSession : public SessionBase {
public:
    DeepseekSession(const DeepseekConfig &config, const Tensor &inputs,
                    int max_output_tokens);

    std::vector<LatentKVCache> kv_caches;
    CudaWeight inv_freq;          // [rope_dim/2] float，YARN 校正后的频率（device）
    float attn_softmax_scale = 0; // = mscale^2 / sqrt(qk_head)

    size_t kv_state_bytes() const override;
};

#endif // LOCAL_LLM_DEEPSEEK_SESSION_H
