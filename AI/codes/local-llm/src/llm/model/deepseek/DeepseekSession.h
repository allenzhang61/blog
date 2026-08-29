//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEK_SESSION_H
#define LOCAL_LLM_DEEPSEEK_SESSION_H

#include "backend/cuda/mem/SessionBase.h"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "tensor/StorageTensor.h"
#include "tensor/GPUTensor.h"

#include <cstddef>
#include <vector>

// 每层 latent KV cache：布局 [max_seq_len, kv_lora + qk_rope]，float。
struct LatentKVCache {
    GPUTensor g_cache_f32;
    int seq_len = 0;
};

// 一次推理请求作用域：持有 latent KV cache（每层一份）、YARN inv_freq（device 常量）、
// 前向 scratch 与已生成 token。prefill 时重建，decode 时复用。
class DeepseekSession : public SessionBase {
public:
    DeepseekSession(const DeepseekConfig &config, std::vector<int> h_input_i32, int max_output_tokens);
    ~DeepseekSession() override;

    // decode 单步 token id 常驻 device：embedding 从此读取输入 token，
    // greedy argmax 将下一 token 写回此处，避免每步回传整行 logits。
    int *d_token() const { return d_token_; }

    std::vector<LatentKVCache> kv_caches;
    GPUTensor g_inv_freq_f32;         // [rope_dim/2] float，YARN 校正后的频率（scratch device view）
    float attn_softmax_scale = 0; // = mscale^2 / sqrt(qk_head)
    int trace_pos = -1;
    int trace_layer = -1;

    size_t kv_state_bytes() const override;

private:
    int *d_token_ = nullptr;
};

#endif // LOCAL_LLM_DEEPSEEK_SESSION_H
