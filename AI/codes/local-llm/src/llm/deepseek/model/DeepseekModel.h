//
// Created by zhangyoulun on 9/8/2026.
//

#ifndef LOCAL_LLM_DEEPSEEKMODEL_H
#define LOCAL_LLM_DEEPSEEKMODEL_H

#include "llm/BaseModel.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "format/gguf/GgufFile.h"
#include "llm/deepseek/DeepseekConfig.h"
#include "llm/deepseek/DeepseekTokenizer.h"
#include "llm/deepseek/DeepseekWeights.h"
#include "llm/deepseek/model/DeepseekSession.h"

#include <memory>
#include <string>
#include <vector>

// DeepSeek-V2-Lite（MLA + DeepSeekMoE，Q4_K 量化，GGUF 权重）推理模型，实现 BaseModel。
class DeepseekModel : public BaseModel {
public:
    DeepseekModel(const std::string &model_dir, int max_output_tokens);
    ~DeepseekModel() override;

    const char *name() const override { return "deepseek"; }
    int eos_token_id() const override { return config_.eos_token_id; }
    std::vector<int> encode(const std::string &text) const override { return tokenizer_.Encode(text); }
    std::string decode_text(const std::vector<int> &ids) const override { return tokenizer_.Decode(ids); }
    int prefill(const std::vector<int> &input_ids) override;
    int decode(int prev_token_id, int pos) override;
    void append_output(int token_id) override;
    const std::vector<int> &outputs() const override;
    const MemoryUsageProvider &memory_usage() const override;
    CudaWeightPool &weight_pool() override { return pool_; }

private:
    // 前向：tokens=N（prefill）或 1（decode）。start_pos 为首 token 的绝对位置。
    int forward(DeepseekSession &session, const std::vector<int> &token_ids, int start_pos);
    void layer_forward(DeepseekSession &session, int layer, int tokens, int start_pos);
    void mla_forward(DeepseekSession &session, int layer, int tokens, int start_pos);
    void ffn_dense_forward(DeepseekSession &session, int layer, int tokens);
    void ffn_moe_forward(DeepseekSession &session, int layer, int tokens);

    // 权重 GPU 化辅助：常驻量化 + 反量化到 f16 视图（可直接喂 gemm_weight）。
    CudaWeight dequant_weight(const TensorView *t, CudaScratchBuffer<uint16_t> &buf,
                              const std::string &tag);
    // 3D 专家权重的第 e 个专家切片反量化。
    CudaWeight dequant_expert(const TensorView *t, int expert, int n_experts,
                              CudaScratchBuffer<uint16_t> &buf, const std::string &tag);
    // 上传 F32 norm 权重（返回 device float* via CudaWeight，type=CUDA_R_8I 原始字节）。
    const float *resident_f32(const TensorView *t, const std::string &tag);

    GgufFile gguf_;
    DeepseekConfig config_;
    DeepseekWeights weights_;
    DeepseekTokenizer tokenizer_;
    CudaWeightPool pool_;
    int max_output_tokens_ = 0;
    std::unique_ptr<DeepseekSession> session_;
};

#endif // LOCAL_LLM_DEEPSEEKMODEL_H
