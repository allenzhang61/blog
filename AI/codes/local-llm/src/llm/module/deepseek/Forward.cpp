//
// Created by zhangyoulun on 15/8/2026.
//

#include "Forward.h"

#include "backend/cuda/common.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "backend/cuda/ops/gemm.h"
#include "backend/cuda/ops/kernel.cuh"
#include "llm/model/deepseek/DeepseekConfig.h"
#include "llm/model/deepseek/DeepseekSession.h"
#include "llm/model/deepseek/DeepseekWeights.h"
#include "llm/model/deepseek/DeepseekModel.h"
#include "llm/module/deepseek/MLA.h"
#include "llm/module/deepseek/MLP.h"
#include "utils/sampling/Sampler.h"

#include <cuda_runtime.h>
#include <memory>

Forward::Forward(const DeepseekConfig &config, const DeepseekWeights &weights, CudaWeightPool *pool,
                 MLA *mla, MLP *mlp, Sampler *sampler)
    : config_(config),
      weights_(weights),
      pool_(pool),
      mla_(mla),
      mlp_(mlp),
      sampler_(sampler) {}

int Forward::forward(DeepseekSession &session, const std::vector<int> &token_ids, int start_pos) {
    auto &s = session.scratch;
    const int tokens = static_cast<int>(token_ids.size());
    const int H = config_.hidden_size;
    const int vocab = config_.vocab_size;

    float *d_hidden = s.hidden.ensure(static_cast<size_t>(tokens) * H, "ds.hidden");
    CudaWeight embd = pool_->cached_weight(*weights_.token_embd)->try_dequant();
    int *d_ids = s.top_idx.ensure(static_cast<size_t>(tokens), "ds.ids");
    check_cuda(cudaMemcpy(d_ids, token_ids.data(), tokens * sizeof(int), cudaMemcpyHostToDevice), "ds.ids.h2d");
    launch_embedding_lookup(static_cast<const uint16_t *>(embd.ptr), d_ids, d_hidden, tokens, vocab, H, 1, nullptr);

    for (int i = 0; i < config_.num_layers; ++i) {
        mla_->forward(session, i, tokens, start_pos);
        mlp_->forward(session, i, tokens);
    }

    const int last = tokens - 1;
    float *d_last = d_hidden + static_cast<size_t>(last) * H;
    float *d_normed = s.normed.ensure(static_cast<size_t>(H), "ds.final_normed");
    CudaWeight *output_norm = pool_->cached_weight(*weights_.output_norm);
    launch_rms_norm(d_last, output_norm->ptr, 2, d_normed, 1, H,
                    config_.rms_norm_eps, false, nullptr);

    float *d_logits = s.logits.ensure(static_cast<size_t>(vocab), "ds.logits");
    CudaWeight w = pool_->cached_weight(*weights_.output)->try_dequant();
    uint16_t *xlow = s.logits_in_lowp.ensure(static_cast<size_t>(H), "ds.logits_in_lowp");
    to_weight_lowp(d_normed, xlow, H, w, nullptr);
    gemm_weight(pool_->handle, w, vocab, H, xlow, w.type, 1, d_logits, "ds.gemm.lm_head");

    s.h_logits.resize(static_cast<size_t>(vocab));
    check_cuda(cudaMemcpy(s.h_logits.data(), d_logits, static_cast<size_t>(vocab) * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "ds.logits.d2h");
    return sampler_->sample(s.h_logits.data(), vocab, session.h_outputs);
}

int DeepseekModel::prefill(const std::vector<int> &input_ids) {
    session_ = std::make_unique<DeepseekSession>(config_, input_ids, max_output_tokens_);
    return forward_.forward(*session_, input_ids, 0);
}

int DeepseekModel::decode(int prev_token_id, int pos) {
    return forward_.forward(*session_, {prev_token_id}, pos);
}

void DeepseekModel::append_output(int token_id) { session_->h_outputs.push_back(token_id); }

const std::vector<int> &DeepseekModel::outputs() const { return session_->h_outputs; }

const MemoryUsageProvider &DeepseekModel::memory_usage() const { return *session_; }
