//
// Created by zhangyoulun on 9/8/2026.
//

#include "DeepseekModel.h"

#include <utility>

DeepseekModel::DeepseekModel(std::unique_ptr<MF> mf, int max_output_tokens, const SamplingConfig &sampling)
    : mf_(std::move(mf)),
      config_(*mf_),
      weights_(*mf_, config_),
      max_output_tokens_(max_output_tokens),
      sampler_(sampling),
      mla_(config_, weights_, &global_cuda_weight_pool()),
      mlp_(config_, weights_, &global_cuda_weight_pool()),
      forward_(config_, weights_, &global_cuda_weight_pool(), &mla_, &mlp_, &sampler_) {
    config_.DebugDump();
}

DeepseekModel::~DeepseekModel() = default;
