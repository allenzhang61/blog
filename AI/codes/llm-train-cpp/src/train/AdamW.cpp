#include "llm/train/AdamW.hpp"

#include <cmath>
#include <utility>

#if LLM_CPP_ENABLE_CUDA_COMPILED
#include "../kernels/cuda/cuda_runtime.hpp"
#endif
#if LLM_CPP_ENABLE_METAL_COMPILED
#include "../kernels/metal/metal_runtime.hpp"
#endif

namespace llm {

AdamW::AdamW(std::vector<Tensor*> params, double lr, double weight_decay, double beta1, double beta2, double eps)
    : params_(std::move(params)),
      lr_(lr),
      weight_decay_(weight_decay),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps) {
    m_.reserve(params_.size());
    v_.reserve(params_.size());
    for (const auto* p : params_) {
        m_.push_back(std::vector<double>(static_cast<size_t>(p->numel()), 0.0));
        v_.push_back(std::vector<double>(static_cast<size_t>(p->numel()), 0.0));
#if LLM_CPP_ENABLE_CUDA_COMPILED
        if (p->device().type == DeviceType::CUDA && cuda::detail::CudaRuntime::instance().available()) {
            auto m = cuda::detail::CudaRuntime::instance().create_tensor_storage();
            auto v = cuda::detail::CudaRuntime::instance().create_tensor_storage();
            cuda::detail::CudaRuntime::instance().fill_data_buffer(*m, static_cast<size_t>(p->numel()), 0.0f);
            cuda::detail::CudaRuntime::instance().fill_data_buffer(*v, static_cast<size_t>(p->numel()), 0.0f);
            cuda_m_.push_back(std::move(m));
            cuda_v_.push_back(std::move(v));
        } else {
            cuda_m_.push_back(nullptr);
            cuda_v_.push_back(nullptr);
        }
#else
        cuda_m_.push_back(nullptr);
        cuda_v_.push_back(nullptr);
#endif
#if LLM_CPP_ENABLE_METAL_COMPILED
        if (p->device().type == DeviceType::Metal && metal::detail::runtime_available()) {
            auto m = metal::detail::create_tensor_storage();
            auto v = metal::detail::create_tensor_storage();
            metal::detail::fill_data_buffer(*m, static_cast<size_t>(p->numel()), 0.0f);
            metal::detail::fill_data_buffer(*v, static_cast<size_t>(p->numel()), 0.0f);
            metal_m_.push_back(std::move(m));
            metal_v_.push_back(std::move(v));
        } else {
            metal_m_.push_back(nullptr);
            metal_v_.push_back(nullptr);
        }
#else
        metal_m_.push_back(nullptr);
        metal_v_.push_back(nullptr);
#endif
    }
}

void AdamW::zero_grad() {
    for (auto* p : params_) {
        p->zero_grad();
    }
}

void AdamW::step() {
    ++step_;
    double bias_correction1 = 1.0 - std::pow(beta1_, static_cast<double>(step_));
    double bias_correction2 = 1.0 - std::pow(beta2_, static_cast<double>(step_));
    for (size_t pi = 0; pi < params_.size(); ++pi) {
        auto* p = params_[pi];
#if LLM_CPP_ENABLE_CUDA_COMPILED
        if (p->device().type == DeviceType::CUDA && p->node->cuda_storage && cuda_m_[pi] && cuda_v_[pi]) {
            if (p->node->host_data_dirty || !p->node->cuda_storage->data) {
                p->node->cuda_storage->copy_data_from_host(*p->node->cuda_storage, p->node->data);
                p->node->host_data_dirty = false;
                p->node->device_data_dirty = false;
            }
            if (p->node->host_grad_dirty || !p->node->cuda_storage->grad) {
                p->node->cuda_storage->copy_grad_from_host(*p->node->cuda_storage, p->node->grad);
                p->node->host_grad_dirty = false;
                p->node->device_grad_dirty = true;
            }
            cuda::detail::CudaRuntime::instance().adamw_update(
                *p->node->cuda_storage, *p->node->cuda_storage, *cuda_m_[pi], *cuda_v_[pi],
                static_cast<size_t>(p->numel()), static_cast<float>(lr_), static_cast<float>(weight_decay_),
                static_cast<float>(beta1_), static_cast<float>(beta2_), static_cast<float>(eps_),
                static_cast<float>(bias_correction1), static_cast<float>(bias_correction2));
            p->node->host_data_dirty = false;
            p->node->device_data_dirty = true;
            continue;
        }
#endif
#if LLM_CPP_ENABLE_METAL_COMPILED
        if (p->device().type == DeviceType::Metal && p->node->metal_storage && metal_m_[pi] && metal_v_[pi]) {
            if (p->node->host_data_dirty || !p->node->metal_storage->data) {
                p->node->metal_storage->copy_data_from_host(*p->node->metal_storage, p->node->data);
                p->node->host_data_dirty = false;
                p->node->device_data_dirty = false;
            }
            if (p->node->host_grad_dirty || !p->node->metal_storage->grad) {
                p->node->metal_storage->copy_grad_from_host(*p->node->metal_storage, p->node->grad);
                p->node->host_grad_dirty = false;
                p->node->device_grad_dirty = true;
            }
            metal::detail::adamw_update(
                *p->node->metal_storage, *p->node->metal_storage, *metal_m_[pi], *metal_v_[pi],
                static_cast<size_t>(p->numel()), static_cast<float>(lr_), static_cast<float>(weight_decay_),
                static_cast<float>(beta1_), static_cast<float>(beta2_), static_cast<float>(eps_),
                static_cast<float>(bias_correction1), static_cast<float>(bias_correction2));
            p->node->host_data_dirty = false;
            p->node->device_data_dirty = true;
            continue;
        }
#endif
        if (p->grad().empty()) {
            continue;
        }
        for (int64_t i = 0; i < p->numel(); ++i) {
            double g = p->grad()[i];
            m_[pi][i] = beta1_ * m_[pi][i] + (1.0 - beta1_) * g;
            v_[pi][i] = beta2_ * v_[pi][i] + (1.0 - beta2_) * g * g;
            double m_hat = m_[pi][i] / bias_correction1;
            double v_hat = v_[pi][i] / bias_correction2;
            p->data()[i] -= lr_ * weight_decay_ * p->data()[i];
            p->data()[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

} // namespace llm
