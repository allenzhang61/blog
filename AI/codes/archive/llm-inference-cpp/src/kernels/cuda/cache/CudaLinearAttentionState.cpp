#include "CudaLinearAttentionState.h"

#include "../cuda_common.h"

#include <cstddef>
#include <stdexcept>

#include <cuda_runtime.h>

namespace llm_inference {

namespace {

template <typename T>
void cuda_free_if_set(T * ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

} // namespace

CudaLinearAttentionState::~CudaLinearAttentionState() {
    cuda_free_if_set(conv_state);
    cuda_free_if_set(recurrent_state);
    cuda_free_if_set(mixed);
    cuda_free_if_set(projection);
    cuda_free_if_set(z);
    cuda_free_if_set(b);
    cuda_free_if_set(a);
    cuda_free_if_set(conv_out);
    cuda_free_if_set(gated);
    cuda_free_if_set(gated_bf16);
}

void CudaLinearAttentionState::release_batch_buffers() {
    batch_projection.reset();
    batch_conv_out.reset();
    batch_gated.reset();
    batch_gated_lowp.reset();
    batch_z.reset();
    batch_b.reset();
    batch_a.reset();
}

CudaLinearAttentionState * CudaLinearAttentionState::ensure(
        void *& state_handle,
        int key_heads,
        int value_heads,
        int k_dim,
        int v_dim,
        int kernel) {
    auto * state = static_cast<CudaLinearAttentionState *>(state_handle);
    if (state) {
        if (state->key_heads != key_heads ||
            state->value_heads != value_heads ||
            state->k_dim != k_dim ||
            state->v_dim != v_dim ||
            state->kernel != kernel) {
            throw std::runtime_error("CUDA linear attention state 维度变化，无法复用");
        }
        return state;
    }

    state = new CudaLinearAttentionState();
    state->key_heads = key_heads;
    state->value_heads = value_heads;
    state->k_dim = k_dim;
    state->v_dim = v_dim;
    state->kernel = kernel;
    const int key_total = key_heads * k_dim;
    const int value_total = value_heads * v_dim;
    const int conv_dim = key_total * 2 + value_total;
    check_cuda(cudaMalloc(&state->conv_state, static_cast<size_t>(conv_dim) * kernel * sizeof(float)), "cudaMalloc linear conv state 失败");
    check_cuda(cudaMalloc(&state->recurrent_state, static_cast<size_t>(value_heads) * k_dim * v_dim * sizeof(float)), "cudaMalloc linear recurrent state 失败");
    check_cuda(cudaMalloc(&state->mixed, static_cast<size_t>(conv_dim) * sizeof(float)), "cudaMalloc linear mixed 失败");
    check_cuda(cudaMalloc(&state->projection, static_cast<size_t>(conv_dim + value_total + value_heads * 2) * sizeof(float)), "cudaMalloc linear projection 失败");
    check_cuda(cudaMalloc(&state->z, static_cast<size_t>(value_total) * sizeof(float)), "cudaMalloc linear z 失败");
    check_cuda(cudaMalloc(&state->b, static_cast<size_t>(value_heads) * sizeof(float)), "cudaMalloc linear b 失败");
    check_cuda(cudaMalloc(&state->a, static_cast<size_t>(value_heads) * sizeof(float)), "cudaMalloc linear a 失败");
    check_cuda(cudaMalloc(&state->conv_out, static_cast<size_t>(conv_dim) * sizeof(float)), "cudaMalloc linear conv out 失败");
    check_cuda(cudaMalloc(&state->gated, static_cast<size_t>(value_total) * sizeof(float)), "cudaMalloc linear gated 失败");
    check_cuda(cudaMalloc(&state->gated_bf16, static_cast<size_t>(value_total) * sizeof(uint16_t)), "cudaMalloc linear gated bf16 失败");
    check_cuda(cudaMemset(state->conv_state, 0, static_cast<size_t>(conv_dim) * kernel * sizeof(float)), "cudaMemset linear conv state 失败");
    check_cuda(cudaMemset(state->recurrent_state, 0, static_cast<size_t>(value_heads) * k_dim * v_dim * sizeof(float)), "cudaMemset linear recurrent state 失败");
    state_handle = state;
    return state;
}

void CudaLinearAttentionState::destroy(void * state) {
    delete static_cast<CudaLinearAttentionState *>(state);
}

} // namespace llm_inference
