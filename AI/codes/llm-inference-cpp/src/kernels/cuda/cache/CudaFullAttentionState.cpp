#include "CudaFullAttentionState.h"

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

CudaFullAttentionState::~CudaFullAttentionState() {
    cuda_free_if_set(q_and_gate);
    cuda_free_if_set(projection);
    cuda_free_if_set(k);
    cuda_free_if_set(v);
    cuda_free_if_set(q);
    cuda_free_if_set(gate);
    cuda_free_if_set(key_cache);
    cuda_free_if_set(value_cache);
    cuda_free_if_set(attn);
    cuda_free_if_set(attn_bf16);
}

void CudaFullAttentionState::release_batch_buffers() {
    batch_projection.reset();
    batch_q.reset();
    batch_gate.reset();
    batch_attn.reset();
    batch_attn_lowp.reset();
    batch_k.reset();
    batch_v.reset();
}

CudaFullAttentionState * CudaFullAttentionState::ensure(
        void *& state_handle,
        int n_heads,
        int kv_heads,
        int head_dim,
        int max_seq_len) {
    auto * state = static_cast<CudaFullAttentionState *>(state_handle);
    if (state) {
        if (state->n_heads != n_heads ||
            state->kv_heads != kv_heads ||
            state->head_dim != head_dim ||
            state->max_seq_len != max_seq_len) {
            throw std::runtime_error("CUDA full attention state 维度变化，无法复用");
        }
        return state;
    }

    state = new CudaFullAttentionState();
    state->n_heads = n_heads;
    state->kv_heads = kv_heads;
    state->head_dim = head_dim;
    state->max_seq_len = max_seq_len;
    const int q_total = n_heads * head_dim;
    const int kv_total = kv_heads * head_dim;
    check_cuda(cudaMalloc(&state->q_and_gate, static_cast<size_t>(q_total) * 2 * sizeof(float)), "cudaMalloc full q_and_gate 失败");
    check_cuda(cudaMalloc(&state->projection, static_cast<size_t>(q_total * 2 + kv_total * 2) * sizeof(float)), "cudaMalloc full projection 失败");
    check_cuda(cudaMalloc(&state->k, static_cast<size_t>(kv_total) * sizeof(float)), "cudaMalloc full k 失败");
    check_cuda(cudaMalloc(&state->v, static_cast<size_t>(kv_total) * sizeof(float)), "cudaMalloc full v 失败");
    check_cuda(cudaMalloc(&state->q, static_cast<size_t>(q_total) * sizeof(float)), "cudaMalloc full q 失败");
    check_cuda(cudaMalloc(&state->gate, static_cast<size_t>(q_total) * sizeof(float)), "cudaMalloc full gate 失败");
    check_cuda(cudaMalloc(&state->key_cache, static_cast<size_t>(max_seq_len) * kv_total * sizeof(float)), "cudaMalloc full key cache 失败");
    check_cuda(cudaMalloc(&state->value_cache, static_cast<size_t>(max_seq_len) * kv_total * sizeof(float)), "cudaMalloc full value cache 失败");
    check_cuda(cudaMalloc(&state->attn, static_cast<size_t>(q_total) * sizeof(float)), "cudaMalloc full attn 失败");
    check_cuda(cudaMalloc(&state->attn_bf16, static_cast<size_t>(q_total) * sizeof(uint16_t)), "cudaMalloc full attn bf16 失败");
    state_handle = state;
    return state;
}

void CudaFullAttentionState::destroy(void * state) {
    delete static_cast<CudaFullAttentionState *>(state);
}

} // namespace llm_inference
