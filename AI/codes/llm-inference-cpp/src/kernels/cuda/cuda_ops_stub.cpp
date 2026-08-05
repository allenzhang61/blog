#include "cuda_ops.h"

namespace llm_inference {

bool cuda_matvec(const TensorRef &, const std::vector<float> &, std::vector<float> &) {
    return false;
}

bool cuda_argmax_matvec(const TensorRef &, const std::vector<float> &, int &) {
    return false;
}

void * cuda_token_hidden_buffer(int, int) {
    return nullptr;
}

void * cuda_generated_token_buffer(int) {
    return nullptr;
}

bool cuda_embedding_lookup_device(const TensorRef &, int, void *) {
    return false;
}

bool cuda_embedding_lookup_device_token(const TensorRef &, const void *, void *) {
    return false;
}

bool cuda_final_norm_argmax_device(const TensorRef &, const TensorRef &, const void *, int, float, bool, int &) {
    return false;
}

bool cuda_final_norm_argmax_to_device(const TensorRef &, const TensorRef &, const void *, int, float, bool, void *) {
    return false;
}

bool cuda_copy_generated_tokens_to_host(const void *, int, std::vector<int> &) {
    return false;
}

bool cuda_synchronize_device() {
    return false;
}

const void * cuda_prefill_batch(
    const ModelConfig &,
    const ModelWeights &,
    const std::vector<int> &,
    std::vector<void *> &,
    std::vector<void *> &,
    const std::vector<int> &,
    int &) {
    return nullptr;
}

bool cuda_cublas_enabled() {
    return false;
}

bool cuda_fused_mlp_enabled() {
    return false;
}

bool cuda_project_attention_enabled() {
    return false;
}

bool cuda_full_layer_enabled() {
    return false;
}

bool cuda_rmsnorm_mlp_enabled() {
    return false;
}

void cuda_free_linear_attention_state(void *) {}

void cuda_free_full_attention_state(void *) {}

bool cuda_mlp_layer(const TensorRef &, const TensorRef &, const TensorRef &, const std::vector<float> &, std::vector<float> &) {
    return false;
}

bool cuda_linear_attention_layer(
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const std::vector<float> &,
    const std::vector<float> &,
    const std::vector<float> &,
    const std::vector<float> &,
    void *&,
    int,
    int,
    int,
    int,
    int,
    float,
    std::vector<float> &) {
    return false;
}

bool cuda_linear_attention_project_layer(
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const std::vector<float> &,
    void *&,
    int,
    int,
    int,
    int,
    int,
    float,
    std::vector<float> &) {
    return false;
}

bool cuda_rmsnorm_linear_attention_project_layer(
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const std::vector<float> &,
    void *&,
    int,
    int,
    int,
    int,
    int,
    float,
    bool,
    std::vector<float> &) {
    return false;
}

bool cuda_linear_attention_full_layer(
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const std::vector<float> &,
    void *&,
    int,
    int,
    int,
    int,
    int,
    float,
    bool,
    std::vector<float> &) {
    return false;
}

bool cuda_linear_attention_full_layer_device(
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const void *,
    void *,
    int,
    void *&,
    int,
    int,
    int,
    int,
    int,
    float,
    bool) {
    return false;
}

bool cuda_full_attention_layer(
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const std::vector<float> &,
    const std::vector<float> &,
    const std::vector<float> &,
    void *&,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
    float,
    std::vector<float> &) {
    return false;
}

bool cuda_full_attention_project_layer(
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const std::vector<float> &,
    void *&,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
    float,
    std::vector<float> &) {
    return false;
}

bool cuda_rmsnorm_full_attention_project_layer(
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const std::vector<float> &,
    void *&,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
    float,
    bool,
    std::vector<float> &) {
    return false;
}

bool cuda_full_attention_full_layer(
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const std::vector<float> &,
    void *&,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
    float,
    bool,
    std::vector<float> &) {
    return false;
}

bool cuda_full_attention_full_layer_device(
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const void *,
    void *,
    int,
    void *&,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
    float,
    bool) {
    return false;
}

bool cuda_rmsnorm_mlp_layer(
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const TensorRef &,
    const std::vector<float> &,
    float,
    bool,
    std::vector<float> &) {
    return false;
}

} // namespace llm_inference
