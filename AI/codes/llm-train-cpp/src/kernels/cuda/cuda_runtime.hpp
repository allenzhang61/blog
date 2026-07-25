#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "llm/tensor.hpp"

namespace llm::cuda::detail {

// CUDA host 端调度与内存搬运封装。
class CudaRuntime {
public:
    static CudaRuntime& instance();

    bool available() const;
    std::string status() const;
    void require() const;

    std::shared_ptr<TensorCudaStorage> create_tensor_storage();
    void ensure_data_buffer(TensorCudaStorage& storage, size_t count);
    void ensure_grad_buffer(TensorCudaStorage& storage, size_t count);
    void copy_data_from_host(TensorCudaStorage& storage, const std::vector<double>& host);
    void copy_data_to_host(TensorCudaStorage& storage, std::vector<double>& host);
    void copy_grad_from_host(TensorCudaStorage& storage, const std::vector<double>& host);
    void copy_grad_to_host(TensorCudaStorage& storage, std::vector<double>& host);
    void fill_data_buffer(TensorCudaStorage& storage, size_t count, float value);
    void fill_grad_buffer(TensorCudaStorage& storage, size_t count, float value);
    void set_data_scalar(TensorCudaStorage& storage, float value);
    void set_grad_scalar(TensorCudaStorage& storage, float value);
    void elementwise2_buffer(const char* op, TensorCudaStorage& out, const TensorCudaStorage& a,
                             const TensorCudaStorage& b, unsigned int b_size, size_t count);
    void mul_scalar_buffer(TensorCudaStorage& out, const TensorCudaStorage& a, float scalar, size_t count);
    void unary_buffer(const char* op, TensorCudaStorage& out, const TensorCudaStorage& a, float scalar, size_t count);
    void gather_buffer(TensorCudaStorage& out, const TensorCudaStorage& a,
                       const std::vector<unsigned int>& index);
    void scale_data_buffer(TensorCudaStorage& storage, size_t count, float scalar);
    void reduce_buffer(const char* op, TensorCudaStorage& out, const TensorCudaStorage& a, size_t count);
    void matmul_buffer(TensorCudaStorage& out, const TensorCudaStorage& a, const TensorCudaStorage& b,
                       unsigned int m, unsigned int k, unsigned int n);
    void batch_matmul_buffer(TensorCudaStorage& out, const TensorCudaStorage& a, const TensorCudaStorage& b,
                             unsigned int batches, unsigned int heads, unsigned int m,
                             unsigned int k, unsigned int n);
    void softmax_buffer(TensorCudaStorage& out, const TensorCudaStorage& a, unsigned int rows, unsigned int width);
    void log_softmax_buffer(TensorCudaStorage& out, const TensorCudaStorage& a, unsigned int rows, unsigned int width);
    void layernorm_buffer(TensorCudaStorage& out, const TensorCudaStorage& x, const TensorCudaStorage& scale,
                          const TensorCudaStorage& shift, unsigned int rows, unsigned int width, float eps);
    void embedding_buffer(TensorCudaStorage& out, const TensorCudaStorage& ids, const TensorCudaStorage& weight,
                          unsigned int count, unsigned int dim);
    void cross_entropy_loss_buffer(TensorCudaStorage& out, const TensorCudaStorage& logits,
                                   const TensorCudaStorage& targets, unsigned int rows, unsigned int vocab);
    void add_grad(TensorCudaStorage& target, const TensorCudaStorage& out_grad, unsigned int target_size,
                  size_t count, float scale = 1.0f);
    void elementwise_grad(const char* op, TensorCudaStorage* a_grad, TensorCudaStorage* b_grad,
                          const TensorCudaStorage& a, const TensorCudaStorage& b,
                          const TensorCudaStorage& out_grad, size_t count);
    void mul_scalar_grad(TensorCudaStorage& a_grad, const TensorCudaStorage& out_grad, float scalar, size_t count);
    void pow_grad(TensorCudaStorage& a_grad, const TensorCudaStorage& a, const TensorCudaStorage& out_grad,
                  float exponent, size_t count);
    void reduce_grad(TensorCudaStorage& a_grad, const TensorCudaStorage& out_grad, size_t count, float scale);
    void scatter_add_grad(TensorCudaStorage& a_grad, const TensorCudaStorage& out_grad,
                          const std::vector<unsigned int>& index);
    void matmul_grad(TensorCudaStorage* a_grad, TensorCudaStorage* b_grad, const TensorCudaStorage& a,
                     const TensorCudaStorage& b, const TensorCudaStorage& out_grad,
                     unsigned int m, unsigned int k, unsigned int n);
    void batch_matmul_grad(TensorCudaStorage* a_grad, TensorCudaStorage* b_grad, const TensorCudaStorage& a,
                           const TensorCudaStorage& b, const TensorCudaStorage& out_grad,
                           unsigned int batches, unsigned int heads, unsigned int m,
                           unsigned int k, unsigned int n);
    void softmax_grad(TensorCudaStorage& a_grad, const TensorCudaStorage& out,
                      const TensorCudaStorage& out_grad, unsigned int rows, unsigned int width);
    void cross_entropy_grad(TensorCudaStorage& logits_grad, const TensorCudaStorage& logits,
                            const TensorCudaStorage& targets, const TensorCudaStorage& out_grad,
                            unsigned int rows, unsigned int vocab);
    void embedding_grad(TensorCudaStorage& weight_grad, const TensorCudaStorage& ids,
                        const TensorCudaStorage& out_grad, unsigned int count, unsigned int dim);
    void layernorm_grad(TensorCudaStorage* x_grad, TensorCudaStorage* scale_grad, TensorCudaStorage* shift_grad,
                        const TensorCudaStorage& x, const TensorCudaStorage& scale,
                        const TensorCudaStorage& out_grad, unsigned int rows, unsigned int width, float eps);
    void gelu_grad(TensorCudaStorage& x_grad, const TensorCudaStorage& x,
                   const TensorCudaStorage& out_grad, size_t count);
    void adamw_update(TensorCudaStorage& param, TensorCudaStorage& grad, TensorCudaStorage& m, TensorCudaStorage& v,
                      size_t count, float lr, float weight_decay, float beta1, float beta2,
                      float eps, float bias_correction1, float bias_correction2);
    void causal_mask_buffer(TensorCudaStorage& out, const TensorCudaStorage& scores,
                            unsigned int batches, unsigned int heads, unsigned int sequence_length,
                            float mask_value);
    void causal_mask_grad(TensorCudaStorage& scores_grad, const TensorCudaStorage& out_grad,
                          unsigned int batches, unsigned int heads, unsigned int sequence_length);

    std::vector<float> elementwise2(const char* op, const std::vector<float>& a,
                                    const std::vector<float>& b, unsigned int b_size);
    std::vector<float> mul_scalar(const std::vector<float>& a, float scalar);
    std::vector<float> unary(const char* op, const std::vector<float>& a, float scalar = 0.0f);
    std::vector<float> gather(const std::vector<float>& a, const std::vector<unsigned int>& index);
    std::vector<float> matmul(const std::vector<float>& a, const std::vector<float>& b,
                              unsigned int m, unsigned int k, unsigned int n);
    std::vector<float> batch_matmul(const std::vector<float>& a, const std::vector<float>& b,
                                    unsigned int batches, unsigned int heads, unsigned int m,
                                    unsigned int k, unsigned int n);
    std::vector<float> softmax(const std::vector<float>& x, unsigned int rows, unsigned int width);
    std::vector<float> layernorm(const std::vector<float>& x, const std::vector<float>& scale,
                                 const std::vector<float>& shift, unsigned int rows,
                                 unsigned int width, float eps);
    std::vector<float> embedding(const std::vector<float>& ids, const std::vector<float>& weight,
                                 unsigned int count, unsigned int dim);
    std::vector<float> cross_entropy_row_losses(const std::vector<float>& logits,
                                                const std::vector<float>& targets,
                                                unsigned int rows, unsigned int vocab);
    float reduce(const char* op, const std::vector<float>& a);

private:
    CudaRuntime();
    void sync();

    bool available_{false};
    std::string status_;
};

} // namespace llm::cuda::detail
