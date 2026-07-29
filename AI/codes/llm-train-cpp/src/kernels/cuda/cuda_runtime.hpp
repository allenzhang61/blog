#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "llm/tensor.hpp"

namespace llm::cuda::detail {

// 设备内存 RAII 包装，避免手动 cudaFree 泄漏。
struct DeviceBuffer {
    float* ptr{nullptr};
    size_t count{0};

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n);
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    ~DeviceBuffer();
};

// CUDA host 端调度与内存搬运封装。
class CudaRuntime {
public:
    static CudaRuntime& instance();
    ~CudaRuntime();

    bool available() const;
    std::string status() const;
    void require() const;

    std::shared_ptr<TensorStorage> create_tensor_storage();
    void ensure_data_buffer(TensorStorage& storage, size_t count);
    void ensure_grad_buffer(TensorStorage& storage, size_t count);
    void copy_data_from_host(TensorStorage& storage, const std::vector<double>& host);
    void copy_data_to_host(TensorStorage& storage, std::vector<double>& host);
    void copy_grad_from_host(TensorStorage& storage, const std::vector<double>& host);
    void copy_grad_to_host(TensorStorage& storage, std::vector<double>& host);
    void fill_data_buffer(TensorStorage& storage, size_t count, float value);
    void fill_grad_buffer(TensorStorage& storage, size_t count, float value);
    void set_data_scalar(TensorStorage& storage, float value);
    void set_grad_scalar(TensorStorage& storage, float value);
    void elementwise2_buffer(const char* op, TensorStorage& out, const TensorStorage& a,
                             const TensorStorage& b, unsigned int b_size, size_t count);
    void mul_scalar_buffer(TensorStorage& out, const TensorStorage& a, float scalar, size_t count);
    void unary_buffer(const char* op, TensorStorage& out, const TensorStorage& a, float scalar, size_t count);
    void gather_buffer(TensorStorage& out, const TensorStorage& a,
                       const std::vector<unsigned int>& index);
    void transpose_buffer(TensorStorage& out, const TensorStorage& a,
                          const std::vector<int64_t>& shape, int64_t dim0, int64_t dim1);
    void transpose_add_grad(TensorStorage& target_grad, const TensorStorage& out_grad,
                            const std::vector<int64_t>& shape, int64_t dim0, int64_t dim1);
    void scale_data_buffer(TensorStorage& storage, size_t count, float scalar);
    void reduce_buffer(const char* op, TensorStorage& out, const TensorStorage& a, size_t count);
    void matmul_buffer(TensorStorage& out, const TensorStorage& a, const TensorStorage& b,
                       unsigned int m, unsigned int k, unsigned int n);
    void batch_matmul_buffer(TensorStorage& out, const TensorStorage& a, const TensorStorage& b,
                             unsigned int batches, unsigned int heads, unsigned int m,
                             unsigned int k, unsigned int n);
    void softmax_buffer(TensorStorage& out, const TensorStorage& a, unsigned int rows, unsigned int width);
    void log_softmax_buffer(TensorStorage& out, const TensorStorage& a, unsigned int rows, unsigned int width);
    void layernorm_buffer(TensorStorage& out, const TensorStorage& x, const TensorStorage& scale,
                          const TensorStorage& shift, unsigned int rows, unsigned int width, float eps);
    void embedding_buffer(TensorStorage& out, const TensorStorage& ids, const TensorStorage& weight,
                          unsigned int count, unsigned int dim);
    void cross_entropy_loss_buffer(TensorStorage& out, const TensorStorage& logits,
                                   const TensorStorage& targets, unsigned int rows, unsigned int vocab);
    void add_grad(TensorStorage& target, const TensorStorage& out_grad, unsigned int target_size,
                  size_t count, float scale = 1.0f);
    void elementwise_grad(const char* op, TensorStorage* a_grad, TensorStorage* b_grad,
                          const TensorStorage& a, const TensorStorage& b,
                          const TensorStorage& out_grad, size_t count);
    void mul_scalar_grad(TensorStorage& a_grad, const TensorStorage& out_grad, float scalar, size_t count);
    void pow_grad(TensorStorage& a_grad, const TensorStorage& a, const TensorStorage& out_grad,
                  float exponent, size_t count);
    void reduce_grad(TensorStorage& a_grad, const TensorStorage& out_grad, size_t count, float scale);
    void scatter_add_grad(TensorStorage& a_grad, const TensorStorage& out_grad,
                          const std::vector<unsigned int>& index);
    void matmul_grad(TensorStorage* a_grad, TensorStorage* b_grad, const TensorStorage& a,
                     const TensorStorage& b, const TensorStorage& out_grad,
                     unsigned int m, unsigned int k, unsigned int n);
    void batch_matmul_grad(TensorStorage* a_grad, TensorStorage* b_grad, const TensorStorage& a,
                           const TensorStorage& b, const TensorStorage& out_grad,
                           unsigned int batches, unsigned int heads, unsigned int m,
                           unsigned int k, unsigned int n);
    void softmax_grad(TensorStorage& a_grad, const TensorStorage& out,
                      const TensorStorage& out_grad, unsigned int rows, unsigned int width);
    void cross_entropy_grad(TensorStorage& logits_grad, const TensorStorage& logits,
                            const TensorStorage& targets, const TensorStorage& out_grad,
                            unsigned int rows, unsigned int vocab);
    void embedding_grad(TensorStorage& weight_grad, const TensorStorage& ids,
                        const TensorStorage& out_grad, unsigned int count, unsigned int dim);
    void layernorm_grad(TensorStorage* x_grad, TensorStorage* scale_grad, TensorStorage* shift_grad,
                        const TensorStorage& x, const TensorStorage& scale,
                        const TensorStorage& out_grad, unsigned int rows, unsigned int width, float eps);
    void gelu_grad(TensorStorage& x_grad, const TensorStorage& x,
                   const TensorStorage& out_grad, size_t count);
    void adamw_update(TensorStorage& param, TensorStorage& grad, TensorStorage& m, TensorStorage& v,
                      size_t count, float lr, float weight_decay, float beta1, float beta2,
                      float eps, float bias_correction1, float bias_correction2);
    void causal_mask_buffer(TensorStorage& out, const TensorStorage& scores,
                            unsigned int batches, unsigned int heads, unsigned int sequence_length,
                            float mask_value);
    void causal_mask_grad(TensorStorage& scores_grad, const TensorStorage& out_grad,
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
    void* cublas_handle_{nullptr};
};

} // namespace llm::cuda::detail
