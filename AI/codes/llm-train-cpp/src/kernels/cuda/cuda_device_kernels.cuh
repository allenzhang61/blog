#pragma once

#include <cuda_runtime.h>

namespace llm::cuda::detail {

__global__ void add_kernel(const float* a, const float* b, float* out, unsigned int b_size,
                           long long count);
__global__ void mul_kernel(const float* a, const float* b, float* out, long long count);
__global__ void div_kernel(const float* a, const float* b, float* out, long long count);
__global__ void mul_scalar_kernel(const float* a, float* out, float scalar, long long count);
__global__ void neg_kernel(const float* a, float* out, long long count);
__global__ void pow_kernel(const float* a, float* out, float exponent, long long count);
__global__ void copy_kernel(const float* a, float* out, long long count);
__global__ void log_kernel(const float* a, float* out, long long count);
__global__ void gather_kernel(const float* a, const unsigned int* index, float* out,
                              long long count);
__global__ void gelu_kernel(const float* x, float* out, long long count);
__global__ void matmul_kernel(const float* a, const float* b, float* out, unsigned int m,
                              unsigned int k, unsigned int n);
__global__ void batch_matmul_kernel(const float* a, const float* b, float* out,
                                    unsigned int batches, unsigned int heads, unsigned int m,
                                    unsigned int k, unsigned int n);
__global__ void softmax_kernel(const float* x, float* out, unsigned int rows, unsigned int width);
__global__ void layernorm_kernel(const float* x, const float* scale, const float* shift,
                                 float* out, unsigned int rows, unsigned int width, float eps);
__global__ void embedding_kernel(const float* ids, const float* weight, float* out,
                                 unsigned int count, unsigned int dim);
__global__ void cross_entropy_row_loss_kernel(const float* logits, const float* targets,
                                              float* row_losses, unsigned int rows,
                                              unsigned int vocab);
__global__ void sum_kernel(const float* a, float* out, unsigned int count);
__global__ void max_kernel(const float* a, float* out, unsigned int count);
__global__ void fill_kernel(float* out, float value, long long count);
__global__ void log_softmax_kernel(const float* x, float* out, unsigned int rows, unsigned int width);
__global__ void cross_entropy_loss_kernel(const float* logits, const float* targets, float* loss,
                                          unsigned int rows, unsigned int vocab);
__global__ void add_grad_kernel(float* target_grad, const float* out_grad, unsigned int target_size,
                                long long count, float scale);
__global__ void mul_grad_kernel(float* a_grad, float* b_grad, const float* a, const float* b,
                                const float* out_grad, long long count);
__global__ void div_grad_kernel(float* a_grad, float* b_grad, const float* a, const float* b,
                                const float* out_grad, long long count);
__global__ void mul_scalar_grad_kernel(float* a_grad, const float* out_grad, float scalar,
                                       long long count);
__global__ void pow_grad_kernel(float* a_grad, const float* a, const float* out_grad,
                                float exponent, long long count);
__global__ void reduce_grad_kernel(float* a_grad, const float* out_grad, long long count, float scale);
__global__ void scatter_add_grad_kernel(float* a_grad, const float* out_grad,
                                        const unsigned int* index, long long count);
__global__ void matmul_grad_a_kernel(float* a_grad, const float* b, const float* out_grad,
                                     unsigned int m, unsigned int k, unsigned int n);
__global__ void matmul_grad_b_kernel(float* b_grad, const float* a, const float* out_grad,
                                     unsigned int m, unsigned int k, unsigned int n);
__global__ void batch_matmul_grad_a_kernel(float* a_grad, const float* b, const float* out_grad,
                                           unsigned int batches, unsigned int heads, unsigned int m,
                                           unsigned int k, unsigned int n);
__global__ void batch_matmul_grad_b_kernel(float* b_grad, const float* a, const float* out_grad,
                                           unsigned int batches, unsigned int heads, unsigned int m,
                                           unsigned int k, unsigned int n);
__global__ void softmax_grad_kernel(float* a_grad, const float* out, const float* out_grad,
                                    unsigned int rows, unsigned int width);
__global__ void cross_entropy_grad_kernel(float* logits_grad, const float* logits,
                                          const float* targets, const float* out_grad,
                                          unsigned int rows, unsigned int vocab);
__global__ void embedding_grad_kernel(float* weight_grad, const float* ids, const float* out_grad,
                                      unsigned int count, unsigned int dim);
__global__ void layernorm_grad_x_kernel(float* x_grad, const float* x, const float* scale,
                                        const float* out_grad, unsigned int rows, unsigned int width,
                                        float eps);
__global__ void layernorm_grad_scale_shift_kernel(float* scale_grad, float* shift_grad,
                                                  const float* x, const float* out_grad,
                                                  unsigned int rows, unsigned int width, float eps);
__global__ void gelu_grad_kernel(float* x_grad, const float* x, const float* out_grad,
                                 long long count);
__global__ void adamw_update_kernel(float* param, const float* grad, float* m, float* v,
                                    long long count, float lr, float weight_decay, float beta1,
                                    float beta2, float eps, float bias_correction1,
                                    float bias_correction2);
__global__ void causal_mask_kernel(const float* scores, float* out, unsigned int batches,
                                   unsigned int heads, unsigned int sequence_length, float mask_value);
__global__ void causal_mask_grad_kernel(float* scores_grad, const float* out_grad,
                                        unsigned int batches, unsigned int heads,
                                        unsigned int sequence_length);

} // namespace llm::cuda::detail
