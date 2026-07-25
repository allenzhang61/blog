#pragma once

#include <cuda_runtime.h>

namespace llm::cuda::detail {

// 逐元素加法；b_size 为 1 时支持标量广播，否则按 b_size 周期广播。
__global__ void add_kernel(const float* a, const float* b, float* out, unsigned int b_size,
                           long long count);

// 逐元素乘法，要求 a/b/out 长度一致。
__global__ void mul_kernel(const float* a, const float* b, float* out, long long count);

// 逐元素除法，要求 a/b/out 长度一致。
__global__ void div_kernel(const float* a, const float* b, float* out, long long count);

// 张量逐元素乘以标量 scalar。
__global__ void mul_scalar_kernel(const float* a, float* out, float scalar, long long count);

// 张量逐元素取负。
__global__ void neg_kernel(const float* a, float* out, long long count);

// 张量逐元素幂运算，out = pow(a, exponent)。
__global__ void pow_kernel(const float* a, float* out, float exponent, long long count);

// 连续拷贝 count 个 float 元素。
__global__ void copy_kernel(const float* a, float* out, long long count);

// 张量逐元素取自然对数，内部会做最小值保护。
__global__ void log_kernel(const float* a, float* out, long long count);

// 按 index 做 gather：out[i] = a[index[i]]。
__global__ void gather_kernel(const float* a, const unsigned int* index, float* out,
                              long long count);

// GELU 前向激活。
__global__ void gelu_kernel(const float* x, float* out, long long count);

// 二维矩阵乘法：[m,k] x [k,n] -> [m,n]。
__global__ void matmul_kernel(const float* a, const float* b, float* out, unsigned int m,
                              unsigned int k, unsigned int n);

// 批量多头矩阵乘法：[B,H,M,K] x [B,H,K,N] -> [B,H,M,N]。
__global__ void batch_matmul_kernel(const float* a, const float* b, float* out,
                                    unsigned int batches, unsigned int heads, unsigned int m,
                                    unsigned int k, unsigned int n);

// 按行 softmax；输入视为 rows x width。
__global__ void softmax_kernel(const float* x, float* out, unsigned int rows, unsigned int width);

// LayerNorm 前向；每行 width 个元素，用 scale/shift 做仿射变换。
__global__ void layernorm_kernel(const float* x, const float* scale, const float* shift,
                                 float* out, unsigned int rows, unsigned int width, float eps);

// Embedding 查表；ids 中每个 token id 取 weight 的一行。
__global__ void embedding_kernel(const float* ids, const float* weight, float* out,
                                 unsigned int count, unsigned int dim);

// 逐行交叉熵 loss，输出每一行的 loss 值。
__global__ void cross_entropy_row_loss_kernel(const float* logits, const float* targets,
                                              float* row_losses, unsigned int rows,
                                              unsigned int vocab);

// 对 count 个元素求和，输出一个标量。
__global__ void sum_kernel(const float* a, float* out, unsigned int count);

// 对 count 个元素求最大值，输出一个标量。
__global__ void max_kernel(const float* a, float* out, unsigned int count);

// 将 out 的前 count 个元素填充为 value。
__global__ void fill_kernel(float* out, float value, long long count);

// 按行 log_softmax；输入视为 rows x width。
__global__ void log_softmax_kernel(const float* x, float* out, unsigned int rows, unsigned int width);

// 交叉熵 loss 归约，直接输出 batch/token 平均 loss。
__global__ void cross_entropy_loss_kernel(const float* logits, const float* targets, float* loss,
                                          unsigned int rows, unsigned int vocab);

// 将 out_grad 累加到 target_grad；target_size 非 0 时按目标大小做广播归约。
__global__ void add_grad_kernel(float* target_grad, const float* out_grad, unsigned int target_size,
                                long long count, float scale);

// 逐元素乘法反向，分别累加到 a_grad 和 b_grad。
__global__ void mul_grad_kernel(float* a_grad, float* b_grad, const float* a, const float* b,
                                const float* out_grad, long long count);

// 逐元素除法反向，分别累加到 a_grad 和 b_grad。
__global__ void div_grad_kernel(float* a_grad, float* b_grad, const float* a, const float* b,
                                const float* out_grad, long long count);

// 标量乘法反向，a_grad += out_grad * scalar。
__global__ void mul_scalar_grad_kernel(float* a_grad, const float* out_grad, float scalar,
                                       long long count);

// 幂运算反向，a_grad += out_grad * exponent * pow(a, exponent - 1)。
__global__ void pow_grad_kernel(float* a_grad, const float* a, const float* out_grad,
                                float exponent, long long count);

// sum/mean 这类归约算子的反向，将标量梯度按 scale 分发到每个输入元素。
__global__ void reduce_grad_kernel(float* a_grad, const float* out_grad, long long count, float scale);

// gather/transpose 类索引操作的反向，将 out_grad 按 index scatter-add 回 a_grad。
__global__ void scatter_add_grad_kernel(float* a_grad, const float* out_grad,
                                        const unsigned int* index, long long count);

// 二维 matmul 对 a 的梯度：[m,n] x [k,n]^T -> [m,k]。
__global__ void matmul_grad_a_kernel(float* a_grad, const float* b, const float* out_grad,
                                     unsigned int m, unsigned int k, unsigned int n);

// 二维 matmul 对 b 的梯度：[m,k]^T x [m,n] -> [k,n]。
__global__ void matmul_grad_b_kernel(float* b_grad, const float* a, const float* out_grad,
                                     unsigned int m, unsigned int k, unsigned int n);

// 批量多头 matmul 对 a 的梯度。
__global__ void batch_matmul_grad_a_kernel(float* a_grad, const float* b, const float* out_grad,
                                           unsigned int batches, unsigned int heads, unsigned int m,
                                           unsigned int k, unsigned int n);

// 批量多头 matmul 对 b 的梯度。
__global__ void batch_matmul_grad_b_kernel(float* b_grad, const float* a, const float* out_grad,
                                           unsigned int batches, unsigned int heads, unsigned int m,
                                           unsigned int k, unsigned int n);

// softmax 反向；out 是前向 softmax 输出。
__global__ void softmax_grad_kernel(float* a_grad, const float* out, const float* out_grad,
                                    unsigned int rows, unsigned int width);

// cross_entropy 对 logits 的反向。
__global__ void cross_entropy_grad_kernel(float* logits_grad, const float* logits,
                                          const float* targets, const float* out_grad,
                                          unsigned int rows, unsigned int vocab);

// embedding 对 weight 的反向，将 token 位置梯度累加回对应词表行。
__global__ void embedding_grad_kernel(float* weight_grad, const float* ids, const float* out_grad,
                                      unsigned int count, unsigned int dim);

// LayerNorm 对输入 x 的反向。
__global__ void layernorm_grad_x_kernel(float* x_grad, const float* x, const float* scale,
                                        const float* out_grad, unsigned int rows, unsigned int width,
                                        float eps);

// LayerNorm 对 scale/shift 的反向。
__global__ void layernorm_grad_scale_shift_kernel(float* scale_grad, float* shift_grad,
                                                  const float* x, const float* out_grad,
                                                  unsigned int rows, unsigned int width, float eps);

// GELU 反向。
__global__ void gelu_grad_kernel(float* x_grad, const float* x, const float* out_grad,
                                 long long count);

// AdamW 参数更新，同时维护一阶矩 m 和二阶矩 v。
__global__ void adamw_update_kernel(float* param, const float* grad, float* m, float* v,
                                    long long count, float lr, float weight_decay, float beta1,
                                    float beta2, float eps, float bias_correction1,
                                    float bias_correction2);

// causal mask 前向，将未来位置写成 mask_value。
__global__ void causal_mask_kernel(const float* scores, float* out, unsigned int batches,
                                   unsigned int heads, unsigned int sequence_length, float mask_value);

// causal mask 反向，只把允许关注的位置梯度传回 scores_grad。
__global__ void causal_mask_grad_kernel(float* scores_grad, const float* out_grad,
                                        unsigned int batches, unsigned int heads,
                                        unsigned int sequence_length);

} // namespace llm::cuda::detail
