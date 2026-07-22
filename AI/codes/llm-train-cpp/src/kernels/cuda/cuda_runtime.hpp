#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llm::cuda::detail {

// CUDA host 端调度与内存搬运封装。
class CudaRuntime {
public:
    static CudaRuntime& instance();

    bool available() const;
    std::string status() const;
    void require() const;

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
