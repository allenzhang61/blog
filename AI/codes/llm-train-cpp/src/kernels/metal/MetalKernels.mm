#include "metal_ops.hpp"
#include "metal_runtime.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>

namespace {

struct ScalarParams {
    float scalar;
};

struct MatmulParams {
    uint32_t m;
    uint32_t k;
    uint32_t n;
};

struct BatchMatmulParams {
    uint32_t batches;
    uint32_t heads;
    uint32_t m;
    uint32_t k;
    uint32_t n;
};

struct SoftmaxParams {
    uint32_t rows;
    uint32_t width;
};

struct LayerNormParams {
    uint32_t rows;
    uint32_t width;
    float eps;
};

struct EmbeddingParams {
    uint32_t count;
    uint32_t dim;
};

struct CrossEntropyParams {
    uint32_t rows;
    uint32_t vocab;
};

struct GradParams {
    uint32_t target_size;
    uint32_t count;
    float scale;
};

struct AdamWParams {
    uint32_t count;
    float lr;
    float weight_decay;
    float beta1;
    float beta2;
    float eps;
    float bias_correction1;
    float bias_correction2;
};

struct ElementwiseGradParams {
    uint32_t count;
    uint32_t has_a;
    uint32_t has_b;
};

struct CausalMaskParams {
    uint32_t batches;
    uint32_t heads;
    uint32_t sequence_length;
    float mask_value;
};

class MetalRuntime {
public:
    static MetalRuntime& instance() {
        static MetalRuntime runtime;
        return runtime;
    }

    bool available() const {
        return device_ != nil && queue_ != nil && library_ != nil;
    }

    std::string status() const {
        if (available()) {
            return "Metal backend available";
        }
        return status_;
    }

    ~MetalRuntime() {
        sync();
    }

    std::vector<float> run1d(const char* kernel_name,
                             const std::vector<float>& a,
                             const std::vector<float>& b,
                             const void* params,
                             size_t params_size,
                             uint32_t b_size,
                             int64_t count) {
        if (!available()) {
            throw std::runtime_error(status_);
        }
        if (count <= 0) {
            return {};
        }
        id<MTLComputePipelineState> pipeline_state = get_pipeline(kernel_name);
        id<MTLBuffer> a_buffer = buffer(a);
        id<MTLBuffer> b_buffer = b.empty() ? nil : buffer(b);
        id<MTLBuffer> out_buffer = [device_ newBufferWithLength:sizeof(float) * static_cast<size_t>(count)
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> params_buffer = nil;
        if (params != nullptr && params_size > 0) {
            params_buffer = [device_ newBufferWithBytes:params length:params_size options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> b_size_buffer = [device_ newBufferWithBytes:&b_size length:sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_state];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        if (std::string(kernel_name) == "add_kernel") {
            [encoder setBuffer:b_buffer offset:0 atIndex:1];
            [encoder setBuffer:out_buffer offset:0 atIndex:2];
            [encoder setBuffer:b_size_buffer offset:0 atIndex:3];
        } else if (std::string(kernel_name) == "mul_kernel") {
            [encoder setBuffer:b_buffer offset:0 atIndex:1];
            [encoder setBuffer:out_buffer offset:0 atIndex:2];
        } else if (std::string(kernel_name) == "div_kernel") {
            [encoder setBuffer:b_buffer offset:0 atIndex:1];
            [encoder setBuffer:out_buffer offset:0 atIndex:2];
        } else {
            [encoder setBuffer:out_buffer offset:0 atIndex:1];
            [encoder setBuffer:params_buffer offset:0 atIndex:2];
        }
        dispatch1d(encoder, pipeline_state, static_cast<NSUInteger>(count));
        [encoder endEncoding];
        submit(command);
        return read(out_buffer, count);
    }

    std::vector<float> matmul(const std::vector<float>& a,
                              const std::vector<float>& b,
                              uint32_t m,
                              uint32_t k,
                              uint32_t n) {
        if (!available()) {
            throw std::runtime_error(status_);
        }
        id<MTLComputePipelineState> pipeline_state = get_pipeline("matmul_kernel");
        id<MTLBuffer> a_buffer = buffer(a);
        id<MTLBuffer> b_buffer = buffer(b);
        id<MTLBuffer> out_buffer = [device_ newBufferWithLength:sizeof(float) * static_cast<size_t>(m) * n
                                                        options:MTLResourceStorageModeShared];
        MatmulParams params{m, k, n};
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:&params length:sizeof(params)
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_state];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:b_buffer offset:0 atIndex:1];
        [encoder setBuffer:out_buffer offset:0 atIndex:2];
        [encoder setBuffer:params_buffer offset:0 atIndex:3];
        MTLSize grid = MTLSizeMake(n, m, 1);
        NSUInteger width = pipeline_state.threadExecutionWidth;
        NSUInteger height = std::max<NSUInteger>(1, pipeline_state.maxTotalThreadsPerThreadgroup / width);
        MTLSize threads = MTLSizeMake(width, height, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        [encoder endEncoding];
        submit(command);
        return read(out_buffer, static_cast<int64_t>(m) * n);
    }

    std::vector<float> batch_matmul(const std::vector<float>& a,
                                    const std::vector<float>& b,
                                    uint32_t batches,
                                    uint32_t heads,
                                    uint32_t m,
                                    uint32_t k,
                                    uint32_t n) {
        BatchMatmulParams params{batches, heads, m, k, n};
        return run3buffer1d("batch_matmul_kernel", a, b, &params, sizeof(params),
                            static_cast<int64_t>(batches) * heads * m * n);
    }

    std::vector<float> softmax(const std::vector<float>& x, uint32_t rows, uint32_t width) {
        SoftmaxParams params{rows, width};
        return run1input_rows("softmax_kernel", x, &params, sizeof(params), rows,
                              static_cast<int64_t>(rows) * width);
    }

    std::vector<float> layernorm(const std::vector<float>& x,
                                 const std::vector<float>& scale,
                                 const std::vector<float>& shift,
                                 uint32_t rows,
                                 uint32_t width,
                                 float eps) {
        if (!available()) {
            throw std::runtime_error(status_);
        }
        id<MTLComputePipelineState> pipeline_state = get_pipeline("layernorm_kernel");
        id<MTLBuffer> x_buffer = buffer(x);
        id<MTLBuffer> scale_buffer = buffer(scale);
        id<MTLBuffer> shift_buffer = buffer(shift);
        id<MTLBuffer> out_buffer = [device_ newBufferWithLength:sizeof(float) * x.size()
                                                        options:MTLResourceStorageModeShared];
        LayerNormParams params{rows, width, eps};
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:&params length:sizeof(params)
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_state];
        [encoder setBuffer:x_buffer offset:0 atIndex:0];
        [encoder setBuffer:scale_buffer offset:0 atIndex:1];
        [encoder setBuffer:shift_buffer offset:0 atIndex:2];
        [encoder setBuffer:out_buffer offset:0 atIndex:3];
        [encoder setBuffer:params_buffer offset:0 atIndex:4];
        dispatch1d(encoder, pipeline_state, rows);
        [encoder endEncoding];
        submit(command);
        return read(out_buffer, static_cast<int64_t>(rows) * width);
    }

    std::vector<float> embedding(const std::vector<float>& ids,
                                 const std::vector<float>& weight,
                                 uint32_t count,
                                 uint32_t dim) {
        EmbeddingParams params{count, dim};
        return run3buffer1d("embedding_kernel", ids, weight, &params, sizeof(params),
                            static_cast<int64_t>(count) * dim);
    }

    std::vector<float> cross_entropy_row_losses(const std::vector<float>& logits,
                                                const std::vector<float>& targets,
                                                uint32_t rows,
                                                uint32_t vocab) {
        CrossEntropyParams params{rows, vocab};
        return run3buffer1d("cross_entropy_row_loss_kernel", logits, targets, &params, sizeof(params), rows);
    }

    std::shared_ptr<llm::TensorCudaStorage> create_tensor_storage() {
        if (!available()) {
            throw std::runtime_error(status_);
        }
        auto storage = std::make_shared<llm::TensorCudaStorage>();
        storage->release = [](llm::TensorCudaStorage& s) {
            release_buffer(s.data);
            release_buffer(s.grad);
            s.data = nullptr;
            s.grad = nullptr;
            s.data_count = 0;
            s.grad_count = 0;
        };
        storage->copy_data_from_host = [](llm::TensorCudaStorage& s, const std::vector<double>& host) {
            MetalRuntime::instance().copy_data_from_host(s, host);
        };
        storage->copy_data_to_host = [](llm::TensorCudaStorage& s, std::vector<double>& host) {
            MetalRuntime::instance().copy_data_to_host(s, host);
        };
        storage->copy_grad_from_host = [](llm::TensorCudaStorage& s, const std::vector<double>& host) {
            MetalRuntime::instance().copy_grad_from_host(s, host);
        };
        storage->copy_grad_to_host = [](llm::TensorCudaStorage& s, std::vector<double>& host) {
            MetalRuntime::instance().copy_grad_to_host(s, host);
        };
        storage->fill_grad = [](llm::TensorCudaStorage& s, size_t count, float value) {
            MetalRuntime::instance().fill_grad_buffer(s, count, value);
        };
        return storage;
    }

    void ensure_data_buffer(llm::TensorCudaStorage& storage, size_t count) {
        ensure_float_buffer(storage.data, storage.data_count, count);
    }

    void ensure_grad_buffer(llm::TensorCudaStorage& storage, size_t count) {
        ensure_float_buffer(storage.grad, storage.grad_count, count);
    }

    void copy_data_from_host(llm::TensorCudaStorage& storage, const std::vector<double>& host) {
        ensure_data_buffer(storage, host.size());
        copy_from_host(storage.data, host);
    }

    void copy_data_to_host(llm::TensorCudaStorage& storage, std::vector<double>& host) {
        copy_to_host(storage.data, storage.data_count, host);
    }

    void copy_grad_from_host(llm::TensorCudaStorage& storage, const std::vector<double>& host) {
        ensure_grad_buffer(storage, host.size());
        copy_from_host(storage.grad, host);
    }

    void copy_grad_to_host(llm::TensorCudaStorage& storage, std::vector<double>& host) {
        copy_to_host(storage.grad, storage.grad_count, host);
    }

    void fill_data_buffer(llm::TensorCudaStorage& storage, size_t count, float value) {
        ensure_data_buffer(storage, count);
        run_fill(buffer_from(storage.data), count, value);
    }

    void fill_grad_buffer(llm::TensorCudaStorage& storage, size_t count, float value) {
        ensure_grad_buffer(storage, count);
        run_fill(buffer_from(storage.grad), count, value);
    }

    void set_grad_scalar(llm::TensorCudaStorage& storage, float value) {
        fill_grad_buffer(storage, 1, value);
    }

    void elementwise2_buffer(const char* op, llm::TensorCudaStorage& out, const llm::TensorCudaStorage& a,
                             const llm::TensorCudaStorage& b, uint32_t b_size, size_t count) {
        ensure_data_buffer(out, count);
        id<MTLComputePipelineState> pipeline = get_pipeline(std::string(op) == "add" ? "add_kernel" :
                                                            std::string(op) == "mul" ? "mul_kernel" : "div_kernel");
        id<MTLBuffer> b_size_buffer = [device_ newBufferWithBytes:&b_size length:sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(a.data) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(b.data) offset:0 atIndex:1];
        [encoder setBuffer:buffer_from(out.data) offset:0 atIndex:2];
        if (std::string(op) == "add") {
            [encoder setBuffer:b_size_buffer offset:0 atIndex:3];
        }
        dispatch1d(encoder, pipeline, count);
        [encoder endEncoding];
        submit(command);
    }

    void mul_scalar_buffer(llm::TensorCudaStorage& out, const llm::TensorCudaStorage& a, float scalar, size_t count) {
        ScalarParams params{scalar};
        unary_like("mul_scalar_kernel", out, a, &params, sizeof(params), count);
    }

    void unary_buffer(const char* op, llm::TensorCudaStorage& out, const llm::TensorCudaStorage& a,
                      float scalar, size_t count) {
        ScalarParams params{scalar};
        std::string name(op);
        const char* kernel = name == "pow" ? "pow_kernel" : name == "copy" ? "copy_kernel" :
                             name == "log" ? "log_kernel" : name == "gelu" ? "gelu_kernel" : "neg_kernel";
        unary_like(kernel, out, a, name == "pow" ? &params : nullptr, name == "pow" ? sizeof(params) : 0, count);
    }

    void gather_buffer(llm::TensorCudaStorage& out, const llm::TensorCudaStorage& a,
                       const std::vector<unsigned int>& index) {
        ensure_data_buffer(out, index.size());
        id<MTLComputePipelineState> pipeline = get_pipeline("gather_kernel");
        id<MTLBuffer> index_buffer = [device_ newBufferWithBytes:index.data()
                                                          length:sizeof(uint32_t) * index.size()
                                                         options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(a.data) offset:0 atIndex:0];
        [encoder setBuffer:index_buffer offset:0 atIndex:1];
        [encoder setBuffer:buffer_from(out.data) offset:0 atIndex:2];
        dispatch1d(encoder, pipeline, index.size());
        [encoder endEncoding];
        submit(command);
    }

    void scale_data_buffer(llm::TensorCudaStorage& storage, size_t count, float scalar) {
        llm::TensorCudaStorage out;
        out.data = storage.data;
        out.data_count = storage.data_count;
        mul_scalar_buffer(out, storage, scalar, count);
        storage.data = out.data;
        storage.data_count = out.data_count;
    }

    void reduce_buffer(const char* op, llm::TensorCudaStorage& out, const llm::TensorCudaStorage& a, size_t count) {
        ensure_data_buffer(out, 1);
        uint32_t c = static_cast<uint32_t>(count);
        id<MTLComputePipelineState> pipeline = get_pipeline(std::string(op) == "max" ? "max_kernel" : "sum_kernel");
        id<MTLBuffer> count_buffer = [device_ newBufferWithBytes:&c length:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(a.data) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(out.data) offset:0 atIndex:1];
        [encoder setBuffer:count_buffer offset:0 atIndex:2];
        dispatch1d(encoder, pipeline, 1);
        [encoder endEncoding];
        submit(command);
    }

    void matmul_buffer(llm::TensorCudaStorage& out, const llm::TensorCudaStorage& a, const llm::TensorCudaStorage& b,
                       uint32_t m, uint32_t k, uint32_t n) {
        ensure_data_buffer(out, static_cast<size_t>(m) * n);
        MatmulParams params{m, k, n};
        run_matmul_buffers(out, a, b, params);
    }

    void batch_matmul_buffer(llm::TensorCudaStorage& out, const llm::TensorCudaStorage& a,
                             const llm::TensorCudaStorage& b, uint32_t batches, uint32_t heads,
                             uint32_t m, uint32_t k, uint32_t n) {
        ensure_data_buffer(out, static_cast<size_t>(batches) * heads * m * n);
        BatchMatmulParams params{batches, heads, m, k, n};
        run_3buffer_params("batch_matmul_kernel", out, a, b, &params, sizeof(params),
                           static_cast<size_t>(batches) * heads * m * n);
    }

    void causal_mask_buffer(llm::TensorCudaStorage& out, const llm::TensorCudaStorage& scores,
                            uint32_t batches, uint32_t heads, uint32_t sequence_length, float mask_value) {
        size_t count = static_cast<size_t>(batches) * heads * sequence_length * sequence_length;
        ensure_data_buffer(out, count);
        CausalMaskParams params{batches, heads, sequence_length, mask_value};
        id<MTLComputePipelineState> pipeline = get_pipeline("causal_mask_kernel");
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:&params length:sizeof(params)
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(scores.data) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(out.data) offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        dispatch1d(encoder, pipeline, count);
        [encoder endEncoding];
        submit(command);
    }

    void softmax_buffer(llm::TensorCudaStorage& out, const llm::TensorCudaStorage& a, uint32_t rows, uint32_t width) {
        ensure_data_buffer(out, static_cast<size_t>(rows) * width);
        SoftmaxParams params{rows, width};
        run_1input_params("softmax_kernel", out, a, &params, sizeof(params), rows);
    }

    void log_softmax_buffer(llm::TensorCudaStorage& out, const llm::TensorCudaStorage& a, uint32_t rows, uint32_t width) {
        ensure_data_buffer(out, static_cast<size_t>(rows) * width);
        SoftmaxParams params{rows, width};
        run_1input_params("log_softmax_kernel", out, a, &params, sizeof(params), rows);
    }

    void layernorm_buffer(llm::TensorCudaStorage& out, const llm::TensorCudaStorage& x,
                          const llm::TensorCudaStorage& scale, const llm::TensorCudaStorage& shift,
                          uint32_t rows, uint32_t width, float eps) {
        ensure_data_buffer(out, static_cast<size_t>(rows) * width);
        LayerNormParams params{rows, width, eps};
        id<MTLComputePipelineState> pipeline = get_pipeline("layernorm_kernel");
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:&params length:sizeof(params)
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(x.data) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(scale.data) offset:0 atIndex:1];
        [encoder setBuffer:buffer_from(shift.data) offset:0 atIndex:2];
        [encoder setBuffer:buffer_from(out.data) offset:0 atIndex:3];
        [encoder setBuffer:params_buffer offset:0 atIndex:4];
        dispatch1d(encoder, pipeline, rows);
        [encoder endEncoding];
        submit(command);
    }

    void embedding_buffer(llm::TensorCudaStorage& out, const llm::TensorCudaStorage& ids,
                          const llm::TensorCudaStorage& weight, uint32_t count, uint32_t dim) {
        ensure_data_buffer(out, static_cast<size_t>(count) * dim);
        EmbeddingParams params{count, dim};
        run_3buffer_params("embedding_kernel", out, ids, weight, &params, sizeof(params), static_cast<size_t>(count) * dim);
    }

    void cross_entropy_loss_buffer(llm::TensorCudaStorage& out, const llm::TensorCudaStorage& logits,
                                   const llm::TensorCudaStorage& targets, uint32_t rows, uint32_t vocab) {
        ensure_data_buffer(out, 1);
        CrossEntropyParams params{rows, vocab};
        run_3buffer_params("cross_entropy_loss_kernel", out, logits, targets, &params, sizeof(params), 1);
    }

    void add_grad(llm::TensorCudaStorage& target, const llm::TensorCudaStorage& out_grad,
                  uint32_t target_size, size_t count, float scale = 1.0f) {
        ensure_grad_buffer(target, target_size == 0 ? count : target_size);
        GradParams params{target_size, static_cast<uint32_t>(count), scale};
        run_grad1("add_grad_kernel", target, out_grad, &params, sizeof(params), count);
    }

    void elementwise_grad(const char* op, llm::TensorCudaStorage* a_grad, llm::TensorCudaStorage* b_grad,
                          const llm::TensorCudaStorage& a, const llm::TensorCudaStorage& b,
                          const llm::TensorCudaStorage& out_grad, size_t count) {
        ElementwiseGradParams params{static_cast<uint32_t>(count), a_grad ? 1u : 0u, b_grad ? 1u : 0u};
        id<MTLComputePipelineState> pipeline = get_pipeline(std::string(op) == "div" ? "div_grad_kernel" : "mul_grad_kernel");
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:&params length:sizeof(params)
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(a_grad ? a_grad->grad : out_grad.grad) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(b_grad ? b_grad->grad : out_grad.grad) offset:0 atIndex:1];
        [encoder setBuffer:buffer_from(a.data) offset:0 atIndex:2];
        [encoder setBuffer:buffer_from(b.data) offset:0 atIndex:3];
        [encoder setBuffer:buffer_from(out_grad.grad) offset:0 atIndex:4];
        [encoder setBuffer:params_buffer offset:0 atIndex:5];
        dispatch1d(encoder, pipeline, count);
        [encoder endEncoding];
        submit(command);
    }

    void mul_scalar_grad(llm::TensorCudaStorage& a_grad, const llm::TensorCudaStorage& out_grad,
                         float scalar, size_t count) {
        GradParams params{0, static_cast<uint32_t>(count), scalar};
        run_grad1("mul_scalar_grad_kernel", a_grad, out_grad, &params, sizeof(params), count);
    }

    void pow_grad(llm::TensorCudaStorage& a_grad, const llm::TensorCudaStorage& a,
                  const llm::TensorCudaStorage& out_grad, float exponent, size_t count) {
        ensure_grad_buffer(a_grad, count);
        GradParams params{0, static_cast<uint32_t>(count), exponent};
        id<MTLComputePipelineState> pipeline = get_pipeline("pow_grad_kernel");
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:&params length:sizeof(params)
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(a_grad.grad) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(a.data) offset:0 atIndex:1];
        [encoder setBuffer:buffer_from(out_grad.grad) offset:0 atIndex:2];
        [encoder setBuffer:params_buffer offset:0 atIndex:3];
        dispatch1d(encoder, pipeline, count);
        [encoder endEncoding];
        submit(command);
    }

    void reduce_grad(llm::TensorCudaStorage& a_grad, const llm::TensorCudaStorage& out_grad,
                     size_t count, float scale) {
        ensure_grad_buffer(a_grad, count);
        GradParams params{0, static_cast<uint32_t>(count), scale};
        run_grad1("reduce_grad_kernel", a_grad, out_grad, &params, sizeof(params), count);
    }

    void scatter_add_grad(llm::TensorCudaStorage& a_grad, const llm::TensorCudaStorage& out_grad,
                          const std::vector<unsigned int>& index) {
        ensure_grad_buffer(a_grad, index.size());
        uint32_t count = static_cast<uint32_t>(index.size());
        id<MTLComputePipelineState> pipeline = get_pipeline("scatter_add_grad_kernel");
        id<MTLBuffer> index_buffer = [device_ newBufferWithBytes:index.data()
                                                          length:sizeof(uint32_t) * index.size()
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> count_buffer = [device_ newBufferWithBytes:&count length:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(a_grad.grad) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(out_grad.grad) offset:0 atIndex:1];
        [encoder setBuffer:index_buffer offset:0 atIndex:2];
        [encoder setBuffer:count_buffer offset:0 atIndex:3];
        dispatch1d(encoder, pipeline, count);
        [encoder endEncoding];
        submit(command);
    }

    void matmul_grad(llm::TensorCudaStorage* a_grad, llm::TensorCudaStorage* b_grad,
                     const llm::TensorCudaStorage& a, const llm::TensorCudaStorage& b,
                     const llm::TensorCudaStorage& out_grad, uint32_t m, uint32_t k, uint32_t n) {
        MatmulParams params{m, k, n};
        if (a_grad) {
            ensure_grad_buffer(*a_grad, static_cast<size_t>(m) * k);
            run_matmul_grad2d("matmul_grad_a_kernel", *a_grad, b, out_grad, params, k, m);
        }
        if (b_grad) {
            ensure_grad_buffer(*b_grad, static_cast<size_t>(k) * n);
            run_matmul_grad2d("matmul_grad_b_kernel", *b_grad, a, out_grad, params, n, k);
        }
    }

    void batch_matmul_grad(llm::TensorCudaStorage* a_grad, llm::TensorCudaStorage* b_grad,
                           const llm::TensorCudaStorage& a, const llm::TensorCudaStorage& b,
                           const llm::TensorCudaStorage& out_grad, uint32_t batches, uint32_t heads,
                           uint32_t m, uint32_t k, uint32_t n) {
        BatchMatmulParams params{batches, heads, m, k, n};
        if (a_grad) {
            ensure_grad_buffer(*a_grad, static_cast<size_t>(batches) * heads * m * k);
            run_grad3("batch_matmul_grad_a_kernel", *a_grad, b, out_grad, &params, sizeof(params),
                      static_cast<size_t>(batches) * heads * m * k);
        }
        if (b_grad) {
            ensure_grad_buffer(*b_grad, static_cast<size_t>(batches) * heads * k * n);
            run_grad3("batch_matmul_grad_b_kernel", *b_grad, a, out_grad, &params, sizeof(params),
                      static_cast<size_t>(batches) * heads * k * n);
        }
    }

    void softmax_grad(llm::TensorCudaStorage& a_grad, const llm::TensorCudaStorage& out,
                      const llm::TensorCudaStorage& out_grad, uint32_t rows, uint32_t width) {
        ensure_grad_buffer(a_grad, static_cast<size_t>(rows) * width);
        SoftmaxParams params{rows, width};
        run_grad3("softmax_grad_kernel", a_grad, out, out_grad, &params, sizeof(params), rows);
    }

    void cross_entropy_grad(llm::TensorCudaStorage& logits_grad, const llm::TensorCudaStorage& logits,
                            const llm::TensorCudaStorage& targets, const llm::TensorCudaStorage& out_grad,
                            uint32_t rows, uint32_t vocab) {
        ensure_grad_buffer(logits_grad, static_cast<size_t>(rows) * vocab);
        CrossEntropyParams params{rows, vocab};
        id<MTLComputePipelineState> pipeline = get_pipeline("cross_entropy_grad_kernel");
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:&params length:sizeof(params)
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(logits_grad.grad) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(logits.data) offset:0 atIndex:1];
        [encoder setBuffer:buffer_from(targets.data) offset:0 atIndex:2];
        [encoder setBuffer:buffer_from(out_grad.grad) offset:0 atIndex:3];
        [encoder setBuffer:params_buffer offset:0 atIndex:4];
        dispatch1d(encoder, pipeline, static_cast<size_t>(rows) * vocab);
        [encoder endEncoding];
        submit(command);
    }

    void embedding_grad(llm::TensorCudaStorage& weight_grad, const llm::TensorCudaStorage& ids,
                        const llm::TensorCudaStorage& out_grad, uint32_t count, uint32_t dim) {
        ensure_grad_buffer(weight_grad, weight_grad.data_count);
        EmbeddingParams params{count, dim};
        run_grad3("embedding_grad_kernel", weight_grad, ids, out_grad, &params, sizeof(params),
                  static_cast<size_t>(count) * dim);
    }

    void layernorm_grad(llm::TensorCudaStorage* x_grad, llm::TensorCudaStorage* scale_grad,
                        llm::TensorCudaStorage* shift_grad, const llm::TensorCudaStorage& x,
                        const llm::TensorCudaStorage& scale, const llm::TensorCudaStorage& out_grad,
                        uint32_t rows, uint32_t width, float eps) {
        LayerNormParams params{rows, width, eps};
        if (x_grad) {
            ensure_grad_buffer(*x_grad, static_cast<size_t>(rows) * width);
            id<MTLComputePipelineState> pipeline = get_pipeline("layernorm_grad_x_kernel");
            id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:&params length:sizeof(params)
                                                              options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> command = current_command();
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:buffer_from(x_grad->grad) offset:0 atIndex:0];
            [encoder setBuffer:buffer_from(x.data) offset:0 atIndex:1];
            [encoder setBuffer:buffer_from(scale.data) offset:0 atIndex:2];
            [encoder setBuffer:buffer_from(out_grad.grad) offset:0 atIndex:3];
            [encoder setBuffer:params_buffer offset:0 atIndex:4];
            dispatch1d(encoder, pipeline, static_cast<size_t>(rows) * width);
            [encoder endEncoding];
            submit(command);
        }
        if (scale_grad || shift_grad) {
            if (scale_grad) ensure_grad_buffer(*scale_grad, width);
            if (shift_grad) ensure_grad_buffer(*shift_grad, width);
            ElementwiseGradParams flags{static_cast<uint32_t>(rows * width), scale_grad ? 1u : 0u, shift_grad ? 1u : 0u};
            id<MTLComputePipelineState> pipeline = get_pipeline("layernorm_grad_scale_shift_kernel");
            id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:&params length:sizeof(params)
                                                              options:MTLResourceStorageModeShared];
            id<MTLBuffer> flags_buffer = [device_ newBufferWithBytes:&flags length:sizeof(flags)
                                                             options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> command = current_command();
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:buffer_from(scale_grad ? scale_grad->grad : out_grad.grad) offset:0 atIndex:0];
            [encoder setBuffer:buffer_from(shift_grad ? shift_grad->grad : out_grad.grad) offset:0 atIndex:1];
            [encoder setBuffer:buffer_from(x.data) offset:0 atIndex:2];
            [encoder setBuffer:buffer_from(out_grad.grad) offset:0 atIndex:3];
            [encoder setBuffer:params_buffer offset:0 atIndex:4];
            [encoder setBuffer:flags_buffer offset:0 atIndex:5];
            dispatch1d(encoder, pipeline, static_cast<size_t>(rows) * width);
            [encoder endEncoding];
            submit(command);
        }
    }

    void gelu_grad(llm::TensorCudaStorage& x_grad, const llm::TensorCudaStorage& x,
                   const llm::TensorCudaStorage& out_grad, size_t count) {
        ensure_grad_buffer(x_grad, count);
        uint32_t c = static_cast<uint32_t>(count);
        id<MTLComputePipelineState> pipeline = get_pipeline("gelu_grad_kernel");
        id<MTLBuffer> count_buffer = [device_ newBufferWithBytes:&c length:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(x_grad.grad) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(x.data) offset:0 atIndex:1];
        [encoder setBuffer:buffer_from(out_grad.grad) offset:0 atIndex:2];
        [encoder setBuffer:count_buffer offset:0 atIndex:3];
        dispatch1d(encoder, pipeline, count);
        [encoder endEncoding];
        submit(command);
    }

    void causal_mask_grad(llm::TensorCudaStorage& scores_grad, const llm::TensorCudaStorage& out_grad,
                          uint32_t batches, uint32_t heads, uint32_t sequence_length) {
        size_t count = static_cast<size_t>(batches) * heads * sequence_length * sequence_length;
        ensure_grad_buffer(scores_grad, count);
        CausalMaskParams params{batches, heads, sequence_length, 0.0f};
        run_grad1("causal_mask_grad_kernel", scores_grad, out_grad, &params, sizeof(params), count);
    }

    void adamw_update(llm::TensorCudaStorage& param, llm::TensorCudaStorage& grad,
                      llm::TensorCudaStorage& m, llm::TensorCudaStorage& v, size_t count,
                      float lr, float weight_decay, float beta1, float beta2, float eps,
                      float bias_correction1, float bias_correction2) {
        ensure_data_buffer(param, count);
        ensure_grad_buffer(grad, count);
        ensure_data_buffer(m, count);
        ensure_data_buffer(v, count);
        AdamWParams params{static_cast<uint32_t>(count), lr, weight_decay, beta1, beta2, eps,
                           bias_correction1, bias_correction2};
        id<MTLComputePipelineState> pipeline = get_pipeline("adamw_update_kernel");
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:&params length:sizeof(params)
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(param.data) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(grad.grad) offset:0 atIndex:1];
        [encoder setBuffer:buffer_from(m.data) offset:0 atIndex:2];
        [encoder setBuffer:buffer_from(v.data) offset:0 atIndex:3];
        [encoder setBuffer:params_buffer offset:0 atIndex:4];
        dispatch1d(encoder, pipeline, count);
        [encoder endEncoding];
        submit(command);
    }

private:
    MetalRuntime() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (device_ == nil) {
                status_ = "Metal backend is unavailable: no Metal device was found";
                return;
            }
            queue_ = [device_ newCommandQueue];
            if (queue_ == nil) {
                status_ = "Metal backend is unavailable: failed to create command queue";
                return;
            }
            NSError* error = nil;
#ifndef LLM_CPP_METALLIB_PATH
            status_ = "Metal backend is unavailable: LLM_CPP_METALLIB_PATH was not defined at build time";
            return;
#else
            NSString* metallib_path = [NSString stringWithUTF8String:LLM_CPP_METALLIB_PATH];
            NSURL* metallib_url = [NSURL fileURLWithPath:metallib_path];
            library_ = [device_ newLibraryWithURL:metallib_url error:&error];
            if (library_ == nil) {
                status_ = "Metal backend is unavailable: failed to load Metal library from " +
                          std::string(LLM_CPP_METALLIB_PATH);
                if (error != nil) {
                    status_ += ": " + std::string([[error localizedDescription] UTF8String]);
                }
                return;
            }
            status_ = "Metal backend available";
#endif
        }
    }

    id<MTLComputePipelineState> get_pipeline(const char* name) {
        std::string key(name);
        auto it = pipelines_.find(key);
        if (it != pipelines_.end()) {
            return it->second;
        }
        NSString* function_name = [NSString stringWithUTF8String:name];
        id<MTLFunction> function = [library_ newFunctionWithName:function_name];
        if (function == nil) {
            throw std::runtime_error("Metal function not found: " + key);
        }
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            std::string message = "failed to create Metal pipeline: " + key;
            if (error != nil) {
                message += ": " + std::string([[error localizedDescription] UTF8String]);
            }
            throw std::runtime_error(message);
        }
        pipelines_[key] = pipeline;
        return pipeline;
    }

    id<MTLBuffer> buffer(const std::vector<float>& values) {
        return [device_ newBufferWithBytes:values.data()
                                    length:sizeof(float) * values.size()
                                   options:MTLResourceStorageModeShared];
    }

    std::vector<float> read(id<MTLBuffer> buffer, int64_t count) {
        sync();
        auto* ptr = static_cast<float*>([buffer contents]);
        return std::vector<float>(ptr, ptr + count);
    }

    std::vector<float> run3buffer1d(const char* kernel_name,
                                    const std::vector<float>& a,
                                    const std::vector<float>& b,
                                    const void* params,
                                    size_t params_size,
                                    int64_t count) {
        if (!available()) {
            throw std::runtime_error(status_);
        }
        id<MTLComputePipelineState> pipeline_state = get_pipeline(kernel_name);
        id<MTLBuffer> a_buffer = buffer(a);
        id<MTLBuffer> b_buffer = buffer(b);
        id<MTLBuffer> out_buffer = [device_ newBufferWithLength:sizeof(float) * static_cast<size_t>(count)
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:params length:params_size
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_state];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:b_buffer offset:0 atIndex:1];
        [encoder setBuffer:out_buffer offset:0 atIndex:2];
        [encoder setBuffer:params_buffer offset:0 atIndex:3];
        dispatch1d(encoder, pipeline_state, static_cast<NSUInteger>(count));
        [encoder endEncoding];
        submit(command);
        return read(out_buffer, count);
    }

    std::vector<float> run1input_rows(const char* kernel_name,
                                      const std::vector<float>& x,
                                      const void* params,
                                      size_t params_size,
                                      uint32_t rows,
                                      int64_t output_count) {
        if (!available()) {
            throw std::runtime_error(status_);
        }
        id<MTLComputePipelineState> pipeline_state = get_pipeline(kernel_name);
        id<MTLBuffer> x_buffer = buffer(x);
        id<MTLBuffer> out_buffer = [device_ newBufferWithLength:sizeof(float) * static_cast<size_t>(output_count)
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:params length:params_size
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_state];
        [encoder setBuffer:x_buffer offset:0 atIndex:0];
        [encoder setBuffer:out_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        dispatch1d(encoder, pipeline_state, rows);
        [encoder endEncoding];
        submit(command);
        return read(out_buffer, output_count);
    }

public:
    // 单线程全量规约（sum_kernel / max_kernel），输出 1 个标量。
    float reduce(const char* kernel_name, const std::vector<float>& a) {
        if (!available()) {
            throw std::runtime_error(status_);
        }
        id<MTLComputePipelineState> pipeline_state = get_pipeline(kernel_name);
        id<MTLBuffer> a_buffer = buffer(a);
        id<MTLBuffer> out_buffer = [device_ newBufferWithLength:sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        uint32_t count = static_cast<uint32_t>(a.size());
        id<MTLBuffer> count_buffer = [device_ newBufferWithBytes:&count length:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_state];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:out_buffer offset:0 atIndex:1];
        [encoder setBuffer:count_buffer offset:0 atIndex:2];
        dispatch1d(encoder, pipeline_state, 1);
        [encoder endEncoding];
        submit(command);
        return read(out_buffer, 1)[0];
    }

    // 按 host 端预计算的 index 表做数据重排（gather_kernel），供 transpose 使用。
    std::vector<float> gather(const std::vector<float>& a, const std::vector<uint32_t>& index) {
        if (!available()) {
            throw std::runtime_error(status_);
        }
        id<MTLComputePipelineState> pipeline_state = get_pipeline("gather_kernel");
        id<MTLBuffer> a_buffer = buffer(a);
        id<MTLBuffer> index_buffer = [device_ newBufferWithBytes:index.data()
                                                          length:sizeof(uint32_t) * index.size()
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer = [device_ newBufferWithLength:sizeof(float) * index.size()
                                                        options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_state];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:index_buffer offset:0 atIndex:1];
        [encoder setBuffer:out_buffer offset:0 atIndex:2];
        dispatch1d(encoder, pipeline_state, static_cast<NSUInteger>(index.size()));
        [encoder endEncoding];
        submit(command);
        return read(out_buffer, static_cast<int64_t>(index.size()));
    }

private:
    static id<MTLBuffer> buffer_from(void* ptr) {
        return (__bridge id<MTLBuffer>)ptr;
    }

    static id<MTLBuffer> buffer_from(const void* ptr) {
        return (__bridge id<MTLBuffer>)const_cast<void*>(ptr);
    }

    static void release_buffer(void* ptr) {
        if (ptr != nullptr) {
            [(__bridge id)ptr release];
        }
    }

    void ensure_float_buffer(void*& ptr, size_t& current_count, size_t requested_count) {
        if (current_count >= requested_count) {
            return;
        }
        release_buffer(ptr);
        ptr = nullptr;
        current_count = 0;
        if (requested_count > 0) {
            id<MTLBuffer> buffer = [device_ newBufferWithLength:sizeof(float) * requested_count
                                                        options:MTLResourceStorageModeShared];
            ptr = (void*)buffer;
        }
        current_count = requested_count;
    }

    static std::vector<float> double_to_float(const std::vector<double>& values) {
        return std::vector<float>(values.begin(), values.end());
    }

    void copy_from_host(void* ptr, const std::vector<double>& host) {
        if (host.empty()) {
            return;
        }
        std::vector<float> values = double_to_float(host);
        std::memcpy([buffer_from(ptr) contents], values.data(), sizeof(float) * values.size());
    }

    void copy_to_host(void* ptr, size_t count, std::vector<double>& host) {
        sync();
        host.assign(count, 0.0);
        if (count == 0) {
            return;
        }
        auto* values = static_cast<float*>([buffer_from(ptr) contents]);
        for (size_t i = 0; i < count; ++i) {
            host[i] = values[i];
        }
    }

    void run_fill(id<MTLBuffer> target, size_t count, float value) {
        ScalarParams params{value};
        id<MTLComputePipelineState> pipeline = get_pipeline("fill_kernel");
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:&params length:sizeof(params)
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:target offset:0 atIndex:0];
        [encoder setBuffer:params_buffer offset:0 atIndex:1];
        dispatch1d(encoder, pipeline, count);
        [encoder endEncoding];
        submit(command);
    }

    void unary_like(const char* kernel_name, llm::TensorCudaStorage& out, const llm::TensorCudaStorage& a,
                    const void* params, size_t params_size, size_t count) {
        ensure_data_buffer(out, count);
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        id<MTLBuffer> params_buffer = nil;
        if (params != nullptr && params_size > 0) {
            params_buffer = [device_ newBufferWithBytes:params length:params_size options:MTLResourceStorageModeShared];
        }
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(a.data) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(out.data) offset:0 atIndex:1];
        if (params_buffer != nil) {
            [encoder setBuffer:params_buffer offset:0 atIndex:2];
        }
        dispatch1d(encoder, pipeline, count);
        [encoder endEncoding];
        submit(command);
    }

    void run_matmul_buffers(llm::TensorCudaStorage& out, const llm::TensorCudaStorage& a,
                            const llm::TensorCudaStorage& b, const MatmulParams& params) {
        id<MTLComputePipelineState> pipeline = get_pipeline("matmul_kernel");
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:&params length:sizeof(params)
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(a.data) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(b.data) offset:0 atIndex:1];
        [encoder setBuffer:buffer_from(out.data) offset:0 atIndex:2];
        [encoder setBuffer:params_buffer offset:0 atIndex:3];
        MTLSize grid = MTLSizeMake(params.n, params.m, 1);
        NSUInteger width = pipeline.threadExecutionWidth;
        NSUInteger height = std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup / width);
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(width, height, 1)];
        [encoder endEncoding];
        submit(command);
    }

    void run_3buffer_params(const char* kernel_name, llm::TensorCudaStorage& out,
                            const llm::TensorCudaStorage& a, const llm::TensorCudaStorage& b,
                            const void* params, size_t params_size, size_t count) {
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:params length:params_size
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(a.data) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(b.data) offset:0 atIndex:1];
        [encoder setBuffer:buffer_from(out.data) offset:0 atIndex:2];
        [encoder setBuffer:params_buffer offset:0 atIndex:3];
        dispatch1d(encoder, pipeline, count);
        [encoder endEncoding];
        submit(command);
    }

    void run_1input_params(const char* kernel_name, llm::TensorCudaStorage& out,
                           const llm::TensorCudaStorage& a, const void* params,
                           size_t params_size, size_t rows) {
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:params length:params_size
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(a.data) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(out.data) offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        dispatch1d(encoder, pipeline, rows);
        [encoder endEncoding];
        submit(command);
    }

    void run_matmul_grad2d(const char* kernel_name, llm::TensorCudaStorage& target,
                           const llm::TensorCudaStorage& input, const llm::TensorCudaStorage& out_grad,
                           const MatmulParams& params, uint32_t grid_x, uint32_t grid_y) {
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:&params length:sizeof(params)
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(target.grad) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(input.data) offset:0 atIndex:1];
        [encoder setBuffer:buffer_from(out_grad.grad) offset:0 atIndex:2];
        [encoder setBuffer:params_buffer offset:0 atIndex:3];
        NSUInteger width = pipeline.threadExecutionWidth;
        NSUInteger height = std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup / width);
        [encoder dispatchThreads:MTLSizeMake(grid_x, grid_y, 1) threadsPerThreadgroup:MTLSizeMake(width, height, 1)];
        [encoder endEncoding];
        submit(command);
    }

    void run_grad3(const char* kernel_name, llm::TensorCudaStorage& target,
                   const llm::TensorCudaStorage& input, const llm::TensorCudaStorage& out_grad,
                   const void* params, size_t params_size, size_t count) {
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:params length:params_size
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(target.grad) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(input.data) offset:0 atIndex:1];
        [encoder setBuffer:buffer_from(out_grad.grad) offset:0 atIndex:2];
        [encoder setBuffer:params_buffer offset:0 atIndex:3];
        dispatch1d(encoder, pipeline, count);
        [encoder endEncoding];
        submit(command);
    }

    void run_grad1(const char* kernel_name, llm::TensorCudaStorage& target, const llm::TensorCudaStorage& out_grad,
                   const void* params, size_t params_size, size_t count) {
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:params length:params_size
                                                          options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = current_command();
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_from(target.grad) offset:0 atIndex:0];
        [encoder setBuffer:buffer_from(out_grad.grad) offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        dispatch1d(encoder, pipeline, count);
        [encoder endEncoding];
        submit(command);
    }

    void dispatch1d(id<MTLComputeCommandEncoder> encoder,
                    id<MTLComputePipelineState> pipeline,
                    NSUInteger count) {
        NSUInteger width = pipeline.threadExecutionWidth;
        MTLSize grid = MTLSizeMake(count, 1, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    }

    id<MTLCommandBuffer> current_command() {
        if (active_command_ == nil) {
            active_command_ = [[queue_ commandBuffer] retain];
            pending_encoder_count_ = 0;
        }
        return active_command_;
    }

    void submit(id<MTLCommandBuffer> command) {
        (void)command;
        ++pending_encoder_count_;
        constexpr NSUInteger kMaxEncodersPerCommandBuffer = 64;
        if (pending_encoder_count_ >= kMaxEncodersPerCommandBuffer) {
            commit_active();
        }
    }

    void commit_active() {
        if (active_command_ == nil) {
            return;
        }
        [active_command_ commit];
        [last_command_ release];
        last_command_ = active_command_;
        active_command_ = nil;
        pending_encoder_count_ = 0;
    }

    void sync() {
        commit_active();
        if (last_command_ == nil) {
            return;
        }
        [last_command_ waitUntilCompleted];
        [last_command_ release];
        last_command_ = nil;
    }

    id<MTLDevice> device_{nil};
    id<MTLCommandQueue> queue_{nil};
    id<MTLLibrary> library_{nil};
    id<MTLCommandBuffer> active_command_{nil};
    id<MTLCommandBuffer> last_command_{nil};
    NSUInteger pending_encoder_count_{0};
    std::map<std::string, id<MTLComputePipelineState>> pipelines_;
    std::string status_;
};

std::vector<float> to_float(const std::vector<double>& values) {
    return std::vector<float>(values.begin(), values.end());
}

std::vector<double> to_double(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

llm::TensorCudaStorage& ensure_metal_storage(const llm::Tensor& t) {
    if (!t.node->metal_storage) {
        t.node->metal_storage = MetalRuntime::instance().create_tensor_storage();
    }
    return *t.node->metal_storage;
}

void ensure_metal_data(const llm::Tensor& t) {
    llm::TensorCudaStorage& storage = ensure_metal_storage(t);
    if (t.node->host_data_dirty || storage.data == nullptr || storage.data_count < static_cast<size_t>(t.numel())) {
        MetalRuntime::instance().copy_data_from_host(storage, t.node->data);
        t.node->host_data_dirty = false;
        t.node->device_data_dirty = false;
    }
}

void ensure_metal_grad(const llm::Tensor& t) {
    llm::TensorCudaStorage& storage = ensure_metal_storage(t);
    if (t.node->grad.empty()) {
        t.node->grad.assign(static_cast<size_t>(t.numel()), 0.0);
        t.node->host_grad_dirty = true;
    }
    if (t.node->host_grad_dirty || storage.grad == nullptr || storage.grad_count < static_cast<size_t>(t.numel())) {
        MetalRuntime::instance().copy_grad_from_host(storage, t.node->grad);
        t.node->host_grad_dirty = false;
        t.node->device_grad_dirty = true;
    }
}

void mark_metal_grad_dirty(const llm::Tensor& t) {
    t.node->host_grad_dirty = false;
    t.node->device_grad_dirty = true;
}

llm::Tensor make_metal_output(const std::vector<int64_t>& shape, llm::Device device, bool requires_grad) {
    llm::Tensor out(shape, llm::DType::Float32, device, requires_grad);
    out.node->metal_storage = MetalRuntime::instance().create_tensor_storage();
    out.node->host_data_dirty = false;
    out.node->device_data_dirty = true;
    return out;
}

} // namespace

namespace llm::metal {

namespace detail {

bool runtime_available() {
    return MetalRuntime::instance().available();
}

std::shared_ptr<TensorCudaStorage> create_tensor_storage() {
    return MetalRuntime::instance().create_tensor_storage();
}

void fill_data_buffer(TensorCudaStorage& storage, size_t count, float value) {
    MetalRuntime::instance().fill_data_buffer(storage, count, value);
}

void adamw_update(TensorCudaStorage& param, TensorCudaStorage& grad, TensorCudaStorage& m, TensorCudaStorage& v,
                  size_t count, float lr, float weight_decay, float beta1, float beta2,
                  float eps, float bias_correction1, float bias_correction2) {
    MetalRuntime::instance().adamw_update(param, grad, m, v, count, lr, weight_decay, beta1, beta2, eps,
                                          bias_correction1, bias_correction2);
}

} // namespace detail

bool available() {
    return MetalRuntime::instance().available();
}

std::string status() {
    return MetalRuntime::instance().status();
}

Tensor add(const Tensor& a, const Tensor& b) {
    bool same_shape = a.shape() == b.shape();
    bool broadcast_batch = a.shape().size() == 3 && b.shape().size() == 2 &&
                           a.shape()[1] == b.shape()[0] && a.shape()[2] == b.shape()[1];
    bool broadcast_last = !a.shape().empty() && b.shape().size() == 1 && a.shape().back() == b.shape()[0];
    if (!same_shape && !broadcast_batch && !broadcast_last) {
        throw std::runtime_error("Metal add shape mismatch");
    }
    ensure_metal_data(a);
    ensure_metal_data(b);
    Tensor out = make_metal_output(a.shape(), a.device(), a.requires_grad() || b.requires_grad());
    MetalRuntime::instance().elementwise2_buffer("add", *out.node->metal_storage, *a.node->metal_storage,
                                                 *b.node->metal_storage, static_cast<uint32_t>(b.numel()),
                                                 static_cast<size_t>(a.numel()));
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out, broadcast_batch, broadcast_last]() mutable {
            ensure_metal_grad(out);
            if (a.requires_grad()) {
                ensure_metal_grad(a);
                MetalRuntime::instance().add_grad(*a.node->metal_storage, *out.node->metal_storage, 0,
                                                  static_cast<size_t>(out.numel()));
                mark_metal_grad_dirty(a);
            }
            if (b.requires_grad()) {
                ensure_metal_grad(b);
                MetalRuntime::instance().add_grad(*b.node->metal_storage, *out.node->metal_storage,
                                                  (broadcast_batch || broadcast_last)
                                                      ? static_cast<uint32_t>(b.numel())
                                                      : 0,
                                                  static_cast<size_t>(out.numel()));
                mark_metal_grad_dirty(b);
            }
        };
    }
    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Metal mul expects same shape");
    }
    ensure_metal_data(a);
    ensure_metal_data(b);
    Tensor out = make_metal_output(a.shape(), a.device(), a.requires_grad() || b.requires_grad());
    MetalRuntime::instance().elementwise2_buffer("mul", *out.node->metal_storage, *a.node->metal_storage,
                                                 *b.node->metal_storage, 0, static_cast<size_t>(a.numel()));
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out]() mutable {
            ensure_metal_grad(out);
            ensure_metal_data(a);
            ensure_metal_data(b);
            llm::TensorCudaStorage* a_grad = nullptr;
            llm::TensorCudaStorage* b_grad = nullptr;
            if (a.requires_grad()) {
                ensure_metal_grad(a);
                a_grad = a.node->metal_storage.get();
            }
            if (b.requires_grad()) {
                ensure_metal_grad(b);
                b_grad = b.node->metal_storage.get();
            }
            MetalRuntime::instance().elementwise_grad("mul", a_grad, b_grad, *a.node->metal_storage,
                                                      *b.node->metal_storage, *out.node->metal_storage,
                                                      static_cast<size_t>(out.numel()));
            if (a_grad) mark_metal_grad_dirty(a);
            if (b_grad) mark_metal_grad_dirty(b);
        };
    }
    return out;
}

Tensor mul_scalar(const Tensor& a, double scalar) {
    ScalarParams params{static_cast<float>(scalar)};
    ensure_metal_data(a);
    Tensor out = make_metal_output(a.shape(), a.device(), a.requires_grad());
    MetalRuntime::instance().mul_scalar_buffer(*out.node->metal_storage, *a.node->metal_storage,
                                               static_cast<float>(scalar), static_cast<size_t>(a.numel()));
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, scalar]() mutable {
            ensure_metal_grad(out);
            ensure_metal_grad(a);
            MetalRuntime::instance().mul_scalar_grad(*a.node->metal_storage, *out.node->metal_storage,
                                                     static_cast<float>(scalar),
                                                     static_cast<size_t>(out.numel()));
            mark_metal_grad_dirty(a);
        };
    }
    return out;
}

Tensor div(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Metal div expects same shape");
    }
    ensure_metal_data(a);
    ensure_metal_data(b);
    Tensor out = make_metal_output(a.shape(), a.device(), a.requires_grad() || b.requires_grad());
    MetalRuntime::instance().elementwise2_buffer("div", *out.node->metal_storage, *a.node->metal_storage,
                                                 *b.node->metal_storage, 0, static_cast<size_t>(a.numel()));
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out]() mutable {
            ensure_metal_grad(out);
            ensure_metal_data(a);
            ensure_metal_data(b);
            llm::TensorCudaStorage* a_grad = nullptr;
            llm::TensorCudaStorage* b_grad = nullptr;
            if (a.requires_grad()) {
                ensure_metal_grad(a);
                a_grad = a.node->metal_storage.get();
            }
            if (b.requires_grad()) {
                ensure_metal_grad(b);
                b_grad = b.node->metal_storage.get();
            }
            MetalRuntime::instance().elementwise_grad("div", a_grad, b_grad, *a.node->metal_storage,
                                                      *b.node->metal_storage, *out.node->metal_storage,
                                                      static_cast<size_t>(out.numel()));
            if (a_grad) mark_metal_grad_dirty(a);
            if (b_grad) mark_metal_grad_dirty(b);
        };
    }
    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2 || a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("Metal matmul expects [m,k] x [k,n]");
    }
    uint32_t m = static_cast<uint32_t>(a.shape()[0]);
    uint32_t k = static_cast<uint32_t>(a.shape()[1]);
    uint32_t n = static_cast<uint32_t>(b.shape()[1]);
    ensure_metal_data(a);
    ensure_metal_data(b);
    Tensor out = make_metal_output({static_cast<int64_t>(m), static_cast<int64_t>(n)}, a.device(),
                                   a.requires_grad() || b.requires_grad());
    MetalRuntime::instance().matmul_buffer(*out.node->metal_storage, *a.node->metal_storage, *b.node->metal_storage,
                                           m, k, n);
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out, m, k, n]() mutable {
            ensure_metal_grad(out);
            ensure_metal_data(a);
            ensure_metal_data(b);
            llm::TensorCudaStorage* a_grad = nullptr;
            llm::TensorCudaStorage* b_grad = nullptr;
            if (a.requires_grad()) {
                ensure_metal_grad(a);
                a_grad = a.node->metal_storage.get();
            }
            if (b.requires_grad()) {
                ensure_metal_grad(b);
                b_grad = b.node->metal_storage.get();
            }
            MetalRuntime::instance().matmul_grad(a_grad, b_grad, *a.node->metal_storage, *b.node->metal_storage,
                                                 *out.node->metal_storage, m, k, n);
            if (a_grad) mark_metal_grad_dirty(a);
            if (b_grad) mark_metal_grad_dirty(b);
        };
    }
    return out;
}

Tensor batch_matmul(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 4 || b.shape().size() != 4) {
        throw std::runtime_error("Metal batch_matmul expects 4D tensors");
    }
    uint32_t B = static_cast<uint32_t>(a.shape()[0]);
    uint32_t H = static_cast<uint32_t>(a.shape()[1]);
    uint32_t M = static_cast<uint32_t>(a.shape()[2]);
    uint32_t K = static_cast<uint32_t>(a.shape()[3]);
    if (b.shape()[0] != B || b.shape()[1] != H || b.shape()[2] != K) {
        throw std::runtime_error("Metal batch_matmul shape mismatch");
    }
    uint32_t N = static_cast<uint32_t>(b.shape()[3]);
    ensure_metal_data(a);
    ensure_metal_data(b);
    Tensor out = make_metal_output({B, H, M, N}, a.device(), a.requires_grad() || b.requires_grad());
    MetalRuntime::instance().batch_matmul_buffer(*out.node->metal_storage, *a.node->metal_storage,
                                                 *b.node->metal_storage, B, H, M, K, N);
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out, B, H, M, K, N]() mutable {
            ensure_metal_grad(out);
            ensure_metal_data(a);
            ensure_metal_data(b);
            llm::TensorCudaStorage* a_grad = nullptr;
            llm::TensorCudaStorage* b_grad = nullptr;
            if (a.requires_grad()) {
                ensure_metal_grad(a);
                a_grad = a.node->metal_storage.get();
            }
            if (b.requires_grad()) {
                ensure_metal_grad(b);
                b_grad = b.node->metal_storage.get();
            }
            MetalRuntime::instance().batch_matmul_grad(a_grad, b_grad, *a.node->metal_storage,
                                                       *b.node->metal_storage, *out.node->metal_storage,
                                                       B, H, M, K, N);
            if (a_grad) mark_metal_grad_dirty(a);
            if (b_grad) mark_metal_grad_dirty(b);
        };
    }
    return out;
}

Tensor causal_mask(const Tensor& scores, int64_t sequence_length, double mask_value) {
    if (scores.shape().size() != 4 || scores.shape()[2] != sequence_length || scores.shape()[3] != sequence_length) {
        throw std::runtime_error("Metal causal_mask expects scores [B,H,T,T]");
    }
    uint32_t B = static_cast<uint32_t>(scores.shape()[0]);
    uint32_t H = static_cast<uint32_t>(scores.shape()[1]);
    uint32_t T = static_cast<uint32_t>(sequence_length);
    ensure_metal_data(scores);
    Tensor out = make_metal_output(scores.shape(), scores.device(), scores.requires_grad());
    MetalRuntime::instance().causal_mask_buffer(*out.node->metal_storage, *scores.node->metal_storage,
                                                B, H, T, static_cast<float>(mask_value));
    if (scores.requires_grad()) {
        out.node->parents = {scores};
        out.node->backward_fn = [scores, out, B, H, T]() mutable {
            ensure_metal_grad(out);
            ensure_metal_grad(scores);
            MetalRuntime::instance().causal_mask_grad(*scores.node->metal_storage, *out.node->metal_storage, B, H, T);
            mark_metal_grad_dirty(scores);
        };
    }
    return out;
}

Tensor softmax(const Tensor& a, int64_t dim) {
    int64_t rank = static_cast<int64_t>(a.shape().size());
    dim = canonical_dim(dim, rank);
    if (dim != rank - 1) {
        throw std::runtime_error("Metal softmax currently supports last dim only");
    }
    uint32_t width = static_cast<uint32_t>(a.shape().back());
    uint32_t rows = static_cast<uint32_t>(a.numel() / width);
    ensure_metal_data(a);
    Tensor out = make_metal_output(a.shape(), a.device(), a.requires_grad());
    MetalRuntime::instance().softmax_buffer(*out.node->metal_storage, *a.node->metal_storage, rows, width);
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, rows, width]() mutable {
            ensure_metal_grad(out);
            ensure_metal_grad(a);
            MetalRuntime::instance().softmax_grad(*a.node->metal_storage, *out.node->metal_storage,
                                                  *out.node->metal_storage, rows, width);
            mark_metal_grad_dirty(a);
        };
    }
    return out;
}

Tensor layernorm(const Tensor& x, const Tensor& scale, const Tensor& shift, double eps) {
    uint32_t C = static_cast<uint32_t>(x.shape().back());
    uint32_t rows = static_cast<uint32_t>(x.numel() / C);
    ensure_metal_data(x);
    ensure_metal_data(scale);
    ensure_metal_data(shift);
    Tensor out = make_metal_output(x.shape(), x.device(),
                                   x.requires_grad() || scale.requires_grad() || shift.requires_grad());
    MetalRuntime::instance().layernorm_buffer(*out.node->metal_storage, *x.node->metal_storage,
                                              *scale.node->metal_storage, *shift.node->metal_storage,
                                              rows, C, static_cast<float>(eps));

    if (out.requires_grad()) {
        out.node->parents = {x, scale, shift};
        out.node->backward_fn = [x, scale, shift, out, C, rows, eps]() mutable {
            ensure_metal_grad(out);
            ensure_metal_data(x);
            ensure_metal_data(scale);
            llm::TensorCudaStorage* x_grad = nullptr;
            llm::TensorCudaStorage* scale_grad = nullptr;
            llm::TensorCudaStorage* shift_grad = nullptr;
            if (x.requires_grad()) {
                ensure_metal_grad(x);
                x_grad = x.node->metal_storage.get();
            }
            if (scale.requires_grad()) {
                ensure_metal_grad(scale);
                scale_grad = scale.node->metal_storage.get();
            }
            if (shift.requires_grad()) {
                ensure_metal_grad(shift);
                shift_grad = shift.node->metal_storage.get();
            }
            MetalRuntime::instance().layernorm_grad(x_grad, scale_grad, shift_grad, *x.node->metal_storage,
                                                    *scale.node->metal_storage, *out.node->metal_storage,
                                                    rows, C, static_cast<float>(eps));
            if (x_grad) mark_metal_grad_dirty(x);
            if (scale_grad) mark_metal_grad_dirty(scale);
            if (shift_grad) mark_metal_grad_dirty(shift);
        };
    }
    return out;
}

Tensor embedding(const Tensor& ids, const Tensor& weight) {
    if (weight.shape().size() != 2) {
        throw std::runtime_error("Metal embedding weight must be 2D");
    }
    uint32_t count = static_cast<uint32_t>(ids.numel());
    uint32_t dim = static_cast<uint32_t>(weight.shape()[1]);
    auto out_shape = ids.shape();
    out_shape.push_back(dim);
    ensure_metal_data(ids);
    ensure_metal_data(weight);
    Tensor out = make_metal_output(out_shape, weight.device(), weight.requires_grad());
    MetalRuntime::instance().embedding_buffer(*out.node->metal_storage, *ids.node->metal_storage,
                                              *weight.node->metal_storage, count, dim);
    if (weight.requires_grad()) {
        out.node->parents = {weight};
        out.node->backward_fn = [ids, weight, out, dim]() mutable {
            ensure_metal_grad(out);
            ensure_metal_data(ids);
            ensure_metal_grad(weight);
            MetalRuntime::instance().embedding_grad(*weight.node->metal_storage, *ids.node->metal_storage,
                                                    *out.node->metal_storage, static_cast<uint32_t>(ids.numel()), dim);
            mark_metal_grad_dirty(weight);
        };
    }
    return out;
}

Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
    if (logits.shape().size() != 3) {
        throw std::runtime_error("Metal cross_entropy expects logits [B,T,V]");
    }
    int64_t B = logits.shape()[0], T = logits.shape()[1], V = logits.shape()[2];
    if (targets.numel() != B * T) {
        throw std::runtime_error("Metal cross_entropy target shape mismatch");
    }

    ensure_metal_data(logits);
    ensure_metal_data(targets);
    Tensor out = make_metal_output({}, logits.device(), logits.requires_grad());
    MetalRuntime::instance().cross_entropy_loss_buffer(*out.node->metal_storage, *logits.node->metal_storage,
                                                       *targets.node->metal_storage,
                                                       static_cast<uint32_t>(B * T), static_cast<uint32_t>(V));

    if (logits.requires_grad()) {
        out.node->parents = {logits};
        out.node->backward_fn = [logits, targets, out, B, T, V]() mutable {
            ensure_metal_grad(out);
            ensure_metal_data(logits);
            ensure_metal_data(targets);
            ensure_metal_grad(logits);
            MetalRuntime::instance().cross_entropy_grad(*logits.node->metal_storage, *logits.node->metal_storage,
                                                        *targets.node->metal_storage, *out.node->metal_storage,
                                                        static_cast<uint32_t>(B * T), static_cast<uint32_t>(V));
            mark_metal_grad_dirty(logits);
        };
    }
    return out;
}

Tensor gelu(const Tensor& x) {
    ensure_metal_data(x);
    Tensor out = make_metal_output(x.shape(), x.device(), x.requires_grad());
    MetalRuntime::instance().unary_buffer("gelu", *out.node->metal_storage, *x.node->metal_storage,
                                          0.0f, static_cast<size_t>(x.numel()));
    if (x.requires_grad()) {
        out.node->parents = {x};
        out.node->backward_fn = [x, out]() mutable {
            ensure_metal_grad(out);
            ensure_metal_data(x);
            ensure_metal_grad(x);
            MetalRuntime::instance().gelu_grad(*x.node->metal_storage, *x.node->metal_storage,
                                               *out.node->metal_storage, static_cast<size_t>(x.numel()));
            mark_metal_grad_dirty(x);
        };
    }
    return out;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    ensure_metal_data(b);
    Tensor neg_b = make_metal_output(b.shape(), b.device(), b.requires_grad());
    MetalRuntime::instance().unary_buffer("neg", *neg_b.node->metal_storage, *b.node->metal_storage,
                                          0.0f, static_cast<size_t>(b.numel()));
    if (b.requires_grad()) {
        neg_b.node->parents = {b};
        neg_b.node->backward_fn = [b, neg_b]() mutable {
            ensure_metal_grad(neg_b);
            ensure_metal_grad(b);
            MetalRuntime::instance().mul_scalar_grad(*b.node->metal_storage, *neg_b.node->metal_storage,
                                                     -1.0f, static_cast<size_t>(neg_b.numel()));
            mark_metal_grad_dirty(b);
        };
    }
    return add(a, neg_b);
}

Tensor pow(const Tensor& a, double exponent) {
    ScalarParams params{static_cast<float>(exponent)};
    ensure_metal_data(a);
    Tensor out = make_metal_output(a.shape(), a.device(), a.requires_grad());
    MetalRuntime::instance().unary_buffer("pow", *out.node->metal_storage, *a.node->metal_storage,
                                          static_cast<float>(exponent), static_cast<size_t>(a.numel()));
    if (out.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, exponent]() mutable {
            ensure_metal_grad(out);
            ensure_metal_data(a);
            ensure_metal_grad(a);
            MetalRuntime::instance().pow_grad(*a.node->metal_storage, *a.node->metal_storage,
                                              *out.node->metal_storage, static_cast<float>(exponent),
                                              static_cast<size_t>(out.numel()));
            mark_metal_grad_dirty(a);
        };
    }
    return out;
}

Tensor sum(const Tensor& a) {
    ensure_metal_data(a);
    Tensor out = make_metal_output({}, a.device(), a.requires_grad());
    MetalRuntime::instance().reduce_buffer("sum", *out.node->metal_storage, *a.node->metal_storage,
                                           static_cast<size_t>(a.numel()));
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out]() mutable {
            ensure_metal_grad(out);
            ensure_metal_grad(a);
            MetalRuntime::instance().reduce_grad(*a.node->metal_storage, *out.node->metal_storage,
                                                 static_cast<size_t>(a.numel()), 1.0f);
            mark_metal_grad_dirty(a);
        };
    }
    return out;
}

Tensor mean(const Tensor& a) {
    Tensor out = sum(a);
    MetalRuntime::instance().scale_data_buffer(*out.node->metal_storage, 1, 1.0f / static_cast<float>(a.numel()));
    out.node->device_data_dirty = true;
    if (a.requires_grad()) {
        out.node->backward_fn = [a, out]() mutable {
            ensure_metal_grad(out);
            ensure_metal_grad(a);
            MetalRuntime::instance().reduce_grad(*a.node->metal_storage, *out.node->metal_storage,
                                                 static_cast<size_t>(a.numel()),
                                                 1.0f / static_cast<float>(a.numel()));
            mark_metal_grad_dirty(a);
        };
    }
    return out;
}

Tensor max(const Tensor& a) {
    ensure_metal_data(a);
    Tensor out = make_metal_output({}, a.device(), false);
    MetalRuntime::instance().reduce_buffer("max", *out.node->metal_storage, *a.node->metal_storage,
                                           static_cast<size_t>(a.numel()));
    return out;
}

Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_shape) {
    if (product(new_shape) != a.numel()) {
        throw std::runtime_error("Metal reshape numel mismatch");
    }
    ensure_metal_data(a);
    Tensor out = make_metal_output(new_shape, a.device(), a.requires_grad());
    MetalRuntime::instance().unary_buffer("copy", *out.node->metal_storage, *a.node->metal_storage,
                                          0.0f, static_cast<size_t>(a.numel()));
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out]() mutable {
            ensure_metal_grad(out);
            ensure_metal_grad(a);
            MetalRuntime::instance().add_grad(*a.node->metal_storage, *out.node->metal_storage, 0,
                                              static_cast<size_t>(out.numel()));
            mark_metal_grad_dirty(a);
        };
    }
    return out;
}

Tensor transpose(const Tensor& a, int64_t dim0, int64_t dim1) {
    auto shape = a.shape();
    int64_t rank = static_cast<int64_t>(shape.size());
    dim0 = canonical_dim(dim0, rank);
    dim1 = canonical_dim(dim1, rank);
    std::vector<int64_t> out_shape = shape;
    std::swap(out_shape[dim0], out_shape[dim1]);
    auto in_strides = strides_for(shape);
    auto out_strides = strides_for(out_shape);
    // host 端预计算「输出扁平位置 -> 输入扁平位置」映射，交给 gather_kernel 在 GPU 上重排。
    std::vector<uint32_t> index(static_cast<size_t>(a.numel()));
    for (int64_t flat = 0; flat < a.numel(); ++flat) {
        int64_t rem = flat;
        std::vector<int64_t> idx(rank);
        for (int64_t d = 0; d < rank; ++d) {
            idx[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }
        std::swap(idx[dim0], idx[dim1]);
        int64_t in_flat = 0;
        for (int64_t d = 0; d < rank; ++d) {
            in_flat += idx[d] * in_strides[d];
        }
        index[flat] = static_cast<uint32_t>(in_flat);
    }
    ensure_metal_data(a);
    Tensor out = make_metal_output(out_shape, a.device(), a.requires_grad());
    MetalRuntime::instance().gather_buffer(*out.node->metal_storage, *a.node->metal_storage, index);
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, index]() mutable {
            ensure_metal_grad(out);
            ensure_metal_grad(a);
            MetalRuntime::instance().scatter_add_grad(*a.node->metal_storage, *out.node->metal_storage, index);
            mark_metal_grad_dirty(a);
        };
    }
    return out;
}

Tensor log_softmax(const Tensor& a, int64_t dim) {
    int64_t rank = static_cast<int64_t>(a.shape().size());
    int64_t cdim = canonical_dim(dim, rank);
    if (cdim != rank - 1) {
        throw std::runtime_error("Metal log_softmax currently supports last dim only");
    }
    uint32_t width = static_cast<uint32_t>(a.shape().back());
    uint32_t rows = static_cast<uint32_t>(a.numel() / width);
    ensure_metal_data(a);
    Tensor out = make_metal_output(a.shape(), a.device(), a.requires_grad());
    MetalRuntime::instance().log_softmax_buffer(*out.node->metal_storage, *a.node->metal_storage, rows, width);
    return out;
}

} // namespace llm::metal
