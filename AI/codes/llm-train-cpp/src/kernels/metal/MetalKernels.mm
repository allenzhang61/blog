#include "metal_ops.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

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

        id<MTLCommandBuffer> command = [queue_ commandBuffer];
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
        [command commit];
        [command waitUntilCompleted];
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
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
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
        [command commit];
        [command waitUntilCompleted];
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
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_state];
        [encoder setBuffer:x_buffer offset:0 atIndex:0];
        [encoder setBuffer:scale_buffer offset:0 atIndex:1];
        [encoder setBuffer:shift_buffer offset:0 atIndex:2];
        [encoder setBuffer:out_buffer offset:0 atIndex:3];
        [encoder setBuffer:params_buffer offset:0 atIndex:4];
        dispatch1d(encoder, pipeline_state, rows);
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
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
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_state];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:b_buffer offset:0 atIndex:1];
        [encoder setBuffer:out_buffer offset:0 atIndex:2];
        [encoder setBuffer:params_buffer offset:0 atIndex:3];
        dispatch1d(encoder, pipeline_state, static_cast<NSUInteger>(count));
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
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
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_state];
        [encoder setBuffer:x_buffer offset:0 atIndex:0];
        [encoder setBuffer:out_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        dispatch1d(encoder, pipeline_state, rows);
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
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
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_state];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:out_buffer offset:0 atIndex:1];
        [encoder setBuffer:count_buffer offset:0 atIndex:2];
        dispatch1d(encoder, pipeline_state, 1);
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
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
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline_state];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:index_buffer offset:0 atIndex:1];
        [encoder setBuffer:out_buffer offset:0 atIndex:2];
        dispatch1d(encoder, pipeline_state, static_cast<NSUInteger>(index.size()));
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        return read(out_buffer, static_cast<int64_t>(index.size()));
    }

private:
    void dispatch1d(id<MTLComputeCommandEncoder> encoder,
                    id<MTLComputePipelineState> pipeline,
                    NSUInteger count) {
        NSUInteger width = pipeline.threadExecutionWidth;
        MTLSize grid = MTLSizeMake(count, 1, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    }

    id<MTLDevice> device_{nil};
    id<MTLCommandQueue> queue_{nil};
    id<MTLLibrary> library_{nil};
    std::map<std::string, id<MTLComputePipelineState>> pipelines_;
    std::string status_;
};

std::vector<float> to_float(const std::vector<double>& values) {
    return std::vector<float>(values.begin(), values.end());
}

std::vector<double> to_double(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

} // namespace

namespace llm::metal {

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
    auto out_data = MetalRuntime::instance().run1d("add_kernel", to_float(a.data()), to_float(b.data()), nullptr, 0,
                                                   static_cast<uint32_t>(b.numel()), a.numel());
    Tensor out(a.shape(), to_double(out_data), DType::Float32, a.device(), a.requires_grad() || b.requires_grad());
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out, broadcast_batch, broadcast_last]() mutable {
            if (a.requires_grad()) {
                for (int64_t i = 0; i < out.numel(); ++i) {
                    a.grad()[i] += out.grad()[i];
                }
            }
            if (b.requires_grad()) {
                if (broadcast_batch || broadcast_last) {
                    for (int64_t i = 0; i < out.numel(); ++i) {
                        b.grad()[i % b.numel()] += out.grad()[i];
                    }
                } else {
                    for (int64_t i = 0; i < out.numel(); ++i) {
                        b.grad()[i] += out.grad()[i];
                    }
                }
            }
        };
    }
    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Metal mul expects same shape");
    }
    auto out_data = MetalRuntime::instance().run1d("mul_kernel", to_float(a.data()), to_float(b.data()), nullptr, 0,
                                                   0, a.numel());
    Tensor out(a.shape(), to_double(out_data), DType::Float32, a.device(), a.requires_grad() || b.requires_grad());
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out]() mutable {
            if (a.requires_grad()) {
                for (int64_t i = 0; i < out.numel(); ++i) {
                    a.grad()[i] += b.data()[i] * out.grad()[i];
                }
            }
            if (b.requires_grad()) {
                for (int64_t i = 0; i < out.numel(); ++i) {
                    b.grad()[i] += a.data()[i] * out.grad()[i];
                }
            }
        };
    }
    return out;
}

Tensor mul_scalar(const Tensor& a, double scalar) {
    ScalarParams params{static_cast<float>(scalar)};
    auto out_data = MetalRuntime::instance().run1d("mul_scalar_kernel", to_float(a.data()), {}, &params, sizeof(params),
                                                   0, a.numel());
    Tensor out(a.shape(), to_double(out_data), DType::Float32, a.device(), a.requires_grad());
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, scalar]() mutable {
            for (int64_t i = 0; i < out.numel(); ++i) {
                a.grad()[i] += out.grad()[i] * scalar;
            }
        };
    }
    return out;
}

Tensor div(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Metal div expects same shape");
    }
    auto out_data = MetalRuntime::instance().run1d("div_kernel", to_float(a.data()), to_float(b.data()), nullptr, 0,
                                                   0, a.numel());
    Tensor out(a.shape(), to_double(out_data), DType::Float32, a.device(), a.requires_grad() || b.requires_grad());
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out]() mutable {
            if (a.requires_grad()) {
                for (int64_t i = 0; i < out.numel(); ++i) {
                    a.grad()[i] += out.grad()[i] / b.data()[i];
                }
            }
            if (b.requires_grad()) {
                for (int64_t i = 0; i < out.numel(); ++i) {
                    b.grad()[i] -= out.grad()[i] * a.data()[i] / (b.data()[i] * b.data()[i]);
                }
            }
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
    auto out_data = MetalRuntime::instance().matmul(to_float(a.data()), to_float(b.data()), m, k, n);
    Tensor out({static_cast<int64_t>(m), static_cast<int64_t>(n)}, to_double(out_data), DType::Float32,
               a.device(), a.requires_grad() || b.requires_grad());
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out, m, k, n]() mutable {
            if (a.requires_grad()) {
                for (int64_t i = 0; i < static_cast<int64_t>(m); ++i) {
                    for (int64_t p = 0; p < static_cast<int64_t>(k); ++p) {
                        for (int64_t j = 0; j < static_cast<int64_t>(n); ++j) {
                            a.grad()[i * k + p] += out.grad()[i * n + j] * b.data()[p * n + j];
                        }
                    }
                }
            }
            if (b.requires_grad()) {
                for (int64_t p = 0; p < static_cast<int64_t>(k); ++p) {
                    for (int64_t j = 0; j < static_cast<int64_t>(n); ++j) {
                        for (int64_t i = 0; i < static_cast<int64_t>(m); ++i) {
                            b.grad()[p * n + j] += a.data()[i * k + p] * out.grad()[i * n + j];
                        }
                    }
                }
            }
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
    auto out_data = MetalRuntime::instance().batch_matmul(to_float(a.data()), to_float(b.data()), B, H, M, K, N);
    Tensor out({B, H, M, N}, to_double(out_data), DType::Float32, a.device(), a.requires_grad() || b.requires_grad());
    if (out.requires_grad()) {
        out.node->parents = {a, b};
        out.node->backward_fn = [a, b, out, B, H, M, K, N]() mutable {
            if (a.requires_grad()) {
                for (int64_t bb = 0; bb < B; ++bb)
                    for (int64_t hh = 0; hh < H; ++hh)
                        for (int64_t i = 0; i < M; ++i)
                            for (int64_t p = 0; p < K; ++p)
                                for (int64_t j = 0; j < N; ++j) {
                                    int64_t ai = ((bb * H + hh) * M + i) * K + p;
                                    int64_t bi = ((bb * H + hh) * K + p) * N + j;
                                    int64_t oi = ((bb * H + hh) * M + i) * N + j;
                                    a.grad()[ai] += out.grad()[oi] * b.data()[bi];
                                }
            }
            if (b.requires_grad()) {
                for (int64_t bb = 0; bb < B; ++bb)
                    for (int64_t hh = 0; hh < H; ++hh)
                        for (int64_t p = 0; p < K; ++p)
                            for (int64_t j = 0; j < N; ++j)
                                for (int64_t i = 0; i < M; ++i) {
                                    int64_t ai = ((bb * H + hh) * M + i) * K + p;
                                    int64_t bi = ((bb * H + hh) * K + p) * N + j;
                                    int64_t oi = ((bb * H + hh) * M + i) * N + j;
                                    b.grad()[bi] += a.data()[ai] * out.grad()[oi];
                                }
            }
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
    auto out_data = MetalRuntime::instance().softmax(to_float(a.data()), rows, width);
    Tensor out(a.shape(), to_double(out_data), DType::Float32, a.device(), a.requires_grad());
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, rows, width]() mutable {
            for (int64_t r = 0; r < rows; ++r) {
                double dot = 0.0;
                for (int64_t c = 0; c < width; ++c) {
                    dot += out.grad()[r * width + c] * out.data()[r * width + c];
                }
                for (int64_t c = 0; c < width; ++c) {
                    a.grad()[r * width + c] += out.data()[r * width + c] * (out.grad()[r * width + c] - dot);
                }
            }
        };
    }
    return out;
}

Tensor layernorm(const Tensor& x, const Tensor& scale, const Tensor& shift, double eps) {
    uint32_t C = static_cast<uint32_t>(x.shape().back());
    uint32_t rows = static_cast<uint32_t>(x.numel() / C);
    auto out_data = MetalRuntime::instance().layernorm(to_float(x.data()), to_float(scale.data()), to_float(shift.data()),
                                                       rows, C, static_cast<float>(eps));
    Tensor out(x.shape(), to_double(out_data), DType::Float32, x.device(),
               x.requires_grad() || scale.requires_grad() || shift.requires_grad());

    std::vector<double> xhat(x.numel(), 0.0);
    std::vector<double> invs(rows, 0.0);
    for (int64_t r = 0; r < rows; ++r) {
        double mean = 0.0;
        for (int64_t c = 0; c < C; ++c) {
            mean += x.data()[r * C + c];
        }
        mean /= C;
        double var = 0.0;
        for (int64_t c = 0; c < C; ++c) {
            double z = x.data()[r * C + c] - mean;
            var += z * z;
        }
        var /= C;
        invs[r] = 1.0 / std::sqrt(var + eps);
        for (int64_t c = 0; c < C; ++c) {
            xhat[r * C + c] = (x.data()[r * C + c] - mean) * invs[r];
        }
    }

    if (out.requires_grad()) {
        out.node->parents = {x, scale, shift};
        out.node->backward_fn = [x, scale, shift, out, C, rows, xhat, invs]() mutable {
            if (scale.requires_grad()) {
                for (int64_t i = 0; i < out.numel(); ++i) {
                    scale.grad()[i % C] += out.grad()[i] * xhat[i];
                }
            }
            if (shift.requires_grad()) {
                for (int64_t i = 0; i < out.numel(); ++i) {
                    shift.grad()[i % C] += out.grad()[i];
                }
            }
            if (x.requires_grad()) {
                for (int64_t r = 0; r < rows; ++r) {
                    double sum_dxhat = 0.0;
                    double sum_dxhat_xhat = 0.0;
                    for (int64_t c = 0; c < C; ++c) {
                        int64_t idx = r * C + c;
                        double dxhat = out.grad()[idx] * scale.data()[c];
                        sum_dxhat += dxhat;
                        sum_dxhat_xhat += dxhat * xhat[idx];
                    }
                    for (int64_t c = 0; c < C; ++c) {
                        int64_t idx = r * C + c;
                        double dxhat = out.grad()[idx] * scale.data()[c];
                        x.grad()[idx] += (static_cast<double>(C) * dxhat - sum_dxhat - xhat[idx] * sum_dxhat_xhat) *
                                         invs[r] / static_cast<double>(C);
                    }
                }
            }
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
    auto out_data = MetalRuntime::instance().embedding(to_float(ids.data()), to_float(weight.data()), count, dim);
    Tensor out(out_shape, to_double(out_data), DType::Float32, weight.device(), weight.requires_grad());
    if (weight.requires_grad()) {
        out.node->parents = {weight};
        out.node->backward_fn = [ids, weight, out, dim]() mutable {
            for (int64_t i = 0; i < ids.numel(); ++i) {
                int64_t id = static_cast<int64_t>(ids.data()[i]);
                for (int64_t d = 0; d < dim; ++d) {
                    weight.grad()[id * dim + d] += out.grad()[i * dim + d];
                }
            }
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

    auto row_losses = MetalRuntime::instance().cross_entropy_row_losses(to_float(logits.data()), to_float(targets.data()),
                                                                        static_cast<uint32_t>(B * T), static_cast<uint32_t>(V));
    double loss = 0.0;
    for (auto value : row_losses) {
        loss += value;
    }
    Tensor out({}, DType::Float32, logits.device(), logits.requires_grad());
    out.data()[0] = loss / static_cast<double>(B * T);

    std::vector<double> probs(logits.numel(), 0.0);
    for (int64_t row = 0; row < B * T; ++row) {
        double mx = -1e100;
        for (int64_t v = 0; v < V; ++v) {
            mx = std::max(mx, logits.data()[row * V + v]);
        }
        double denom = 0.0;
        for (int64_t v = 0; v < V; ++v) {
            probs[row * V + v] = std::exp(logits.data()[row * V + v] - mx);
            denom += probs[row * V + v];
        }
        for (int64_t v = 0; v < V; ++v) {
            probs[row * V + v] /= denom;
        }
    }
    if (logits.requires_grad()) {
        out.node->parents = {logits};
        out.node->backward_fn = [logits, targets, out, probs, B, T, V]() mutable {
            for (int64_t row = 0; row < B * T; ++row) {
                int64_t target = static_cast<int64_t>(targets.data()[row]);
                for (int64_t v = 0; v < V; ++v) {
                    double g = probs[row * V + v];
                    if (v == target) {
                        g -= 1.0;
                    }
                    logits.grad()[row * V + v] += out.grad()[0] * g / static_cast<double>(B * T);
                }
            }
        };
    }
    return out;
}

Tensor gelu(const Tensor& x) {
    auto out_data = MetalRuntime::instance().run1d("gelu_kernel", to_float(x.data()), {}, nullptr, 0, 0, x.numel());
    Tensor out(x.shape(), to_double(out_data), DType::Float32, x.device(), x.requires_grad());
    if (x.requires_grad()) {
        out.node->parents = {x};
        out.node->backward_fn = [x, out]() mutable {
            constexpr double k = 0.7978845608028654;
            for (int64_t i = 0; i < x.numel(); ++i) {
                double v = x.data()[i];
                double u = k * (v + 0.044715 * v * v * v);
                double th = std::tanh(u);
                double du = k * (1.0 + 3.0 * 0.044715 * v * v);
                double g = 0.5 * (1.0 + th) + 0.5 * v * (1.0 - th * th) * du;
                x.grad()[i] += out.grad()[i] * g;
            }
        };
    }
    return out;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    auto neg_data = MetalRuntime::instance().run1d("neg_kernel", to_float(b.data()), {}, nullptr, 0, 0, b.numel());
    Tensor neg_b(b.shape(), to_double(neg_data), DType::Float32, b.device(), b.requires_grad());
    if (b.requires_grad()) {
        neg_b.node->parents = {b};
        neg_b.node->backward_fn = [b, neg_b]() mutable {
            for (int64_t i = 0; i < neg_b.numel(); ++i) {
                b.grad()[i] -= neg_b.grad()[i];
            }
        };
    }
    return add(a, neg_b);
}

Tensor pow(const Tensor& a, double exponent) {
    ScalarParams params{static_cast<float>(exponent)};
    auto out_data = MetalRuntime::instance().run1d("pow_kernel", to_float(a.data()), {}, &params, sizeof(params),
                                                   0, a.numel());
    Tensor out(a.shape(), to_double(out_data), DType::Float32, a.device(), a.requires_grad());
    if (out.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, exponent]() mutable {
            for (int64_t i = 0; i < out.numel(); ++i) {
                a.grad()[i] += exponent * std::pow(a.data()[i], exponent - 1.0) * out.grad()[i];
            }
        };
    }
    return out;
}

Tensor sum(const Tensor& a) {
    float value = MetalRuntime::instance().reduce("sum_kernel", to_float(a.data()));
    Tensor out({}, DType::Float32, a.device(), a.requires_grad());
    out.data()[0] = value;
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out]() mutable {
            for (int64_t i = 0; i < a.numel(); ++i) {
                a.grad()[i] += out.grad()[0];
            }
        };
    }
    return out;
}

Tensor mean(const Tensor& a) {
    Tensor out = sum(a);
    out.data()[0] /= static_cast<double>(a.numel());
    if (a.requires_grad()) {
        out.node->backward_fn = [a, out]() mutable {
            for (int64_t i = 0; i < a.numel(); ++i) {
                a.grad()[i] += out.grad()[0] / static_cast<double>(a.numel());
            }
        };
    }
    return out;
}

Tensor max(const Tensor& a) {
    float value = MetalRuntime::instance().reduce("max_kernel", to_float(a.data()));
    Tensor out({}, DType::Float32, a.device(), false);
    out.data()[0] = value;
    return out;
}

Tensor reshape(const Tensor& a, const std::vector<int64_t>& new_shape) {
    if (product(new_shape) != a.numel()) {
        throw std::runtime_error("Metal reshape numel mismatch");
    }
    auto out_data = MetalRuntime::instance().run1d("copy_kernel", to_float(a.data()), {}, nullptr, 0, 0, a.numel());
    Tensor out(new_shape, to_double(out_data), DType::Float32, a.device(), a.requires_grad());
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out]() mutable {
            for (int64_t i = 0; i < out.numel(); ++i) {
                a.grad()[i] += out.grad()[i];
            }
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
    auto out_data = MetalRuntime::instance().gather(to_float(a.data()), index);
    Tensor out(out_shape, to_double(out_data), DType::Float32, a.device(), a.requires_grad());
    if (a.requires_grad()) {
        out.node->parents = {a};
        out.node->backward_fn = [a, out, index]() mutable {
            for (int64_t flat = 0; flat < out.numel(); ++flat) {
                a.grad()[index[flat]] += out.grad()[flat];
            }
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
    auto probs = MetalRuntime::instance().softmax(to_float(a.data()), rows, width);
    auto out_data = MetalRuntime::instance().run1d("log_kernel", probs, {}, nullptr, 0, 0, a.numel());
    Tensor out(a.shape(), to_double(out_data), DType::Float32, a.device(), a.requires_grad());
    return out;
}

} // namespace llm::metal
