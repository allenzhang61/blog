#include "backend/cuda/ops/kernel.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cmath>
#include <vector>

namespace {

void skip_if_no_cuda_device() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || device_count == 0) {
        GTEST_SKIP() << "CUDA device is not available";
    }
    ASSERT_EQ(cudaSuccess, status) << cudaGetErrorString(status);
}

template <typename T>
class DeviceArray {
public:
    explicit DeviceArray(size_t count) : count_(count) {
        status_ = cudaMalloc(&ptr_, count_ * sizeof(T));
    }

    ~DeviceArray() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceArray(const DeviceArray &) = delete;
    DeviceArray &operator=(const DeviceArray &) = delete;

    cudaError_t status() const { return status_; }
    T *get() const { return ptr_; }

    cudaError_t copy_from(const std::vector<T> &host) {
        return cudaMemcpy(ptr_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice);
    }

    cudaError_t copy_to(std::vector<T> &host) const {
        return cudaMemcpy(host.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost);
    }

private:
    T *ptr_ = nullptr;
    size_t count_ = 0;
    cudaError_t status_ = cudaSuccess;
};

} // namespace

TEST(LaunchAddTest, AddsTwoFloatArrays) {
    skip_if_no_cuda_device();

    const std::vector<float> a{1.0f, -2.0f, 3.5f, 0.0f, 10.0f};
    const std::vector<float> b{4.0f, 5.0f, -1.5f, 7.0f, -3.0f};
    std::vector<float> out(a.size(), 0.0f);

    DeviceArray<float> d_a(a.size());
    DeviceArray<float> d_b(b.size());
    DeviceArray<float> d_out(out.size());
    ASSERT_EQ(cudaSuccess, d_a.status());
    ASSERT_EQ(cudaSuccess, d_b.status());
    ASSERT_EQ(cudaSuccess, d_out.status());
    ASSERT_EQ(cudaSuccess, d_a.copy_from(a));
    ASSERT_EQ(cudaSuccess, d_b.copy_from(b));

    launch_add(d_a.get(), d_b.get(), d_out.get(), static_cast<int>(a.size()), nullptr);

    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_EQ(cudaSuccess, d_out.copy_to(out));

    ASSERT_EQ(a.size(), out.size());
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_FLOAT_EQ(a[i] + b[i], out[i]);
    }
}

TEST(LaunchAddTest, SupportsInPlaceOutput) {
    skip_if_no_cuda_device();

    const std::vector<float> original{2.0f, 4.0f, -8.0f, 1.5f};
    const std::vector<float> residual{0.5f, -1.0f, 3.0f, 2.5f};
    std::vector<float> out(original.size(), 0.0f);

    DeviceArray<float> d_hidden(original.size());
    DeviceArray<float> d_residual(residual.size());
    ASSERT_EQ(cudaSuccess, d_hidden.status());
    ASSERT_EQ(cudaSuccess, d_residual.status());
    ASSERT_EQ(cudaSuccess, d_hidden.copy_from(original));
    ASSERT_EQ(cudaSuccess, d_residual.copy_from(residual));

    launch_add(d_hidden.get(), d_residual.get(), d_hidden.get(),
               static_cast<int>(original.size()), nullptr);

    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_EQ(cudaSuccess, d_hidden.copy_to(out));

    ASSERT_EQ(original.size(), out.size());
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_FLOAT_EQ(original[i] + residual[i], out[i]);
    }
}

TEST(LaunchSiluMulTest, AppliesSiluGateAndMultiply) {
    skip_if_no_cuda_device();

    const std::vector<float> gate{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    const std::vector<float> up{3.0f, -4.0f, 5.0f, 6.0f, -7.0f};
    std::vector<float> out(gate.size(), 0.0f);

    DeviceArray<float> d_gate(gate.size());
    DeviceArray<float> d_up(up.size());
    DeviceArray<float> d_out(out.size());
    ASSERT_EQ(cudaSuccess, d_gate.status());
    ASSERT_EQ(cudaSuccess, d_up.status());
    ASSERT_EQ(cudaSuccess, d_out.status());
    ASSERT_EQ(cudaSuccess, d_gate.copy_from(gate));
    ASSERT_EQ(cudaSuccess, d_up.copy_from(up));

    launch_silu_mul(d_gate.get(), d_up.get(), d_out.get(), static_cast<int>(gate.size()),
                    nullptr);

    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_EQ(cudaSuccess, d_out.copy_to(out));

    ASSERT_EQ(gate.size(), out.size());
    for (size_t i = 0; i < out.size(); ++i) {
        const float silu = gate[i] / (1.0f + std::exp(-gate[i]));
        EXPECT_NEAR(silu * up[i], out[i], 1e-5f);
    }
}
