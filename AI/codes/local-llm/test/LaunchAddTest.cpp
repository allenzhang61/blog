#include "backend/cuda/ops/kernel.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
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

void append_u16_le(std::vector<uint8_t> &bytes, uint16_t v) {
    bytes.push_back(static_cast<uint8_t>(v & 0xff));
    bytes.push_back(static_cast<uint8_t>(v >> 8));
}

void write_u16_le(std::vector<uint8_t> &bytes, size_t offset, uint16_t v) {
    bytes[offset] = static_cast<uint8_t>(v & 0xff);
    bytes[offset + 1] = static_cast<uint8_t>(v >> 8);
}

uint16_t read_u16_le(const std::vector<uint8_t> &bytes, size_t offset) {
    return static_cast<uint16_t>(bytes[offset]) |
           static_cast<uint16_t>(static_cast<uint16_t>(bytes[offset + 1]) << 8);
}

float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;

    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ff;
            out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        out = sign | 0x7f800000 | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float value;
    std::memcpy(&value, &out, sizeof(value));
    return value;
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

    launch_add(d_a.get(), d_b.get(), d_out.get(), static_cast<int>(a.size()));

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
               static_cast<int>(original.size()));

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

    launch_silu_mul(d_gate.get(), d_up.get(), d_out.get(), static_cast<int>(gate.size()));

    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_EQ(cudaSuccess, d_out.copy_to(out));

    ASSERT_EQ(gate.size(), out.size());
    for (size_t i = 0; i < out.size(); ++i) {
        const float silu = gate[i] / (1.0f + std::exp(-gate[i]));
        EXPECT_NEAR(silu * up[i], out[i], 1e-5f);
    }
}

TEST(LaunchQuantMatmulTest, ComputesQ80MultipleRows) {
    skip_if_no_cuda_device();

    constexpr int in_dim = 32;
    constexpr int out_dim = 2;
    constexpr int m = 3;
    constexpr size_t row_bytes = 34;
    std::vector<uint8_t> weight(row_bytes * out_dim, 0);
    for (int row = 0; row < out_dim; ++row) {
        std::vector<uint8_t> row_bytes_host;
        append_u16_le(row_bytes_host, 0x3c00); // f16(1.0)
        for (int k = 0; k < in_dim; ++k) {
            const int8_t q = static_cast<int8_t>(row == 0 ? 1 : (k % 2 == 0 ? 2 : -1));
            row_bytes_host.push_back(static_cast<uint8_t>(q));
        }
        std::copy(row_bytes_host.begin(), row_bytes_host.end(), weight.begin() + row * row_bytes);
    }

    const std::vector<float> x{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
        -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32,
        1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
        1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1};
    std::vector<float> out(static_cast<size_t>(m) * out_dim, 0.0f);

    DeviceArray<uint8_t> d_weight(weight.size());
    DeviceArray<float> d_x(x.size());
    DeviceArray<float> d_out(out.size());
    ASSERT_EQ(cudaSuccess, d_weight.status());
    ASSERT_EQ(cudaSuccess, d_x.status());
    ASSERT_EQ(cudaSuccess, d_out.status());
    ASSERT_EQ(cudaSuccess, d_weight.copy_from(weight));
    ASSERT_EQ(cudaSuccess, d_x.copy_from(x));

    launch_quant_matmul(DType::Q8_0, d_weight.get(), row_bytes, d_x.get(), d_out.get(),
                        out_dim, in_dim, m, false);

    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_EQ(cudaSuccess, d_out.copy_to(out));

    for (int row = 0; row < m; ++row) {
        float expected0 = 0.0f;
        float expected1 = 0.0f;
        for (int k = 0; k < in_dim; ++k) {
            const float xv = x[static_cast<size_t>(row) * in_dim + k];
            expected0 += xv;
            expected1 += xv * (k % 2 == 0 ? 2.0f : -1.0f);
        }
        EXPECT_NEAR(expected0, out[static_cast<size_t>(row) * out_dim + 0], 1e-5f);
        EXPECT_NEAR(expected1, out[static_cast<size_t>(row) * out_dim + 1], 1e-5f);
    }
}

TEST(LaunchQuantMatmulQ81Test, ComputesQ80MultipleRows) {
    skip_if_no_cuda_device();

    constexpr int in_dim = 32;
    constexpr int out_dim = 2;
    constexpr int m = 2;
    constexpr size_t row_bytes = 34;
    std::vector<uint8_t> weight(row_bytes * out_dim, 0);
    for (int row = 0; row < out_dim; ++row) {
        std::vector<uint8_t> row_bytes_host;
        append_u16_le(row_bytes_host, 0x3c00); // f16(1.0)
        for (int k = 0; k < in_dim; ++k) {
            const int8_t q = static_cast<int8_t>(row == 0 ? 1 : (k % 2 == 0 ? 2 : -1));
            row_bytes_host.push_back(static_cast<uint8_t>(q));
        }
        std::copy(row_bytes_host.begin(), row_bytes_host.end(), weight.begin() + row * row_bytes);
    }

    std::vector<float> x(static_cast<size_t>(m) * in_dim, 0.0f);
    x[0] = 127.0f;
    x[in_dim] = -127.0f;
    for (int k = 1; k < in_dim; ++k) {
        x[k] = static_cast<float>(k);
        x[static_cast<size_t>(in_dim) + k] = static_cast<float>(-k);
    }
    std::vector<float> out(static_cast<size_t>(m) * out_dim, 0.0f);

    DeviceArray<uint8_t> d_weight(weight.size());
    DeviceArray<float> d_x(x.size());
    DeviceArray<uint8_t> d_x_q8_1(static_cast<size_t>(m) * q8_1_row_bytes(in_dim));
    DeviceArray<float> d_out(out.size());
    ASSERT_EQ(cudaSuccess, d_weight.status());
    ASSERT_EQ(cudaSuccess, d_x.status());
    ASSERT_EQ(cudaSuccess, d_x_q8_1.status());
    ASSERT_EQ(cudaSuccess, d_out.status());
    ASSERT_EQ(cudaSuccess, d_weight.copy_from(weight));
    ASSERT_EQ(cudaSuccess, d_x.copy_from(x));

    launch_quantize_q8_1(d_x.get(), d_x_q8_1.get(), in_dim, m);
    launch_quant_matmul_q8_1(DType::Q8_0, d_weight.get(), row_bytes, d_x_q8_1.get(), d_out.get(),
                             out_dim, in_dim, m);

    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_EQ(cudaSuccess, d_out.copy_to(out));

    for (int row = 0; row < m; ++row) {
        float expected0 = 0.0f;
        float expected1 = 0.0f;
        for (int k = 0; k < in_dim; ++k) {
            const float xv = x[static_cast<size_t>(row) * in_dim + k];
            expected0 += xv;
            expected1 += xv * (k % 2 == 0 ? 2.0f : -1.0f);
        }
        EXPECT_NEAR(expected0, out[static_cast<size_t>(row) * out_dim + 0], 1e-5f);
        EXPECT_NEAR(expected1, out[static_cast<size_t>(row) * out_dim + 1], 1e-5f);
    }
}

TEST(LaunchQuantizeQ81Test, SupportsRawAndQuantizedBlockSums) {
    skip_if_no_cuda_device();

    constexpr int in_dim = 32;
    constexpr int m = 1;
    std::vector<float> x(in_dim, 0.01f);
    x[0] = 1.0f;

    DeviceArray<float> d_x(x.size());
    DeviceArray<uint8_t> d_x_q8_1(q8_1_row_bytes(in_dim));
    ASSERT_EQ(cudaSuccess, d_x.status());
    ASSERT_EQ(cudaSuccess, d_x_q8_1.status());
    ASSERT_EQ(cudaSuccess, d_x.copy_from(x));

    std::vector<uint8_t> encoded(q8_1_row_bytes(in_dim), 0);

    launch_quantize_q8_1(d_x.get(), d_x_q8_1.get(), in_dim, m, false);
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_EQ(cudaSuccess, d_x_q8_1.copy_to(encoded));
    const float quantized_sum = half_to_float(read_u16_le(encoded, 2));

    launch_quantize_q8_1(d_x.get(), d_x_q8_1.get(), in_dim, m, true);
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_EQ(cudaSuccess, d_x_q8_1.copy_to(encoded));
    const float raw_sum = half_to_float(read_u16_le(encoded, 2));

    EXPECT_NEAR(158.0f / 127.0f, quantized_sum, 1e-3f);
    EXPECT_NEAR(1.31f, raw_sum, 1e-3f);
}

TEST(LaunchQuantMatmulQ81Test, ComputesQ4KScaleAndMinCompensation) {
    skip_if_no_cuda_device();

    constexpr int in_dim = 256;
    constexpr int out_dim = 1;
    constexpr int m = 1;
    constexpr size_t row_bytes = 144;

    std::vector<uint8_t> weight(row_bytes, 0);
    write_u16_le(weight, 0, 0x3c00); // d = f16(1.0)
    write_u16_le(weight, 2, 0x3800); // dmin = f16(0.5)
    weight[4 + 0] = 3;             // sub-block 0 scale
    weight[4 + 4] = 2;             // sub-block 0 min
    for (int k = 0; k < 32; ++k) {
        weight[16 + k] = 0x05;     // sub-block 0 q=5, sub-block 1 q=0
    }

    std::vector<float> x(in_dim, 0.0f);
    x[0] = 127.0f;
    for (int k = 1; k < 32; ++k) {
        x[k] = 1.0f;
    }
    std::vector<float> out(out_dim, 0.0f);

    DeviceArray<uint8_t> d_weight(weight.size());
    DeviceArray<float> d_x(x.size());
    DeviceArray<uint8_t> d_x_q8_1(q8_1_row_bytes(in_dim));
    DeviceArray<float> d_out(out.size());
    ASSERT_EQ(cudaSuccess, d_weight.status());
    ASSERT_EQ(cudaSuccess, d_x.status());
    ASSERT_EQ(cudaSuccess, d_x_q8_1.status());
    ASSERT_EQ(cudaSuccess, d_out.status());
    ASSERT_EQ(cudaSuccess, d_weight.copy_from(weight));
    ASSERT_EQ(cudaSuccess, d_x.copy_from(x));

    launch_quantize_q8_1(d_x.get(), d_x_q8_1.get(), in_dim, m);
    launch_quant_matmul_q8_1(DType::Q4_K, d_weight.get(), row_bytes, d_x_q8_1.get(), d_out.get(),
                             out_dim, in_dim, m);

    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_EQ(cudaSuccess, d_out.copy_to(out));

    // For sub-block 0, dequantized Q4_K weight is 1.0*3*5 - 0.5*2 = 14.
    EXPECT_NEAR(14.0f * 158.0f, out[0], 1e-3f);
}

TEST(LaunchQuantMatmulQ81Test, ComputesQ6KScaleGroups) {
    skip_if_no_cuda_device();

    constexpr int in_dim = 256;
    constexpr int out_dim = 1;
    constexpr int m = 1;
    constexpr size_t row_bytes = 210;

    std::vector<uint8_t> weight(row_bytes, 0);
    for (int k = 0; k < 32; ++k) {
        weight[k] = 0x03;          // low 4 bits of q + 32, for group 0
        weight[128 + k] = 0x02;    // high 2 bits of q + 32, for group 0
    }
    weight[192 + 0] = 2;           // scale for lanes 0..15
    weight[192 + 1] = 2;           // scale for lanes 16..31
    write_u16_le(weight, 208, 0x3c00); // d = f16(1.0)

    std::vector<float> x(in_dim, 0.0f);
    x[0] = 127.0f;
    for (int k = 1; k < 32; ++k) {
        x[k] = 1.0f;
    }
    std::vector<float> out(out_dim, 0.0f);

    DeviceArray<uint8_t> d_weight(weight.size());
    DeviceArray<float> d_x(x.size());
    DeviceArray<uint8_t> d_x_q8_1(q8_1_row_bytes(in_dim));
    DeviceArray<float> d_out(out.size());
    ASSERT_EQ(cudaSuccess, d_weight.status());
    ASSERT_EQ(cudaSuccess, d_x.status());
    ASSERT_EQ(cudaSuccess, d_x_q8_1.status());
    ASSERT_EQ(cudaSuccess, d_out.status());
    ASSERT_EQ(cudaSuccess, d_weight.copy_from(weight));
    ASSERT_EQ(cudaSuccess, d_x.copy_from(x));

    launch_quantize_q8_1(d_x.get(), d_x_q8_1.get(), in_dim, m);
    launch_quant_matmul_q8_1(DType::Q6_K, d_weight.get(), row_bytes, d_x_q8_1.get(), d_out.get(),
                             out_dim, in_dim, m);

    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_EQ(cudaSuccess, d_out.copy_to(out));

    // Stored q is 35, so dequantized Q6_K value is 1.0*2*(35 - 32) = 6.
    EXPECT_NEAR(6.0f * 158.0f, out[0], 1e-3f);
}

TEST(LaunchQuantMatmulQ81MmqTest, ComputesQ4KScaleAndMinCompensation) {
    skip_if_no_cuda_device();

    constexpr int in_dim = 256;
    constexpr int out_dim = 1;
    constexpr int m = 1;
    constexpr size_t row_bytes = 144;

    std::vector<uint8_t> weight(row_bytes, 0);
    write_u16_le(weight, 0, 0x3c00);
    write_u16_le(weight, 2, 0x3800);
    weight[4 + 0] = 3;
    weight[4 + 4] = 2;
    for (int k = 0; k < 32; ++k) {
        weight[16 + k] = 0x05;
    }

    std::vector<float> x(in_dim, 0.0f);
    x[0] = 127.0f;
    for (int k = 1; k < 32; ++k) {
        x[k] = 1.0f;
    }
    std::vector<float> out(out_dim, 0.0f);

    DeviceArray<uint8_t> d_weight(weight.size());
    DeviceArray<float> d_x(x.size());
    DeviceArray<uint8_t> d_x_q8_1(q8_1_mmq_row_bytes(in_dim));
    DeviceArray<float> d_out(out.size());
    ASSERT_EQ(cudaSuccess, d_weight.status());
    ASSERT_EQ(cudaSuccess, d_x.status());
    ASSERT_EQ(cudaSuccess, d_x_q8_1.status());
    ASSERT_EQ(cudaSuccess, d_out.status());
    ASSERT_EQ(cudaSuccess, d_weight.copy_from(weight));
    ASSERT_EQ(cudaSuccess, d_x.copy_from(x));

    launch_quantize_q8_1_mmq(d_x.get(), d_x_q8_1.get(), in_dim, m);
    launch_quant_matmul_q8_1_mmq(DType::Q4_K, d_weight.get(), row_bytes, d_x_q8_1.get(), d_out.get(),
                                 out_dim, in_dim, m);

    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_EQ(cudaSuccess, d_out.copy_to(out));

    EXPECT_NEAR(14.0f * 158.0f, out[0], 1e-3f);
}

TEST(LaunchQuantMatmulQ81MmqTest, ComputesQ6KScaleGroups) {
    skip_if_no_cuda_device();

    constexpr int in_dim = 256;
    constexpr int out_dim = 1;
    constexpr int m = 1;
    constexpr size_t row_bytes = 210;

    std::vector<uint8_t> weight(row_bytes, 0);
    for (int k = 0; k < 32; ++k) {
        weight[k] = 0x03;
        weight[128 + k] = 0x02;
    }
    weight[192 + 0] = 2;
    weight[192 + 1] = 2;
    write_u16_le(weight, 208, 0x3c00);

    std::vector<float> x(in_dim, 0.0f);
    x[0] = 127.0f;
    for (int k = 1; k < 32; ++k) {
        x[k] = 1.0f;
    }
    std::vector<float> out(out_dim, 0.0f);

    DeviceArray<uint8_t> d_weight(weight.size());
    DeviceArray<float> d_x(x.size());
    DeviceArray<uint8_t> d_x_q8_1(q8_1_mmq_row_bytes(in_dim));
    DeviceArray<float> d_out(out.size());
    ASSERT_EQ(cudaSuccess, d_weight.status());
    ASSERT_EQ(cudaSuccess, d_x.status());
    ASSERT_EQ(cudaSuccess, d_x_q8_1.status());
    ASSERT_EQ(cudaSuccess, d_out.status());
    ASSERT_EQ(cudaSuccess, d_weight.copy_from(weight));
    ASSERT_EQ(cudaSuccess, d_x.copy_from(x));

    launch_quantize_q8_1_mmq(d_x.get(), d_x_q8_1.get(), in_dim, m);
    launch_quant_matmul_q8_1_mmq(DType::Q6_K, d_weight.get(), row_bytes, d_x_q8_1.get(), d_out.get(),
                                 out_dim, in_dim, m);

    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_EQ(cudaSuccess, d_out.copy_to(out));

    EXPECT_NEAR(6.0f * 158.0f, out[0], 1e-3f);
}
