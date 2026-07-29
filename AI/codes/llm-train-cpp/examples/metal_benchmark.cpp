#include "llm/llm.hpp"

#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    try {
        using namespace llm;
        int64_t n = argc > 1 ? std::stoll(argv[1]) : 1024;
        int64_t iterations = argc > 2 ? std::stoll(argv[2]) : 100;
        if (n <= 0 || iterations <= 0) throw std::runtime_error("usage: metal_benchmark [matrix_size] [iterations]");

        Device device = select_device_from_arg_or_env("metal");

        std::vector<double> a(static_cast<size_t>(n * n));
        std::vector<double> b(static_cast<size_t>(n * n));
        for (int64_t i = 0; i < n * n; ++i) {
            a[static_cast<size_t>(i)] = static_cast<double>((i % 17) - 8) / 17.0;
            b[static_cast<size_t>(i)] = static_cast<double>((i % 23) - 11) / 23.0;
        }

        Tensor ta = Tensor::from_vector(a, {n, n}, device);
        Tensor tb = Tensor::from_vector(b, {n, n}, device);
        Tensor out;

        auto start = std::chrono::steady_clock::now();
        for (int64_t i = 0; i < iterations; ++i) {
            out = ops::matmul(ta, tb);
        }
        auto end = std::chrono::steady_clock::now();

        double seconds = std::chrono::duration<double>(end - start).count();
        double checksum = 0.0;
        for (int64_t i = 0; i < std::min<int64_t>(out.numel(), 16); ++i) checksum += out.data()[i];

        std::cout << "metal benchmark n=" << n
                  << " iterations=" << iterations
                  << " seconds=" << seconds
                  << " checksum=" << checksum << "\n";
        return 0;
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n";
        return 1;
    }
}
