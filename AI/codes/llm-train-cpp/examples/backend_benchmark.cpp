#include "llm/llm.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    try {
        using namespace llm;

        std::string backend = argc > 1 ? argv[1] : "cpu";
        int64_t n = argc > 2 ? std::stoll(argv[2]) : 1024;
        int64_t iterations = argc > 3 ? std::stoll(argv[3]) : 100;
        if (n <= 0 || iterations <= 0) {
            throw std::runtime_error("usage: backend_benchmark <cpu|metal|cuda> [matrix_size] [iterations]");
        }

        Device device = Device::parse(backend);
        BackendRegistry::get(device);

        std::vector<double> a(static_cast<size_t>(n * n));
        std::vector<double> b(static_cast<size_t>(n * n));
        for (int64_t i = 0; i < n * n; ++i) {
            a[static_cast<size_t>(i)] = static_cast<double>((i % 17) - 8) / 17.0;
            b[static_cast<size_t>(i)] = static_cast<double>((i % 23) - 11) / 23.0;
        }

        Tensor ta = Tensor::from_vector(a, {n, n}, device);
        Tensor tb = Tensor::from_vector(b, {n, n}, device);
        Tensor out;

        // One warm-up call keeps one-time backend setup out of the measured loop.
        out = ops::matmul(ta, tb);
        (void)out.data();

        auto start = std::chrono::steady_clock::now();
        for (int64_t i = 0; i < iterations; ++i) {
            out = ops::matmul(ta, tb);
        }
        (void)out.data();
        auto end = std::chrono::steady_clock::now();

        double seconds = std::chrono::duration<double>(end - start).count();
        double per_iter_ms = seconds * 1000.0 / static_cast<double>(iterations);
        double gflops = (2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n) *
                         static_cast<double>(iterations)) /
                        (seconds * 1e9);
        double checksum = 0.0;
        for (int64_t i = 0; i < std::min<int64_t>(out.numel(), 16); ++i) {
            checksum += out.data()[i];
        }

        std::cout << std::fixed << std::setprecision(6)
                  << "backend=" << device.str()
                  << " n=" << n
                  << " iterations=" << iterations
                  << " seconds=" << seconds
                  << " per_iter_ms=" << per_iter_ms
                  << " gflops=" << gflops
                  << " checksum=" << checksum << "\n";
        return 0;
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n";
        return 1;
    }
}
