#include "llm/backend/Backend.hpp"
#include "llm/cuda_ops.hpp"

// 这个文件是「已启用 CUDA 编译」时使用的实现（对应 Metal 的 MetalBackend.mm）。
// cuda_backend_* 直接委派给 src/kernels/cuda/cuda_kernels.cu 里的 llm::cuda:: 运行时。

namespace llm {

bool cuda_backend_available() {
    return cuda::available();
}

std::string cuda_backend_status() {
    return cuda::status();
}

} // namespace llm
