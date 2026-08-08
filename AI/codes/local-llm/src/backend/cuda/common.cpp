//
// Created by zhangyoulun on 8/8/2026.
//

#include "common.h"

#include <stdexcept>

void check_cuda(cudaError_t status, const std::string &what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(what + "：" + cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const std::string &what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(what + "，cublasStatus=" + std::to_string(static_cast<int>(status)));
    }
}
