//
// Created by zhangyoulun on 21/8/2026.
//

#ifndef LOCAL_LLM_CPUTENSOR_H
#define LOCAL_LLM_CPUTENSOR_H

#include "tensor/GPUTensor.h"

class CudaScratch;

class CPUTensor : public TensorShape {
public:
    void *data = nullptr;

    static CPUTensor host_view(const void *host_ptr, std::vector<int64_t> shape, DType dt);

    const int *host_i32() const;
    GPUTensor to_gpu(CudaScratch &scratch, const std::string &key, const std::string &what) const;
    void to_gpu(void *device_ptr, const std::string &what) const;
};

#endif // LOCAL_LLM_CPUTENSOR_H
