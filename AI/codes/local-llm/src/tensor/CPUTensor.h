//
// Created by zhangyoulun on 21/8/2026.
//

#ifndef LOCAL_LLM_CPUTENSOR_H
#define LOCAL_LLM_CPUTENSOR_H

#include "tensor/TensorCommon.h"

class CudaScratch;
class GPUTensor;

class CPUTensor : public TensorShape {
public:
    CPUTensor() = default;
    CPUTensor(const void *host_ptr, std::vector<int64_t> shape, DType dt);

    const void *data() const { return data_; }
    GPUTensor to_gpu(CudaScratch &scratch, const std::string &key, const std::string &what) const;

private:
    void *data_ = nullptr;
};

#endif // LOCAL_LLM_CPUTENSOR_H
