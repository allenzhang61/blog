//
// Created by zhangyoulun on 21/8/2026.
//

#ifndef LOCAL_LLM_GPUTENSOR_H
#define LOCAL_LLM_GPUTENSOR_H

#include "tensor/TensorCommon.h"

class CudaScratch;

class GPUTensor : public TensorShape {
public:
    void *data = nullptr;

    static GPUTensor gpu_scratch(CudaScratch &scratch, const std::string &key,
                                 std::vector<int64_t> shape, DType dt = DType::F32);
    static GPUTensor gpu_view(void *device_ptr, std::vector<int64_t> shape,
                              DType dt = DType::F32);

    float *gpu_f32() const;
    int *gpu_i32() const;
    void *gpu_data() const { return data; }
    void to_host(void *host_ptr, const std::string &what) const;
};

#endif // LOCAL_LLM_GPUTENSOR_H
