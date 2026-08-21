//
// Created by zhangyoulun on 21/8/2026.
//

#ifndef LOCAL_LLM_DISKTENSOR_H
#define LOCAL_LLM_DISKTENSOR_H

#include "tensor/GPUTensor.h"

#include <memory>

class CudaWeightPool;

class DiskTensor : public TensorShape {
public:
    const uint8_t *data = nullptr;
    CudaWeightPool *pool = nullptr;
    mutable std::shared_ptr<void> weight_view_lease;

    static DiskTensor disk_view(const uint8_t *disk_ptr, std::vector<int64_t> shape,
                                DType dt, size_t bytes);

    void to_gpu() const;
    GPUTensor try_dequant() const;
    const void *weight_gpu_data() const;
};

#endif // LOCAL_LLM_DISKTENSOR_H
