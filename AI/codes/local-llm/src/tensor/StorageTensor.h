//
// Created by zhangyoulun on 21/8/2026.
//

#ifndef LOCAL_LLM_STORAGETENSOR_H
#define LOCAL_LLM_STORAGETENSOR_H

#include "tensor/TensorCommon.h"

class GPUTensor;

class StorageTensor : public TensorShape {
public:
    StorageTensor() = default;
    StorageTensor(const uint8_t *disk_ptr, std::vector<int64_t> shape,
               DType dt, size_t bytes);

    const uint8_t *data() const { return data_; }
    StorageTensor slice(size_t byte_offset, std::vector<int64_t> slice_shape,
                     size_t slice_bytes, std::string slice_name) const;

    GPUTensor to_gpu(bool dequant) const;

private:
    const uint8_t *data_ = nullptr;
};

#endif // LOCAL_LLM_STORAGETENSOR_H
