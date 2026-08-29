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
    template <typename T>
    const std::remove_cv_t<T> *data() const {
        validate_tensor_cpp_type<T>(dtype, name);
        return reinterpret_cast<const std::remove_cv_t<T> *>(data_);
    }

    StorageTensor slice(size_t byte_offset, std::vector<int64_t> slice_shape,
                     size_t slice_bytes, std::string slice_name) const;

    GPUTensor to_gpu(bool dequant) const;

    bool is_storage_slice() const { return !storage_name_.empty() && storage_name_ != name; }
    const std::string &storage_name() const { return storage_name_; }
    const std::vector<int64_t> &storage_shape() const { return storage_shape_; }
    const uint8_t *storage_data() const { return storage_data_; }
    size_t storage_nbytes() const { return storage_nbytes_; }
    size_t storage_byte_offset() const { return storage_byte_offset_; }

private:
    const uint8_t *data_ = nullptr;
    std::string storage_name_;
    std::vector<int64_t> storage_shape_;
    const uint8_t *storage_data_ = nullptr;
    size_t storage_nbytes_ = 0;
    size_t storage_byte_offset_ = 0;
};

#endif // LOCAL_LLM_STORAGETENSOR_H
