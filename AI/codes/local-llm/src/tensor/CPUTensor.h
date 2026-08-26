//
// Created by zhangyoulun on 21/8/2026.
//

#ifndef LOCAL_LLM_CPUTENSOR_H
#define LOCAL_LLM_CPUTENSOR_H

#include "tensor/TensorCommon.h"

class CPUScratch;
class CudaScratch;
class GPUTensor;

class CPUTensor : public TensorShape {
public:
    CPUTensor() = default;
    CPUTensor(const void *host_ptr, std::vector<int64_t> shape, DType dt);
    CPUTensor(CPUScratch &scratch, const std::string &key, std::vector<int64_t> shape, DType dt);

    const void *data() const { return data_; }
    void *data() { return const_cast<void *>(data_); }
    template <typename T>
    const std::remove_cv_t<T> *data() const {
        validate_tensor_cpp_type<T>(dtype, name);
        return static_cast<const std::remove_cv_t<T> *>(data_);
    }
    template <typename T>
    std::remove_cv_t<T> *data() {
        validate_tensor_cpp_type<T>(dtype, name);
        return static_cast<std::remove_cv_t<T> *>(data());
    }

    GPUTensor to_gpu(CudaScratch &scratch, const std::string &key, const std::string &what) const;

private:
    const void *data_ = nullptr;
};

#endif // LOCAL_LLM_CPUTENSOR_H
