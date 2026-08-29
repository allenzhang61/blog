//
// Created by zhangyoulun on 21/8/2026.
//

#ifndef LOCAL_LLM_GPUTENSOR_H
#define LOCAL_LLM_GPUTENSOR_H

#include "tensor/TensorCommon.h"

#include <memory>

class CPUTensor;
class CPUScratch;
class CudaScratch;
class CudaWeight;
class CudaWeightPool;
class StorageTensor;

class GPUTensor : public TensorShape {
public:
    GPUTensor() = default;
    GPUTensor(void *device_ptr, std::vector<int64_t> shape, DType dt, std::string name = "");
    GPUTensor(CudaWeight &&weight, std::vector<int64_t> shape);
    GPUTensor(std::shared_ptr<CudaWeight> weight, std::vector<int64_t> shape);
    GPUTensor(CudaScratch &scratch, const std::string &key,
              std::vector<int64_t> shape, DType dt);
    GPUTensor(const GPUTensor &parent, size_t byte_offset,
              std::vector<int64_t> shape);

    void *data() const { return data_; }
    template <typename T>
    std::remove_cv_t<T> *data() const {
        validate_tensor_cpp_type<T>(dtype, name);
        return static_cast<std::remove_cv_t<T> *>(data_);
    }

    // 将一个 host 标量异步写入当前 CUDA stream 上的单元素 tensor。
    template <typename T>
    void set_data(const T &value, const std::string &what = "GPUTensor::setdata") const {
        using U = std::remove_cv_t<T>;
        validate_tensor_cpp_type<U>(dtype, name);
        if (numel() != 1) {
            throw std::runtime_error("GPUTensor::setdata 仅支持单元素 tensor: " + name);
        }
        setdata_from_host(&value, sizeof(U), what);
    }

    CPUTensor to_host(CPUScratch &scratch, const std::string &key, const std::string &what) const;

private:
    friend class StorageTensor;

    void init_from_weight(const CudaWeight &weight, std::vector<int64_t> shape);
    void setdata_from_host(const void *host_data, size_t bytes, const std::string &what) const;

    void *data_ = nullptr;
    CudaWeightPool *pool_ = nullptr;
    std::shared_ptr<CudaWeight> owned_weight_;
    std::shared_ptr<void> weight_view_lease_;
};

#endif // LOCAL_LLM_GPUTENSOR_H
