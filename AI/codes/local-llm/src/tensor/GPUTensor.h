//
// Created by zhangyoulun on 21/8/2026.
//

#ifndef LOCAL_LLM_GPUTENSOR_H
#define LOCAL_LLM_GPUTENSOR_H

#include "tensor/TensorCommon.h"

#include <memory>

class CPUTensor;
class CudaScratch;
class CudaWeight;
class CudaWeightPool;
class StorageTensor;

class GPUTensor : public TensorShape {
public:
    GPUTensor() = default;
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

    CPUTensor to_host(void *host_ptr, const std::string &what) const;

private:
    friend class StorageTensor;

    void init_from_weight(const CudaWeight &weight, std::vector<int64_t> shape);

    void *data_ = nullptr;
    CudaWeightPool *pool_ = nullptr;
    std::shared_ptr<CudaWeight> owned_weight_;
    std::shared_ptr<void> weight_view_lease_;
};

#endif // LOCAL_LLM_GPUTENSOR_H
