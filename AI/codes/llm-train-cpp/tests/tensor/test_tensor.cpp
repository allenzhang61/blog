#include "llm/tensor.hpp"

#include "llm/llm.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

using namespace llm;

// ---------------------------------------------------------------------------
// 构造函数
// ---------------------------------------------------------------------------

TEST(TensorCtor, DefaultIsEmptyScalarLike) {
    Tensor t;
    // 默认构造只分配空 node，形状为空、无数据。
    EXPECT_TRUE(t.shape().empty());
    EXPECT_EQ(t.numel(), 0);
    EXPECT_FALSE(t.requires_grad());
}

TEST(TensorCtor, ShapeCtorZeroFillsAndAllocsGrad) {
    Tensor t({2, 3}, DType::Float32, {}, true);
    EXPECT_EQ(t.shape(), std::vector<int64_t>({2, 3}));
    EXPECT_EQ(t.numel(), 6);
    EXPECT_TRUE(t.requires_grad());
    for (double v : t.data()) {
        EXPECT_DOUBLE_EQ(v, 0.0);
    }
    // requires_grad=true 时梯度缓冲区应已按数据大小分配。
    EXPECT_EQ(t.grad().size(), 6u);
}

TEST(TensorCtor, ShapeCtorNoGradWhenNotRequested) {
    Tensor t({4}, DType::Float32, {}, false);
    EXPECT_FALSE(t.requires_grad());
    EXPECT_EQ(t.numel(), 4);
}

TEST(TensorCtor, DataCtorKeepsValues) {
    Tensor t({2, 2}, {1.0, 2.0, 3.0, 4.0}, DType::Float32, {}, false);
    EXPECT_EQ(t.numel(), 4);
    EXPECT_DOUBLE_EQ(t.data()[0], 1.0);
    EXPECT_DOUBLE_EQ(t.data()[3], 4.0);
}

TEST(TensorCtor, DataCtorMismatchThrows) {
    // 数据长度与 shape 元素数不一致应抛异常。
    EXPECT_THROW((Tensor({2, 2}, {1.0, 2.0, 3.0}, DType::Float32, {}, false)), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 工厂方法
// ---------------------------------------------------------------------------

TEST(TensorFactory, Zeros) {
    Tensor t = Tensor::zeros({3});
    EXPECT_EQ(t.numel(), 3);
    for (double v : t.data()) {
        EXPECT_DOUBLE_EQ(v, 0.0);
    }
    EXPECT_FALSE(t.requires_grad());
}

TEST(TensorFactory, Ones) {
    Tensor t = Tensor::ones({2, 2}, {}, true);
    EXPECT_EQ(t.numel(), 4);
    for (double v : t.data()) {
        EXPECT_DOUBLE_EQ(v, 1.0);
    }
    EXPECT_TRUE(t.requires_grad());
}

TEST(TensorFactory, RandnShapeAndDeterminism) {
    // randn 用固定种子 123，非全 0，且方差随 scale 变化（这里只验证形状与非平凡取值）。
    Tensor t = Tensor::randn({100}, 1.0);
    EXPECT_EQ(t.numel(), 100);
    bool any_nonzero = false;
    for (double v : t.data()) {
        if (v != 0.0) {
            any_nonzero = true;
        }
    }
    EXPECT_TRUE(any_nonzero);
}

TEST(TensorFactory, UniformStaysInBound) {
    // uniform 取值范围应落在 [-bound, bound]。
    const double bound = 0.25;
    Tensor t = Tensor::uniform({256}, bound);
    EXPECT_EQ(t.numel(), 256);
    for (double v : t.data()) {
        EXPECT_GE(v, -bound);
        EXPECT_LE(v, bound);
    }
}

TEST(TensorFactory, FromVector) {
    Tensor t = Tensor::from_vector({10.0, 20.0}, {2}, {}, true);
    EXPECT_EQ(t.numel(), 2);
    EXPECT_DOUBLE_EQ(t.data()[1], 20.0);
    EXPECT_EQ(t.dtype(), DType::Float32);
    EXPECT_TRUE(t.requires_grad());
}

TEST(TensorFactory, FromInts) {
    Tensor t = Tensor::from_ints({3, 5, 7}, {3});
    EXPECT_EQ(t.numel(), 3);
    EXPECT_EQ(t.dtype(), DType::Int64);
    // 内部按 double 存储，整数值应可无损读回。
    EXPECT_DOUBLE_EQ(t.data()[0], 3.0);
    EXPECT_DOUBLE_EQ(t.data()[2], 7.0);
    EXPECT_FALSE(t.requires_grad());
}

// ---------------------------------------------------------------------------
// 访问器
// ---------------------------------------------------------------------------

TEST(TensorAccessor, ShapeDTypeDeviceRequiresGrad) {
    Tensor t({2, 5}, DType::Float32, {}, true);
    EXPECT_EQ(t.shape(), std::vector<int64_t>({2, 5}));
    EXPECT_EQ(t.dtype(), DType::Float32);
    EXPECT_EQ(t.device().type, DeviceType::CPU);
    EXPECT_TRUE(t.requires_grad());
}

TEST(TensorAccessor, NumelMatchesProduct) {
    Tensor t({2, 3, 4});
    EXPECT_EQ(t.numel(), 24);
}

TEST(TensorAccessor, DataMutableAndConst) {
    Tensor t({2});
    t.data()[0] = 42.0;  // 非 const data() 返回可写引用。
    const Tensor& ct = t;
    EXPECT_DOUBLE_EQ(ct.data()[0], 42.0);  // const data() 读回同一值。
}

TEST(TensorAccessor, MutableDataReturnsWritableRef) {
    Tensor t({3});
    t.mutable_data()[2] = 9.0;
    EXPECT_DOUBLE_EQ(t.data()[2], 9.0);
}

TEST(TensorAccessor, GradLazyAllocates) {
    // 即使构造时未分配梯度，首次访问 grad() 也应按数据大小补齐为 0。
    Tensor t({4}, DType::Float32, {}, false);
    EXPECT_EQ(t.grad().size(), 4u);
    for (double g : t.grad()) {
        EXPECT_DOUBLE_EQ(g, 0.0);
    }
}

TEST(TensorAccessor, MutableGradWritable) {
    Tensor t({2});
    t.mutable_grad()[0] = 3.0;
    EXPECT_DOUBLE_EQ(t.grad()[0], 3.0);
}

TEST(TensorAccessor, MarkHostDirtyIsNoopOnCpu) {
    // CPU 张量上这两个标记函数应为安全的空操作（不抛异常、不改变数据）。
    Tensor t({2});
    t.data()[0] = 1.0;
    EXPECT_NO_THROW(t.mark_data_host_dirty());
    EXPECT_NO_THROW(t.mark_grad_host_dirty());
    EXPECT_DOUBLE_EQ(t.data()[0], 1.0);
}

// ---------------------------------------------------------------------------
// item
// ---------------------------------------------------------------------------

TEST(TensorItem, ReturnsScalarValue) {
    Tensor t = Tensor::from_vector({3.14}, {1});
    EXPECT_DOUBLE_EQ(t.item(), 3.14);
}

TEST(TensorItem, NonScalarThrows) {
    Tensor t = Tensor::from_vector({1.0, 2.0}, {2});
    EXPECT_THROW(t.item(), std::runtime_error);
}

// ---------------------------------------------------------------------------
// zero_grad
// ---------------------------------------------------------------------------

TEST(TensorZeroGrad, ClearsGradient) {
    Tensor t({3}, DType::Float32, {}, true);
    t.grad()[0] = 5.0;
    t.grad()[1] = -2.0;
    t.zero_grad();
    EXPECT_EQ(t.grad().size(), 3u);
    for (double g : t.grad()) {
        EXPECT_DOUBLE_EQ(g, 0.0);
    }
}

// ---------------------------------------------------------------------------
// backward
// ---------------------------------------------------------------------------

TEST(TensorBackward, ScalarSeedsGradientOne) {
    // 对标量自身反向，种子梯度应为 1。
    Tensor a = Tensor::from_vector({2.0, 3.0}, {2}, {}, true);
    Tensor loss = ops::sum(a);
    loss.backward();
    EXPECT_DOUBLE_EQ(loss.grad()[0], 1.0);
    // sum 对每个输入的偏导为 1。
    EXPECT_DOUBLE_EQ(a.grad()[0], 1.0);
    EXPECT_DOUBLE_EQ(a.grad()[1], 1.0);
}

TEST(TensorBackward, NonScalarThrows) {
    Tensor a = Tensor::from_vector({1.0, 2.0}, {2}, {}, true);
    EXPECT_THROW(a.backward(), std::runtime_error);
}

TEST(TensorBackward, PropagatesThroughMatmul) {
    Tensor a = Tensor::from_vector({1, 2, 3, 4}, {2, 2}, {}, true);
    Tensor b = Tensor::from_vector({5, 6, 7, 8}, {2, 2}, {}, true);
    Tensor loss = ops::sum(ops::matmul(a, b));
    loss.backward();
    EXPECT_NEAR(a.grad()[0], 11.0, 1e-9);
    EXPECT_NEAR(b.grad()[0], 4.0, 1e-9);
}

TEST(TensorBackward, BreaksGraphAfterBackward) {
    // 反向后计算图应被打断：中间节点的 backward_fn 清空、parents 清空，避免循环引用泄漏。
    Tensor a = Tensor::from_vector({1.0, 2.0}, {2}, {}, true);
    Tensor s = ops::sum(a);
    s.backward();
    EXPECT_EQ(s.node->backward_fn, nullptr);
    EXPECT_TRUE(s.node->parents.empty());
}

// ---------------------------------------------------------------------------
// strides_for（自由函数）
// ---------------------------------------------------------------------------

TEST(StridesFor, RowMajorContiguous) {
    EXPECT_EQ(strides_for({2, 3, 4}), std::vector<int64_t>({12, 4, 1}));
}

TEST(StridesFor, SingleDim) {
    EXPECT_EQ(strides_for({5}), std::vector<int64_t>({1}));
}

TEST(StridesFor, EmptyShape) {
    EXPECT_TRUE(strides_for({}).empty());
}

// ---------------------------------------------------------------------------
// 设备（Metal/CUDA）分支
//
// 这些分支在纯 CPU 张量上不会被触达。为了在没有真实 GPU 的情况下覆盖它们，
// 这里构造一个假的 TensorStorage（回调只记账、不碰真实设备内存），手动挂到
// node->metal_storage 上，并把 device 设成 Metal，从而走完 sync / mark_dirty /
// zero_grad / backward 里的设备代码路径。
// ---------------------------------------------------------------------------

namespace {

// 记录各回调被调用次数的账本。
struct FakeStorageStats {
    int release = 0;
    int copy_data_to_host = 0;
    int copy_grad_to_host = 0;
    int copy_grad_from_host = 0;
    int fill_grad = 0;
};

FakeStorageStats g_stats;

std::shared_ptr<TensorStorage> make_fake_storage() {
    g_stats = FakeStorageStats{};
    auto s = std::make_shared<TensorStorage>();
    s->release = [](TensorStorage&) { g_stats.release++; };
    s->copy_data_to_host = [](TensorStorage&, std::vector<double>& host) {
        g_stats.copy_data_to_host++;
        std::fill(host.begin(), host.end(), 7.0);
    };
    s->copy_grad_to_host = [](TensorStorage&, std::vector<double>& host) {
        g_stats.copy_grad_to_host++;
        std::fill(host.begin(), host.end(), 3.0);
    };
    s->copy_grad_from_host = [](TensorStorage&, const std::vector<double>&) {
        g_stats.copy_grad_from_host++;
    };
    s->fill_grad = [](TensorStorage&, size_t, float) { g_stats.fill_grad++; };
    return s;
}

// 把一个 CPU 张量伪装成 Metal 张量并挂上假 storage。
Tensor make_metal_tensor(const std::vector<double>& values, const std::vector<int64_t>& shape) {
    Tensor t = Tensor::from_vector(values, shape, {}, true);
    t.node->device = Device{DeviceType::Metal, 0};
    t.node->metal_storage = make_fake_storage();
    return t;
}

// 把一个 CPU 张量伪装成 CUDA 张量并挂上假 storage（覆盖 CUDA 专用分支）。
Tensor make_cuda_tensor(const std::vector<double>& values, const std::vector<int64_t>& shape) {
    Tensor t = Tensor::from_vector(values, shape, {}, true);
    t.node->device = Device{DeviceType::CUDA, 0};
    t.node->cuda_storage = make_fake_storage();
    return t;
}

} // namespace

TEST(TensorStorageDtor, CallsReleaseCallback) {
    int released = 0;
    {
        TensorStorage s;
        s.release = [](TensorStorage& self) { (*static_cast<int*>(self.data))++; };
        s.data = &released;
    }  // 离开作用域触发析构，release 应被调用一次。
    EXPECT_EQ(released, 1);
}

TEST(TensorStorageDtor, NoReleaseWhenNull) {
    // release 为空指针时析构不应崩溃。
    EXPECT_NO_THROW({ TensorStorage s; });
}

TEST(TensorDevice, ConstructorSetsHostDirtyOnMetal) {
    // 在 Metal 设备上按 shape 构造张量应把 host_*_dirty 标记为脏。
    Tensor t({2}, DType::Float32, Device{DeviceType::Metal, 0}, true);
    EXPECT_TRUE(t.node->host_data_dirty);
    EXPECT_TRUE(t.node->host_grad_dirty);
}

TEST(TensorDevice, DataCtorSetsHostDirtyOnMetal) {
    Tensor t({2}, {1.0, 2.0}, DType::Float32, Device{DeviceType::Metal, 0}, true);
    EXPECT_TRUE(t.node->host_data_dirty);
    EXPECT_TRUE(t.node->host_grad_dirty);
}

TEST(TensorDevice, SyncDataToHostPullsFromDevice) {
    Tensor t = make_metal_tensor({1.0, 2.0}, {2});
    t.node->device_data_dirty = true;  // 设备端数据较新，读取时应回拉。
    const std::vector<double>& d = t.data();
    EXPECT_EQ(g_stats.copy_data_to_host, 1);
    EXPECT_DOUBLE_EQ(d[0], 7.0);  // 回调把 host 填成了 7.0。
    EXPECT_FALSE(t.node->device_data_dirty);  // 同步后应清除脏标记。
}

TEST(TensorDevice, SyncDataThrowsWhenCopyMissing) {
    Tensor t = make_metal_tensor({1.0}, {1});
    t.node->device_data_dirty = true;
    t.node->metal_storage->copy_data_to_host = nullptr;  // 缺少回调应抛异常。
    EXPECT_THROW(t.data(), std::runtime_error);
}

TEST(TensorDevice, SyncGradToHostPullsFromDevice) {
    Tensor t = make_metal_tensor({1.0, 2.0}, {2});
    t.node->device_grad_dirty = true;
    const std::vector<double>& g = t.grad();
    EXPECT_EQ(g_stats.copy_grad_to_host, 1);
    EXPECT_DOUBLE_EQ(g[0], 3.0);
    EXPECT_FALSE(t.node->device_grad_dirty);
}

TEST(TensorDevice, SyncGradThrowsWhenCopyMissing) {
    Tensor t = make_metal_tensor({1.0}, {1});
    t.node->device_grad_dirty = true;
    t.node->metal_storage->copy_grad_to_host = nullptr;
    EXPECT_THROW(t.grad(), std::runtime_error);
}

TEST(TensorDevice, MarkHostDirtyFlipsFlagsOnMetal) {
    Tensor t = make_metal_tensor({1.0}, {1});
    t.node->device_data_dirty = true;
    t.node->device_grad_dirty = true;
    t.mark_data_host_dirty();
    t.mark_grad_host_dirty();
    EXPECT_TRUE(t.node->host_data_dirty);
    EXPECT_FALSE(t.node->device_data_dirty);
    EXPECT_TRUE(t.node->host_grad_dirty);
    EXPECT_FALSE(t.node->device_grad_dirty);
}

TEST(TensorDevice, ZeroGradUsesDeviceFillWhenAvailable) {
    Tensor t = make_metal_tensor({1.0, 2.0}, {2});
    t.zero_grad();
    EXPECT_EQ(g_stats.fill_grad, 1);      // 有 fill_grad 时应走设备端填零。
    EXPECT_FALSE(t.node->host_grad_dirty);
    EXPECT_TRUE(t.node->device_grad_dirty);
}

TEST(TensorDevice, ZeroGradFallsBackToHostWhenNoFill) {
    Tensor t = make_metal_tensor({1.0, 2.0}, {2});
    t.node->metal_storage->fill_grad = nullptr;  // 无 fill_grad：仅 host 清零并标脏。
    t.zero_grad();
    EXPECT_EQ(g_stats.fill_grad, 0);
    EXPECT_TRUE(t.node->host_grad_dirty);
    EXPECT_FALSE(t.node->device_grad_dirty);
}

TEST(TensorDevice, BackwardPushesSeedGradToDevice) {
    // Metal 标量张量反向时，种子梯度应通过 copy_grad_from_host 推到设备端。
    Tensor t = make_metal_tensor({5.0}, {1});
    t.backward();
    EXPECT_EQ(g_stats.copy_grad_from_host, 1);
    EXPECT_FALSE(t.node->host_grad_dirty);
    EXPECT_TRUE(t.node->device_grad_dirty);
}

// ---- CUDA 专用分支（选取 CUDA storage 的代码路径）----

TEST(TensorDeviceCuda, SyncDataToHostPullsFromDevice) {
    Tensor t = make_cuda_tensor({1.0, 2.0}, {2});
    t.node->device_data_dirty = true;
    const std::vector<double>& d = t.data();
    EXPECT_EQ(g_stats.copy_data_to_host, 1);
    EXPECT_DOUBLE_EQ(d[0], 7.0);
}

TEST(TensorDeviceCuda, SyncGradToHostPullsFromDevice) {
    Tensor t = make_cuda_tensor({1.0, 2.0}, {2});
    t.node->device_grad_dirty = true;
    const std::vector<double>& g = t.grad();
    EXPECT_EQ(g_stats.copy_grad_to_host, 1);
    EXPECT_DOUBLE_EQ(g[0], 3.0);
}

TEST(TensorDeviceCuda, BackwardPushesSeedGradToDevice) {
    Tensor t = make_cuda_tensor({5.0}, {1});
    t.backward();
    EXPECT_EQ(g_stats.copy_grad_from_host, 1);
    EXPECT_TRUE(t.node->device_grad_dirty);
}

// ---- topo_visit 去重：菱形计算图应对共享节点只访问一次 ----

TEST(TensorBackward, DiamondGraphVisitsSharedNodeOnce) {
    // a 被两条路径共享：loss = sum(a) + sum(a)，对 a 的梯度应累加为 2。
    // 反向能正常完成即说明 topo_visit 的去重（seen）逻辑正确处理了共享节点。
    Tensor a = Tensor::from_vector({1.0, 2.0, 3.0}, {3}, {}, true);
    Tensor loss = ops::add(ops::sum(a), ops::sum(a));
    loss.backward();
    EXPECT_DOUBLE_EQ(a.grad()[0], 2.0);
    EXPECT_DOUBLE_EQ(a.grad()[2], 2.0);
}
