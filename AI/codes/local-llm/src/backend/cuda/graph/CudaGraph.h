//
// Created by zhangyoulun on 26/8/2026.
//

#ifndef LOCAL_LLM_CUDAGRAPH_H
#define LOCAL_LLM_CUDAGRAPH_H

#include <cuda_runtime.h>

// 已 capture 并 instantiate 的 CUDA Graph 句柄封装。
// 只负责 graph / executable graph 的生命周期；具体 capture 的算子序列由调用方组织。
class CudaGraph {
public:
    CudaGraph() = default;
    // 析构时释放 graph 与 executable graph。
    ~CudaGraph();

    CudaGraph(const CudaGraph &) = delete;
    CudaGraph &operator=(const CudaGraph &) = delete;

    // 是否已有可 launch 的 executable graph。
    bool ready() const { return ready_; }

private:
    friend void destroy_cuda_graph(CudaGraph &graph);
    friend void mark_cuda_graph_stale(CudaGraph &graph);
    friend void end_cuda_graph_capture_and_instantiate(CudaGraph &graph, const char *what);
    friend void launch_cuda_graph(const CudaGraph &graph, const char *what);

    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    bool ready_ = false;
};

// 释放 graph 持有的 cudaGraphExec_t / cudaGraph_t，并把 ready 状态清空。
void destroy_cuda_graph(CudaGraph &graph);

// 标记 graph 已不可复用。用于 scratch/device buffer 可能增长或地址语义变化后，
// 让下次使用前重新 capture。
void mark_cuda_graph_stale(CudaGraph &graph);

// 在当前 stream 上开始 ThreadLocal capture。
// 调用方应在 capture 前确保后续算子都走同一 stream，且不会触发 cudaMalloc 等不可捕获操作。
void begin_thread_local_cuda_graph_capture(const char *what);

// 结束当前 stream 上的 capture，并立即 instantiate 成可 launch 的 executable graph。
// 调用前应先 destroy_cuda_graph(graph)，避免覆盖仍持有的旧 graph 句柄。
void end_cuda_graph_capture_and_instantiate(CudaGraph &graph, const char *what);

// 在当前 stream 上启动已经 ready() 的 executable graph。
void launch_cuda_graph(const CudaGraph &graph, const char *what);

#endif // LOCAL_LLM_CUDAGRAPH_H
