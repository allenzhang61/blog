//
// Created by zhangyoulun on 26/8/2026.
//

#include "backend/cuda/graph/CudaGraph.h"

#include "backend/cuda/common.h"

#include <stdexcept>

CudaGraph::~CudaGraph() {
    destroy_cuda_graph(*this);
}

void destroy_cuda_graph(CudaGraph &graph) {
    if (graph.exec_) {
        cudaGraphExecDestroy(graph.exec_);
        graph.exec_ = nullptr;
    }
    if (graph.graph_) {
        cudaGraphDestroy(graph.graph_);
        graph.graph_ = nullptr;
    }
    graph.ready_ = false;
}

void mark_cuda_graph_stale(CudaGraph &graph) {
    graph.ready_ = false;
}

void begin_thread_local_cuda_graph_capture(const char *what) {
    check_cuda(cudaStreamBeginCapture(get_current_cuda_stream(), cudaStreamCaptureModeThreadLocal), what);
}

void end_cuda_graph_capture_and_instantiate(CudaGraph &graph, const char *what) {
    check_cuda(cudaStreamEndCapture(get_current_cuda_stream(), &graph.graph_), what);
    check_cuda(cudaGraphInstantiate(&graph.exec_, graph.graph_, nullptr, nullptr, 0), what);
    graph.ready_ = true;
}

void launch_cuda_graph(const CudaGraph &graph, const char *what) {
    if (!graph.ready_ || graph.exec_ == nullptr) {
        throw std::runtime_error("launch_cuda_graph 需要已实例化的 graph");
    }
    check_cuda(cudaGraphLaunch(graph.exec_, get_current_cuda_stream()), what);
}
