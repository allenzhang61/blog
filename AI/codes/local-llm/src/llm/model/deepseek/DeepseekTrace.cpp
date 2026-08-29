#include "DeepseekTrace.h"

#include "llm/model/deepseek/DeepseekSession.h"
#include "tensor/CPUTensor.h"
#include "tensor/GPUTensor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {
int env_int(const char *key, int fallback) {
    const char *env = std::getenv(key);
    if (env == nullptr || env[0] == '\0') return fallback;
    return std::atoi(env);
}

int trace_row_for(const GPUTensor &g_tensor, int requested) {
    const int64_t rows = g_tensor.rows();
    if (rows <= 0) return 0;
    int row = requested;
    if (row < 0) row = env_int("LOCAL_LLM_DEEPSEEK_TRACE_ROW", static_cast<int>(rows - 1));
    if (row < 0) row = static_cast<int>(rows - 1);
    if (row >= rows) row = static_cast<int>(rows - 1);
    return row;
}

void print_prefix(const char *kind, const char *stage, int pos, int layer, int row) {
    std::cerr << "[ds.trace] kind=" << kind
              << " tag=" << deepseek_trace::tag()
              << " pos=" << pos
              << " layer=" << layer
              << " stage=" << stage
              << " row=" << row;
}
} // namespace

namespace deepseek_trace {
bool enabled() {
    const char *env = std::getenv("LOCAL_LLM_DEEPSEEK_TRACE");
    return env != nullptr && std::atoi(env) > 0;
}

bool pos_match(int pos) {
    const char *env = std::getenv("LOCAL_LLM_DEEPSEEK_TRACE_POS");
    if (env == nullptr || env[0] == '\0') return true;
    return pos == std::atoi(env);
}

const char *tag() {
    const char *env = std::getenv("LOCAL_LLM_DEEPSEEK_TRACE_TAG");
    return (env != nullptr && env[0] != '\0') ? env : "run";
}

void tensor(DeepseekSession &session, const GPUTensor &g_tensor,
            const char *stage, int pos, int layer, int row) {
    if (!enabled() || !pos_match(pos) || g_tensor.dtype != DType::F32) return;
    CPUTensor c_tensor = g_tensor.to_host(session.cpu_scratch, "ds.trace.tensor", "ds.trace.tensor");
    const float *h = c_tensor.data<float>();
    const int64_t rows = c_tensor.rows();
    const int64_t cols = c_tensor.cols();
    if (rows <= 0 || cols <= 0) return;
    const int trace_row = trace_row_for(g_tensor, row);
    const float *r = h + static_cast<size_t>(trace_row) * cols;

    double sum = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    int finite_count = 0;
    int nan_count = 0;
    int inf_count = 0;
    for (int64_t i = 0; i < cols; ++i) {
        const float v = r[i];
        if (std::isnan(v)) {
            ++nan_count;
            continue;
        }
        if (!std::isfinite(v)) {
            ++inf_count;
            continue;
        }
        ++finite_count;
        sum += v;
        const double av = std::fabs(static_cast<double>(v));
        sum_abs += av;
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
        max_abs = std::max(max_abs, static_cast<float>(av));
    }

    print_prefix("tensor", stage, pos, layer, trace_row);
    std::cerr << " rows=" << rows
              << " cols=" << cols
              << " finite=" << finite_count
              << " nan=" << nan_count
              << " inf=" << inf_count
              << " sum=" << sum
              << " mean_abs=" << (finite_count > 0 ? sum_abs / static_cast<double>(finite_count) : 0.0)
              << " rms=" << (finite_count > 0 ? std::sqrt(sum_sq / static_cast<double>(finite_count)) : 0.0)
              << " max_abs=" << max_abs
              << " first=";
    const int64_t limit = std::min<int64_t>(8, cols);
    for (int64_t i = 0; i < limit; ++i) {
        if (i) std::cerr << ",";
        std::cerr << r[i];
    }
    std::cerr << "\n";
}

void topk(DeepseekSession &session, const GPUTensor &g_idx_i32, const GPUTensor &g_w_f32,
          const char *stage, int pos, int layer, int row) {
    if (!enabled() || !pos_match(pos)) return;
    CPUTensor c_idx = g_idx_i32.to_host(session.cpu_scratch, "ds.trace.topk.idx", "ds.trace.topk.idx");
    CPUTensor c_w = g_w_f32.to_host(session.cpu_scratch, "ds.trace.topk.w", "ds.trace.topk.w");
    const int64_t rows = c_idx.rows();
    const int64_t k = c_idx.cols();
    if (rows <= 0 || k <= 0) return;
    const int trace_row = trace_row_for(g_idx_i32, row);
    const int *idx = c_idx.data<int>() + static_cast<size_t>(trace_row) * k;
    const float *w = c_w.data<float>() + static_cast<size_t>(trace_row) * k;

    print_prefix("topk", stage, pos, layer, trace_row);
    std::cerr << " k=" << k << " ids=";
    for (int64_t i = 0; i < k; ++i) {
        if (i) std::cerr << ",";
        std::cerr << idx[i];
    }
    std::cerr << " weights=";
    for (int64_t i = 0; i < k; ++i) {
        if (i) std::cerr << ",";
        std::cerr << w[i];
    }
    std::cerr << "\n";
}
} // namespace deepseek_trace
