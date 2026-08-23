//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_ARGS_H
#define LOCAL_LLM_ARGS_H
#include <string>
#include "backend/common.h"
#include "utils/sampling/Sampler.h"

class Args {
public:
    // 模型名（决定加载哪种模型 + 报告文件名后缀）。默认 qwen。
    std::string model = "qwen";
    std::string model_dir;
    int max_output_tokens = 20;
    Device device = Device::CUDA;

    // 采样配置：默认 temperature=0 即贪心（argmax），与旧行为一致。
    SamplingConfig sampling;

    // 是否开启性能采集（Profiler / MemoryReporter / DeviceMonitor / 权重懒加载追踪）。
    bool profile = false;
    // profile 报告输出目录（jsonl 原始日志 + json/markdown summary），默认当前目录。
    std::string profile_dir = ".";
    // 采样式 profile：每隔多少个 decode step 采一次全量 GPU event 计时，其余步零
    // event 开销。默认 16（低扰动，稳态吞吐接近无 profile）。设为 1 即每步全量最细。
    int profile_sample_every = 16;

    Args(int argc, char **argv);
    void debug_dump();
};


#endif //LOCAL_LLM_ARGS_H
