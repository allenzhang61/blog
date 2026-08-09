//
// Created by zhangyoulun on 8/8/2026.
//

#ifndef LOCAL_LLM_ARGS_H
#define LOCAL_LLM_ARGS_H
#include <string>
#include "backend/common.h"

class Args {
public:
    std::string model_dir;
    int max_output_tokens = 20;
    Device device = Device::CPU;

    // 是否开启性能采集（Profiler / MemoryReporter / DeviceMonitor / 权重懒加载追踪）。
    bool profile = false;
    // profile 报告输出目录（jsonl 原始日志 + json/markdown summary），默认当前目录。
    std::string profile_dir = ".";

    Args(int argc, char **argv);
    void DebugDump();
};


#endif //LOCAL_LLM_ARGS_H
