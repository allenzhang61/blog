#include <fstream>
#include <iostream>
#include <cstdlib>
#include <chrono>

#include <cuda_runtime.h>

#include "utils/cli/Args.h"
#include "llm/BaseModel.h"
#include "llm/ModelFactory.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "utils/stats/DeviceMonitor.h"
#include "utils/stats/MemoryReporter.h"
#include "utils/stats/Profiler.h"
#include "utils/stats/ScopedTimer.h"
#include "utils/stats/WeightLoadTracker.h"

int main(int argc, char **argv) {
    Args args(argc, argv);
    args.DebugDump();

    // 经工厂按 --model 构造具体模型；主循环之后只依赖 BaseModel 接口。
    std::unique_ptr<BaseModel> model = create_model(args.model, args.model_dir, args.max_output_tokens, args.sampling);
    CudaWeightPool &pool = model->weight_pool();

    std::vector<int> inputs = model->encode(std::getenv("PROMPT") ? std::getenv("PROMPT") : "法国的首都是");
    const int eos = model->eos_token_id();

    // 性能采集：仅 --profile 时开启，所有埋点否则零开销。
    Profiler::instance().enable(args.profile);
    // 权重懒加载追踪：挂到 pool 上，记录“权重一块块搬进显存”的事件时间线。
    InMemoryWeightLoadTracker weight_tracker;
    if (args.profile) {
        pool.set_load_tracker(&weight_tracker);
    }
    // 设备侧时序监控：后台线程按间隔采 SM / 带宽 / 功耗 / 温度。
    DeviceMonitor device_monitor(0);
    // 显存分层时间线：prefill / 每个 decode step 采一条，看清 KV cache 增长曲线。
    MemoryReporter mem_reporter;

    // warmup：正式计时前跑一遍完整前向，触发权重惰性上传、CUDA context / cuBLAS
    // 初始化与 kernel 首次启动，避免首步冷启动污染稳态计时。
    // BaseModel::prefill 每次会重建内部 session，因此 warmup 与正式跑天然隔离。
    {
        int wnext = model->prefill(inputs);
        int wpos = static_cast<int>(inputs.size());
        for (int i = 0; i < 4 && wnext != eos; ++i) {
            model->append_output(wnext);
            wnext = model->decode(wnext, wpos);
            ++wpos;
        }
    }
    // warmup 已把权重全部搬入显存并触发首启，清零聚合与时间线，只统计稳态。
    Profiler::instance().reset();

    if (args.profile) {
        device_monitor.start(100);
    }

    // prefill：喂入整段 prompt，跑完各层，返回首个生成 token。
    int next;
    {
        ScopedCpuTimer t("prefill");
        next = model->prefill(inputs);
    }
    if (args.profile) {
        mem_reporter.sample(pool, model->memory_usage(), "prefill");
    }

    // decode：从 prompt 之后的位置开始逐 token 生成。
    int pos = static_cast<int>(inputs.size());
    int decode_tokens = 0;
    // 基础墙钟计时：与 profile 埋点无关，无 CUDA 同步、无落盘，恒定开销可忽略。
    const auto decode_wall_start = std::chrono::steady_clock::now();
    {
        ScopedCpuTimer t("decode_total");
        for (int step = 0; step < args.max_output_tokens; ++step) {
            if (next == eos) break;
            model->append_output(next);
            {
                ScopedCpuTimer td("decode_token");
                next = model->decode(next, pos);
            }
            ++pos;
            ++decode_tokens;
            if (args.profile) {
                mem_reporter.sample(pool, model->memory_usage(), "decode");
            }
        }
    }
    // 确保所有 decode kernel 落地后再停表，得到真实稳态墙钟。
    cudaDeviceSynchronize();
    const double decode_wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - decode_wall_start).count();

    if (args.profile) {
        device_monitor.stop();
    }

    std::cout << "生成结果：" << model->decode_text(model->outputs()) << std::endl;

    // 基础耗时：无论是否 profile 都打印（仅墙钟，无逐算子埋点/同步/落盘）。
    if (decode_tokens > 0 && decode_wall_ms > 0.0) {
        const double avg_ms = decode_wall_ms / decode_tokens;
        const double tps = decode_tokens / decode_wall_ms * 1000.0;
        std::cout << "[decode] tokens=" << decode_tokens
                  << " wall_ms=" << decode_wall_ms
                  << " avg_ms_per_token=" << avg_ms
                  << " tokens_per_sec=" << tps << std::endl;
    }

    if (!args.profile) {
        return 0;
    }

    // 报告文件名加模型 name 区分，便于多模型对比：profile_<name>.{jsonl,_summary.json,_summary.md}。
    const std::string prefix = args.profile_dir + "/profile_" + model->name();

    // === 三类报告落盘 ===
    // 1) JSONL 原始日志：各采集器的明细时间线逐行追加到同一文件。
    {
        std::ofstream jsonl(prefix + ".jsonl");
        Profiler::instance().write_jsonl(jsonl);
        weight_tracker.write_jsonl(jsonl);
        device_monitor.write_jsonl(jsonl);
        mem_reporter.write_jsonl(jsonl);
    }
    // 2) JSON summary：各采集器聚合结论逐行写入（每行一个 JSON 对象）。
    {
        std::ofstream json(prefix + "_summary.json");
        Profiler::instance().write_json_summary(json);
        weight_tracker.write_json_summary(json);
        device_monitor.write_json_summary(json);
        mem_reporter.write_json_summary(json);
    }
    // 3) Markdown summary：人类可读汇总，可直接贴进 doc / MR。
    {
        std::ofstream md(prefix + "_summary.md");
        md << "# Profile summary (" << model->name() << ")\n\n";
        Profiler::instance().write_markdown_summary(md);
        mem_reporter.write_markdown_summary(md);
        device_monitor.write_markdown_summary(md);
        weight_tracker.write_markdown_summary(md);
    }

    // 同时在 stdout 打印稳态端到端指标，便于快速查看。
    std::cout << "PROFILE model=" << model->name()
              << " input_tokens=" << inputs.size()
              << " decode_tokens=" << decode_tokens
              << " reports=" << prefix << ".{jsonl,_summary.json,_summary.md}"
              << std::endl;
    return 0;
}
