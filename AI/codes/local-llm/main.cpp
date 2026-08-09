#include <fstream>
#include <iostream>

#include "utils/cli/Args.h"
#include "llm/qwen/QwenConfig.h"
#include "llm/qwen/QwenSession.h"
#include "llm/qwen/QwenTokenizer.h"
#include "llm/qwen/QwenWeights.h"
#include "llm/qwen/model/QwenModel.h"
#include "backend/cuda/mem/CudaWeightPool.h"
#include "utils/stats/DeviceMonitor.h"
#include "utils/stats/MemoryReporter.h"
#include "utils/stats/Profiler.h"
#include "utils/stats/ScopedTimer.h"
#include "utils/stats/WeightLoadTracker.h"

int main(int argc, char **argv) {
    Args args(argc, argv);
    args.DebugDump();

    QwenConfig config(args.model_dir + "/config.json");
    config.DebugDump();

    QwenWeights weights(args.model_dir, config);
    weights.DebugDump();

    QwenTokenizer tokenizer(args.model_dir + "/tokenizer.json");
    tokenizer.DebugDump();

    std::vector<int> inputs = tokenizer.Encode("法国的首都是");

    QwenSession session(config, inputs, args.max_output_tokens);

    // device 权重缓存（惰性上传）+ 模型（串起各 Module，无 per-request 状态）。
    CudaWeightPool pool;
    QwenModel model(config, weights, &pool);

    const int eos = config.data.text.eos_token_id;

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

    // warmup：正式计时前跑一遍完整前向（独立 session），触发权重惰性上传、
    // CUDA context / cuBLAS 初始化与 kernel 首次启动，避免首步冷启动污染稳态计时。
    {
        QwenSession warm(config, inputs, 4);
        int wnext = model.prefill(warm, inputs);
        int wpos = static_cast<int>(inputs.size());
        for (int i = 0; i < 4 && wnext != eos; ++i) {
            warm.h_outputs.push_back(wnext);
            wnext = model.decode(warm, wnext, wpos);
            ++wpos;
        }
    }
    // warmup 已把权重全部搬入显存并触发首启，清零聚合与时间线，只统计稳态。
    Profiler::instance().reset();

    if (args.profile) {
        device_monitor.start(100);
    }

    // prefill：喂入整段 prompt，跑完 32 层，返回首个生成 token。
    int next;
    {
        ScopedCpuTimer t("prefill");
        next = model.prefill(session, inputs);
    }
    if (args.profile) {
        mem_reporter.sample(pool, session, "prefill");
    }

    // decode：从 prompt 之后的位置开始逐 token 生成。
    int pos = static_cast<int>(inputs.size());
    int decode_tokens = 0;
    {
        ScopedCpuTimer t("decode_total");
        for (int step = 0; step < args.max_output_tokens; ++step) {
            if (next == eos) break;
            session.h_outputs.push_back(next);
            {
                ScopedCpuTimer td("decode_token");
                next = model.decode(session, next, pos);
            }
            ++pos;
            ++decode_tokens;
            if (args.profile) {
                mem_reporter.sample(pool, session, "decode");
            }
        }
    }

    if (args.profile) {
        device_monitor.stop();
    }

    std::cout << "生成结果：" << tokenizer.Decode(session.h_outputs) << std::endl;

    if (!args.profile) {
        return 0;
    }

    // === 三类报告落盘 ===
    // 1) JSONL 原始日志：各采集器的明细时间线逐行追加到同一文件。
    {
        std::ofstream jsonl(args.profile_dir + "/profile.jsonl");
        Profiler::instance().write_jsonl(jsonl);
        weight_tracker.write_jsonl(jsonl);
        device_monitor.write_jsonl(jsonl);
        mem_reporter.write_jsonl(jsonl);
    }
    // 2) JSON summary：各采集器聚合结论逐行写入（每行一个 JSON 对象）。
    {
        std::ofstream json(args.profile_dir + "/profile_summary.json");
        Profiler::instance().write_json_summary(json);
        weight_tracker.write_json_summary(json);
        device_monitor.write_json_summary(json);
        mem_reporter.write_json_summary(json);
    }
    // 3) Markdown summary：人类可读汇总，可直接贴进 doc / MR。
    {
        std::ofstream md(args.profile_dir + "/profile_summary.md");
        md << "# Profile summary\n\n";
        Profiler::instance().write_markdown_summary(md);
        mem_reporter.write_markdown_summary(md);
        device_monitor.write_markdown_summary(md);
        weight_tracker.write_markdown_summary(md);
    }

    // 同时在 stdout 打印稳态端到端指标，便于快速查看。
    std::cout << "PROFILE input_tokens=" << inputs.size()
              << " decode_tokens=" << decode_tokens
              << " reports=" << args.profile_dir << "/profile.{jsonl,_summary.json,_summary.md}"
              << std::endl;
    return 0;
}
