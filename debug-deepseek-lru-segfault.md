# Debug Session: deepseek-lru-segfault

Status: OPEN

## Symptom

`LOCAL_LLM_EXPERIMENTAL_CUDA_WEIGHT_POOL_LRU=1` 下 DeepSeek 128 token 连续运行偶发 segfault。问题对时序敏感，gdb 下不易复现，普通 shell 连续运行曾在第 5 次崩溃。

## Hypotheses

1. `CudaWeightDequantPool` LRU 淘汰释放了仍被异步 cuBLAS GEMM 读取的 F16 dequant buffer。
2. `CudaWeightPool` LRU 淘汰释放了仍被异步 kernel/cuBLAS 读取的 device weight buffer。
3. safe dequant GEMM 中 dequant/copy/GEMM 存在跨 stream 未排序，导致使用未完成写入或已释放缓冲区。
4. 连续运行中的 GPU 显存压力或残留进程触发 OOM/非法访问，被表现为 segfault。
5. host 侧缓存裸指针在一次 operator 调用内跨淘汰后悬挂。

## Evidence Plan

- 先跑默认 `ctest`，确认非实验路径仍稳定。
- 再跑 LRU 128 token 连续验证，观察是否仍复现 segfault。
- 若仍崩溃，收集 exit code、尾部日志、必要时启用 `compute-sanitizer` 或 backtrace。
- 根据证据决定是否保留同步修复，或升级为 delayed-free / stream-event 释放策略。

## Changes So Far

- `CudaWeightPool` LRU 淘汰前同步，降低释放仍在异步使用的 weight buffer 风险。
- `CudaWeightDequantPool` LRU 淘汰前同步，降低释放仍在异步使用的 F16 dequant buffer 风险。

