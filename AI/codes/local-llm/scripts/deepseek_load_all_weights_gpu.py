#!/usr/bin/env python3
"""Probe whether GGUF tensors can be resident in GPU memory at once.

This script parses the GGUF tensor table, then cudaMallocs each tensor as a
separate allocation and copies its raw bytes to GPU. It intentionally keeps all
device pointers alive until the end to approximate CudaWeightPool's no-eviction
behavior.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import mmap
import os
import struct
import sys
import time
from dataclasses import dataclass


GGML_TYPES: dict[int, tuple[str, int, int, bool]] = {
    0: ("F32", 1, 4, False),
    1: ("F16", 1, 2, False),
    2: ("Q4_0", 32, 18, True),
    3: ("Q4_1", 32, 20, True),
    6: ("Q5_0", 32, 22, True),
    7: ("Q5_1", 32, 24, True),
    8: ("Q8_0", 32, 34, True),
    9: ("Q8_1", 32, 36, True),
    10: ("Q2_K", 256, 84, True),
    11: ("Q3_K", 256, 110, True),
    12: ("Q4_K", 256, 144, True),
    13: ("Q5_K", 256, 176, True),
    14: ("Q6_K", 256, 210, True),
    15: ("Q8_K", 256, 292, True),
    30: ("BF16", 1, 2, False),
}


@dataclass
class TensorInfo:
    name: str
    shape: list[int]
    ggml_type: int
    rel_offset: int
    nbytes: int
    file_offset: int = 0


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def read(self, fmt: str):
        size = struct.calcsize(fmt)
        if self.pos + size > len(self.data):
            raise ValueError("unexpected EOF")
        value = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return value[0] if len(value) == 1 else value

    def read_bytes(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError("unexpected EOF")
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def read_string(self) -> str:
        n = self.read("<Q")
        return self.read_bytes(n).decode("utf-8", errors="replace")


def align_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


def skip_metadata_value(r: Reader, value_type: int):
    if value_type in (0, 1, 7):      # u8, i8, bool
        r.pos += 1
    elif value_type in (2, 3):       # u16, i16
        r.pos += 2
    elif value_type in (4, 5, 6):    # u32, i32, f32
        r.pos += 4
    elif value_type == 8:            # string
        r.read_string()
    elif value_type in (10, 11, 12): # u64, i64, f64
        r.pos += 8
    elif value_type == 9:            # array
        elem_type = r.read("<I")
        count = r.read("<Q")
        for _ in range(count):
            skip_metadata_value(r, elem_type)
    else:
        raise ValueError(f"unsupported GGUF metadata type: {value_type}")


def parse_gguf(path: str) -> tuple[list[TensorInfo], int]:
    with open(path, "rb") as f:
        header = f.read()
    r = Reader(header)
    if r.read_bytes(4) != b"GGUF":
        raise ValueError(f"not a GGUF file: {path}")
    version = r.read("<I")
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version: {version}")
    tensor_count = r.read("<Q")
    metadata_count = r.read("<Q")

    alignment = 32
    for _ in range(metadata_count):
        key = r.read_string()
        value_type = r.read("<I")
        if key == "general.alignment" and value_type in (4, 10):
            alignment = int(r.read("<I" if value_type == 4 else "<Q"))
        else:
            skip_metadata_value(r, value_type)

    tensors: list[TensorInfo] = []
    for _ in range(tensor_count):
        name = r.read_string()
        n_dims = r.read("<I")
        dims = [r.read("<Q") for _ in range(n_dims)]
        ggml_type = r.read("<I")
        rel_offset = r.read("<Q")
        if ggml_type not in GGML_TYPES:
            raise ValueError(f"unsupported GGML tensor type {ggml_type} for {name}")
        type_name, block_size, type_size, _ = GGML_TYPES[ggml_type]
        elements = 1
        for d in dims:
            elements *= d
        if elements % block_size != 0:
            raise ValueError(f"tensor element count is not block aligned: {name} {type_name} {dims}")
        nbytes = elements // block_size * type_size
        tensors.append(TensorInfo(name, dims, ggml_type, rel_offset, nbytes))

    data_start = align_up(r.pos, alignment)
    for t in tensors:
        t.file_offset = data_start + t.rel_offset
    return tensors, data_start


class Cuda:
    def __init__(self):
        lib_name = ctypes.util.find_library("cudart")
        candidates = [lib_name, "libcudart.so", "/usr/local/cuda/lib64/libcudart.so"]
        self.lib = None
        for name in candidates:
            if not name:
                continue
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                pass
        if self.lib is None:
            raise RuntimeError("cannot load libcudart")
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def check(self, code: int, what: str):
        if code != 0:
            msg = self.lib.cudaGetErrorString(code).decode("utf-8", errors="replace")
            raise RuntimeError(f"{what}: {msg} ({code})")

    def mem_info(self) -> tuple[int, int]:
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        self.check(self.lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)), "cudaMemGetInfo")
        return int(free.value), int(total.value)

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes), f"cudaMalloc {nbytes} bytes")
        return ptr

    def free(self, ptr: ctypes.c_void_p):
        self.lib.cudaFree(ptr)

    def copy_h2d(self, dst: ctypes.c_void_p, src: bytes, offset: int):
        buf = ctypes.create_string_buffer(src)
        dst_addr = ctypes.c_void_p(dst.value + offset)
        self.check(self.lib.cudaMemcpy(dst_addr, ctypes.cast(buf, ctypes.c_void_p), len(src), 1), "cudaMemcpy H2D")


def mib(n: int) -> float:
    return n / 1024 / 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("LOCAL_LLM_DEEPSEEK_MODEL", "/home/zyl/models/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"))
    parser.add_argument("--quantized-only", action="store_true", help="load only GGML quantized tensors")
    parser.add_argument("--skip-copy", action="store_true", help="only cudaMalloc, do not copy bytes to GPU")
    parser.add_argument("--chunk-mib", type=int, default=64, help="H2D copy chunk size")
    args = parser.parse_args()

    tensors, data_start = parse_gguf(args.model)
    selected = [t for t in tensors if (not args.quantized_only or GGML_TYPES[t.ggml_type][3])]
    selected.sort(key=lambda t: t.nbytes, reverse=True)
    expected = sum(t.nbytes for t in selected)

    cuda = Cuda()
    free0, total = cuda.mem_info()
    print(f"model={args.model}")
    print(f"data_start={data_start} tensors={len(tensors)} selected={len(selected)} quantized_only={args.quantized_only}")
    print(f"expected_tensor_bytes={expected} ({mib(expected):.2f} MiB)")
    print(f"cuda_before free={mib(free0):.2f} MiB total={mib(total):.2f} MiB")

    ptrs: list[ctypes.c_void_p] = []
    loaded = 0
    start = time.time()
    try:
        with open(args.model, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            for index, t in enumerate(selected, start=1):
                free_before, _ = cuda.mem_info()
                ptr = cuda.malloc(t.nbytes)
                ptrs.append(ptr)
                if not args.skip_copy:
                    copied = 0
                    chunk = args.chunk_mib * 1024 * 1024
                    while copied < t.nbytes:
                        n = min(chunk, t.nbytes - copied)
                        cuda.copy_h2d(ptr, mm[t.file_offset + copied:t.file_offset + copied + n], copied)
                        copied += n
                loaded += t.nbytes
                free_after, _ = cuda.mem_info()
                free_drop = free0 - free_after
                allocator_overhead = free_drop - loaded
                print(
                    f"loaded {index}/{len(selected)} {t.name} type={GGML_TYPES[t.ggml_type][0]} "
                    f"bytes={mib(t.nbytes):.2f} MiB total_loaded={mib(loaded):.2f} MiB "
                    f"free_before={mib(free_before):.2f} MiB free_after={mib(free_after):.2f} MiB "
                    f"free_drop={mib(free_drop):.2f} MiB allocator_overhead={mib(allocator_overhead):.2f} MiB",
                    flush=True,
                )
            mm.close()
    except Exception as exc:
        free_now, _ = cuda.mem_info()
        print(f"FAILED loaded={mib(loaded):.2f} MiB free_now={mib(free_now):.2f} MiB error={exc}", file=sys.stderr)
        return 2
    finally:
        elapsed = time.time() - start
        free_now, _ = cuda.mem_info()
        free_drop = free0 - free_now
        allocator_overhead = free_drop - loaded
        print(
            f"summary loaded={mib(loaded):.2f} MiB allocations={len(ptrs)} "
            f"free_now={mib(free_now):.2f} MiB free_drop={mib(free_drop):.2f} MiB "
            f"allocator_overhead={mib(allocator_overhead):.2f} MiB "
            f"device_used_from_total={mib(total - free_now):.2f} MiB elapsed_sec={elapsed:.2f}"
        )
        for ptr in reversed(ptrs):
            cuda.free(ptr)

    free_end, _ = cuda.mem_info()
    print(f"SUCCESS free_after_release={mib(free_end):.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
