#!/usr/bin/env python3
import argparse
import re


TRACE_RE = re.compile(
    r"\[ds\.trace\] kind=(\w+) tag=(\w+) pos=(-?\d+) layer=(-?\d+) stage=([^ ]+).*?"
    r"(?:sum=([-+eE0-9.]+).*?mean_abs=([-+eE0-9.]+).*?rms=([-+eE0-9.]+).*?"
    r"max_abs=([-+eE0-9.]+).*?first=([^\n]+)|ids=([^ ]+) weights=([^\n]+))"
)


def parse_values(text, cast):
    if not text:
        return []
    return [cast(x) for x in text.split(",") if x]


def load_trace(path, pos):
    traces = {}
    order = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = TRACE_RE.search(line)
            if not m:
                continue
            kind = m.group(1)
            trace_pos = int(m.group(3))
            if pos is not None and trace_pos != pos:
                continue
            key = (kind, trace_pos, int(m.group(4)), m.group(5))
            if kind == "tensor":
                traces[key] = {
                    "sum": float(m.group(6)),
                    "mean_abs": float(m.group(7)),
                    "rms": float(m.group(8)),
                    "max_abs": float(m.group(9)),
                    "first": parse_values(m.group(10), float),
                }
            else:
                traces[key] = {
                    "ids": parse_values(m.group(11), int),
                    "weights": parse_values(m.group(12), float),
                }
            order.append(key)
    return traces, order


def tensor_diff(a, b):
    first_diff = 0.0
    if a["first"] and b["first"]:
        first_diff = max(abs(x - y) for x, y in zip(a["first"], b["first"]))
    rms_rel = abs(a["rms"] - b["rms"]) / (abs(a["rms"]) + 1e-12)
    return abs(a["sum"] - b["sum"]), rms_rel, first_diff


def main():
    parser = argparse.ArgumentParser(description="Compare DeepSeek [ds.trace] logs.")
    parser.add_argument("safe_log")
    parser.add_argument("quant_log")
    parser.add_argument("--pos", type=int, default=None)
    parser.add_argument("--first-diff-threshold", type=float, default=0.05)
    parser.add_argument("--rms-rel-threshold", type=float, default=0.01)
    args = parser.parse_args()

    safe, order = load_trace(args.safe_log, args.pos)
    quant, _ = load_trace(args.quant_log, args.pos)

    print("topk mismatches:")
    for key in order:
        if key not in quant or key[0] != "topk":
            continue
        s = safe[key]
        q = quant[key]
        if s["ids"] != q["ids"]:
            max_w = max(abs(x - y) for x, y in zip(s["weights"], q["weights"])) if s["weights"] else 0.0
            print(f"pos={key[1]} layer={key[2]} stage={key[3]} safe={s['ids']} quant={q['ids']} max_w_diff={max_w:.6g}")

    print("\nlarge tensor diffs:")
    for key in order:
        if key not in quant or key[0] != "tensor":
            continue
        sum_diff, rms_rel, first_diff = tensor_diff(safe[key], quant[key])
        if first_diff > args.first_diff_threshold or rms_rel > args.rms_rel_threshold:
            print(
                f"pos={key[1]} layer={key[2]:2d} stage={key[3]:14s} "
                f"first_diff={first_diff:.6g} rms_rel={rms_rel:.6g} sum_diff={sum_diff:.6g}"
            )


if __name__ == "__main__":
    main()
