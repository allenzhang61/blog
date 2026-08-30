#!/usr/bin/env python3
"""Generate a DeepSeek profile flame-style SVG.

Default output:
    temp/deepseek_profile_hotspots.svg

Usage:
    python3 tools/deepseek_profile_flame_svg.py
    python3 tools/deepseek_profile_flame_svg.py profile_deepseek_summary.json
    python3 tools/deepseek_profile_flame_svg.py profile_deepseek_summary.json -o temp/foo.svg

The input summary is expected to be the profiler summary JSON/JSONL emitted by
local-llm. If no input is provided, the script uses the latest hand-copied
DeepSeek default-path profile values.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "temp" / "deepseek_profile_hotspots.svg"


DEFAULT_METRICS: dict[str, float] = {
    "prefill": 154.919,
    "decode_window": 102.7,
    "ds.decode.mla_forward": 58.5928,
    "ds.decode.mlp.moe_forward": 174.458,
    "lm_head": 28.9115,
    "ds.gemm.d_attn_q": 29.3476,
    "ds.gemm.d_attn_output": 21.4321,
    "moe_router_topk": 6.85206,
    "ds.gemm.e_indexed_moe": 86.8759,
    "ds.gemm.e_swiglu": 65.8601,
    "ds.gemm.edown": 37.4755,
    "ds.gemm.s_swiglu": 23.9368,
    "ds.gemm.sdown_add": 16.6682,
    "moe_accumulate": 6.08179,
}


def load_summary(path: Path | None) -> tuple[dict[str, float], str]:
    metrics = dict(DEFAULT_METRICS)
    subtitle = "Default path: 62.74 tokens/s"
    if path is None:
        return metrics, subtitle

    profiler: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("kind") == "profiler_summary":
                profiler = obj
                break
            if "items" in obj:
                profiler = obj
                break

    if profiler is None:
        raise ValueError(f"no profiler_summary found in {path}")

    for item in profiler.get("items", []):
        name = item.get("name")
        total_ms = item.get("total_ms")
        if isinstance(name, str) and isinstance(total_ms, (int, float)):
            metrics[name] = float(total_ms)

    tps = profiler.get("decode_tokens_per_sec")
    avg = profiler.get("decode_avg_ms_per_token")
    if isinstance(tps, (int, float)) and isinstance(avg, (int, float)):
        subtitle = f"Profile input: {path.name}, {tps:.2f} tokens/s, avg {avg:.2f} ms/token"
    else:
        subtitle = f"Profile input: {path.name}"
    return metrics, subtitle


def fmt_ms(v: float) -> str:
    return f"{v:.2f} ms"


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def rect(cls: str, x: float, y: float, w: float, h: float, rx: float = 10) -> str:
    return f'<rect class="stage {cls}" x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx:g}"/>'


def label(x: float, y: float, text: str, cls: str = "label") -> str:
    return f'<text class="{cls}" x="{x:g}" y="{y:g}">{esc(text)}</text>'


def metric(m: dict[str, float], key: str, fallback: float = 0.0) -> float:
    return float(m.get(key, fallback))


def build_svg(metrics: dict[str, float], subtitle_value: str) -> str:
    scale = 2.2
    h = lambda key, fallback=0.0, min_h=12.0: max(min_h, metric(metrics, key, fallback) * scale)

    prefill_h = h("prefill")
    decode_h = 226
    mla_h = h("ds.decode.mla_forward")
    moe_h = h("ds.decode.mlp.moe_forward")
    lm_h = h("lm_head")
    d_attn_q_h = h("ds.gemm.d_attn_q")
    d_attn_out_h = h("ds.gemm.d_attn_output")
    router_h = h("moe_router_topk", min_h=14)
    e_indexed_h = h("ds.gemm.e_indexed_moe")
    e_swiglu_h = h("ds.gemm.e_swiglu")
    edown_h = h("ds.gemm.edown")
    s_swiglu_h = h("ds.gemm.s_swiglu")
    sdown_h = h("ds.gemm.sdown_add")
    accum_h = h("moe_accumulate", min_h=14)

    lines: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1360" viewBox="0 0 1600 1360">',
        "  <defs>",
        '    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        '      <path d="M0,0 L0,6 L9,3 z" fill="#ff8a4c"/>',
        "    </marker>",
        '    <marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        '      <path d="M0,0 L0,6 L9,3 z" fill="#62d0ff"/>',
        "    </marker>",
        "    <style>",
        "      .bg { fill: #0b1020; }",
        "      .panel { fill: #121a2f; stroke: #26324d; stroke-width: 1; rx: 18; }",
        '      .title { fill: #f5f7fb; font: 700 28px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }',
        '      .subtitle { fill: #aab4c8; font: 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }',
        "      .axis { stroke: #3b4663; stroke-width: 1.2; }",
        "      .grid { stroke: #24304a; stroke-width: 1; stroke-dasharray: 4 6; }",
        "      .label { fill: #dce3f2; font: 13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }",
        "      .small-label { fill: #dce3f2; font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }",
        "      .value { fill: #f5f7fb; font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }",
        '      .note { fill: #8995ad; font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }',
        '      .column-title { fill: #f5f7fb; font: 700 15px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }',
        "      .stage { stroke: rgba(255,255,255,0.18); stroke-width: 1; }",
        "      .prefill { fill: #f5c542; }",
        "      .decode { fill: #8b7cff; }",
        "      .moe { fill: #ff8a4c; }",
        "      .mla { fill: #62d0ff; }",
        "      .lm { fill: #7ddc8a; }",
        "      .router { fill: #ffb36e; }",
        "      .shared { fill: #ffa56d; }",
        "      .pill { fill: #1d2944; stroke: #33405f; }",
        '      .pill-text { fill: #dce3f2; font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }',
        "    </style>",
        "  </defs>",
        '  <rect class="bg" width="1600" height="1360"/>',
        '  <text class="title" x="48" y="52">DeepSeek Flame Timeline</text>',
        '  <text class="subtitle" x="48" y="78">从左到右：大粒度 -&gt; 小粒度；从上到下：执行先后顺序；矩形高度：total_ms</text>',
        '  <rect class="pill" x="48" y="102" width="300" height="34" rx="17"/>',
        f'  <text class="pill-text" x="66" y="124">{esc(subtitle_value)}</text>',
        '  <rect class="pill" x="368" y="102" width="270" height="34" rx="17"/>',
        f'  <text class="pill-text" x="386" y="124">Scale: 1 ms = {scale:.1f} px height</text>',
        '  <rect class="pill" x="658" y="102" width="390" height="34" rx="17"/>',
        '  <text class="pill-text" x="676" y="124">Nested timers are separated by granularity column</text>',
        '  <rect class="pill" x="1068" y="102" width="214" height="34" rx="17"/>',
        '  <text class="pill-text" x="1086" y="124">Generated by tools script</text>',
        '  <rect class="panel" x="40" y="160" width="1520" height="1080" rx="18"/>',
        label(72, 196, "Coarse Phase", "column-title"),
        label(340, 196, "Decode Stage", "column-title"),
        label(610, 196, "Stage Detail", "column-title"),
        label(900, 196, "Component", "column-title"),
        label(1200, 196, "Leaf Hotspots", "column-title"),
        '  <line class="grid" x1="300" y1="215" x2="300" y2="1215"/>',
        '  <line class="grid" x1="570" y1="215" x2="570" y2="1215"/>',
        '  <line class="grid" x1="860" y1="215" x2="860" y2="1215"/>',
        '  <line class="grid" x1="1160" y1="215" x2="1160" y2="1215"/>',
        '  <line class="axis" x1="72" y1="215" x2="1488" y2="215"/>',
        '  <line class="axis" x1="72" y1="1215" x2="1488" y2="1215"/>',
        '  <path d="M72 142 L1488 142" stroke="#ff8a4c" stroke-width="3" marker-end="url(#arrow)"/>',
        label(600, 136, "granularity: coarse -> fine", "note"),
        '  <path d="M1518 230 L1518 1215" stroke="#62d0ff" stroke-width="3" marker-end="url(#arrow-blue)"/>',
        '  <text class="note" x="1462" y="730" transform="rotate(90 1462 730)">execution order: earlier -> later</text>',
        "  <!-- Coarse phase column. -->",
        rect("prefill", 72, 230, 210, prefill_h),
        label(92, 260, "prefill"),
        label(92, 282, fmt_ms(metric(metrics, "prefill")), "value"),
        rect("decode", 72, 592, 210, decode_h),
        label(92, 622, "decode window"),
        label(92, 644, "sampled hot path", "value"),
        "  <!-- Decode stage column, ordered by layer execution sequence. -->",
        rect("mla", 340, 592, 190, mla_h),
        label(360, 622, "MLA forward"),
        label(360, 644, fmt_ms(metric(metrics, "ds.decode.mla_forward")), "value"),
        rect("moe", 340, 742, 190, moe_h),
        label(360, 772, "MLP / MoE"),
        label(360, 794, fmt_ms(metric(metrics, "ds.decode.mlp.moe_forward")), "value"),
        label(360, 816, "largest stage", "note"),
        rect("lm", 340, 1140, 190, lm_h),
        label(360, 1168, "LM head"),
        label(360, 1190, fmt_ms(metric(metrics, "lm_head")), "value"),
        "  <!-- Stage detail column. -->",
        rect("mla", 610, 592, 220, d_attn_q_h),
        label(630, 621, "d_attn_q", "small-label"),
        label(630, 641, fmt_ms(metric(metrics, "ds.gemm.d_attn_q")), "value"),
        rect("mla", 610, 670, 220, d_attn_out_h),
        label(630, 698, "d_attn_output", "small-label"),
        label(630, 714, fmt_ms(metric(metrics, "ds.gemm.d_attn_output")), "value"),
        rect("moe", 610, 742, 220, moe_h),
        label(630, 772, "moe_forward"),
        label(630, 794, fmt_ms(metric(metrics, "ds.decode.mlp.moe_forward")), "value"),
        label(630, 816, "dominant decode component", "note"),
        "  <!-- Component and leaf hotspots. -->",
        rect("router", 900, 742, 220, router_h, rx=7),
        label(1132, 754, f"router/topk {fmt_ms(metric(metrics, 'moe_router_topk'))}", "small-label"),
        rect("moe", 900, 770, 220, e_indexed_h),
        label(920, 802, "e_indexed_moe", "small-label"),
        label(920, 822, fmt_ms(metric(metrics, "ds.gemm.e_indexed_moe")), "value"),
        label(920, 842, "routed expert fast path", "note"),
        rect("moe", 1200, 770, 250, e_swiglu_h),
        label(1220, 800, "e_swiglu", "small-label"),
        label(1220, 820, fmt_ms(metric(metrics, "ds.gemm.e_swiglu")), "value"),
        rect("moe", 1200, 935, 250, edown_h),
        label(1220, 965, "edown", "small-label"),
        label(1220, 985, fmt_ms(metric(metrics, "ds.gemm.edown")), "value"),
        rect("shared", 900, 980, 220, s_swiglu_h),
        label(920, 1010, "shared s_swiglu", "small-label"),
        label(920, 1030, fmt_ms(metric(metrics, "ds.gemm.s_swiglu")), "value"),
        rect("shared", 900, 1050, 220, sdown_h),
        label(920, 1076, f"shared sdown_add {fmt_ms(metric(metrics, 'ds.gemm.sdown_add'))}", "small-label"),
        rect("moe", 900, 1105, 220, accum_h, rx=7),
        label(1132, 1117, f"accumulate {fmt_ms(metric(metrics, 'moe_accumulate'))}", "small-label"),
        '  <rect class="panel" x="40" y="1260" width="1520" height="72" rx="18"/>',
        '  <circle cx="72" cy="1294" r="6" fill="#f5c542"/>',
        label(90, 1299, "Yellow: prefill", "note"),
        '  <circle cx="212" cy="1294" r="6" fill="#62d0ff"/>',
        label(230, 1299, "Blue: MLA", "note"),
        '  <circle cx="332" cy="1294" r="6" fill="#ff8a4c"/>',
        label(350, 1299, "Orange: MoE / routed experts", "note"),
        '  <circle cx="570" cy="1294" r="6" fill="#7ddc8a"/>',
        label(588, 1299, "Green: LM head", "note"),
        label(760, 1299, "Note: nested profiler timers are not additive; compare heights within the same granularity column.", "note"),
        "</svg>",
        "",
    ]
    return "\n".join("  " + line if line and not line.startswith("<svg") else line for line in lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", type=Path, help="profile_deepseek_summary.json or JSONL")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="output SVG path")
    args = parser.parse_args()

    metrics, subtitle = load_summary(args.summary)
    svg = build_svg(metrics, subtitle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
