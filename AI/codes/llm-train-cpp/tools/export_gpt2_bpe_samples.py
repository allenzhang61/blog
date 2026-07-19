#!/usr/bin/env python3
"""Export GPT-2 BPE ranks and reference samples for the C++ tokenizer tests."""

from pathlib import Path

import tiktoken


SAMPLES = [
    "Hello, world!",
    "Every effort moves you",
    "Every effort moves you forward. Every step teaches.",
    "GPT-2 BPE",
]


def escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


def main() -> None:
    enc = tiktoken.get_encoding("gpt2")
    data_dir = Path(__file__).resolve().parents[1] / "data"
    ranks_out = data_dir / "gpt2_bpe_ranks.tsv"
    sample_out = data_dir / "gpt2_bpe_samples.tsv"

    rank_rows = []
    for token_bytes, rank in sorted(enc._mergeable_ranks.items(), key=lambda item: item[1]):
        rank_rows.append(f"{token_bytes.hex()}\t{rank}")
    ranks_out.write_text("\n".join(rank_rows) + "\n", encoding="utf-8")

    rows = []
    for sample in SAMPLES:
        rows.append(f"{escape(sample)}\t{','.join(str(x) for x in enc.encode(sample))}")
    sample_out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(ranks_out)
    print(sample_out)


if __name__ == "__main__":
    main()
