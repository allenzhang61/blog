#!/usr/bin/env python3
"""Export GPT-2 BPE ranks for the C++ tokenizer tests."""

from pathlib import Path

import tiktoken


def main() -> None:
    enc = tiktoken.get_encoding("gpt2")
    data_dir = Path(__file__).resolve().parents[1] / "data"
    ranks_out = data_dir / "gpt2_bpe_ranks.tsv"

    rank_rows = []
    for token_bytes, rank in sorted(enc._mergeable_ranks.items(), key=lambda item: item[1]):
        rank_rows.append(f"{token_bytes.hex()}\t{rank}")
    ranks_out.write_text("\n".join(rank_rows) + "\n", encoding="utf-8")
    print(ranks_out)


if __name__ == "__main__":
    main()
