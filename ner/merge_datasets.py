#!/usr/bin/env python
"""Merge multiple NER JSONL datasets into unified train/test splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def merge_files(file_paths: Iterable[Path]) -> List[dict]:
    merged: List[dict] = []
    for file_path in file_paths:
        if not file_path.exists():
            continue
        merged.extend(read_jsonl(file_path))
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-files",
        type=Path,
        nargs="+",
        required=True,
        help="List of JSONL files whose contents will be merged prior to splitting.",
    )
    parser.add_argument(
        "--test-files",
        type=Path,
        nargs="*",
        default=None,
        help="Optional additional JSONL files that will be merged before re-splitting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ner/datasets/combined"),
        help="Directory where merged train/test files will be written.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="When no test-files are provided, this ratio controls how much of the merged data stays in train.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    all_paths: List[Path] = list(args.train_files)
    if args.test_files:
        all_paths.extend(args.test_files)

    merged_rows = merge_files(all_paths)
    if not merged_rows:
        raise SystemExit("No rows were loaded from the provided files.")

    if not args.no_shuffle:
        rng.shuffle(merged_rows)

    if len(merged_rows) == 1:
        train_rows = merged_rows[:]
        test_rows = merged_rows[:]
    else:
        split_idx = int(len(merged_rows) * args.train_ratio)
        split_idx = max(1, min(split_idx, len(merged_rows) - 1))
        train_rows = merged_rows[:split_idx]
        test_rows = merged_rows[split_idx:]

    if not args.no_shuffle:
        rng.shuffle(train_rows)
        rng.shuffle(test_rows)

    output_train = args.output_dir / "train.jsonl"
    output_test = args.output_dir / "test.jsonl"

    write_jsonl(output_train, train_rows)
    write_jsonl(output_test, test_rows)

    print(f"Train rows: {len(train_rows)} -> {output_train}")
    print(f"Test rows:  {len(test_rows)} -> {output_test}")


if __name__ == "__main__":
    main()
