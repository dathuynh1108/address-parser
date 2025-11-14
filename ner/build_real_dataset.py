#!/usr/bin/env python
"""Convert raw address dumps into token-level NER supervision using the fuzzy parser."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fuzz.inexus_parser import AddressParser
from ner.build_standard_dataset import label_tokens, clean_text

JsonValue = Union[Dict[str, Any], str]


def detect_file_kind(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        while True:
            char = handle.read(1)
            if not char:
                return "empty"
            if char.isspace():
                continue
            if char == "[":
                return "json_array"
            return "json_lines"


def iter_json_objects(path: Path, mode: str) -> Iterator[JsonValue]:
    kind = detect_file_kind(path)
    if kind == "json_array":
        yield from _iter_json_array(path, mode=mode)
    elif kind == "json_lines":
        yield from _iter_json_lines(path)
    else:
        return


def _iter_json_lines(path: Path) -> Iterator[JsonValue]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line in {"[", "]", ","}:
                continue
            if line.endswith(","):
                line = line[:-1].rstrip()
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _iter_json_array(path: Path, mode: str) -> Iterator[JsonValue]:
    with path.open("r", encoding="utf-8") as handle:
        if mode == "memory":
            # Read the entire file into memory
            content = handle.read()
            try:
                data = json.loads(content)
                print(f"Loaded {len(data)} records into memory from {path}", file=sys.stderr)
                if isinstance(data, list):
                    for item in data:
                        yield item
            except json.JSONDecodeError:
                return
            return
        
        # Stream parsing
        buffer: list[str] = []
        depth = 0
        capturing = False
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line == ",":
                continue
            if line.startswith("]"):
                break
            if not capturing:
                brace_idx = line.find("{")
                if brace_idx == -1:
                    continue
                line = line[brace_idx:]
                capturing = True
            buffer.append(line)
            depth += line.count("{")
            depth -= line.count("}")
            if depth == 0 and buffer:
                text = " ".join(buffer).strip()
                if text.endswith(","):
                    text = text[:-1].rstrip()
                try:
                    yield json.loads(text)
                except json.JSONDecodeError:
                    pass
                buffer = []
                capturing = False


def extract_address(entry: JsonValue, *, field: str) -> Optional[str]:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        value = entry.get(field)
        if isinstance(value, str):
            return value.strip()
    return None


def load_addresses(
    path: Path,
    *,
    field: str,
    limit: Optional[int],
    mode: str,
    batch_size: Optional[int],
) -> Iterable[Union[str, List[str]]]:
    def generator() -> Iterator[str]:
        count = 0
        for entry in iter_json_objects(path, mode):
            if limit is not None and count >= limit:
                break
            count += 1
            address = extract_address(entry, field=field)
            if address:
                yield address

    if mode == "memory":
        return list(generator())
    if mode == "batch":
        if not batch_size or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer when load_mode='batch'")

        def batch_iter() -> Iterator[List[str]]:
            batch: List[str] = []
            for address in generator():
                batch.append(address)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        return batch_iter()
    return generator()


def write_record(handle, record: Dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False))
    handle.write("\n")


def build_dataset(
    *,
    address_file: Path,
    address_field: str,
    output_dir: Path,
    train_ratio: float,
    limit: Optional[int],
    load_mode: str,
    batch_size: Optional[int],
    seed: int,
) -> Dict[str, Any]:
    parser = AddressParser()
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_real.jsonl"
    test_path = output_dir / "test_real.jsonl"

    total = parser_hits = labeled = 0
    train_count = test_count = 0
    addresses = load_addresses(
        address_file,
        field=address_field,
        limit=limit,
        mode=load_mode,
        batch_size=batch_size,
    )

    with train_path.open("w", encoding="utf-8") as train_handle, test_path.open("w", encoding="utf-8") as test_handle:
        if load_mode == "batch":
            def address_iterator():
                for chunk in addresses:  # type: ignore
                    for addr in chunk:
                        yield addr
            iterator = address_iterator()
        else:
            iterator = iter(addresses) if not isinstance(addresses, list) else iter(addresses)

        for address in iterator:
            total += 1
            try:
                parsed_result = parser.process(address)
            except Exception:
                continue
            province = (parsed_result.get("province") or {}).get("name")
            district = (parsed_result.get("district") or {}).get("name")
            ward = (parsed_result.get("ward") or {}).get("name")
            street = parsed_result.get("street_address")
            if not (province and district and ward):
                continue
            parser_hits += 1
            labeling = label_tokens(address, street=street, province=province, district=district, ward=ward)
            required_matches = ["PROVINCE", "DISTRICT", "WARD"]
            if street:
                required_matches.append("STREET")
            if not all(labeling.matches.get(key, False) for key in required_matches):
                continue
            tokens = labeling.tokens
            tags = labeling.ner_tags
            matches = labeling.matches
            text = clean_text(address, remove_slash=False)
            labeled += 1
            payload = {
                "id": f"real_{labeled}",
                "text": text,
                "tokens": tokens,
                "ner_tags": tags,
                "matches": matches,
            }
            if rng.random() < train_ratio:
                write_record(train_handle, payload)
                train_count += 1
            else:
                write_record(test_handle, payload)
                test_count += 1

            if total % 1000 == 0:
                print(f"Processed {total} addresses...")
    
    print("Build done")    
    
    return {
        "total": total,
        "parser_hits": parser_hits,
        "labeled": labeled,
        "train": train_count,
        "test": test_count,
        "train_path": str(train_path),
        "test_path": str(test_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--address-file",
        type=Path,
        default=Path("ner/datasets/addresses.jsonl"),
        help="JSON array or JSONL file containing raw address records.",
    )
    parser.add_argument(
        "--address-field",
        type=str,
        default="mst_address",
        help="Name of the field that stores the address string inside each JSON object.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ner/datasets/real"),
        help="Directory where the labeled train/test splits will be written.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Probability of sending a labeled sample to the train split (rest goes to test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many address rows to read from the source file.",
    )
    parser.add_argument(
        "--load-mode",
        choices=("memory", "stream", "batch"),
        default="memory",
        help="memory: load all addresses before parsing (faster, higher RAM). batch: load chunks of addresses into memory. stream: process lazily for huge files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="How many addresses to read into memory per batch when load-mode=batch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_dataset(
        address_file=args.address_file,
        address_field=args.address_field,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        limit=args.limit,
        load_mode=args.load_mode,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
