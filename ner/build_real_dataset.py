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

from fuzz.parser import AddressParser
from ner.build_standard_dataset import label_tokens, clean_text, strip_accents

JsonValue = Union[Dict[str, Any], str]


def detect_file_kind(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig") as handle:
        while True:
            char = handle.read(1)
            if not char:
                return "empty"
            if char == "\ufeff":
                continue
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
    with path.open("r", encoding="utf-8-sig") as handle:
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
    with path.open("r", encoding="utf-8-sig") as handle:
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

        decoder = json.JSONDecoder()
        buffer = ""
        in_array = False
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            buffer += chunk
            idx = 0
            if not in_array:
                while idx < len(buffer) and buffer[idx].isspace():
                    idx += 1
                if idx < len(buffer) and buffer[idx] == "\ufeff":
                    idx += 1
                if idx < len(buffer) and buffer[idx] == "[":
                    idx += 1
                    in_array = True
                else:
                    buffer = buffer[idx:]
                    continue
            while True:
                while idx < len(buffer) and buffer[idx].isspace():
                    idx += 1
                if idx < len(buffer) and buffer[idx] == ",":
                    idx += 1
                    continue
                if idx < len(buffer) and buffer[idx] == "]":
                    return
                if idx >= len(buffer):
                    break
                try:
                    item, end = decoder.raw_decode(buffer, idx)
                except json.JSONDecodeError:
                    break
                yield item
                idx = end
            buffer = buffer[idx:]

        idx = 0
        if not in_array:
            while idx < len(buffer) and buffer[idx].isspace():
                idx += 1
            if idx < len(buffer) and buffer[idx] == "\ufeff":
                idx += 1
            if idx < len(buffer) and buffer[idx] == "[":
                idx += 1
                in_array = True
        while in_array:
            while idx < len(buffer) and buffer[idx].isspace():
                idx += 1
            if idx < len(buffer) and buffer[idx] == ",":
                idx += 1
                continue
            if idx < len(buffer) and buffer[idx] == "]":
                return
            if idx >= len(buffer):
                return
            try:
                item, end = decoder.raw_decode(buffer, idx)
            except json.JSONDecodeError:
                return
            yield item
            idx = end


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
            address = extract_address(entry, field=field)
            if not address:
                continue
            if limit is not None and count >= limit:
                break
            count += 1
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


TYPE_ALIASES = {
    "province": (
        (("tinh",), ("Tinh", "T.")),
        (("thanh", "pho"), ("TP.", "TP")),
    ),
    "district": (
        (("quan",), ("Q.", "Q")),
        (("huyen",), ("H.", "H")),
        (("thi", "xa"), ("TX.", "TX")),
        (("thi", "tran"), ("TT.", "TT")),
        (("thanh", "pho"), ("TP.", "TP")),
    ),
    "ward": (
        (("phuong",), ("P.", "P")),
        (("xa",), ("X.", "X")),
        (("thi", "tran"), ("TT.", "TT")),
    ),
}


def _normalize_token(text: str) -> str:
    return strip_accents(text or "").lower()


def _add_abbr_variants(abbr: str, body: str, dest: List[str]) -> None:
    abbr_base = abbr.rstrip(".")
    bodies = [body] if body else [""]
    for current_body in bodies:
        if current_body:
            compact = current_body.replace(" ", "")
            dest.extend(
                [
                    f"{abbr} {current_body}".strip(),
                    f"{abbr}{current_body}",
                    f"{abbr}{compact}",
                ]
            )
        else:
            dest.append(abbr)
        if abbr.endswith(".") and abbr_base != abbr:
            if current_body:
                compact = current_body.replace(" ", "")
                dest.extend(
                    [
                        f"{abbr_base} {current_body}".strip(),
                        f"{abbr_base}{current_body}",
                        f"{abbr_base}{compact}",
                    ]
                )
            else:
                dest.append(abbr_base)


def expand_component_alias(name: Optional[str], *, kind: str) -> List[str]:
    if not name:
        return []
    cleaned = clean_text(name, remove_slash=False)
    tokens = cleaned.split()
    norm_tokens = [_normalize_token(tok) for tok in tokens]
    abbrs: List[str] = []
    drop = 0
    for prefix_tokens, candidates in TYPE_ALIASES.get(kind, ()):
        if norm_tokens[: len(prefix_tokens)] == list(prefix_tokens):
            abbrs = list(candidates)
            drop = len(prefix_tokens)
            break
    rest_tokens = tokens[drop:]
    rest = " ".join(rest_tokens).strip()
    variants: List[str] = [cleaned]
    for abbr in abbrs:
        _add_abbr_variants(abbr, rest, variants)
    seen = set()
    deduped = []
    for item in variants:
        normalized_item = clean_text(item, remove_slash=False)
        if normalized_item and normalized_item not in seen:
            deduped.append(normalized_item)
            seen.add(normalized_item)
    return deduped


def collect_aliases(
    primary: Optional[str],
    full_name: Optional[str],
    *,
    kind: str,
    extra_aliases: Optional[Iterable[str]] = None,
) -> List[str]:
    bases: List[str] = []
    for candidate in (full_name, primary):
        if candidate and candidate not in bases:
            bases.append(candidate)
    aliases: List[str] = []
    for base in bases:
        aliases.extend(expand_component_alias(base, kind=kind))
    if extra_aliases:
        for alias in extra_aliases:
            if isinstance(alias, str):
                cleaned = clean_text(alias, remove_slash=False)
                if cleaned:
                    aliases.append(cleaned)
    seen: set[str] = set()
    merged: List[str] = []
    for alias in aliases:
        if alias not in seen:
            merged.append(alias)
            seen.add(alias)
    return merged


def extract_aliases(entry: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(entry, dict):
        return []
    result: List[str] = []

    def _add(raw: Optional[Union[str, List[str]]]) -> None:
        if isinstance(raw, str):
            candidate = raw.strip()
            if candidate:
                result.append(candidate)
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    candidate = item.strip()
                    if candidate:
                        result.append(candidate)

    _add(entry.get("aliases"))
    _add(entry.get("legacy_names"))
    return result


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
    log_skipped: bool,
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
            province_entry = parsed_result.get("province") or {}
            district_entry = parsed_result.get("district") or {}
            ward_entry = parsed_result.get("ward") or {}

            province = province_entry.get("name")
            district = district_entry.get("name")
            ward = ward_entry.get("name")
            province_full = province_entry.get("full_name")
            district_full = district_entry.get("full_name")
            ward_full = ward_entry.get("full_name")
            street = parsed_result.get("street_address")
            if not (province and ward):
                continue
            parser_hits += 1
            province_aliases = collect_aliases(
                province,
                province_full,
                kind="province",
                extra_aliases=extract_aliases(province_entry),
            )
            district_aliases = collect_aliases(
                district,
                district_full,
                kind="district",
                extra_aliases=extract_aliases(district_entry),
            )
            ward_aliases = collect_aliases(
                ward,
                ward_full,
                kind="ward",
                extra_aliases=extract_aliases(ward_entry),
            )

            def _iter_or_default(values: List[str], fallback: Optional[str]) -> Iterator[str]:
                if values:
                    yield from values
                elif fallback:
                    yield fallback

            province_candidates = list(_iter_or_default(province_aliases, province))
            ward_candidates = list(_iter_or_default(ward_aliases, ward))
            district_candidates = list(_iter_or_default(district_aliases, district))
            if not district_candidates:
                district_candidates = [None]

            def _component_present(label: str, values: List[Optional[str]]) -> bool:
                key_map = {
                    "PROVINCE": "province",
                    "DISTRICT": "district",
                    "WARD": "ward",
                    "STREET": "street",
                }
                key = key_map[label]
                for value in values:
                    if not value:
                        continue
                    candidate = label_tokens(address, **{key: value})
                    if candidate.matches.get(label):
                        return True
                return False

            district_present = bool(district and _component_present("DISTRICT", district_candidates))
            street_present = bool(street and _component_present("STREET", [street]))

            labeling: Optional[Any] = None
            required_matches = ["PROVINCE", "WARD"]
            if district_present:
                required_matches.append("DISTRICT")
            if street_present:
                required_matches.append("STREET")

            for province_value in province_candidates:
                for ward_value in ward_candidates:
                    for district_value in district_candidates:
                        candidate = label_tokens(
                            address,
                            street=street,
                            province=province_value,
                            district=district_value,
                            ward=ward_value,
                        )
                        if all(candidate.matches.get(key, False) for key in required_matches):
                            labeling = candidate
                            break
                    if labeling:
                        break
                if labeling:
                    break

            if not labeling:
                if log_skipped:
                    skip_payload = {
                        "event": "skip",
                        "address": address,
                        "parsed": parsed_result,
                        "candidates": {
                            "province": province_candidates,
                            "district": district_candidates,
                            "ward": ward_candidates,
                        },
                    }
                    print(json.dumps(skip_payload, ensure_ascii=False))
                    
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
    parser.add_argument(
        "--log-skipped",
        action="store_true",
        help="Print parser outputs for addresses that fail labeling",
    )
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
        log_skipped=args.log_skipped,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
