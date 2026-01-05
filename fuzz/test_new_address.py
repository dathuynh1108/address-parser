from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from parser import AddressParser


def _is_province_level_city(parser: AddressParser, city_key: str) -> bool:
    city_key = (city_key or "").strip()
    if not city_key:
        return False
    normalized = parser._detect_special_province_token(city_key) or city_key
    province_id_new = parser._lookup_new_province_id_by_name(normalized)
    if not province_id_new:
        return False
    record = (
        parser.external_new_province_records.get(province_id_new)
        or parser.new_province_records.get(province_id_new)
    )
    return bool(
        isinstance(record, dict) and record.get("administrative_unit_id") == 1
    )


def _load_addresses(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and isinstance(data.get("addresses"), list):
        rows = data["addresses"]
    else:
        raise ValueError(
            f"Unsupported JSON shape in {path}: expected list or {{'addresses': [...]}}"
        )

    addresses: List[str] = []
    for row in rows:
        if isinstance(row, str):
            addr = row
        elif isinstance(row, dict):
            addr = row.get("mst_address") or row.get("address") or row.get("raw")
        else:
            addr = None

        if isinstance(addr, str) and addr.strip():
            addresses.append(addr.strip())

    return addresses


def _get_component_name(component: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(component, dict):
        return None
    name = component.get("name")
    return name if isinstance(name, str) and name.strip() else None


def _has_district_prefix(parser: AddressParser, address: str) -> bool:
    raw_lower = address.lower()
    full_std = parser.standardize_name(address, False)
    tokens_full = [tok for tok in full_std.split() if tok]
    has_province_segment = False
    for segment_std, _ in parser._split_address_segments(address):
        if not segment_std:
            continue
        if segment_std in parser.province_names_std and not segment_std.startswith(
            ("tp ", "thanh pho ")
        ):
            has_province_segment = True
            break
    for idx, tok in enumerate(tokens_full):
        if tok in {"quan", "q", "tx"}:
            return True
        if tok == "dac" and idx + 1 < len(tokens_full) and tokens_full[idx + 1] == "khu":
            return True
        if tok in {"huyen", "h"}:
            prev = tokens_full[idx - 1] if idx > 0 else ""
            if prev == "duong":
                continue
            if "huyện" in raw_lower or re.search(r"\bhuyen\b", raw_lower):
                return True
        if tok == "thi" and idx + 1 < len(tokens_full) and tokens_full[idx + 1] == "xa":
            return True

    for segment_std, _ in parser._split_address_segments(address):
        if not segment_std:
            continue
        tokens = [tok for tok in segment_std.split() if tok]
        if not tokens:
            continue
        for i, prefix in enumerate(tokens):
            if prefix == "tp" and i + 1 < len(tokens):
                city_tokens = tokens[i + 1 :]
                while city_tokens and city_tokens[-1] in {"viet", "nam", "vietnam"}:
                    city_tokens.pop()
                city_key = " ".join(city_tokens)
                if city_key and (
                    has_province_segment
                    or not _is_province_level_city(parser, city_key)
                ):
                    return True
            if (
                prefix == "thanh"
                and i + 2 < len(tokens)
                and tokens[i + 1] == "pho"
            ):
                city_tokens = tokens[i + 2 :]
                while city_tokens and city_tokens[-1] in {"viet", "nam", "vietnam"}:
                    city_tokens.pop()
                city_key = " ".join(city_tokens)
                if city_key and (
                    has_province_segment
                    or not _is_province_level_city(parser, city_key)
                ):
                    return True
    return False


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--file",
        type=str,
        default="",
        help="Path to JSON file (default: fuzz/test_data/new_addresses.json)",
    )
    ap.add_argument("--start", type=int, default=0, help="0-based start index")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N cases (0 = all)")
    ap.add_argument(
        "--strict-prefix",
        action="store_true",
        help="Fail if any address contains a district prefix (Huyện/Quận/Tx...)",
    )
    ap.add_argument(
        "--show",
        type=int,
        default=5,
        help="Show up to N failing cases (0 = none)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Write failing cases to JSONL file",
    )
    args = ap.parse_args()

    default_path = Path(__file__).parent / "test_data" / "new_addresses.json"
    path = Path(args.file).expanduser() if args.file else default_path

    addresses = _load_addresses(path)
    if not addresses:
        print(f"No addresses found in {path}")
        return 1

    start_idx = max(args.start or 0, 0)
    if start_idx:
        addresses = addresses[start_idx:]

    limit = max(args.limit or 0, 0)
    if limit:
        addresses = addresses[:limit]

    parser = AddressParser()
    t0 = time.time()

    failures: List[Dict[str, Any]] = []
    cases_with_district_prefix = 0

    for idx, addr in enumerate(addresses, start=start_idx + 1):
        parsed = parser.process(addr)
        is_new = parsed.get("is_new")
        fmt = parsed.get("format")
        has_prefix = _has_district_prefix(parser, addr)
        if has_prefix:
            cases_with_district_prefix += 1
            if args.strict_prefix:
                failures.append(
                    {
                        "i": idx,
                        "mst_address": addr,
                        "reason": "district prefix present in new dataset",
                        "parsed": parsed,
                    }
                )
                continue

        # For curated "new" dataset: when there is no district prefix, the parser must
        # return new format (2-level) and not invent a district.
        #
        # Exception: some real-world "new-looking" inputs may contain wards that only exist
        # in the legacy registry; in those cases the parser may recover the district from
        # old metadata and mark the address as old format.
        if not has_prefix and is_new is not True and fmt != "new":
            ward_id = (parsed.get("ward") or {}).get("id")
            ward_key = parser._normalize_id_token(ward_id)
            if (
                ward_key
                and ward_key in parser.old_ward_records
                and ward_key not in parser.new_ward_records
            ):
                continue
            failures.append(
                {
                    "i": idx,
                    "mst_address": addr,
                    "reason": f"not_new: format={fmt!r} is_new={is_new!r}",
                    "parsed": parsed,
                }
            )

    elapsed_s = time.time() - t0
    summary = {
        "cases": len(addresses),
        "cases_with_district_prefix": cases_with_district_prefix,
        "failures": len(failures),
        "elapsed_s": round(elapsed_s, 3),
    }
    print(json.dumps(summary, ensure_ascii=False))

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in failures:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.show and args.show > 0 and failures:
        for row in failures[: args.show]:
            parsed = row.get("parsed") or {}
            prov = _get_component_name(parsed.get("province"))
            dist = _get_component_name(parsed.get("district"))
            ward = _get_component_name(parsed.get("ward"))
            print("-" * 40)
            print(f"#{row['i']} {row['reason']}")
            print(row["mst_address"])
            print(
                json.dumps(
                    {
                        "province": prov,
                        "district": dist,
                        "ward": ward,
                        "format": parsed.get("format"),
                        "is_new": parsed.get("is_new"),
                    },
                    ensure_ascii=False,
                )
            )

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
