from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from inexus_parser import AddressParser


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


def _is_old_format(parsed: Dict[str, Any]) -> bool:
    is_new = parsed.get("is_new")
    fmt = parsed.get("format")
    return is_new is False or fmt == "old"


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
        help="Path to JSON file (default: fuzz/test_data/old_addresses.json)",
    )
    ap.add_argument("--start", type=int, default=0, help="0-based start index")
    ap.add_argument(
        "--limit", type=int, default=0, help="Only process first N cases (0 = all)"
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="district_prefix",
        choices=("district_prefix", "non_old"),
        help="district_prefix = only flag is_new=True when district prefix present; non_old = flag any non-old parse",
    )
    ap.add_argument(
        "--show",
        type=int,
        default=5,
        help="Show up to N non-old cases (0 = none)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Write non-old cases to JSONL file",
    )
    ap.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print full parsed output for shown cases",
    )
    args = ap.parse_args()

    default_path = Path(__file__).parent / "test_data" / "old_addresses.json"
    path = Path(args.file).expanduser() if args.file else default_path

    addresses = _load_addresses(path)
    if not addresses:
        print(f"No addresses found in {path}")
        return 1

    parser = AddressParser()

    start_idx = max(args.start or 0, 0)
    if start_idx:
        addresses = addresses[start_idx:]

    limit = max(args.limit or 0, 0)
    if limit:
        addresses = addresses[:limit]

    t0 = time.time()
    flagged: List[Dict[str, Any]] = []
    cases_with_district_prefix = 0
    cases_without_district_prefix = 0
    no_prefix_is_new_true = 0
    no_prefix_is_new_false = 0
    no_prefix_is_new_none = 0

    for idx, addr in enumerate(addresses, start=start_idx + 1):
        parsed = parser.process(addr)
        fmt = parsed.get("format")
        is_new = parsed.get("is_new")

        if args.mode == "district_prefix":
            segments = parser._split_address_segments(addr)
            has_district_prefix = False
            for segment_std, _ in segments:
                if not segment_std:
                    continue
                tokens = [tok for tok in segment_std.split() if tok]
                if not tokens:
                    continue
                first = tokens[0]
                if first in {"huyen", "h", "quan", "q", "tx"}:
                    has_district_prefix = True
                    break
                if len(tokens) >= 2 and f"{tokens[0]} {tokens[1]}" == "thi xa":
                    has_district_prefix = True
                    break
                if first == "tp" and len(tokens) >= 2:
                    city_key = " ".join(tokens[1:])
                    if city_key and not _is_province_level_city(parser, city_key):
                        has_district_prefix = True
                        break
                if (
                    len(tokens) >= 3
                    and f"{tokens[0]} {tokens[1]}" == "thanh pho"
                ):
                    city_key = " ".join(tokens[2:])
                    if city_key and not _is_province_level_city(parser, city_key):
                        has_district_prefix = True
                        break
            if has_district_prefix:
                cases_with_district_prefix += 1
            else:
                cases_without_district_prefix += 1
                if is_new is True:
                    no_prefix_is_new_true += 1
                elif is_new is False:
                    no_prefix_is_new_false += 1
                else:
                    no_prefix_is_new_none += 1
            if not has_district_prefix or is_new is not True:
                continue
            reason = "is_new=True but district prefix present"
        else:
            if _is_old_format(parsed):
                continue
            if fmt != "old":
                reason = f"format={fmt!r}"
            else:
                reason = f"is_new={is_new!r}"

        flagged.append(
            {
                "i": idx,
                "mst_address": addr,
                "reason": reason,
                "parsed": parsed,
            }
        )

    elapsed_s = time.time() - t0
    summary: Dict[str, Any] = {
        "cases": len(addresses),
        "elapsed_s": round(elapsed_s, 3),
    }
    if args.mode == "district_prefix":
        summary["cases_with_district_prefix"] = cases_with_district_prefix
        summary["cases_without_district_prefix"] = cases_without_district_prefix
        summary["no_prefix_is_new_true"] = no_prefix_is_new_true
        summary["no_prefix_is_new_false"] = no_prefix_is_new_false
        summary["no_prefix_is_new_none"] = no_prefix_is_new_none
        summary["failures"] = len(flagged)
    else:
        summary["non_old"] = len(flagged)
    print(json.dumps(summary, ensure_ascii=False))

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in flagged:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.show and args.show > 0 and flagged:
        indent = 2 if args.pretty else None
        for row in flagged[: args.show]:
            parsed = row.get("parsed") or {}
            prov = _get_component_name(parsed.get("province"))
            dist = _get_component_name(parsed.get("district"))
            ward = _get_component_name(parsed.get("ward"))
            print("-" * 40)
            print(f"#{row['i']} {row['reason']}")
            print(row["mst_address"])
            if args.pretty:
                print(json.dumps(parsed, ensure_ascii=False, indent=indent))
            else:
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
