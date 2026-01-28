from __future__ import annotations

# Kept for ad-hoc experiments; main test runner is `fuzz/test.py`.

import argparse
import json
import sys
from typing import Dict, List

from parser import AddressParser


tests = [
    {
        "mst_address": "Huyện Phú Quốc",
    },
]

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Only process first N cases")
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Write dataset JSONL (id + mst_address) for tests/eval_parsers.py",
    )
    ap.add_argument(
        "--summary",
        action="store_true",
        help="Print only summary stats instead of full parsed output",
    )
    ap.add_argument(
        "--show",
        type=int,
        default=5,
        help="Show up to N suspicious cases (0 = none)",
    )
    args = ap.parse_args()

    parser = AddressParser()

    addresses: List[str] = [
        t.get("mst_address") for t in tests if isinstance(t, dict) and t.get("mst_address")
    ]
    if args.limit and args.limit > 0:
        addresses = addresses[: args.limit]

    def _is_province_level_city(city_key: str) -> bool:
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
        return bool(isinstance(record, dict) and record.get("administrative_unit_id") == 1)

    def _has_district_prefix(addr: str) -> bool:
        for segment_std, _ in parser._split_address_segments(addr):
            if not segment_std:
                continue
            tokens = [tok for tok in segment_std.split() if tok]
            if not tokens:
                continue
            first = tokens[0]
            if first in {"huyen", "h", "quan", "q", "tx"}:
                return True
            if len(tokens) >= 2 and f"{tokens[0]} {tokens[1]}" == "thi xa":
                return True
            if first == "tp" and len(tokens) >= 2:
                city_tokens = tokens[1:]
                while city_tokens and city_tokens[-1] in {"viet", "nam", "vietnam"}:
                    city_tokens.pop()
                if city_tokens and not _is_province_level_city(" ".join(city_tokens)):
                    return True
            if len(tokens) >= 3 and f"{tokens[0]} {tokens[1]}" == "thanh pho":
                city_tokens = tokens[2:]
                while city_tokens and city_tokens[-1] in {"viet", "nam", "vietnam"}:
                    city_tokens.pop()
                if city_tokens and not _is_province_level_city(" ".join(city_tokens)):
                    return True
        return False

    fmt_counts: Dict[str, int] = {"new": 0, "old": 0, "unknown": 0}
    prefix_counts: Dict[str, int] = {"with_prefix": 0, "no_prefix": 0}
    suspicious: List[Dict] = []

    for i, mst_address in enumerate(addresses, start=1):
        parsed = parser.process(mst_address)
        fmt = parsed.get("format")
        if fmt == "new":
            fmt_counts["new"] += 1
        elif fmt == "old":
            fmt_counts["old"] += 1
        else:
            fmt_counts["unknown"] += 1

        has_prefix = _has_district_prefix(mst_address)
        if has_prefix:
            prefix_counts["with_prefix"] += 1
        else:
            prefix_counts["no_prefix"] += 1

        # "Suspicious" heuristics for quick manual review
        if has_prefix and parsed.get("is_new") is True:
            suspicious.append(
                {
                    "i": i,
                    "reason": "is_new=True but district prefix present",
                    "mst_address": mst_address,
                    "parsed": parsed,
                }
            )
        elif (not has_prefix) and parsed.get("is_new") is False:
            ward_id = (parsed.get("ward") or {}).get("id")
            ward_key = parser._normalize_id_token(ward_id)
            is_legacy_only_ward = bool(
                ward_key
                and ward_key in parser.old_ward_records
                and ward_key not in parser.new_ward_records
            )
            if not is_legacy_only_ward:
                suspicious.append(
                    {
                        "i": i,
                        "reason": "is_new=False but no district prefix",
                        "mst_address": mst_address,
                        "parsed": parsed,
                    }
                )

        if not args.summary:
            print(f"Address: {mst_address}")
            print(json.dumps(parsed, ensure_ascii=False, indent=2))
            print("-" * 40)

    if args.summary:
        print(
            json.dumps(
                {
                    "cases": len(addresses),
                    "format_counts": fmt_counts,
                    "district_prefix_counts": prefix_counts,
                    "suspicious": len(suspicious),
                },
                ensure_ascii=False,
            )
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for idx, addr in enumerate(addresses, start=1):
                f.write(
                    json.dumps({"id": idx, "mst_address": addr}, ensure_ascii=False) + "\n"
                )

    if args.show and args.show > 0 and suspicious:
        for row in suspicious[: args.show]:
            parsed = row.get("parsed") or {}
            prov = (parsed.get("province") or {}).get("name")
            dist = (parsed.get("district") or {}).get("name")
            ward = (parsed.get("ward") or {}).get("name")
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
