from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fuzz.parser import AddressParser
from ner.build_real_dataset import collect_aliases, extract_aliases, extract_address, iter_json_objects
from ner.build_standard_dataset import label_tokens


def _component_present(address: str, label: str, values: Sequence[Optional[str]]) -> bool:
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


def _iter_or_default(values: List[str], fallback: Optional[str]):
    if values:
        for item in values:
            yield item
    elif fallback:
        yield fallback


def label_address(parser: AddressParser, address: str):
    parsed_result = parser.process(address)
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
        return {
            "status": "skip",
            "reason": "missing province or ward",
            "parsed": parsed_result,
        }

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

    province_candidates = list(_iter_or_default(province_aliases, province))
    ward_candidates = list(_iter_or_default(ward_aliases, ward))
    district_candidates = list(_iter_or_default(district_aliases, district))
    if not district_candidates:
        district_candidates = [None]

    district_present = bool(district and _component_present(address, "DISTRICT", district_candidates))
    street_present = bool(street and _component_present(address, "STREET", [street]))

    required_matches = ["PROVINCE", "WARD"]
    if district_present:
        required_matches.append("DISTRICT")
    if street_present:
        required_matches.append("STREET")

    last_candidate = None
    for province_value in province_candidates:
        for ward_value in ward_candidates:
            for district_value in district_candidates:
                last_candidate = label_tokens(
                    address,
                    street=street,
                    province=province_value,
                    district=district_value,
                    ward=ward_value,
                )
                if all(last_candidate.matches.get(key, False) for key in required_matches):
                    return {
                        "status": "ok",
                        "matches": last_candidate.matches,
                        "required": required_matches,
                        "parsed": parsed_result,
                    }

    return {
        "status": "fail",
        "matches": last_candidate.matches if last_candidate else {},
        "required": required_matches,
        "parsed": parsed_result,
        "candidates": {
            "province": province_candidates,
            "district": district_candidates,
            "ward": ward_candidates,
        },
    }


def load_addresses(path: Path, field: str, limit: int) -> List[str]:
    if not path.exists():
        return []
    addresses: List[str] = []
    for entry in iter_json_objects(path, mode="stream"):
        address = extract_address(entry, field=field)
        if not address:
            continue
        addresses.append(address)
        if limit and len(addresses) >= limit:
            break
    return addresses


def _build_dataset_map(root: Path) -> Dict[str, Path]:
    datasets = {
        "new": root / "fuzz" / "test_data" / "new_addresses.json",
        "old": root / "fuzz" / "test_data" / "old_addresses.json",
    }
    genai = root / "fuzz" / "test_data" / "genai_mst_public_masothue.json"
    if genai.exists():
        datasets["genai"] = genai
    return datasets


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        description="Sample fuzz datasets and validate labeling coverage."
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset to scan: new, old, genai, or all.",
    )
    ap.add_argument("--field", type=str, default="mst_address")
    ap.add_argument("--limit", type=int, default=0, help="Limit rows loaded per dataset.")
    ap.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Random sample size per dataset (0 = use all).",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show", type=int, default=5, help="Show up to N failures.")
    args = ap.parse_args()

    dataset_map = _build_dataset_map(REPO_ROOT)
    if args.dataset == "all":
        selected = list(dataset_map.items())
    else:
        if args.dataset not in dataset_map:
            print(f"Unknown dataset: {args.dataset}", file=sys.stderr)
            return 2
        selected = [(args.dataset, dataset_map[args.dataset])]

    parser = AddressParser()
    rng = random.Random(args.seed)

    for name, path in selected:
        addresses = load_addresses(path, args.field, args.limit)
        if not addresses:
            print(json.dumps({"dataset": name, "status": "empty"}, ensure_ascii=False))
            continue
        if args.sample and args.sample < len(addresses):
            addresses = rng.sample(addresses, args.sample)
        stats = {"ok": 0, "fail": 0, "skip": 0}
        failures = []
        for address in addresses:
            result = label_address(parser, address)
            stats[result["status"]] += 1
            if result["status"] != "ok" and len(failures) < args.show:
                parsed = result.get("parsed") or {}
                failures.append(
                    {
                        "address": address,
                        "status": result["status"],
                        "required": result.get("required"),
                        "matches": result.get("matches"),
                        "province": (parsed.get("province") or {}).get("name"),
                        "district": (parsed.get("district") or {}).get("name"),
                        "ward": (parsed.get("ward") or {}).get("name"),
                        "street": parsed.get("street_address"),
                    }
                )

        print(
            json.dumps(
                {
                    "dataset": name,
                    "count": len(addresses),
                    "stats": stats,
                },
                ensure_ascii=False,
            )
        )
        for row in failures:
            print(json.dumps(row, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
