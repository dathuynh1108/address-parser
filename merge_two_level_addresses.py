#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LIST_ADDRESS = BASE_DIR / "list_address.json"
DEFAULT_TWO_LEVEL = BASE_DIR / "genai_public_standard_address_vn_new.json"
PLACEHOLDER_DISTRICT = ""

PROVINCE_OVERRIDES = {
    "Huế": "Thừa Thiên Huế",
}

PROVINCE_PREFIXES = (
    "Thành phố trực thuộc trung ương ",
    "Thành phố ",
    "Tỉnh ",
    "TP. ",
)


def normalize_province(name: str) -> str:
    stripped = name.strip()
    for prefix in PROVINCE_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):].strip()
            break
    return PROVINCE_OVERRIDES.get(stripped, stripped)


def merge_addresses(existing: dict, flat_records: list) -> int:
    added = 0

    for record in flat_records:
        province_raw = record.get("province")
        ward = record.get("ward")

        if not province_raw or not ward:
            continue

        province_name = normalize_province(province_raw)
        province_entry = existing.setdefault(province_name, {})

        # Skip if ward already present in any district for this province
        if any(ward in wards for wards in province_entry.values()):
            continue

        wards_list = province_entry.setdefault(PLACEHOLDER_DISTRICT, [])
        if ward in wards_list:
            continue

        wards_list.append(ward)
        added += 1

    return added


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge two-level address records (province + ward) into the 3-level "
            "list_address.json structure by placing them under an empty district key."
        )
    )
    parser.add_argument(
        "--list-address",
        type=Path,
        default=DEFAULT_LIST_ADDRESS,
        help="Path to the existing list_address.json file to update in place.",
    )
    parser.add_argument(
        "--two-level",
        type=Path,
        default=DEFAULT_TWO_LEVEL,
        help="Path to the two-level address JSON file to merge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output path. When omitted, the merged data overwrites "
            "the --list-address file."
        ),
    )

    args = parser.parse_args()

    list_address_path = args.list_address
    two_level_path = args.two_level
    output_path = args.output or list_address_path

    with list_address_path.open(encoding="utf-8") as fh:
        list_address = json.load(fh)

    with two_level_path.open(encoding="utf-8") as fh:
        two_level = json.load(fh)

    if not isinstance(list_address, dict):
        raise TypeError(f"{list_address_path} must contain a JSON object")

    if not isinstance(two_level, list):
        raise TypeError(f"{two_level_path} must contain a JSON array")

    added = merge_addresses(list_address, two_level)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(list_address, fh, ensure_ascii=False, indent=2)

    print(f"Added {added} new wards under the '{PLACEHOLDER_DISTRICT}' district placeholder.")
    print(f"Merged data written to {output_path}")


if __name__ == "__main__":
    main()
