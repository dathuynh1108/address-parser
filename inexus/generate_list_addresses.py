from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT = BASE_DIR / "list_addresses.jsonl"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _index_by_code(records: Iterable[Dict[str, Any]], code_field: str = "code") -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        code = record.get(code_field)
        if not code:
            continue
        indexed[str(code)] = record
    return indexed


def _pick_name(record: Dict[str, Any] | None) -> str | None:
    if not record:
        return None
    return record.get("full_name") or record.get("name")


def _sort_dict_by_key(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: data[key] for key in sorted(data.keys(), key=str)}


def _build_mapping_payload(
    ward_mappings: List[Dict[str, Any]],
    old_wards: Dict[str, Dict[str, Any]],
    old_districts: Dict[str, Dict[str, Any]],
    old_provinces: Dict[str, Dict[str, Any]],
    new_wards: Dict[str, Dict[str, Any]],
    new_provinces: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    old_to_new: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    new_to_old: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in ward_mappings:
        old_code = str(row.get("old_ward_code") or "").strip()
        new_code = str(row.get("new_ward_code") or "").strip()
        if not old_code and not new_code:
            continue

        old_ward = old_wards.get(old_code) if old_code else None
        old_district = (
            old_districts.get(str(old_ward.get("district_code")))
            if old_ward and old_ward.get("district_code")
            else None
        )
        old_province = (
            old_provinces.get(str(old_district.get("province_code")))
            if old_district and old_district.get("province_code")
            else None
        )

        new_ward = new_wards.get(new_code) if new_code else None
        new_province = (
            new_provinces.get(str(new_ward.get("province_code")))
            if new_ward and new_ward.get("province_code")
            else None
        )

        entry = {
            "city_id_old": str(old_district.get("province_code")) if old_district and old_district.get("province_code") else None,
            "district_id_old": str(old_ward.get("district_code")) if old_ward and old_ward.get("district_code") else None,
            "ward_id_old": old_code or None,
            "city_id_new": str(new_ward.get("province_code")) if new_ward and new_ward.get("province_code") else None,
            "ward_id_new": new_code or None,
            "old_ward_name": row.get("old_ward_name") or _pick_name(old_ward),
            "new_ward_name": row.get("new_ward_name") or _pick_name(new_ward),
            "old_province_name": row.get("old_province_name") or _pick_name(old_province),
            "new_province_name": row.get("new_province_name") or _pick_name(new_province),
            "old_district_name": row.get("old_district_name") or _pick_name(old_district),
        }

        # Drop None values for cleaner payloads
        entry = {key: value for key, value in entry.items() if value is not None}

        if old_code:
            old_to_new[old_code].append(entry)
        if new_code:
            new_to_old[new_code].append(entry)

    return dict(old_to_new), dict(new_to_old)


def build_payload() -> Dict[str, Any]:
    old_provinces = _index_by_code(_load_json(DATA_DIR / "old_provinces.json"))
    old_districts = _index_by_code(_load_json(DATA_DIR / "old_districts.json"))
    old_wards = _index_by_code(_load_json(DATA_DIR / "old_wards.json"))

    new_provinces = _index_by_code(_load_json(DATA_DIR / "provinces.json"))
    new_wards = _index_by_code(_load_json(DATA_DIR / "wards.json"))

    ward_mappings = _load_json(DATA_DIR / "ward_mappings.json")
    old_to_new_map, new_to_old_map = _build_mapping_payload(
        ward_mappings,
        old_wards,
        old_districts,
        old_provinces,
        new_wards,
        new_provinces,
    )

    return {
        "old": {
            "provinces": _sort_dict_by_key(old_provinces),
            "districts": _sort_dict_by_key(old_districts),
            "wards": _sort_dict_by_key(old_wards),
        },
        "new": {
            "provinces": _sort_dict_by_key(new_provinces),
            "wards": _sort_dict_by_key(new_wards),
        },
        "mapping": {
            "ward_old_to_new": _sort_dict_by_key(old_to_new_map),
            "ward_new_to_old": _sort_dict_by_key(new_to_old_map),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate list_addresses.jsonl payload.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination path for the generated JSON (default: inexus/list_addresses.jsonl).",
    )
    args = parser.parse_args()

    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Generated {args.output}")


if __name__ == "__main__":
    main()
