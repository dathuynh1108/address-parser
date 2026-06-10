import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from parser import AddressParser


DEFAULT_OUTPUT_PATH = Path("data/generated_full_dataset_regression_cases.json")
SYNTHETIC_STREET_TEMPLATE = "Số {number} Đường Kiểm Thử"


def _sorted_codes(records: Dict[str, Dict[str, Any]]) -> List[str]:
    return sorted(records.keys(), key=lambda code: int(str(code)))


def _synthetic_street(number: int) -> str:
    return SYNTHETIC_STREET_TEMPLATE.format(number=number)


def _component_snapshot(component: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not component:
        return None
    return {
        "id": str(component.get("id")),
        "code": str(component.get("code")),
        "full_name": str(component.get("full_name")),
    }


def _required_component(component: Optional[Dict[str, Any]], label: str) -> Dict[str, Any]:
    if not component:
        raise ValueError(f"Missing {label} component while building regression cases")
    return component


def _build_old_case(
    parser: AddressParser,
    ward_code: str,
    sequence: int,
) -> Dict[str, Any]:
    ward_entry = _required_component(parser.old_ward_records.get(ward_code), "old ward")
    district_code = str(ward_entry.get("parent_code") or "")
    district_entry = _required_component(
        parser.old_district_records.get(district_code),
        "old district",
    )
    province_code = str(district_entry.get("parent_code") or "")

    components = parser.get_address_components_from_ids(
        province_id=province_code,
        district_id=district_code,
        ward_id=ward_code,
        is_new_format=False,
    )
    province = _required_component(components["province"], "old province")
    district = _required_component(components["district"], "old district")
    ward = _required_component(components["ward"], "old ward")

    street = _synthetic_street(100000 + sequence)
    address = ", ".join(
        [street, ward["full_name"], district["full_name"], province["full_name"]]
    )

    return {
        "case_id": f"old-{ward_code}",
        "address": address,
        "expected": {
            "format": "old",
            "is_new": False,
            "street_address": street,
            "province": _component_snapshot(province),
            "district": _component_snapshot(district),
            "ward": _component_snapshot(ward),
        },
    }


def _build_old_compact_numeric_ward_case(
    parser: AddressParser,
    ward_code: str,
    sequence: int,
) -> Optional[Dict[str, Any]]:
    ward_entry = _required_component(parser.old_ward_records.get(ward_code), "old ward")
    ward_name = str(ward_entry.get("full_name") or ward_entry.get("name") or "")
    ward_match = re.fullmatch(r"Phường\s+(\d+)", ward_name, flags=re.IGNORECASE)
    if not ward_match:
        return None

    district_code = str(ward_entry.get("parent_code") or "")
    district_entry = _required_component(
        parser.old_district_records.get(district_code),
        "old district",
    )
    province_code = str(district_entry.get("parent_code") or "")

    components = parser.get_address_components_from_ids(
        province_id=province_code,
        district_id=district_code,
        ward_id=ward_code,
        is_new_format=False,
    )
    province = _required_component(components["province"], "old province")
    district = _required_component(components["district"], "old district")
    ward = _required_component(components["ward"], "old ward")

    street = _synthetic_street(300000 + sequence)
    district_name = str(district["name"])
    province_name = str(province["name"])
    if district_name.isdigit() or district_name == province_name:
        district_label = str(district["full_name"])
    else:
        district_label = district_name
    address = ", ".join(
        [street, f"P.{ward_match.group(1)} {district_label}", province["full_name"]]
    )

    return {
        "case_id": f"old-compact-numeric-ward-{ward_code}",
        "address": address,
        "expected": {
            "format": "old",
            "is_new": False,
            "street_address": street,
            "province": _component_snapshot(province),
            "district": _component_snapshot(district),
            "ward": _component_snapshot(ward),
        },
    }


def _build_new_case(
    parser: AddressParser,
    ward_code: str,
    sequence: int,
) -> Dict[str, Any]:
    ward_entry = _required_component(parser.new_ward_records.get(ward_code), "new ward")
    province_code = str(ward_entry.get("parent_code") or "")

    components = parser.get_address_components_from_ids(
        province_id=province_code,
        ward_id=ward_code,
        is_new_format=True,
    )
    province = _required_component(components["province"], "new province")
    ward = _required_component(components["ward"], "new ward")

    street = _synthetic_street(200000 + sequence)
    address = ", ".join([street, ward["full_name"], province["full_name"]])

    return {
        "case_id": f"new-{ward_code}",
        "address": address,
        "expected": {
            "format": "new",
            "is_new": True,
            "street_address": street,
            "province": _component_snapshot(province),
            "district": None,
            "ward": _component_snapshot(ward),
        },
    }


def build_regression_cases(parser: Optional[AddressParser] = None) -> Dict[str, Any]:
    parser = parser or AddressParser()

    old_cases = [
        _build_old_case(parser, ward_code, sequence)
        for sequence, ward_code in enumerate(_sorted_codes(parser.old_ward_records), start=1)
    ]
    old_compact_numeric_ward_cases = [
        case
        for sequence, ward_code in enumerate(
            _sorted_codes(parser.old_ward_records),
            start=1,
        )
        if (
            case := _build_old_compact_numeric_ward_case(parser, ward_code, sequence)
        )
        is not None
    ]
    new_cases = [
        _build_new_case(parser, ward_code, sequence)
        for sequence, ward_code in enumerate(_sorted_codes(parser.new_ward_records), start=1)
    ]
    old_cases.extend(old_compact_numeric_ward_cases)

    return {
        "metadata": {
            "old_case_count": len(old_cases),
            "old_compact_numeric_ward_case_count": len(
                old_compact_numeric_ward_cases
            ),
            "new_case_count": len(new_cases),
            "total_case_count": len(old_cases) + len(new_cases),
        },
        "old_cases": old_cases,
        "new_cases": new_cases,
    }


def write_regression_cases(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    parser: Optional[AddressParser] = None,
) -> Dict[str, Any]:
    payload = build_regression_cases(parser=parser)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build exhaustive old/new parser regression cases from the dataset."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to write the generated regression case JSON.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    payload = write_regression_cases(output_path)
    metadata = payload["metadata"]
    print(
        "Wrote "
        f"{metadata['total_case_count']} cases "
        f"({metadata['old_case_count']} old, {metadata['new_case_count']} new) "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
