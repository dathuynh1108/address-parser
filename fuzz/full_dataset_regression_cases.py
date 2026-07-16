import argparse
import json
import re
from pathlib import Path
from typing import Literal, cast

from address_parser import AddressParser
from address_parser.contracts import (
    AddressCode,
    AdministrativeRecord,
    AdministrativeRecordsByCode,
    ComponentSnapshot,
    ProjectedAddressComponent,
    RegressionCase,
    RegressionCorpus,
)

DEFAULT_OUTPUT_PATH = Path(__file__).with_name("generated_full_dataset_regression_cases.json")
SYNTHETIC_STREET_TEMPLATE = "Số {number} Đường Kiểm Thử"


class _Arguments(argparse.Namespace):
    output: str


def _sorted_codes(records: AdministrativeRecordsByCode) -> list[AddressCode]:
    return sorted(records, key=_numeric_code_sort_key)


def _numeric_code_sort_key(code: AddressCode) -> int:
    return int(code)


def _synthetic_street(number: int) -> str:
    return SYNTHETIC_STREET_TEMPLATE.format(number=number)


def _component_snapshot(
    component: ProjectedAddressComponent | None,
) -> ComponentSnapshot | None:
    if not component:
        return None
    return {
        "id": _required_component_text(component, "id", "snapshot"),
        "code": _required_component_text(component, "code", "snapshot"),
        "full_name": _required_component_text(component, "full_name", "snapshot"),
    }


def _required_component_text(
    component: ProjectedAddressComponent,
    field: Literal["id", "code", "name", "full_name"],
    label: str,
) -> str:
    value = component[field]
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {label} {field} while building regression cases")
    return value


def _required_record(
    component: AdministrativeRecord | None,
    label: str,
) -> AdministrativeRecord:
    if not component:
        raise ValueError(f"Missing {label} component while building regression cases")
    return component


def _required_projected_component(
    component: ProjectedAddressComponent | None,
    label: str,
) -> ProjectedAddressComponent:
    if not component:
        raise ValueError(f"Missing {label} component while building regression cases")
    return component


def _build_old_case(
    parser: AddressParser,
    ward_code: AddressCode,
    sequence: int,
) -> RegressionCase:
    ward_entry = _required_record(parser.old_ward_records.get(ward_code), "old ward")
    district_code = str(ward_entry.get("parent_code") or "")
    district_entry = _required_record(
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
    province = _required_projected_component(components["province"], "old province")
    district = _required_projected_component(components["district"], "old district")
    ward = _required_projected_component(components["ward"], "old ward")

    street = _synthetic_street(100000 + sequence)
    address = ", ".join(
        [
            street,
            _required_component_text(ward, "full_name", "old ward"),
            _required_component_text(district, "full_name", "old district"),
            _required_component_text(province, "full_name", "old province"),
        ]
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
    ward_code: AddressCode,
    sequence: int,
) -> RegressionCase | None:
    ward_entry = _required_record(parser.old_ward_records.get(ward_code), "old ward")
    ward_name = str(ward_entry.get("full_name") or ward_entry.get("name") or "")
    ward_match = re.fullmatch(r"Phường\s+(\d+)", ward_name, flags=re.IGNORECASE)
    if not ward_match:
        return None

    district_code = str(ward_entry.get("parent_code") or "")
    district_entry = _required_record(
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
    province = _required_projected_component(components["province"], "old province")
    district = _required_projected_component(components["district"], "old district")
    ward = _required_projected_component(components["ward"], "old ward")

    street = _synthetic_street(300000 + sequence)
    district_name = _required_component_text(district, "name", "old district")
    province_name = _required_component_text(province, "name", "old province")
    if district_name.isdigit() or district_name == province_name:
        district_label = _required_component_text(district, "full_name", "old district")
    else:
        district_label = district_name
    address = ", ".join(
        [
            street,
            f"P.{ward_match.group(1)} {district_label}",
            _required_component_text(province, "full_name", "old province"),
        ]
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
    ward_code: AddressCode,
    sequence: int,
) -> RegressionCase:
    ward_entry = _required_record(parser.new_ward_records.get(ward_code), "new ward")
    province_code = str(ward_entry.get("parent_code") or "")

    components = parser.get_address_components_from_ids(
        province_id=province_code,
        ward_id=ward_code,
        is_new_format=True,
    )
    province = _required_projected_component(components["province"], "new province")
    ward = _required_projected_component(components["ward"], "new ward")

    street = _synthetic_street(200000 + sequence)
    address = ", ".join(
        [
            street,
            _required_component_text(ward, "full_name", "new ward"),
            _required_component_text(province, "full_name", "new province"),
        ]
    )

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


def build_regression_cases(parser: AddressParser | None = None) -> RegressionCorpus:
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
        if (case := _build_old_compact_numeric_ward_case(parser, ward_code, sequence)) is not None
    ]
    new_cases = [
        _build_new_case(parser, ward_code, sequence)
        for sequence, ward_code in enumerate(_sorted_codes(parser.new_ward_records), start=1)
    ]
    old_cases.extend(old_compact_numeric_ward_cases)

    return {
        "metadata": {
            "old_case_count": len(old_cases),
            "old_compact_numeric_ward_case_count": len(old_compact_numeric_ward_cases),
            "new_case_count": len(new_cases),
            "total_case_count": len(old_cases) + len(new_cases),
        },
        "old_cases": old_cases,
        "new_cases": new_cases,
    }


def write_regression_cases(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    parser: AddressParser | None = None,
) -> RegressionCorpus:
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
    args = cast(_Arguments, parser.parse_args())

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
