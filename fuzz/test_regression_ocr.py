from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from address_parser import AddressParser


def _name(component):
    return (component or {}).get("name") if isinstance(component, dict) else None


def _check_case(parser: AddressParser, address: str, expected: dict) -> None:
    parsed = parser.process(address)
    actual = {
        "province": _name(parsed.get("province")),
        "district": _name(parsed.get("district")),
        "ward": _name(parsed.get("ward")),
        "street": parsed.get("street_address"),
        "format": parsed.get("format"),
        "is_new": parsed.get("is_new"),
    }
    for key, value in expected.items():
        assert actual.get(key) == value, (
            f"Mismatch for {address!r} on {key}: expected {value!r}, got {actual.get(key)!r}.\n"
            f"Actual parsed: {json.dumps(actual, ensure_ascii=False)}"
        )


def _extract_street(address: str, province: str, district: str, ward: str, aliases: dict) -> str:
    parser = AddressParser.__new__(AddressParser)
    node = AddressParser.AddressNode(province, district, ward)
    return parser._extract_street_address(address, node, aliases)


def test_street_extractor_regressions() -> None:
    cases = [
        (
            "Thôn 1B Hòa Tiến, Krông Pắc, Đắk Lắk",
            {
                "province": ["Đắk Lắk", "Tỉnh Đắk Lắk"],
                "district": ["Krông Pắc", "Huyện Krông Pắc"],
                "ward": ["Xã Hoà Tiến", "Xã Hòa Tiến", "Hoà Tiến", "Hòa Tiến"],
            },
            ("Đắk Lắk", "Krông Pắc", "Xã Hoà Tiến"),
            "Thôn 1B",
        ),
        (
            "Ấp 5 An Lục Long, Châu Thành, Long An",
            {
                "province": ["Long An", "Tỉnh Long An"],
                "district": ["Châu Thành", "Huyện Châu Thành"],
                "ward": ["Xã An Lục Long", "An Lục Long"],
            },
            ("Long An", "Châu Thành", "Xã An Lục Long"),
            "Ấp 5",
        ),
        (
            "Đường Hòa Tiến, Krông Pắc, Đắk Lắk",
            {
                "province": ["Đắk Lắk", "Tỉnh Đắk Lắk"],
                "district": ["Krông Pắc", "Huyện Krông Pắc"],
                "ward": ["Xã Hoà Tiến", "Xã Hòa Tiến", "Hoà Tiến", "Hòa Tiến"],
            },
            ("Đắk Lắk", "Krông Pắc", "Xã Hoà Tiến"),
            "Đường Hòa Tiến",
        ),
        (
            "50 Nguyễn Trãi Dịch Vọng Hà Nội",
            {
                "province": ["Hà Nội", "Thành phố Hà Nội"],
                "district": ["Cầu Giấy", "Quận Cầu Giấy"],
                "ward": ["Phường Dịch Vọng", "Dịch Vọng"],
            },
            ("Hà Nội", "Cầu Giấy", "Phường Dịch Vọng"),
            "50 Nguyễn Trãi",
        ),
    ]
    for address, aliases, components, expected in cases:
        province, district, ward = components
        actual = _extract_street(address, province, district, ward, aliases)
        assert actual == expected, (
            f"Street extraction mismatch for {address!r}: expected {expected!r}, got {actual!r}"
        )


def test_ocr_regressions() -> None:
    parser = AddressParser()
    cases = [
        (
            "Kỳ Sơn, Xã Tân Yên, Tỉnh Bắc Ninh, Việt Nam",
            {
                "province": "Bắc Ninh",
                "district": None,
                "ward": "Xã Tân Yên",
                "format": "new",
                "is_new": True,
            },
        ),
        (
            "Kỳ Sơn, X. Tân Yên, Bắc Ninh",
            {
                "province": "Bắc Ninh",
                "district": None,
                "ward": "Xã Tân Yên",
                "street": "Kỳ Sơn",
                "format": "new",
                "is_new": True,
            },
        ),
        (
            "Thôn Làng Hạ, X. Quan Sơn, Tỉnh Lạng Sơn, Việt Nam",
            {
                "province": "Lạng Sơn",
                "ward": "Xã Quan Sơn",
                "street": "Thôn Làng Hạ",
            },
        ),
        (
            "Số 156/8 Ấp Cầu Đúc Xã An Lục Long Huyện Châu Thành Tỉnh Long An Việt Nam",
            {
                "province": "Long An",
                "district": "Châu Thành",
                "ward": "Xã An Lục Long",
                "street": "Số 156/8 Ấp Cầu Đúc",
                "format": "old",
                "is_new": False,
            },
        ),
        (
            "50 Nguyễn Trãi Dịch Vọng Hà Nội",
            {
                "province": "Hà Nội",
                "district": "Cầu Giấy",
                "ward": "Phường Dịch Vọng",
                "street": "50 Nguyễn Trãi",
                "format": "old",
                "is_new": False,
            },
        ),
        (
            "123 Nguyễn Trãi Dịch Vọng Hà Nội",
            {
                "province": "Hà Nội",
                "district": "Cầu Giấy",
                "ward": "Phường Dịch Vọng",
                "street": "123 Nguyễn Trãi",
                "format": "old",
                "is_new": False,
            },
        ),
    ]
    for address, expected in cases:
        _check_case(parser, address, expected)


if __name__ == "__main__":
    test_street_extractor_regressions()
    test_ocr_regressions()
    print(json.dumps({"status": "ok", "suite": "ocr_regressions"}, ensure_ascii=False))
