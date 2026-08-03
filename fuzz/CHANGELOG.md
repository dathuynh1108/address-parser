# Changelog

All notable updates to the address parser are documented here. Examples show **before** (raw input) and **after** (parsed output) so you can see how the parser normalizes Vietnamese addresses.

---

## Update 2026-08-03

- Added an optional Cython native wheel kernel for packed n-gram candidate
  ranking; the public parser input and `ParseResult` contracts are unchanged.
- Moved deterministic normalization into a strict typed module with a bounded
  cache and precomputed stable fuzzy-match/province alias profiles.
- Added explicit native availability and startup requirement checks. Production
  builds can fail fast with `VN_ADDRESS_PARSER_NATIVE=required`.
- Bumped the wheel version to `0.2.0` so native and earlier Python-only artifacts
  cannot collide in package caches.

---

## Update 2026-07-15

- Package is now installed and imported as `address_parser`.
- Public parser, mapping, lookup, and search contracts are fully typed in
  `address_parser.contracts` and distributed with `py.typed`.
- Removed ambiguous top-level imports such as `from parser import AddressParser`;
  use `from address_parser import AddressParser`.
- Added explicit `normalize_address_code()` and `get_administrative_record()`
  boundaries for consumers that need registry access.

---

## Update 2026-03-07
- Fix ward_code is None for case like this address:
```
Số 27, Ngõ 92 đường Nguyễn Khánh Toàn, Phường Quan Hoa, Quận Cầu Giấy, Thành phố Hà Nội, Việt Nam
```

- Previous output:
```
{
    "province": {
        "name": "Hà Nội",
        "id": "01",
        "full_name": "Thành phố Hà Nội",
        "code": "01",
        "aliases": ["Hà Nội", "Thành phố Hà Nội", "ha noi"],
    },
    "district": {
        "name": "Cầu Giấy",
        "id": "005",
        "full_name": "Quận Cầu Giấy",
        "code": "005",
        "aliases": ["Cầu Giấy", "Quận Cầu Giấy", "cau giay"],
    },
    "ward": {"name": "Phường Quan Hoa", "aliases": ["Phường Quan Hoa", "quan hoa"]},
    "street_address": "Số 27 Ngõ 92 đường Nguyễn Khánh Toàn",
    "format": "old",
    "is_new": False,
}
```

- After output:
```
{
    "province": {
        "name": "Hà Nội",
        "id": "01",
        "full_name": "Thành phố Hà Nội",
        "code": "01",
        "aliases": ["Hà Nội", "Thành phố Hà Nội", "ha noi"],
    },
    "district": {
        "name": "Cầu Giấy",
        "id": "005",
        "full_name": "Quận Cầu Giấy",
        "code": "005",
        "aliases": ["Cầu Giấy", "Quận Cầu Giấy", "cau giay"],
    },
    "ward": {
        "name": "Phường Quan Hoa",
        "id": "00169",
        "full_name": "Phường Quan Hoa",
        "code": "00169",
        "aliases": ["Phường Quan Hoa", "Quan Hoa", "quan hoa"],
    },
    "street_address": "Số 27 Ngõ 92 đường Nguyễn Khánh Toàn",
    "format": "old",
    "is_new": False,
}
```

- Fix ward_code is None for case where old_ward_id == new_ward_id:
```
Trang trại cam vinh Kỳ Yến, xóm Thọ Thành, Xã Minh Hợp, Huyện Quỳ Hợp, Tỉnh Nghệ An, Việt Nam
```
