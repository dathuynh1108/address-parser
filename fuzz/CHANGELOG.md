# Changelog

All notable updates to the address parser are documented here. Examples show **before** (raw input) and **after** (parsed output) so you can see how the parser normalizes Vietnamese addresses.

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
