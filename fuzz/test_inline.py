from __future__ import annotations

# Kept for ad-hoc experiments; main test runner is `fuzz/test.py`.

import argparse
import json
import sys
from typing import Dict, List

from parser import AddressParser


tests = [
    {
        "mst_address": "Số 20, ngõ 151, đường Hồng Hà, Phường Hồng Hà, TP Hà Nội, Việt Nam"
    },
    {"mst_address": "Lô P, KCN Bình Xuyên, Xã Bình Nguyên, Tỉnh Phú Thọ, Việt Nam"},
    {
        "mst_address": "Số 29, Lô 03, Khu đất dịch vụ DV03, Phường Hà Đông, TP Hà Nội, Việt Nam"
    },
    {"mst_address": "Khu 11 Khải Xuân, Xã Đông Thành, Tỉnh Phú Thọ, Việt Nam"},
    {
        "mst_address": "158/58 Vũng Việt, Khu Phố Đông Chiêu, Phường Dĩ An, TP Hồ Chí Minh, Việt Nam"
    },
    {"mst_address": "Thôn Bờ Mận, Xã Bố Hạ, Tỉnh Bắc Ninh, Việt Nam"},
    {"mst_address": "Thôn Đạo Khê, Xã Yên Mỹ, Tỉnh Hưng Yên, Việt Nam"},
    {
        "mst_address": "12/10/17 Đường Trường Học, Phường Dĩ An, TP Hồ Chí Minh, Việt Nam"
    },
    {
        "mst_address": "Số 424 Hải Thượng Lãn Ông, TDP Tiền, Phường Hoa Lư, Tỉnh Ninh Bình, Việt Nam"
    },
    {"mst_address": "Cụm CN Ba Động, Xã Ba Động, Tỉnh Quảng Ngãi, Việt Nam"},
    {"mst_address": "1/7 K3, Ô1, Xã Thủ Thừa, Tỉnh Tây Ninh, Việt Nam"},
    {"mst_address": "179 phố Tân Sơn, Phường Đông Quang, Tỉnh Thanh Hóa, Việt Nam"},
    {
        "mst_address": "Lầu 3, Tòa nhà Đại Phát, 185B Hà Huy Giáp, Phường An Phú Đông, TP Hồ Chí Minh, Việt Nam"
    },
    {
        "mst_address": "Số nhà 259, Đường Đinh Tiên Hoàng, Tổ 1D, Phường Phủ Lý, Tỉnh Ninh Bình, Việt Nam"
    },
    {
        "mst_address": "Lô T1-27, TTTM Gemek Tower, Khu đô thị hai bên đường Lê Trọng Tấn – Geleximco, Xã An Khánh, TP Hà Nội, Việt Nam"
    },
    {"mst_address": "Số 1 Cầu Đông, Phường Nam Định, Tỉnh Ninh Bình, Việt Nam"},
    {
        "mst_address": "Căn SH3, tầng 1, Tòa nhà thương mại Lotus Central, số 28 đường Lý Thái Tổ, Phường Kinh Bắc, Tỉnh Bắc Ninh, Việt Nam"
    },
    {"mst_address": "470 Tân Kỳ Tân Quý, Phường Tân Sơn Nhì, TP Hồ Chí Minh, Việt Nam"},
    {"mst_address": "Thôn Gạch, Xã Sơn Đồng, TP Hà Nội, Việt Nam"},
    {"mst_address": "Thôn Hải Thành, Xã Xuân Hòa, Tỉnh Thanh Hóa, Việt Nam"},
    {
        "mst_address": "Số nhà 84 Đường Trần Anh Tông, Phường Yên Tử, Tỉnh Quảng Ninh, Việt Nam"
    },
    {
        "mst_address": "824/8, đường Giao Thông Nông Thôn, Phường Bình Minh, Tỉnh Tây Ninh, Việt Nam"
    },
    {"mst_address": "Tổ 18, Phường Phan Thiết, Thành Phố Tuyên Quang, Tuyên Quang"},
    {"mst_address": "Số 79 Hoàng Phan Thái, Phường Vinh Phú, Tỉnh Nghệ An, Việt Nam"},
    {
        "mst_address": "Số 51, Đường Lưu Bình Thái, Xã Quảng Bình, Tỉnh Thanh Hóa, Việt Nam"
    },
    {"mst_address": "Thôn Chiêu, Xã Sơn Đồng, TP Hà Nội, Việt Nam"},
    {"mst_address": "Thôn Ngã Tư, Xã Sơn Đồng, TP Hà Nội, Việt Nam"},
    {"mst_address": "174 Ngô Gia Tự, Phường Nha Trang, Tỉnh Khánh Hòa, Việt Nam"},
    {"mst_address": "Ấp 4, Xã Châu Pha, TP Hồ Chí Minh, Việt Nam"},
    {
        "mst_address": "Tầng 3, Tòa nhà Interserco, Số 17 Phạm Hùng, Phường Cầu Giấy, TP Hà Nội, Việt Nam"
    },
    {
        "mst_address": "Phòng 5.09, Lầu 5, Toà nhà ST.Moritz, 1014 Phạm Văn Đồng, Phường Hiệp Bình, TP Hồ Chí Minh, Việt Nam"
    },
    {"mst_address": "84 Cô Bắc, Xã Đức Trọng, Tỉnh Lâm Đồng, Việt Nam"},
    {"mst_address": "Khu phố Tân Ngọc, Phường Phú Mỹ, TP Hồ Chí Minh, Việt Nam"},
    {"mst_address": "Tổ 14A, Khu 2, Phường Vàng Danh, Thành phố Uông Bí, Quảng Ninh"},
    {
        "mst_address": "Số nhà 5, ngõ 113 phố Hoàng Cầu, Phường Đống Đa, TP Hà Nội, Việt Nam"
    },
    {
        "mst_address": "Số 168, Đường Uyên Hưng 27, Khu 06, Phường Tân Uyên, TP Hồ Chí Minh, Việt Nam"
    },
    {"mst_address": "Sô 64 Đường số 4, Phường An Khánh, TP Hồ Chí Minh, Việt Nam"},
    {
        "mst_address": "Số 1522 - Đường Hùng vương - Phường Gia cẩm, , Thành phố Việt Trì, Phú Thọ"
    },
    {"mst_address": "Thôn Đồng Bé, Xã Diên Lạc, Tỉnh Khánh Hòa, Việt Nam"},
    {
        "mst_address": "860/60E Đường Xô Viết Nghệ Tĩnh, Phường Thạnh Mỹ Tây, TP Hồ Chí Minh, Việt Nam"
    },
    {
        "mst_address": "Số 94, Đường DT747, khu phố 1, Phường Tân Uyên, TP Hồ Chí Minh, Việt Nam"
    },
    {"mst_address": "Thôn Mới, Xã Hoàng An, Tỉnh Phú Thọ, Việt Nam"},
    {"mst_address": "Tổ 4, ấp Hội Mỹ, Xã Phước Hải, TP Hồ Chí Minh, Việt Nam"},
    {"mst_address": "E12/29P Đường Thới Hòa, Xã Vĩnh Lộc, TP Hồ Chí Minh, Việt Nam"},
    {"mst_address": "271/1A Lê Đình Cẩn, Phường Tân Tạo, TP Hồ Chí Minh, Việt Nam"},
    {"mst_address": "80-82 Lý Chiêu Hoàng, Phường Bình Phú, TP Hồ Chí Minh, Việt Nam"},
    {"mst_address": "Số 1, Cộng Hoà 3, Phường Phú Thọ Hòa, TP Hồ Chí Minh, Việt Nam"},
    {
        "mst_address": "Tầng 1, số 24 đường Phan Bội Châu, Phường Hải Châu, TP Đà Nẵng, Việt Nam"
    },
    {"mst_address": "TDP Ninh Tịnh, Phường Đông Ninh Hòa, Tỉnh Khánh Hòa, Việt Nam"},
    {
        "mst_address": "Nhà ông Vũ Văn Cường, xóm Nam Thành, Xã Yên Từ, Tỉnh Ninh Bình, Việt Nam"
    },
    {
        "mst_address": "Lô B17, LK2, khu A, Khu Đô thị mới An Vân Dương, Phường Vỹ Dạ, TP Huế, Việt Nam"
    },
    {"mst_address": "Pác Riệu, Xã Cô Ba, Tỉnh Cao Bằng, Việt Nam"},
    {
        "mst_address": "76 Đường số 11, KDC Bình Hưng, Xã Bình Hưng, TP Hồ Chí Minh, Việt Nam"
    },
    {"mst_address": "69 Ngô Đức Kế, Phường Nha Trang, Tỉnh Khánh Hòa, Việt Nam"},
    {"mst_address": "256/53 Liên Khu 4-5, Phường Bình Tân, TP Hồ Chí Minh, Việt Nam"},
    {
        "mst_address": "Số 33, đường Bái Tử Long, phường Cẩm Trung, Phường Cẩm Trung, Thành phố Cẩm Phả, Quảng Ninh"
    },
    {
        "mst_address": "Tầng 1 & 2, Khu thương mại chung cư cao tầng, Phường Trần Hưng Đạo, Thành phố Hạ Long, Quảng Ninh"
    },
    {
        "mst_address": "121 đường ĐHT06, Phường Đông Hưng Thuận, TP Hồ Chí Minh, Việt Nam"
    },
    {"mst_address": "190A Tân Thành, Phường Chợ Lớn, TP Hồ Chí Minh, Việt Nam"},
    {
        "mst_address": "153 Đường Tỉnh Lộ 2, Ấp Đình, Xã Củ Chi, TP Hồ Chí Minh, Việt Nam"
    },
    {"mst_address": "Xóm 6, Phường Vinh Lộc, Tỉnh Nghệ An, Việt Nam"},
    {
        "mst_address": "Số 24 ngõ 163, Khu TT Đài Phát Tín, đường Đại Mỗ, Phường Đại Mỗ, TP Hà Nội, Việt Nam"
    },
    {"mst_address": "Số 10 Ngõ 87 Phố Vĩnh Phúc, Phường Ngọc Hà, TP Hà Nội, Việt Nam"},
    {
        "mst_address": "Số 120E1 đường Nguyễn Thị Tám, ấp Láng Cát A, Xã Củ Chi, TP Hồ Chí Minh, Việt Nam"
    },
    {"mst_address": "ấP Hưng Thới 2, Xã Phú Tân, Tỉnh An Giang, Việt Nam"},
    {"mst_address": "Khu 7, Xã Võ Miếu, Tỉnh Phú Thọ, Việt Nam"},
    {
        "mst_address": "Ô 16A Đường D23 Khu Dân Cư Việt Sing, Khu Phố 4, Phường An Phú, TP Hồ Chí Minh, Việt Nam"
    },
    {
        "mst_address": "300/34/37A Nguyễn Văn Linh, Phường Tân Thuận, TP Hồ Chí Minh, Việt Nam"
    },
    {"mst_address": "3/53/6 Thành Thái, Phường Diên Hồng, TP Hồ Chí Minh, Việt Nam"},
    {"mst_address": "Thôn 2, Xã Quảng Đức, Tỉnh Quảng Ninh, Việt Nam"},
    {
        "mst_address": "Thửa đất số 102, Tờ bản đồ số 41, Ngõ 31 Đường Nam Bình, Phường Hoa Lư, Tỉnh Ninh Bình, Việt Nam"
    },
    {"mst_address": "147 Đường số 1, Phường Hạnh Thông, TP Hồ Chí Minh, Việt Nam"},
    {
        "mst_address": "Nhà ông Trần Đại Phong, Xóm 3 Ngọc Động, Xã Gia Phong, Tỉnh Ninh Bình, Việt Nam"
    },
    {
        "mst_address": "2581/27/3 Ấp 17, Huỳnh Tấn Phát, Xã Nhà Bè, TP Hồ Chí Minh, Việt Nam"
    },
    {"mst_address": "Khu 7, Xã Vạn Xuân, Tỉnh Phú Thọ, Việt Nam"},
    {"mst_address": "Số 27 Thôn 3B, Xã Ea Ô, Tỉnh Đắk Lắk, Việt Nam"},
    {"mst_address": "Số 25, đường 2 - 9, Phường Hòa Cường, TP Đà Nẵng, Việt Nam"},
    {"mst_address": "Số 89 đường Vị Xuyên, Phường Nam Định, Tỉnh Ninh Bình, Việt Nam"},
    {
        "mst_address": "Tầng 1 nhà J, nhà khách Chính phủ La Thành, số 226 Vạn Phúc, Phường Ngọc Hà, TP Hà Nội, Việt Nam"
    },
    {"mst_address": "163 Lê Hoàn, Phường Lam Sơn, Thành phố Thanh Hoá, Thanh Hoá"},
    {
        "mst_address": "Ánh Dương 20-21, KĐT Vinhomes Ocean park 3, Xã Nghĩa Trụ, Tỉnh Hưng Yên, Việt Nam"
    },
    {"mst_address": "Thôn 3 Kênh Gà, Xã Gia Viễn, Tỉnh Ninh Bình, Việt Nam"},
    {
        "mst_address": "Số 155, Hẻm 7, đường Nguyễn Hữu Thọ, Tổ 11, Khu phố Hiệp Thạnh, Phường Tân Ninh, Tỉnh Tây Ninh, Việt Nam"
    },
    {
        "mst_address": "Tầng 29, Tòa Đông, Hà Nội Lotte Center, Số 54 Liễu Giai, Phường Giảng Võ, TP Hà Nội, Việt Nam"
    },
    {"mst_address": "240/5 đường Lê Duẩn, ấp 3, Xã An Phước, Tỉnh Đồng Nai, Việt Nam"},
    {"mst_address": "Số 497A, Tỉnh Lộ 8, Ấp 5, Xã Củ Chi, TP Hồ Chí Minh, Việt Nam"},
    {"mst_address": "Xóm 4, Xã Đông Hiếu, Tỉnh Nghệ An, Việt Nam"},
    {"mst_address": "41 Trịnh Phong, Phường Nha Trang, Tỉnh Khánh Hòa, Việt Nam"},
    {"mst_address": "Thôn Lãi, Xã Tân Dĩnh, Tỉnh Bắc Ninh, Việt Nam"},
    {
        "mst_address": "Số 1744 Trần Văn Giàu, Ấp 1, Xã Tân Vĩnh Lộc, TP Hồ Chí Minh, Việt Nam"
    },
    {"mst_address": "B8/1F Ấp 2, Xã Tân Vĩnh Lộc, TP Hồ Chí Minh, Việt Nam"},
    {"mst_address": "C10/17 Ấp 32, Xã Vĩnh Lộc, TP Hồ Chí Minh, Việt Nam"},
    {
        "mst_address": "103-105 Nguyễn Đình Chiểu, Phường Xuân Hòa, TP Hồ Chí Minh, Việt Nam"
    },
    {"mst_address": "45/2 Tổ 1, Đò Quan, Phường Nam Định, Tỉnh Ninh Bình, Việt Nam"},
    {
        "mst_address": "Số 58/27/20, Đường TX 22, Khu phố 44, Phường Thới An, TP Hồ Chí Minh, Việt Nam"
    },
    {"mst_address": "292. tổ 9, ấp Đông Tiến, Xã Tân Đông, Tỉnh Tây Ninh, Việt Nam"},
    {
        "mst_address": "164/13/14 Trịnh Đình Trọng, Phường Tân Phú, TP Hồ Chí Minh, Việt Nam"
    },
    {
        "mst_address": "Thửa đất 1232, TBĐ số 5, Tỉnh lộ 830, Ấp 9, Xã Lương Hòa, Tây Ninh, Việt Nam"
    },
    {"mst_address": "Thôn Yang Hăn, Xã Yang Mao, Tỉnh Đắk Lắk, Việt Nam"},
    {
        "mst_address": "507A2, Ngõ 199 Đường Hồ Tùng Mậu, Phường Từ Liêm, TP Hà Nội, Việt Nam"
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
