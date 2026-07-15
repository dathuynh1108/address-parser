import unittest
from unittest.mock import patch

from parser import AddressParser
from review_regression_cases import (
    CCCD_REVIEWED_CORRECT_CASES,
    HEAD_OFFICE_REVIEWED_CORRECT_CASES,
)


class AddressParserRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = AddressParser()

    def assertComponent(self, component, expected_full_name):
        if expected_full_name is None:
            self.assertIsNone(component)
            return
        self.assertIsNotNone(component)
        self.assertEqual(component["full_name"], expected_full_name)

    def assertReviewedCase(self, result, case):
        self.assertEqual(result["is_new"], case["is_new"])
        self.assertEqual(result["format"], case["format"])
        self.assertEqual(result["street_address"], case["street_address"])
        self.assertReviewedComponent(
            result["province"],
            case["province"],
            case["province_id"],
        )
        self.assertReviewedComponent(
            result["district"],
            case["district"],
            case["district_id"],
        )
        self.assertReviewedComponent(result["ward"], case["ward"], case["ward_id"])

    def assertReviewedComponent(self, component, expected_full_name, expected_id):
        if expected_full_name is None:
            self.assertIsNone(component)
            return
        self.assertIsNotNone(component)
        self.assertEqual(component["full_name"], expected_full_name)
        self.assertEqual(str(component.get("id")), expected_id)

    def test_prefix_detection_reuses_precomputed_matchers(self):
        province_choices = getattr(self.parser, "_province_detection_choices", None)
        district_choices = getattr(self.parser, "_district_detection_choices", None)
        ward_choices = getattr(self.parser, "_ward_detection_choices", None)
        self.assertIsInstance(province_choices, tuple)
        self.assertIsInstance(district_choices, tuple)
        self.assertIsInstance(ward_choices, tuple)

        with (
            patch(
                "parser.re.compile",
                side_effect=AssertionError("compiled regex on the request path"),
            ),
            patch(
                "parser.re.search",
                side_effect=AssertionError("module-level regex search on the request path"),
            ),
        ):
            segmented = self.parser._detect_by_prefix(
                "phuong long an | tay ninh"
            )
            unsegmented = self.parser._detect_by_prefix(
                "phuong quan hoa quan cau giay thanh pho ha noi"
            )

        self.assertEqual(segmented, (None, None, "phuong long an"))
        self.assertEqual(unsegmented, ("ha noi", "cau giay", "quan hoa"))
        self.assertIs(province_choices, self.parser._province_detection_choices)
        self.assertIs(district_choices, self.parser._district_detection_choices)
        self.assertIs(ward_choices, self.parser._ward_detection_choices)

    def test_new_format_regression_cases(self):
        cases = [
            {
                "address": "Số 20, ngõ 151, đường Hồng Hà, Phường Hồng Hà, TP Hà Nội, Việt Nam",
                "province": "Thành phố Hà Nội",
                "district": None,
                "ward": "Phường Hồng Hà",
                "street": "Số 20 ngõ 151 đường Hồng Hà",
            },
            {
                "address": "158/58 Vũng Việt, Khu Phố Đông Chiêu, Phường Dĩ An, TP Hồ Chí Minh, Việt Nam",
                "province": "Thành phố Hồ Chí Minh",
                "district": None,
                "ward": "Phường Dĩ An",
                "street": "158/58 Vũng Việt Khu Phố Đông Chiêu",
            },
            {
                "address": "Thôn Tân Tiến, Xã Ba Chẽ, Tỉnh Quảng Ninh, Việt Nam",
                "province": "Tỉnh Quảng Ninh",
                "district": None,
                "ward": "Xã Ba Chẽ",
                "street": "Thôn Tân Tiến",
            },
            {
                "address": "Khu phố 8, Đặc khu Côn Đảo, TP Hồ Chí Minh",
                "province": "Thành phố Hồ Chí Minh",
                "district": None,
                "ward": "Đặc khu Côn Đảo",
                "street": "Khu phố 8",
            },
            {
                "address": "Đặc khu Côn Đảo, TP Hồ Chí Minh",
                "province": "Thành phố Hồ Chí Minh",
                "district": None,
                "ward": "Đặc khu Côn Đảo",
                "street": "",
            },
            {
                "address": "464 Quốc Lộ 1, Phường Long An, Tây Ninh",
                "province": "Tỉnh Tây Ninh",
                "district": None,
                "ward": "Phường Long An",
                "street": "464 Quốc Lộ 1",
            },
            {
                "address": "Chương Mỹ, TP Hà Nội",
                "province": "Thành phố Hà Nội",
                "district": None,
                "ward": "Phường Chương Mỹ",
                "street": "",
            },
            {
                "address": "Phường Chương Mỹ, TP Hà Nội",
                "province": "Thành phố Hà Nội",
                "district": None,
                "ward": "Phường Chương Mỹ",
                "street": "",
            },
            {
                "address": "Đội 4, Tiến Lữ, Phường Chương Mỹ, Thành phố Hà Nội, Việt Nam",
                "province": "Thành phố Hà Nội",
                "district": None,
                "ward": "Phường Chương Mỹ",
                "street": "Đội 4 Tiến Lữ",
            },
        ]

        for case in cases:
            with self.subTest(address=case["address"]):
                result = self.parser.process(case["address"])
                self.assertTrue(result["is_new"])
                self.assertEqual(result["format"], "new")
                self.assertComponent(result["province"], case["province"])
                self.assertComponent(result["district"], case["district"])
                self.assertComponent(result["ward"], case["ward"])
                self.assertEqual(result["street_address"], case["street"])

    def test_old_format_regression_cases(self):
        cases = [
            {
                "address": "Số 1522 - Đường Hùng vương - Phường Gia cẩm, , Thành phố Việt Trì, Phú Thọ",
                "province": "Tỉnh Phú Thọ",
                "district": "Thành phố Việt Trì",
                "ward": "Phường Gia Cẩm",
                "street": "Số 1522 Đường Hùng vương",
            },
            {
                "address": "Tầng 1 & 2, Khu thương mại chung cư cao tầng, Phường Trần Hưng Đạo, Thành phố Hạ Long, Quảng Ninh",
                "province": "Tỉnh Quảng Ninh",
                "district": "Thành phố Hạ Long",
                "ward": "Phường Trần Hưng Đạo",
                "street": "Tầng 1 & 2 Khu thương mại chung cư cao tầng",
            },
            {
                "address": "Lô 223, Đường Amata, KCN Amata, P.Long Bình, , Thành phố Biên Hoà, Đồng Nai",
                "province": "Tỉnh Đồng Nai",
                "district": "Thành phố Biên Hoà",
                "ward": "Phường Long Bình",
                "street": "Lô 223 Đường Amata KCN Amata",
            },
            {
                "address": "Tầng 10 tháp Tây, Hancorp Plaza 72T Trần Đăng Ninh, Phường Dịch Vọng, Quận Cầu Giấy, Thành phố Hà Nội, Việt Nam",
                "province": "Thành phố Hà Nội",
                "district": "Quận Cầu Giấy",
                "ward": "Phường Dịch Vọng",
                "street": "Tầng 10 tháp Tây Hancorp Plaza 72T Trần Đăng Ninh",
            },
            {
                "address": "Tổ 1, Ấp An Thành, Xã Bình An, Huyện Châu Thành, Tỉnh Kiên Giang, Việt Nam",
                "province": "Tỉnh Kiên Giang",
                "district": "Huyện Châu Thành",
                "ward": "Xã Bình An",
                "street": "Tổ 1 Ấp An Thành",
            },
            {
                "address": "Ô1 4/40 Khu Phố Nhà Dài, Thị Trấn Thủ Thừa, Huyện Thủ Thừa, Tỉnh Long An, Việt Nam",
                "province": "Tỉnh Long An",
                "district": "Huyện Thủ Thừa",
                "ward": "Thị trấn Thủ Thừa",
                "street": "Ô1 4/40 Khu Phố Nhà Dài",
            },
            {
                "address": "Số 84A, ngõ 261, đường Xã Đàn, Phường Nam Đồng, Quận Đống Đa, Thành phố Hà Nội, Việt Nam",
                "province": "Thành phố Hà Nội",
                "district": "Quận Đống Đa",
                "ward": "Phường Nam Đồng",
                "street": "Số 84A ngõ 261 đường Xã Đàn",
            },
            {
                "address": "Tân Phúc, Xã Sơn Đông, Thị Xã Sơn Tây, Thành phố Hà Nội, Việt Nam",
                "province": "Thành phố Hà Nội",
                "district": "Thị xã Sơn Tây",
                "ward": "Xã Sơn Đông",
                "street": "Tân Phúc",
            },
        ]

        for case in cases:
            with self.subTest(address=case["address"]):
                result = self.parser.process(case["address"])
                self.assertFalse(result["is_new"])
                self.assertEqual(result["format"], "old")
                self.assertComponent(result["province"], case["province"])
                self.assertComponent(result["district"], case["district"])
                self.assertComponent(result["ward"], case["ward"])
                self.assertEqual(result["street_address"], case["street"])

    def test_document_ocr_raw_address_regression_cases(self):
        cases = [
            {
                "address": "91/34A Trần Tân, Tân Sơn Nhì, Tân Phú, Hồ Chí Minh",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Hồ Chí Minh",
                "district": "Quận Tân Phú",
                "ward": "Phường Tân Sơn Nhì",
                "street": "91/34A Trần Tân",
            },
            {
                "address": "44, Khu Dân Cư 7, Áp Cây Xăng Phú Túc, Định Quán, Đồng Nai",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Đồng Nai",
                "district": "Huyện Định Quán",
                "ward": "Xã Phú Túc",
                "street": "44 Khu Dân Cư 7 Áp Cây Xăng",
            },
            {
                "address": "68 Đường Tôn Thất Hiệp, Phường 11, Quận 11, TP. Hồ Chí Minh",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Hồ Chí Minh",
                "district": "Quận 11",
                "ward": "Phường 11",
                "street": "68 Đường Tôn Thất Hiệp",
            },
            {
                "address": "Thôn Mã Hoa Thủy, Lệ Thủy, Quảng Bình",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Quảng Bình",
                "district": "Huyện Lệ Thuỷ",
                "ward": "Xã Hoa Thuỷ",
                "street": "Thôn Mã",
            },
            {
                "address": "Khu Phố Phú Nghị, Hòa Lợi, Thị xã Bến Cát, Bình Dương",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Bình Dương",
                "district": "Thành phố Bến Cát",
                "ward": "Phường Hoà Lợi",
                "street": "Khu Phố Phú Nghị",
            },
            {
                "address": "Thôn Giang Ché, Giang Hải, Phú Lộc, Thừa Thiên Huế",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Huế",
                "district": "Huyện Phú Lộc",
                "ward": "Xã Giang Hải",
                "street": "Thôn Giang Ché",
            },
            {
                "address": "Nguyễn Văn Dung, P.6 Gò Vấp, TP.HCM",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Hồ Chí Minh",
                "district": "Quận Gò Vấp",
                "ward": "Phường 6",
                "street": "Nguyễn Văn Dung",
            },
            {
                "address": "Vịnh An Cơ, Châu Thành, Tây Ninh",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Tây Ninh",
                "district": "Huyện Châu Thành",
                "ward": "Xã An Cơ",
                "street": "Vịnh",
            },
            {
                "address": "Nhóm 9 - Cao Cường Đông Quang, Ba Vì, Hà Nội",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Hà Nội",
                "district": "Huyện Ba Vì",
                "ward": "Xã Đông Quang",
                "street": "Nhóm 9 Cao Cường",
            },
            {
                "address": "Trạch Thượng 1 TT. Phong Điền, Phong Điền, TT - Huế",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Huế",
                "district": "Thị xã Phong Điền",
                "ward": "Phường Phong Thu",
                "street": "Trạch Thượng 1",
            },
            {
                "address": "15 Tổng Duy Tân, Thuận Thành, TP Huế, Thừa Thiên Huế",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Huế",
                "district": "Quận Phú Xuân",
                "ward": "Phường Đông Ba",
                "street": "15 Tổng Duy Tân",
            },
            {
                "address": "13 Phò Trạch, TT. Phong Điền, Phong Điền, TT - Huế",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Huế",
                "district": "Thị xã Phong Điền",
                "ward": "Phường Phong Thu",
                "street": "13 Phò Trạch",
            },
            {
                "address": "Minh Thanh Hương Vinh, Hương Trà, Thừa Thiên Huế",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Huế",
                "district": "Thị xã Hương Trà",
                "ward": "Phường Hương Vinh",
                "street": "Minh Thanh",
            },
            {
                "address": "Khóm 2 Thị trấn Tri Tôn, Tri Tôn, An Giang",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh An Giang",
                "district": "Huyện Tri Tôn",
                "ward": "Thị trấn Tri Tôn",
                "street": "Khóm 2",
            },
            {
                "address": "Tân Phong, Tân Biên, Tây Ninh, Áp Xóm Tháp",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Tây Ninh",
                "district": "Huyện Tân Biên",
                "ward": "Xã Tân Phong",
                "street": "Áp Xóm Tháp",
            },
            {
                "address": "P106b Tt Mai Động C27a, Mai Động, Hoàng Mai, Hà Nội",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Hà Nội",
                "district": "Quận Hoàng Mai",
                "ward": "Phường Mai Động",
                "street": "P106b Tt Mai Động C27a",
            },
            {
                "address": "Tổ 22, K. Mỹ Phú, Mỹ Phú, Thành phố Cao Lãnh, Đồng Tháp",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Đồng Tháp",
                "district": "Thành phố Cao Lãnh",
                "ward": "Phường Mỹ Phú",
                "street": "Tổ 22 K. Mỹ Phú",
            },
            {
                "address": "123 đường abc Mỹ Phú, Mỹ Phú, Thành phố Cao Lãnh, Đồng Tháp",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Đồng Tháp",
                "district": "Thành phố Cao Lãnh",
                "ward": "Phường Mỹ Phú",
                "street": "123 đường abc Mỹ Phú",
            },
            {
                "address": "Số 225A, Hưng Lợi Đông Long Hưng B, Lấp Vò, Đồng Tháp",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Đồng Tháp",
                "district": "Huyện Lấp Vò",
                "ward": "Xã Long Hưng B",
                "street": "Số 225A Hưng Lợi Đông",
            },
            {
                "address": "Tổ 16, Khóm 2 Thị trấn Sa Rài, Tân Hồng, Đồng Tháp",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Đồng Tháp",
                "district": "Huyện Tân Hồng",
                "ward": "Thị trấn Sa Rài",
                "street": "Tổ 16 Khóm 2",
            },
            {
                "address": "170/41 Vườn Lài, Tân Thành, Tân Phú, Hồ Chí Minh",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Hồ Chí Minh",
                "district": "Quận Tân Phú",
                "ward": "Phường Tân Thành",
                "street": "170/41 Vườn Lài",
            },
            {
                "address": "75/3, Tổ 07, KP7 Tân Hưng Thuận, Quận 12, TP. Hồ Chí Minh",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Hồ Chí Minh",
                "district": "Quận 12",
                "ward": "Phường Tân Hưng Thuận",
                "street": "75/3 Tổ 07 KP7",
            },
            {
                "address": "499, Tổ 25, KP2 Trung Mỹ Tây, Quận 12, TP.Hồ Chí Minh",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Hồ Chí Minh",
                "district": "Quận 12",
                "ward": "Phường Trung Mỹ Tây",
                "street": "499 Tổ 25 KP2",
            },
            {
                "address": "Phú Thượng, Kỳ Phú, Kỳ Anh, Hà Tĩnh",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Hà Tĩnh",
                "district": "Huyện Kỳ Anh",
                "ward": "Xã Kỳ Phú",
                "street": "Phú Thượng",
            },
            {
                "address": "402, Lô A1, C/ Cư, Hòa Bình, P14, Quận 10, TP. Hồ Chí Minh",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Hồ Chí Minh",
                "district": "Quận 10",
                "ward": "Phường 14",
                "street": "402 Lô A1 C/ Cư Hòa Bình",
            },
            {
                "address": "Số 255A, Hưng Lợi Đông, Long Hưng B, Lấp Vò, Đồng Tháp",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Đồng Tháp",
                "district": "Huyện Lấp Vò",
                "ward": "Xã Long Hưng B",
                "street": "Số 255A Hưng Lợi Đông",
            },
            {
                "address": "Thôn 4, Cư Yang, Ea Kar, Đắk Lắk",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Đắk Lắk",
                "district": "Huyện Ea Kar",
                "ward": "Xã Cư Yang",
                "street": "Thôn 4",
            },
            {
                "address": "Thanh Trì\nĐông Sơn, Chương Mỹ, Hà Nội",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Hà Nội",
                "district": "Huyện Chương Mỹ",
                "ward": "Xã Đông Sơn",
                "street": "Thanh Trì",
            },
            {
                "address": "Tổ Dân Phố 2, Thị trấn M'Đrắk, M'Drắk, Đắk Lắk",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Đắk Lắk",
                "district": "Huyện MĐrắk",
                "ward": "Thị trấn MĐrắk",
                "street": "Tổ Dân Phố 2",
            },
            {
                "address": "Thôn 2 Bình Hòa, Krông Ana, Đắk Lắk",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Đắk Lắk",
                "district": "Huyện Krông A Na",
                "ward": "Xã Bình Hoà",
                "street": "Thôn 2",
            },
            {
                "address": "Ông Trịnh Tân Phước, Tx. Phú Mỹ, Bà Rịa - Vũng Tàu",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Bà Rịa - Vũng Tàu",
                "district": "Thành phố Phú Mỹ",
                "ward": "Phường Tân Phước",
                "street": "Ông Trịnh Tân Phước",
            },
            {
                "address": "Tổ 6, Khu Phố 7, Hưng Long, Phan Thiết, Bình Thuận",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Bình Thuận",
                "district": "Thành phố Phan Thiết",
                "ward": "Phường Bình Hưng",
                "street": "Tổ 6 Khu Phố 7",
            },
            {
                "address": "Phương Bằn, Phụng Châu, Chương Mỹ, Hà Nội",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Hà Nội",
                "district": "Huyện Chương Mỹ",
                "ward": "Xã Phụng Châu",
                "street": "Phương Bằn",
            },
            {
                "address": "Phượng Lâu, Thành phố Việt Trì, Phú Thọ",
                "format": "old",
                "is_new": False,
                "province": "Tỉnh Phú Thọ",
                "district": "Thành phố Việt Trì",
                "ward": "Xã Phượng Lâu",
                "street": "",
            },
            {
                "address": "Đội 2, Đông Nanh Chương Mỹ, Hà Nội",
                "format": "new",
                "is_new": True,
                "province": "Thành phố Hà Nội",
                "district": None,
                "ward": "Phường Chương Mỹ",
                "street": "Đội 2 Đông Nanh",
            },
        ]

        for case in cases:
            with self.subTest(address=case["address"]):
                result = self.parser.process(case["address"])
                self.assertEqual(result["is_new"], case["is_new"])
                self.assertEqual(result["format"], case["format"])
                self.assertComponent(result["province"], case["province"])
                self.assertComponent(result["district"], case["district"])
                self.assertComponent(result["ward"], case["ward"])
                self.assertEqual(result["street_address"], case["street"])

    def test_same_name_new_ward_does_not_emit_old_district(self):
        result = self.parser.process("Tiên Lữ, Phường Chương Mỹ, TP Hà Nội")

        self.assertTrue(result["is_new"])
        self.assertEqual(result["format"], "new")
        self.assertComponent(result["ward"], "Phường Chương Mỹ")
        self.assertIsNone(result["district"])

    def test_explicit_district_prefix_still_wins(self):
        result = self.parser.process(
            "Đội 4, Tiến Lữ, Huyện Chương Mỹ, Phường Chương Mỹ, Thành phố Hà Nội"
        )

        self.assertFalse(result["is_new"])
        self.assertEqual(result["format"], "old")
        self.assertComponent(result["district"], "Huyện Chương Mỹ")
        self.assertComponent(result["ward"], "Phường Chương Mỹ")

    def test_cccd_reviewed_correct_csv_cases_stay_stable(self):
        self.assertEqual(len(CCCD_REVIEWED_CORRECT_CASES), 194)

        for case in CCCD_REVIEWED_CORRECT_CASES:
            with self.subTest(
                sub_id=case["sub_id"],
                submission_id_retest=case["submission_id_retest"],
                doc_id=case["doc_id"],
            ):
                result = self.parser.process(case["address"])
                self.assertReviewedCase(result, case)

    def test_business_registration_reviewed_correct_csv_cases_stay_stable(self):
        self.assertEqual(len(HEAD_OFFICE_REVIEWED_CORRECT_CASES), 210)

        for case in HEAD_OFFICE_REVIEWED_CORRECT_CASES:
            with self.subTest(
                sub_id=case["sub_id"],
                submission_id=case["submission_id"],
                doc_id=case["doc_id"],
            ):
                result = self.parser.process(case["address"])
                self.assertReviewedCase(result, case)

    def test_business_registration_head_office_street_suffix_regression_cases(self):
        cases = [
            {
                "submission_id": "8863",
                "doc_id": "16233",
                "address": "65 - 67 Đường số 5 - Cư Xá Bình Thới, Phường 08, Quận 11",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Hồ Chí Minh",
                "district": "Quận 11",
                "ward": "Phường 08",
                "street": "65 67 Đường số 5 Cư Xá Bình Thới",
            },
            {
                "submission_id": "9001",
                "doc_id": "16647",
                "address": "Số 926 Đường Láng, phường Láng Thượng, quận Đống Đa, thành phố Hà Nội",
                "format": "old",
                "is_new": False,
                "province": "Thành phố Hà Nội",
                "district": "Quận Đống Đa",
                "ward": "Phường Láng Thượng",
                "street": "Số 926 Đường Láng",
            },
            {
                "submission_id": "9019",
                "doc_id": "16701",
                "address": "Tầng 4 – Tòa nhà CT2 Khu văn phòng cho thuê Sevin Office, Số 609 Phố Trương Định, Phường Hoàng Mai, Thành phố Hà Nội, Việt Nam",
                "format": "new",
                "is_new": True,
                "province": "Thành phố Hà Nội",
                "district": None,
                "ward": "Phường Hoàng Mai",
                "street": "Tầng 4 Tòa nhà CT2 Khu văn phòng cho thuê Sevin Office Số 609 Phố Trương Định",
            },
            {
                "submission_id": "9024",
                "doc_id": "16716",
                "address": "Tầng 3, số 119A Trần Đại Nghĩa, Phường Bạch Mai, Thành phố Hà Nội, Việt Nam",
                "format": "new",
                "is_new": True,
                "province": "Thành phố Hà Nội",
                "district": None,
                "ward": "Phường Bạch Mai",
                "street": "Tầng 3 số 119A Trần Đại Nghĩa",
            },
        ]

        for case in cases:
            with self.subTest(
                submission_id=case["submission_id"], doc_id=case["doc_id"]
            ):
                result = self.parser.process(case["address"])

                self.assertEqual(result["is_new"], case["is_new"])
                self.assertEqual(result["format"], case["format"])
                self.assertComponent(result["province"], case["province"])
                self.assertComponent(result["district"], case["district"])
                self.assertComponent(result["ward"], case["ward"])
                self.assertEqual(result["street_address"], case["street"])

    def test_ethnic_name_variants_are_searchable(self):
        district_hits = self.parser.search_district("Krông Ana", limit=5)
        self.assertTrue(district_hits)
        self.assertEqual(district_hits[0]["record"]["full_name"], "Huyện Krông A Na")

        old_ward_hits = self.parser.search_ward(
            "M'Đrắk",
            district_code="652",
            include_new=False,
            include_old=True,
            limit=5,
        )
        self.assertTrue(old_ward_hits)
        self.assertEqual(old_ward_hits[0]["record"]["full_name"], "Thị trấn MĐrắk")

        new_ward_hits = self.parser.search_ward(
            "M'Đrắk",
            province_code="66",
            include_new=True,
            include_old=False,
            limit=5,
        )
        self.assertTrue(new_ward_hits)
        self.assertEqual(new_ward_hits[0]["record"]["full_name"], "Xã MDrắk")

    def test_name_tokens_do_not_trigger_false_unit_prefixes(self):
        old_ward_hits = self.parser.search_ward(
            "Phượng Lâu",
            district_code="227",
            include_new=False,
            include_old=True,
            limit=5,
        )
        self.assertTrue(old_ward_hits)
        self.assertEqual(old_ward_hits[0]["record"]["full_name"], "Xã Phượng Lâu")


if __name__ == "__main__":
    unittest.main()
