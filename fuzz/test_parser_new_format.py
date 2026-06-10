import unittest

from parser import AddressParser


class AddressParserNewFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = AddressParser()

    def test_new_format_ward_does_not_rescue_same_name_district(self):
        result = self.parser.process("Tiên Lữ, Phường Chương Mỹ, TP Hà Nội")

        self.assertTrue(result["is_new"])
        self.assertEqual(result["format"], "new")
        self.assertEqual(result["ward"]["full_name"], "Phường Chương Mỹ")
        self.assertIsNone(result["district"])

    def test_bare_new_format_ward_does_not_emit_same_name_district(self):
        result = self.parser.process("Phường Chương Mỹ, TP Hà Nội")

        self.assertTrue(result["is_new"])
        self.assertEqual(result["ward"]["full_name"], "Phường Chương Mỹ")
        self.assertIsNone(result["district"])

    def test_new_format_ward_keeps_district_empty_for_full_address(self):
        result = self.parser.process(
            "Đội 4, Tiến Lữ, Phường Chương Mỹ, Thành phố Hà Nội, Việt Nam"
        )

        self.assertTrue(result["is_new"])
        self.assertEqual(result["ward"]["full_name"], "Phường Chương Mỹ")
        self.assertIsNone(result["district"])

    def test_explicit_district_prefix_is_preserved(self):
        result = self.parser.process(
            "Đội 4, Tiến Lữ, Huyện Chương Mỹ, Phường Chương Mỹ, Thành phố Hà Nội"
        )

        self.assertEqual(result["district"]["full_name"], "Huyện Chương Mỹ")
        self.assertEqual(result["ward"]["full_name"], "Phường Chương Mỹ")
        self.assertFalse(result["is_new"])
        self.assertEqual(result["format"], "old")

    def test_bare_district_segment_prevents_new_format_classification(self):
        result = self.parser.process(
            "91/34A Trần Tân, Tân Sơn Nhì, Tân Phú, Hồ Chí Minh"
        )

        self.assertFalse(result["is_new"])
        self.assertEqual(result["format"], "old")
        self.assertEqual(result["district"]["full_name"], "Quận Tân Phú")
        self.assertEqual(result["ward"]["full_name"], "Phường Tân Sơn Nhì")

    def test_ambiguous_bare_token_prefers_new_ward(self):
        result = self.parser.process("Chương Mỹ, TP Hà Nội")

        self.assertTrue(result["is_new"])
        self.assertEqual(result["ward"]["full_name"], "Phường Chương Mỹ")
        self.assertIsNone(result["district"])

    def test_new_format_ward_does_not_backslide_to_old_long_an_match(self):
        result = self.parser.process("464 Quốc Lộ 1, Phường Long An, Tây Ninh")

        self.assertTrue(result["is_new"])
        self.assertEqual(result["format"], "new")
        self.assertEqual(result["province"]["full_name"], "Tỉnh Tây Ninh")
        self.assertEqual(result["ward"]["full_name"], "Phường Long An")
        self.assertIsNone(result["district"])

    def test_explicit_ward_prefix_without_province_does_not_backslide(self):
        result = self.parser.process("Phường Long An")

        self.assertTrue(result["is_new"])
        self.assertEqual(result["format"], "new")
        self.assertIsNone(result["province"])
        self.assertEqual(result["ward"]["full_name"], "Phường Long An")
        self.assertIsNone(result["district"])

    def test_long_an_explicit_ward_stays_stable_after_other_parses(self):
        self.parser.process("Khu Phố Phú Nghị, Hòa Lợi, Thị xã Bến Cát, Bình Dương")
        self.parser.process("499, Tổ 25, KP2 Trung Mỹ Tây, Quận 12, TP.Hồ Chí Minh")

        bare = self.parser.process("Phường Long An")
        full = self.parser.process("464 Quốc Lộ 1, Phường Long An, Tây Ninh")

        self.assertEqual(bare["ward"]["full_name"], "Phường Long An")
        self.assertIsNone(bare["district"])
        self.assertTrue(bare["is_new"])

        self.assertEqual(full["province"]["full_name"], "Tỉnh Tây Ninh")
        self.assertEqual(full["ward"]["full_name"], "Phường Long An")
        self.assertIsNone(full["district"])
        self.assertTrue(full["is_new"])

    def test_cccd_ward_province_context_prefers_new_format_without_district(self):
        cases = [
            {
                "sub_id": "8884",
                "doc_id": "16295",
                "address": "B15.18 A1 Celadon City, Số 2 Đường N4, Tân Sơn Nhì, TP. Hồ Chí Minh",
                "street": "B15.18 A1 Celadon City Số 2 Đường N4",
                "province": "Thành phố Hồ Chí Minh",
                "ward": "Phường Tân Sơn Nhì",
                "ward_id": "27019",
            },
            {
                "sub_id": "8898",
                "doc_id": "16337",
                "address": "199, Kp Long Đức 3 Tam Phước, Đồng Nai",
                "street": "199 Kp Long Đức 3",
                "province": "Tỉnh Đồng Nai",
                "ward": "Phường Tam Phước",
                "ward_id": "26374",
            },
            {
                "sub_id": "8933",
                "doc_id": "16442",
                "address": "Ấp Thạnh Lợi A1, Thanh Hóa, Cần Thơ",
                "street": "Ấp Thạnh Lợi A1",
                "province": "Thành phố Cần Thơ",
                "ward": "Xã Thạnh Hoà",
                "ward_id": "31408",
            },
        ]

        for case in cases:
            with self.subTest(sub_id=case["sub_id"], doc_id=case["doc_id"]):
                result = self.parser.process(case["address"])

                self.assertTrue(result["is_new"])
                self.assertEqual(result["format"], "new")
                self.assertEqual(result["street_address"], case["street"])
                self.assertEqual(result["province"]["full_name"], case["province"])
                self.assertIsNone(result["district"])
                self.assertEqual(result["ward"]["full_name"], case["ward"])
                self.assertEqual(result["ward"]["id"], case["ward_id"])

    def test_bare_district_segment_keeps_old_tan_son_nhi_context(self):
        result = self.parser.process(
            "91/34A Trần Tân, Tân Sơn Nhì, Tân Phú, Hồ Chí Minh"
        )

        self.assertFalse(result["is_new"])
        self.assertEqual(result["format"], "old")
        self.assertEqual(result["ward"]["full_name"], "Phường Tân Sơn Nhì")
        self.assertEqual(result["ward"]["id"], "27010")
        self.assertEqual(result["district"]["full_name"], "Quận Tân Phú")

    def test_unresolved_bare_token_does_not_claim_new_format(self):
        result = self.parser.process("Chương Mỹ")

        self.assertEqual(result["format"], "old")
        self.assertFalse(result["is_new"])
        self.assertIsNone(result["province"])
        self.assertIsNone(result["district"])
        self.assertIsNone(result["ward"])
        self.assertEqual(result["street_address"], "Chương Mỹ")


if __name__ == "__main__":
    unittest.main()
