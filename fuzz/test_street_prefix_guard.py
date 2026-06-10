import unittest

from parser import AddressParser


class AddressParserStreetPrefixGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = AddressParser()

    def test_street_only_fragment_does_not_hallucinate_admin_components(self):
        result = self.parser.process("\u004b\u0068\u1ed1\u0069 3, \u0056\u0069\u1ec7\u0074 \u004e\u0061\u006d")

        self.assertEqual(result["format"], "old")
        self.assertFalse(result["is_new"])
        self.assertIsNone(result["province"])
        self.assertIsNone(result["district"])
        self.assertIsNone(result["ward"])
        self.assertEqual(
            result["street_address"],
            "\u004b\u0068\u1ed1\u0069 3",
        )

    def test_street_prefix_segment_does_not_hallucinate_province(self):
        result = self.parser.process(
            "\u0110\u01b0\u1edd\u006e\u0067 12A, \u004b\u0068\u0075 \u0070\u0068\u1ed1 8, \u0056\u0069\u1ec7\u0074 \u004e\u0061\u006d"
        )

        self.assertEqual(result["format"], "old")
        self.assertFalse(result["is_new"])
        self.assertIsNone(result["province"])
        self.assertIsNone(result["district"])
        self.assertIsNone(result["ward"])
        self.assertEqual(
            result["street_address"],
            "\u0110\u01b0\u1edd\u006e\u0067 12A, \u004b\u0068\u0075 \u0070\u0068\u1ed1 8",
        )

    def test_street_only_fragment_trims_trailing_country_suffix(self):
        result = self.parser.process(
            "\u0054\u1ed5 3, \u004b\u0068\u0075 \u0050\u0068\u1ed1 \u004b\u0068\u00e1\u006e\u0068 \u0048\u1ed9\u0069, \u0056\u0069\u1ec7\u0074 \u004e\u0061\u006d"
        )

        self.assertIsNone(result["province"])
        self.assertIsNone(result["district"])
        self.assertIsNone(result["ward"])
        self.assertEqual(
            result["street_address"],
            "\u0054\u1ed5 3, \u004b\u0068\u0075 \u0050\u0068\u1ed1 \u004b\u0068\u00e1\u006e\u0068 \u0048\u1ed9\u0069",
        )

    def test_kp_locality_fragment_without_region_hint_splits_street_and_ward(self):
        result = self.parser.process("KP7 Tân Hưng Thuận, Việt Nam")

        self.assertEqual(result["format"], "old")
        self.assertFalse(result["is_new"])
        self.assertIsNone(result["province"])
        self.assertIsNone(result["district"])
        self.assertEqual(result["ward"]["full_name"], "Phường Tân Hưng Thuận")
        self.assertEqual(result["street_address"], "KP7")
        self.assertNotIn("KP7 Tân Hưng Thuận", result["ward"].get("aliases", []))

    def test_khu_pho_locality_fragment_without_region_hint_splits_street_and_ward(self):
        result = self.parser.process("Khu phố 7 Tân Hưng Thuận, Việt Nam")

        self.assertEqual(result["format"], "old")
        self.assertFalse(result["is_new"])
        self.assertIsNone(result["province"])
        self.assertIsNone(result["district"])
        self.assertEqual(result["ward"]["full_name"], "Phường Tân Hưng Thuận")
        self.assertEqual(result["street_address"], "Khu phố 7")
        self.assertNotIn(
            "Khu phố 7 Tân Hưng Thuận", result["ward"].get("aliases", [])
        )

    def test_kp_locality_fragment_with_region_hint_keeps_only_kp_in_street(self):
        result = self.parser.process("KP2 Trung Mỹ Tây, Quận 12, TP.Hồ Chí Minh")

        self.assertEqual(result["format"], "old")
        self.assertFalse(result["is_new"])
        self.assertEqual(result["district"]["full_name"], "Quận 12")
        self.assertEqual(result["ward"]["full_name"], "Phường Trung Mỹ Tây")
        self.assertEqual(result["street_address"], "KP2")
        self.assertNotIn("KP2 Trung Mỹ Tây", result["ward"].get("aliases", []))


if __name__ == "__main__":
    unittest.main()
