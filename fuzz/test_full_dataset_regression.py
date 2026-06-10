import unittest

from full_dataset_regression_cases import build_regression_cases
from parser import AddressParser


class AddressParserFullDatasetRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = AddressParser()
        payload = build_regression_cases(parser=cls.parser)
        cls.old_cases = payload["old_cases"]
        cls.new_cases = payload["new_cases"]

    def assertComponent(self, component, expected):
        if expected is None:
            self.assertIsNone(component)
            return
        self.assertIsNotNone(component)
        self.assertEqual(component["id"], expected["id"])
        self.assertEqual(component["code"], expected["code"])
        self.assertEqual(component["full_name"], expected["full_name"])

    def assertCase(self, case):
        result = self.parser.process(case["address"])
        expected = case["expected"]
        self.assertEqual(result["format"], expected["format"])
        self.assertEqual(result["is_new"], expected["is_new"])
        self.assertEqual(result["street_address"], expected["street_address"])
        self.assertComponent(result["province"], expected["province"])
        self.assertComponent(result["district"], expected["district"])
        self.assertComponent(result["ward"], expected["ward"])

    def test_old_dataset_cases(self):
        for case in self.old_cases:
            with self.subTest(case_id=case["case_id"]):
                self.assertCase(case)

    def test_new_dataset_cases(self):
        for case in self.new_cases:
            with self.subTest(case_id=case["case_id"]):
                self.assertCase(case)


if __name__ == "__main__":
    unittest.main()
