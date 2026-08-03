import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar, cast
from unittest.mock import patch

from address_parser import AddressParser
from address_parser.contracts import (
    AddressCode,
    AddressCodeInput,
    AddressFormat,
    AdministrativeLevel,
    AdministrativeRecord,
    NormalizationMode,
    ParsedAddressComponent,
    ParseResult,
    PreprocessedCachePayload,
    PreprocessedState,
    RegistrySource,
    SearchDocument,
    SearchEngineState,
)
from address_parser.search_engine import AddressSearchEngine


def _analyze_test_text(value: str | None) -> list[str]:
    return value.casefold().split() if value else []


def _normalize_test_id(value: AddressCodeInput | None) -> AddressCode | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


class AddressParserContractTests(unittest.TestCase):
    RESULT_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "province",
            "district",
            "ward",
            "street_address",
            "format",
            "is_new",
        }
    )
    EXPECTED_DISCRIMINATORS: ClassVar[dict[AddressFormat, bool | None]] = {
        "old": False,
        "new": True,
        "unknown": None,
    }
    parser: ClassVar[AddressParser]

    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = AddressParser()

    def assert_parse_result_contract(self, result: ParseResult) -> None:
        self.assertEqual(frozenset(result), self.RESULT_KEYS)
        self.assertIs(
            result["is_new"],
            self.EXPECTED_DISCRIMINATORS[result["format"]],
        )
        self.assertIsInstance(result["street_address"], str)
        self.assert_component_contract(result["province"])
        self.assert_component_contract(result["district"])
        self.assert_component_contract(result["ward"])
        if result["format"] == "new":
            self.assertIsNone(result["district"])

    def assert_component_contract(
        self,
        component: ParsedAddressComponent | None,
    ) -> None:
        if component is not None:
            self.assertIsInstance(component["name"], str)

    def test_process_returns_exact_discriminated_result_contract(self) -> None:
        addresses = (
            "Số 84A, ngõ 261, đường Xã Đàn, Phường Nam Đồng, "
            "Quận Đống Đa, Thành phố Hà Nội, Việt Nam",
            "464 Quốc Lộ 1, Phường Long An, Tây Ninh",
        )

        for address in addresses:
            with self.subTest(address=address):
                self.assert_parse_result_contract(self.parser.process(address))

    def test_process_rejects_non_string_inputs(self) -> None:
        invalid_inputs: tuple[object, ...] = (
            None,
            True,
            123,
            1.5,
            object(),
            [],
            {},
        )

        for invalid_input in invalid_inputs:
            with self.subTest(input_type=type(invalid_input).__name__):
                with self.assertRaises(TypeError):
                    self.parser.process(cast(str, invalid_input))

    def test_prefix_detection_reuses_precomputed_matchers(self) -> None:
        province_choices = getattr(self.parser, "_province_detection_choices", None)
        district_choices = getattr(self.parser, "_district_detection_choices", None)
        ward_choices = getattr(self.parser, "_ward_detection_choices", None)
        self.assertIsInstance(province_choices, tuple)
        self.assertIsInstance(district_choices, tuple)
        self.assertIsInstance(ward_choices, tuple)

        with (
            patch(
                "address_parser.parser.re.compile",
                side_effect=AssertionError("compiled regex on the request path"),
            ),
            patch(
                "address_parser.parser.re.search",
                side_effect=AssertionError("module-level regex search on the request path"),
            ),
        ):
            segmented = self.parser._detect_by_prefix("phuong long an | tay ninh")
            unsegmented = self.parser._detect_by_prefix(
                "phuong quan hoa quan cau giay thanh pho ha noi"
            )

        self.assertEqual(segmented, (None, None, "phuong long an"))
        self.assertEqual(unsegmented, ("ha noi", "cau giay", "quan hoa"))
        self.assertIs(province_choices, self.parser._province_detection_choices)
        self.assertIs(district_choices, self.parser._district_detection_choices)
        self.assertIs(ward_choices, self.parser._ward_detection_choices)

    def test_fuzzy_match_preserves_second_query_prefix_pass(self) -> None:
        self.assertEqual(
            self.parser._fuzzy_match_component_key(
                "phuong h ai duong",
                ["h ai duong", "hai duong"],
                cutoff=88,
            ),
            "h ai duong",
        )

    def test_fuzzy_match_preserves_duplicate_core_choice_order(self) -> None:
        choices = ["phuong tan an", "xa tan an"]
        self.assertEqual(
            self.parser._fuzzy_match_component_key("tan ann", choices, cutoff=80),
            choices[0],
        )
        choices.reverse()
        self.assertEqual(
            self.parser._fuzzy_match_component_key("tan ann", choices, cutoff=80),
            choices[0],
        )

    def test_preprocess_rebuilds_derived_packed_index(self) -> None:
        previous_index = self.parser._packed_ngram_index
        expected = self.parser.process(
            "Số 27 Nguyễn Khánh Toàn, Phường Quan Hoa, Quận Cầu Giấy, Hà Nội"
        )

        self.parser.preprocess_address()

        self.assertIsNotNone(previous_index)
        self.assertIsNot(previous_index, self.parser._packed_ngram_index)
        self.assertEqual(
            self.parser.process("Số 27 Nguyễn Khánh Toàn, Phường Quan Hoa, Quận Cầu Giấy, Hà Nội"),
            expected,
        )

    def test_component_lookup_rejects_invalid_identifier_types(self) -> None:
        invalid_identifiers: tuple[object, ...] = (True, 1.5, object())

        for invalid_identifier in invalid_identifiers:
            identifier = cast(AddressCodeInput, invalid_identifier)
            with self.subTest(
                parameter="province_id",
                input_type=type(invalid_identifier).__name__,
            ):
                with self.assertRaises(TypeError):
                    self.parser.get_address_components_from_ids(
                        province_id=identifier,
                    )

            with self.subTest(
                parameter="district_id",
                input_type=type(invalid_identifier).__name__,
            ):
                with self.assertRaises(TypeError):
                    self.parser.get_address_components_from_ids(
                        province_id=None,
                        district_id=identifier,
                    )

            with self.subTest(
                parameter="ward_id",
                input_type=type(invalid_identifier).__name__,
            ):
                with self.assertRaises(TypeError):
                    self.parser.get_address_components_from_ids(
                        province_id=None,
                        ward_id=identifier,
                    )

    def test_public_registry_lookup_has_explicit_typed_boundaries(self) -> None:
        self.assertEqual(self.parser.normalize_address_code(1), "1")
        self.assertEqual(self.parser.normalize_address_code(" 001 "), "001")
        self.assertIsNone(self.parser.normalize_address_code("   "))
        with self.assertRaises(TypeError):
            self.parser.normalize_address_code(cast(AddressCodeInput, True))

        old_province = self.parser.get_administrative_record(
            "01",
            level="province",
            source="old",
        )
        self.assertIsNotNone(old_province)
        if old_province is None:
            self.fail("expected an old-registry province")
        original_name = old_province["name"]
        old_province["name"] = "mutated caller copy"
        fresh_record = self.parser.get_administrative_record(
            "01",
            level="province",
            source="old",
        )
        self.assertIsNotNone(fresh_record)
        if fresh_record is None:
            self.fail("expected an old-registry province")
        self.assertEqual(fresh_record["name"], original_name)

        old_ward = self.parser.get_administrative_record(
            "00004",
            level="ward",
            source="old",
        )
        self.assertIsNotNone(old_ward)
        if old_ward is None:
            self.fail("expected an old-registry ward")
        legacy_names = old_ward.get("legacy_names")
        self.assertIsNotNone(legacy_names)
        if legacy_names is None:
            self.fail("expected legacy ward names")
        legacy_names.append("caller-only alias")
        fresh_ward = self.parser.get_administrative_record(
            "00004",
            level="ward",
            source="old",
        )
        self.assertIsNotNone(fresh_ward)
        if fresh_ward is None:
            self.fail("expected an old-registry ward")
        self.assertNotIn("caller-only alias", fresh_ward.get("legacy_names", []))

        self.assertIsNone(
            self.parser.get_administrative_record(
                "001",
                level="district",
                source="new",
            )
        )
        with self.assertRaises(ValueError):
            self.parser.get_administrative_record(
                "01",
                level=cast(AdministrativeLevel, "city"),
                source="old",
            )
        with self.assertRaises(ValueError):
            self.parser.get_administrative_record(
                "01",
                level="province",
                source=cast(RegistrySource, "legacy"),
            )

    def test_mapping_lookup_rejects_invalid_identifier_types(self) -> None:
        invalid_identifiers: tuple[object, ...] = (True, 1.5, object())

        for invalid_identifier in invalid_identifiers:
            identifier = cast(AddressCodeInput, invalid_identifier)
            with self.subTest(
                method="map_old_ward_to_new",
                input_type=type(invalid_identifier).__name__,
            ):
                with self.assertRaises(TypeError):
                    self.parser.map_old_ward_to_new(identifier)

            with self.subTest(
                method="map_new_ward_to_old",
                input_type=type(invalid_identifier).__name__,
            ):
                with self.assertRaises(TypeError):
                    self.parser.map_new_ward_to_old(identifier)

    def test_registry_selector_rejects_non_boolean_values(self) -> None:
        invalid_selectors: tuple[object, ...] = (0, 1, "false", object())

        for invalid_selector in invalid_selectors:
            with self.subTest(
                method="get_address_components_from_ids",
                input_type=type(invalid_selector).__name__,
            ):
                with self.assertRaises(TypeError):
                    self.parser.get_address_components_from_ids(
                        province_id="01",
                        is_new_format=cast(bool, invalid_selector),
                    )

            with self.subTest(
                method="map_address_ids",
                input_type=type(invalid_selector).__name__,
            ):
                with self.assertRaises(TypeError):
                    self.parser.map_address_ids(
                        province_id="01",
                        district_id="001",
                        ward_id="00001",
                        is_new_format=cast(bool | None, invalid_selector),
                    )

    def test_search_rejects_invalid_query_and_source_flags(self) -> None:
        invalid_queries: tuple[object, ...] = (True, 123, 1.5, object())
        for invalid_query in invalid_queries:
            with self.subTest(input_type=type(invalid_query).__name__):
                with self.assertRaises(TypeError):
                    self.parser.search_province(cast(str, invalid_query))

        with self.assertRaises(TypeError):
            self.parser.search_ward(
                "Long An",
                include_new=cast(bool, "yes"),
            )
        with self.assertRaises(TypeError):
            self.parser.search_province(
                cast(str, 123),
                include_new=False,
                include_old=False,
            )
        with self.assertRaises(TypeError):
            self.parser.search_ward(
                cast(str, 123),
                include_new=False,
                include_old=False,
            )
        with self.assertRaises(TypeError):
            self.parser.search_district("Ba Đình", limit=cast(int, False))

    def test_search_engine_rejects_invalid_document_metadata(self) -> None:
        engine = AddressSearchEngine(
            analyzer=_analyze_test_text,
            normalize_id=_normalize_test_id,
        )
        invalid_document = cast(
            SearchDocument,
            {
                "level": "ward",
                "source": "bogus",
                "record": {"code": "1", "id": "1", "name": "Invalid"},
                "province_code": None,
                "district_code": None,
                "unit_token": None,
            },
        )

        with self.assertRaises(ValueError):
            engine.add_document(text_fields=("Invalid",), metadata=invalid_document)
        with self.assertRaises(TypeError):
            engine.search("Invalid", level=cast(AdministrativeLevel, []))
        with self.assertRaises(TypeError):
            engine.search(
                "Invalid",
                allowed_sources=cast(list[RegistrySource], [[]]),
            )

    def test_standardize_name_uses_explicit_modes(self) -> None:
        for mode in ("basic", "search", "aggressive"):
            with self.subTest(mode=mode):
                result = self.parser.standardize_name(
                    "Phường Quan Hoa",
                    cast(NormalizationMode, mode),
                )
                self.assertIsInstance(result, str)

        with self.assertRaises(TypeError):
            self.parser.standardize_name(cast(str, None))
        with self.assertRaises(ValueError):
            self.parser.standardize_name(
                "Phường Quan Hoa",
                cast(NormalizationMode, "legacy"),
            )

    def test_search_state_round_trip_accepts_all_typed_record_ids(self) -> None:
        engine = AddressSearchEngine(
            analyzer=_analyze_test_text,
            normalize_id=_normalize_test_id,
        )
        record: AdministrativeRecord = {
            "code": "100",
            "id": "100",
            "name": "Phường Kiểm Thử",
            "province_id": "01",
            "district_id": "001",
        }
        document: SearchDocument = {
            "level": "ward",
            "source": "old",
            "record": record,
            "province_code": "01",
            "district_code": "001",
            "unit_token": "phuong",
        }
        engine.add_document(text_fields=(record["name"],), metadata=document)
        engine.finalize()

        restored = AddressSearchEngine(
            analyzer=_analyze_test_text,
            normalize_id=_normalize_test_id,
        )
        restored.restore_state(engine.get_state())

        restored_record = restored.get_state()["metadata"][0]["record"]
        self.assertEqual(restored_record["province_id"], "01")
        self.assertEqual(restored_record["district_id"], "001")

    def test_empty_search_cache_rebuilds_from_nonempty_registries(self) -> None:
        empty_search_state: SearchEngineState = {
            "token_corpus": [],
            "field_tokens": [],
            "metadata": [],
            "token_sets": [],
        }
        captured = self.parser._capture_preprocessed_state()
        state: PreprocessedState = {**captured, "search_engine": empty_search_state}
        restored = AddressParser.__new__(AddressParser)

        restored._apply_preprocessed_state(state)

        self.assertIsNotNone(restored.search_engine)
        assert restored.search_engine is not None
        self.assertGreater(len(restored.search_engine.get_state()["metadata"]), 0)

    def test_mapping_results_have_correlated_direction_contracts(self) -> None:
        old_to_new = self.parser.map_address_ids(
            province_id="01",
            district_id="001",
            ward_id="00001",
            is_new_format=False,
        )
        self.assertIsNotNone(old_to_new)
        if old_to_new is None:
            self.fail("expected old-to-new mapping")
        self.assertEqual(old_to_new["direction"], "old_to_new")
        self.assertIs(old_to_new["source_format_is_new"], False)
        self.assertEqual(old_to_new["mapping"]["ward_id_new"], "00097")
        self.assertEqual(old_to_new["source"]["ward"]["id"], "00001")
        self.assertEqual(old_to_new["target"]["ward"]["id"], "00097")
        self.assertIsNone(old_to_new["target"]["district"])

        new_to_old = self.parser.map_address_ids(
            province_id="01",
            district_id=None,
            ward_id="00097",
            is_new_format=True,
        )
        self.assertIsNotNone(new_to_old)
        if new_to_old is None:
            self.fail("expected new-to-old mapping")
        self.assertEqual(new_to_old["direction"], "new_to_old")
        self.assertIs(new_to_old["source_format_is_new"], True)
        self.assertEqual(new_to_old["mapping"]["ward_id_old"], "00001")
        self.assertEqual(new_to_old["source"]["ward"]["id"], "00097")
        self.assertEqual(new_to_old["target"]["ward"]["id"], "00001")
        self.assertIsNotNone(new_to_old["target"]["district"])

    def test_invalid_nested_search_cache_is_treated_as_cache_miss(self) -> None:
        signature = self.parser._dataset_signature()
        captured = self.parser._capture_preprocessed_state()
        original_search_state = captured["search_engine"]
        if original_search_state is None:
            self.fail("expected cached search state")
        invalid_search_state = cast(
            SearchEngineState,
            {
                "token_corpus": original_search_state["token_corpus"][:3],
                "field_tokens": original_search_state["field_tokens"][:3],
                "metadata": [
                    original_search_state["metadata"][0],
                    {},
                    original_search_state["metadata"][2],
                ],
                "token_sets": original_search_state["token_sets"][:3],
            },
        )
        state: PreprocessedState = {**captured, "search_engine": invalid_search_state}
        payload: PreprocessedCachePayload = {
            "version": self.parser._CACHE_VERSION,
            "signature": signature,
            "state": state,
        }

        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "invalid.pkl"
            with cache_path.open("wb") as stream:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

            restored = AddressParser.__new__(AddressParser)
            restored._cache_path = str(cache_path)
            self.assertFalse(restored._hydrate_persistent_state(signature))
            self.assertFalse(hasattr(restored, "address_node_list"))


if __name__ == "__main__":
    unittest.main()
