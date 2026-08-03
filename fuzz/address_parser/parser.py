from __future__ import annotations

import json
import logging
import os
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from copy import deepcopy
from threading import Lock
from typing import ClassVar, cast

from rapidfuzz import fuzz as rf_fuzz
from rapidfuzz import process as rf_process
from rapidfuzz.fuzz import partial_ratio, ratio

from .contracts import (
    ADMINISTRATIVE_RECORD_KEYS,
    ADMINISTRATIVE_RECORD_OPTIONAL_INTEGER_KEYS,
    ADMINISTRATIVE_RECORD_OPTIONAL_STRING_LIST_KEYS,
    ADMINISTRATIVE_RECORD_REQUIRED_KEYS,
    ADMINISTRATIVE_RECORD_STRING_KEYS,
    SEARCH_ENGINE_STATE_KEYS,
    AddressCode,
    AddressCodeInput,
    AddressComponentsResult,
    AddressMappingSummary,
    AddressNodeState,
    AdministrativeLevel,
    AdministrativeRecord,
    AdministrativeRecordsByCode,
    CandidateHit,
    ComponentSignature,
    ContextualOldWardResult,
    ContextualWardCandidate,
    DatasetSignature,
    DetectedComponents,
    ExternalNewDataset,
    ExternalWardMappingRow,
    FuzzyChoiceProfile,
    ImmediateOldWardResult,
    LegacyAddressDataset,
    LegacyDistrictRecord,
    LegacyProvinceRecord,
    LegacyWardRecord,
    NewAddressMappingResult,
    NewToOldAddressMappingSummary,
    NgramHit,
    NormalizationMode,
    OldAddressMappingResult,
    OldToNewAddressMappingSummary,
    ParsedAddressComponent,
    ParseResult,
    PreprocessedCachePayload,
    PreprocessedState,
    ProjectedAddressComponent,
    PromotedContextualWardResult,
    RawAddressDataset,
    RegistrySource,
    SearchEngineState,
    SearchResult,
    StreetToken,
    WardMappingRow,
    WardMappingsByCode,
)
from .ngram_index import PackedNgramIndex
from .normalization import normalize_address_text
from .search_engine import (
    AddressSearchEngine,
    validate_search_metadata,
    validate_search_request,
)

logger = logging.getLogger(__name__)

SPECIAL_PROVINCE_MAP = {
    ("br vt", "br-vt", "brvt", "ba ria vung tau"): "Bà Rịa - Vũng Tàu",
    (
        "hcm",
        "h.c.m",
        "hcmc",
        "tp hcm",
        "tp.hcm",
        "tphcm",
        "tp ho chi minh",
    ): "Thành phố Hồ Chí Minh",
    (
        "dac lac",
        "dac lak",
        "dak lak",
        "dak lac",
        "daklak",
        "daklac",
        "daclac",
    ): "Đắk Lắk",
    ("con tum",): "Kon Tum",
    ("za lai",): "Gia Lai",
    (
        "tt hue",
        "tt-hue",
        "thua thien hue",
        "thua thien-hue",
        "thua thien - hue",
        "thua thienhue",
    ): "Huế",
    ("tphcm", "tp hcm", "tp. hcm", "hcm", "hcmc"): "Hồ Chí Minh",
}

CUSTOM_WARD_ALIASES_BY_CODE = {
    # Former Huế inner-city ward name retained by address sources.
    "19768": ["Thuận Thành", "Phường Thuận Thành"],
}

CUSTOM_LOCALITY_WARD_SUFFIXES = {
    # OCR often drops the comma in "Ấp Vịnh, An Cơ, Châu Thành, Tây Ninh".
    ("vinh", "an co", "chau thanh", "tay ninh"),
}


class AddressParser:
    _LIT_THI_TRAN: ClassVar[str] = "thi tran"
    _LIT_THI_XA: ClassVar[str] = "thi xa"
    _LIT_THANH_PHO: ClassVar[str] = "thanh pho"
    _LIT_DAC_KHU: ClassVar[str] = "dac khu"
    _LIT_TINH_PREFIX: ClassVar[str] = "tinh "
    _LIT_THANH_PHO_PREFIX: ClassVar[str] = "thanh pho "
    _LIT_STRIP_CHARS: ClassVar[str] = " ,;.-"
    _RE_PROVINCE_PREFIX: ClassVar[str] = r"^(tinh|thanh pho|tp)\s+"
    _RE_WORD_TOKEN: ClassVar[str] = r"\b\w+\b"
    _RE_PREFIX_PROVINCE_SEGMENTED: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:^|\|)\s*\btinh\b\s+([a-z0-9 ]+?)(?=(?:\||$))"
    )
    _RE_PREFIX_PROVINCE_INLINE: ClassVar[re.Pattern[str]] = re.compile(
        r"\btinh\b\s+([a-z0-9 ]+?)(?=(?:\b(?:quan|q|huyen|h|thi xa|tx|thi tran|tt|"
        r"phuong|p|xa|x|tp|tinh|thanh pho)\b|\||$))"
    )
    _RE_PREFIX_CITY_SEGMENTED: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:^|\|)\s*\b(?:thanh pho|tp)\b\s+([a-z0-9 ]+?)(?=(?:\||$))"
    )
    _RE_PREFIX_CITY_INLINE: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:thanh pho|tp)\b\s+([a-z0-9 ]+?)(?=(?:\b(?:quan|q|huyen|h|thi xa|tx|"
        r"thi tran|tt|phuong|p|xa|x|tp|tinh|thanh pho)\b|\||$))"
    )
    _RE_PREFIX_DISTRICT_SEGMENTED: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:^|\|)\s*\b(?P<prefix>quan|q|huyen|h|thi xa|tx|thanh pho|tp)\b\s+"
        r"(?P<fragment>[a-z0-9 ]+?)(?=(?:\||$))"
    )
    _RE_PREFIX_DISTRICT_INLINE: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?P<prefix>quan|q|huyen|h|thi xa|tx|thanh pho|tp)\b\s+"
        r"(?P<fragment>[a-z0-9 ]+?)(?=(?:\b(?:phuong|p|xa|x|thi tran|tt|quan|q|"
        r"huyen|h|thi xa|tx|thanh pho|tinh|tp)\b|\||$))"
    )
    _RE_PREFIX_WARD_SEGMENTED: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:^|\|)\s*\b(?P<prefix>phuong|p|xa|x|tt|thi tran|dac\s*khu)\b\s+"
        r"(?P<fragment>[a-z0-9 ]+?)(?=(?:\||$))"
    )
    _RE_PREFIX_WARD_INLINE: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?P<prefix>phuong|p|xa|x|tt|thi tran|dac\s*khu)\b\s+"
        r"(?P<fragment>[a-z0-9 ]+?)(?=(?:\b(?:phuong|p|xa|x|thi tran|tt|quan|q|"
        r"huyen|h|thi xa|tx|thanh pho|tinh|tp)\b|\||$))"
    )
    _RE_NUMERIC_DISTRICT: ClassVar[re.Pattern[str]] = re.compile(r"\b(?:quan|q\.?)\s*(\d{1,3})\b")
    _RE_LOT_PREFIX: ClassVar[re.Pattern[str]] = re.compile(r"\b(?:lo|lot)\s*$")
    _RE_WHITESPACE: ClassVar[re.Pattern[str]] = re.compile(r"\s+")

    _STATEFUL_ATTRS: ClassVar[tuple[str, ...]] = (
        "address_node_list",
        "invert_ngrams_idx",
        "invert_province_to_indices",
        "invert_district_to_indices",
        "invert_ward_to_indices",
        "province_names_std",
        "district_names_std",
        "ward_names_std",
        "province_lookup",
        "district_lookup",
        "district_lookup_by_name",
        "ward_lookup",
        "ward_lookup_by_name",
        "ward_lookup_by_province_name",
        "ward_lookup_by_district_key",
        "ward_mapping_by_old_code",
        "ward_mapping_by_new_code",
        "old_province_records",
        "old_district_records",
        "old_ward_records",
        "new_province_records",
        "new_ward_records",
        "external_new_province_records",
        "external_new_ward_records",
        "search_engine",
    )
    _ADDRESS_NODE_STATE_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "province_name",
            "district_name",
            "ward_name",
            "province_id",
            "district_id",
            "ward_id",
            "is_new_format",
            "standardized_full_name",
            "ngram_list",
        }
    )
    _CACHE_VERSION: ClassVar[int] = 20
    _CACHE_FILENAME: ClassVar[str] = "address_parser.preprocessed.v104.pkl"
    _PREPROCESSED_CACHE: ClassVar[PreprocessedState | None] = None
    _PREPROCESSED_SIGNATURE: ClassVar[DatasetSignature | None] = None
    _PREPROCESSED_LOCK: ClassVar[Lock] = Lock()

    class AddressNode:
        def __init__(
            self,
            province_name: str,
            district_name: str,
            ward_name: str,
            *,
            province_id: str | None = None,
            district_id: str | None = None,
            ward_id: str | None = None,
            is_new_format: bool | None = None,
        ) -> None:
            self.full_name: str = f"{ward_name} {district_name} {province_name}"
            self.full_name = re.sub(r"\s+", " ", self.full_name).strip()
            self.standardized_full_name: str = ""
            self.province_name: str = province_name
            self.district_name: str = district_name
            self.ward_name: str = ward_name
            self.ngram_list: set[str] = set()  # List of n-grams for fuzzy matching
            # None = unknown; True = new 2-level; False = old 3-level
            self.is_new_format: bool | None = is_new_format
            self.province_id: str | None = province_id
            self.district_id: str | None = district_id
            self.ward_id: str | None = ward_id

    _GENERIC_LOCATION_TOKENS: ClassVar[set[str]] = {
        "phuong",
        "p",
        "quan",
        "q",
        "huyen",
        "h",
        "thi",
        "tran",
        "xa",
        "tx",
        "tt",
        "tinh",
        "tp",
        "thanh",
        "pho",
        "thixa",
        "thitran",
        "thanhpho",
        "khu",
        "khuvuc",
        "khupho",
        "kp",
        "thon",
        "thonxom",
        "xom",
        "ap",
        "to",
        "todanpho",
        "d",
        "w",
    }

    _ADMIN_GENERIC_TOKENS: ClassVar[set[str]] = {
        "phuong",
        "p",
        "xa",
        "thi",
        "tran",
        "quan",
        "q",
        "huyen",
        "tp",
        "tinh",
        "thanh",
    }

    _LOCATION_PREFIX_SINGLE: ClassVar[set[str]] = {
        "phuong",
        "p",
        "quan",
        "q",
        "huyen",
        "h",
        "xa",
        "x",
        "tp",
        "tinh",
        "tx",
        "tt",
    }

    _LOCATION_PREFIX_MULTI: ClassVar[set[str]] = {
        _LIT_THI_TRAN,
        _LIT_THI_XA,
        _LIT_THANH_PHO,
        "khu pho",
        "khu vuc",
        _LIT_DAC_KHU,
        "khu dan cu",
        "to dan pho",
    }

    _STREET_PREFIX_SINGLE: ClassVar[set[str]] = {
        "duong",
        "pho",
        "ngo",
        "ngach",
        "hem",
        "ap",
        "thon",
        "xom",
        "khoi",
        "to",
        "khu",
        "kp",
        "cum",
        "khom",
        "so",
        "nha",
        "lo",
    }

    _STREET_PREFIX_MULTI: ClassVar[set[str]] = {
        "khu pho",
        "khu vuc",
        "to dan pho",
        "khu dan cu",
        "so nha",
    }

    def __init__(self) -> None:
        base_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.data_dir: str = os.path.join(base_dir, "data")
        self._cache_path: str = os.path.join(self.data_dir, self._CACHE_FILENAME)
        self.new_format_provinces_path: str = os.path.join(self.data_dir, "provinces.json")
        self.new_format_wards_path: str = os.path.join(self.data_dir, "wards.json")
        self.new_format_mapping_path: str = os.path.join(self.data_dir, "ward_mappings.json")
        self.old_provinces_path: str = os.path.join(self.data_dir, "old_provinces.json")
        self.old_districts_path: str = os.path.join(self.data_dir, "old_districts.json")
        self.old_wards_path: str = os.path.join(self.data_dir, "old_wards.json")
        self.legacy_virtual_wards_path: str = os.path.join(
            self.data_dir, "legacy_virtual_wards.json"
        )

        self.address_node_list: list[AddressParser.AddressNode] = []
        self.invert_ngrams_idx: dict[str, set[int]] = {}
        self._packed_ngram_index: PackedNgramIndex | None = None

        # Name-level inverted indexes for fast prefiltering by known names
        self.invert_province_to_indices: defaultdict[str, set[int]] = defaultdict(set)
        self.invert_district_to_indices: defaultdict[str, set[int]] = defaultdict(set)
        self.invert_ward_to_indices: defaultdict[str, set[int]] = defaultdict(set)

        # Flat name registries (standardized) to support prefix-based detection
        self.province_names_std: set[str] = set()
        self.district_names_std: set[str] = set()
        self.ward_names_std: set[str] = set()
        self._province_detection_choices: tuple[str, ...] = ()
        self._district_detection_choices: tuple[str, ...] = ()
        self._ward_detection_choices: tuple[str, ...] = ()
        self._fuzzy_choice_profiles: dict[str, FuzzyChoiceProfile] = {}

        # Lookup tables to attach IDs to normalized components at runtime
        self.province_lookup: dict[str, AdministrativeRecord] = {}
        self.district_lookup: dict[tuple[str, str], AdministrativeRecord] = {}
        self.district_lookup_by_name: defaultdict[str, list[AdministrativeRecord]] = defaultdict(
            list
        )
        self.ward_lookup: dict[tuple[str, str, str], AdministrativeRecord] = {}
        self.ward_lookup_by_name: defaultdict[str, list[AdministrativeRecord]] = defaultdict(list)
        self.ward_lookup_by_province_name: defaultdict[
            tuple[str, str], list[AdministrativeRecord]
        ] = defaultdict(list)
        self.ward_lookup_by_district_key: defaultdict[str, list[AdministrativeRecord]] = (
            defaultdict(list)
        )

        # Tunables to cap worst-case latency
        self.TOPK_CANDIDATES: int = 400  # bound number of candidates from inverted index
        self.DICE_GATE: float = 0.4  # only compute partial ratio when Dice >= this
        self.PARTIAL_CUTOFF: int = 40  # minimum acceptable partial ratio
        self.REFERENCE_ACCEPT_RATIO: int = 90  # minimum ratio to accept a reference override

        # Dataset level metadata & mapping snapshots for downstream features
        self.ward_mapping_by_old_code: WardMappingsByCode = {}
        self.ward_mapping_by_new_code: WardMappingsByCode = {}
        self.old_province_records: AdministrativeRecordsByCode = {}
        self.old_district_records: AdministrativeRecordsByCode = {}
        self.old_ward_records: AdministrativeRecordsByCode = {}
        self.new_province_records: AdministrativeRecordsByCode = {}
        self.new_ward_records: AdministrativeRecordsByCode = {}
        self.external_new_province_records: AdministrativeRecordsByCode = {}
        self.external_new_ward_records: AdministrativeRecordsByCode = {}

        self.search_engine: AddressSearchEngine | None = None
        self._old_ward_name_index: dict[str, list[str]] | None = None
        self._old_ward_raw_name_index: dict[str, list[str]] | None = None
        self._canonical_province_key_by_alias: dict[str, str] = (
            self._build_canonical_province_key_map()
        )

        # Pre-process address data once when initializing the Solution object
        dataset_signature: DatasetSignature = self._dataset_signature()
        if not self._hydrate_preprocessed_state(dataset_signature):
            with self._PREPROCESSED_LOCK:
                if not self._hydrate_preprocessed_state(dataset_signature):
                    if not self._hydrate_persistent_state(dataset_signature):
                        self.preprocess_address()
                        self._cache_preprocessed_state(dataset_signature)
                        self._persist_preprocessed_state(dataset_signature)
                    else:
                        # Re-cache in-memory for subsequent instances in the same process
                        self._cache_preprocessed_state(dataset_signature)

    def process(self, input_string: str) -> ParseResult:
        # Chuẩn hóa và tạo n-gram cho input
        input_string_standard: str = self.standardize_name(input_string, "search")
        input_string_basic: str = self.standardize_name(input_string, "basic")
        input_string_ngram_list: list[str] = self.generate_ngrams(input_string_standard)
        input_segments: list[tuple[str, str]] = self._split_address_segments(input_string)
        # Keep segment boundaries to avoid prefix detectors swallowing tokens across commas
        prefix_scan_input: str = (
            " | ".join(seg for seg, _ in input_segments) if input_segments else ""
        ) or input_string_basic
        explicit_raw_ward_segment: str | None = None
        if input_segments:
            for segment_std, segment_raw in input_segments:
                if not segment_std or not segment_raw:
                    continue
                if segment_std.startswith(("phuong ", "p ", "xa ", "x ", "tt ", "thi tran ")):
                    explicit_raw_ward_segment = str(segment_raw).strip(self._LIT_STRIP_CHARS)
                    break

        partial_input_string: bool = False

        def _appears_in_input(component: str | None) -> bool:
            if not component:
                return False
            component_std = self.standardize_name(component, "basic")
            if not component_std:
                return False
            if component_std in input_string_basic:
                return True

            component_core = self._strip_generic_prefix(component_std) or component_std
            if component_core and component_core in input_string_basic:
                return True
            if len(component_core) < 4:
                return False

            fuzzy_cutoff = 88 if len(component_core) >= 6 else 90
            for segment_std, _ in input_segments:
                if not segment_std:
                    continue
                if component_std in segment_std:
                    return True
                segment_core = self._strip_generic_prefix(segment_std) or segment_std
                if component_core and component_core in segment_core:
                    return True
                if self._fuzzy_match_component_key(
                    component_core,
                    [segment_std, segment_core],
                    cutoff=fuzzy_cutoff,
                ):
                    return True
            return False

        # Đếm tần suất xuất hiện của từng ngram
        ngram_counts: Counter[str] = Counter(input_string_ngram_list)

        # Lấy 5 ngram phổ biến nhất
        top_5: list[tuple[str, int]] = ngram_counts.most_common(5)
        # Nếu tổng tần suất top 5 ngram ≤ 15 → partial_input_string = True
        if top_5 and sum(count for _, count in top_5) >= 12:
            partial_input_string = True

        input_ngram_set: set[str] = set(input_string_ngram_list)

        address: AddressParser.AddressNode = self.AddressNode("", "", "")
        province_component: ParsedAddressComponent | None
        district_component: ParsedAddressComponent | None
        ward_component: ParsedAddressComponent | None

        detected_components_raw: DetectedComponents = self._detect_by_prefix(prefix_scan_input)
        detected_prov: str | None = self._validate_detected_value(
            detected_components_raw[0], self.invert_province_to_indices
        )
        detected_dist: str | None = self._validate_detected_value(
            detected_components_raw[1], self.invert_district_to_indices
        )
        detected_ward: str | None = self._validate_detected_value(
            detected_components_raw[2], self.invert_ward_to_indices
        )
        segment_suffix_detected_ward_raw: str | None = None

        def _segment_is_candidate(segment_std: str | None) -> bool:
            if not segment_std:
                return False
            if segment_std.isdigit():
                return False
            return len(segment_std) >= 3

        def _looks_like_known_province(segment_std: str | None) -> bool:
            if not segment_std:
                return False
            return bool(self._validate_detected_value(segment_std, self.invert_province_to_indices))

        single_explicit_ward_segment: bool = bool(
            len(input_segments) == 1
            and any(
                self._detect_unit_token_from_query(segment_raw) for _, segment_raw in input_segments
            )
        )

        if not detected_prov and not single_explicit_ward_segment:
            special_province = self._detect_special_province_token(input_string_basic)
            if special_province:
                detected_prov = self._validate_detected_value(
                    special_province, self.invert_province_to_indices
                )
        if not detected_prov and not single_explicit_ward_segment:
            suffix_province = self._detect_suffix_province_token(input_string_basic)
            if suffix_province:
                detected_prov = self._validate_detected_value(
                    suffix_province, self.invert_province_to_indices
                )

        def _detect_custom_locality_ward_context() -> ImmediateOldWardResult | None:
            if len(input_segments) < 3:
                return None

            descriptor_tokens: set[str] = {
                "thon",
                "xom",
                "ap",
                "to",
                "kp",
                "khu",
                "kdc",
                "kdt",
                "cum",
                "doi",
            }

            def _is_descriptor(token: str) -> bool:
                return token in descriptor_tokens or bool(re.fullmatch(r"kp\d+\w*", token))

            def _segment_matches_key(segment_std: str, expected_key: str) -> bool:
                if not segment_std or not expected_key:
                    return False
                segment_core = self._strip_generic_prefix(segment_std) or segment_std
                return segment_std == expected_key or segment_core == expected_key

            def _segment_matches_province(segment_std: str, expected_key: str) -> bool:
                if not segment_std or not expected_key:
                    return False
                if self._canonicalize_province_key(segment_std) == expected_key:
                    return True
                special = self._detect_special_province_token(segment_std)
                return bool(special and self._canonicalize_province_key(special) == expected_key)

            for (
                locality_key,
                ward_key,
                district_key,
                province_key,
            ) in CUSTOM_LOCALITY_WARD_SUFFIXES:
                province_idx: int | None = None
                for idx, (segment_std, _) in enumerate(input_segments):
                    if _segment_matches_province(segment_std, province_key):
                        province_idx = idx
                        break
                if province_idx is None:
                    continue

                district_idx: int | None = None
                for idx, (segment_std, _) in enumerate(input_segments):
                    if idx == province_idx:
                        continue
                    if _segment_matches_key(segment_std, district_key):
                        district_idx = idx
                        break
                if district_idx is None:
                    continue

                district_info = self._lookup_district_info(district_key, province_key)
                if not district_info:
                    continue
                province_id = self.normalize_address_code(district_info.get("parent_code"))
                province_info = (
                    self.old_province_records.get(province_id) if province_id else None
                ) or self._lookup_province_info(province_key)
                if not province_info:
                    continue

                ward_info = self._lookup_ward_info(
                    ward_key,
                    province_key,
                    district_key,
                    preferred_format=False,
                )
                if not ward_info or ward_info.get("is_new_format") is not False:
                    continue

                ward_tokens = [token for token in ward_key.split() if token]
                if not ward_tokens:
                    continue
                for idx, (segment_std, segment_raw) in enumerate(input_segments):
                    if idx in {province_idx, district_idx} or not segment_std:
                        continue
                    tokens = [token for token in segment_std.split() if token]
                    if len(tokens) <= len(ward_tokens):
                        continue
                    if tokens[-len(ward_tokens) :] != ward_tokens:
                        continue
                    prefix_tokens = tokens[: -len(ward_tokens)]
                    meaningful_prefix = [
                        token
                        for token in prefix_tokens
                        if not _is_descriptor(token) and token not in self._GENERIC_LOCATION_TOKENS
                    ]
                    if " ".join(meaningful_prefix) != locality_key:
                        continue

                    raw_fragment = self._recover_raw_suffix_fragment(
                        segment_raw,
                        len(ward_tokens),
                    ) or self._titleize_token(ward_key)
                    raw_street = self._strip_trailing_component_fragment(
                        str(segment_raw).strip(self._LIT_STRIP_CHARS),
                        raw_fragment,
                    )
                    street_parts: list[str] = []
                    for part_idx, (_, raw_part) in enumerate(input_segments):
                        if part_idx in {province_idx, district_idx}:
                            continue
                        cleaned = str(raw_part).strip(self._LIT_STRIP_CHARS)
                        if not cleaned:
                            continue
                        if part_idx == idx:
                            cleaned = raw_street
                        if cleaned:
                            street_parts.append(cleaned)

                    return {
                        "province_info": province_info,
                        "district_info": district_info,
                        "ward_info": ward_info,
                        "street_address": " ".join(street_parts).strip(),
                    }

            return None

        def _detect_explicit_old_ward_segment_context() -> ImmediateOldWardResult | None:
            if len(input_segments) < 3 or not detected_prov:
                return None

            province_name = self._resolve_detected_component(
                "province",
                detected_prov,
                source_string=input_string_basic,
            )
            if not province_name:
                return None
            province_key = self._canonicalize_province_key(province_name)
            if not province_key:
                return None

            province_idx: int | None = None
            for idx, (segment_std, _) in enumerate(input_segments):
                if not segment_std:
                    continue
                segment_key = self._canonicalize_province_key(segment_std)
                if segment_key == province_key:
                    province_idx = idx
                    break
                special = self._detect_special_province_token(segment_std)
                if special and self._canonicalize_province_key(special) == province_key:
                    province_idx = idx
                    break
            if province_idx is None:
                return None

            district_idx: int | None = None
            district_info: AdministrativeRecord | None = None
            for idx in range(len(input_segments) - 1, -1, -1):
                if idx == province_idx:
                    continue
                segment_std, segment_raw = input_segments[idx]
                if not segment_std:
                    continue
                district_info = self._lookup_district_info(segment_raw, province_name)
                if district_info is None and segment_raw != segment_std:
                    district_info = self._lookup_district_info(segment_std, province_name)
                if district_info:
                    district_idx = idx
                    break
            if district_idx is None or not district_info:
                return None

            district_name = (
                district_info.get("name")
                or district_info.get("full_name")
                or input_segments[district_idx][1]
            )
            best_candidate: tuple[tuple[int, int, int], int, AdministrativeRecord] | None = None
            for idx, (segment_std, segment_raw) in enumerate(input_segments):
                if idx in {province_idx, district_idx} or not segment_std:
                    continue
                fragment = str(segment_raw).strip(self._LIT_STRIP_CHARS)
                if not fragment:
                    continue
                ward_entry = self._lookup_ward_info(
                    fragment,
                    province_name,
                    district_name,
                    preferred_format=False,
                )
                if not ward_entry or ward_entry.get("is_new_format") is not False:
                    continue
                if not self._entry_matches_component_fragment(
                    ward_entry,
                    fragment,
                    level="ward",
                ):
                    continue

                candidate_score = (
                    2 if self._entry_matches_raw_query_name(ward_entry, fragment) else 0,
                    1
                    if self._entry_matches_query_name(
                        ward_entry,
                        fragment,
                        include_aliases=True,
                    )
                    else 0,
                    idx,
                )
                if best_candidate is None or candidate_score > best_candidate[0]:
                    best_candidate = (candidate_score, idx, ward_entry)

            if not best_candidate:
                return None

            _, ward_idx, best_ward_info = best_candidate
            street_parts: list[str] = []
            for idx, (segment_std, segment_raw) in enumerate(input_segments):
                if idx in {province_idx, district_idx, ward_idx}:
                    continue
                if segment_std in {"viet nam", "vietnam"}:
                    continue
                if province_key == "hue" and segment_std == "tt":
                    continue
                cleaned = str(segment_raw).strip(self._LIT_STRIP_CHARS)
                if cleaned:
                    street_parts.append(cleaned)

            province_info = self._lookup_province_info(province_name)
            if province_info is None:
                return None
            return {
                "province_info": province_info,
                "district_info": district_info,
                "ward_info": best_ward_info,
                "street_address": " ".join(street_parts).strip(),
            }

        custom_locality_ward = _detect_custom_locality_ward_context()
        if custom_locality_ward:
            custom_province_info = custom_locality_ward["province_info"]
            custom_district_info = custom_locality_ward["district_info"]
            custom_ward_info = custom_locality_ward["ward_info"]
            custom_province_component: ParsedAddressComponent | None = self._format_component(
                custom_province_info.get("name") or custom_province_info.get("full_name"),
                custom_province_info.get("id") or custom_province_info.get("code"),
                custom_province_info,
            )
            custom_district_component: ParsedAddressComponent | None = self._format_component(
                custom_district_info.get("name") or custom_district_info.get("full_name"),
                custom_district_info.get("id") or custom_district_info.get("code"),
                custom_district_info,
            )
            custom_ward_component: ParsedAddressComponent | None = self._format_component(
                custom_ward_info.get("name") or custom_ward_info.get("full_name"),
                custom_ward_info.get("id") or custom_ward_info.get("code"),
                custom_ward_info,
            )
            return {
                "province": custom_province_component,
                "district": custom_district_component,
                "ward": custom_ward_component,
                "street_address": custom_locality_ward["street_address"],
                "format": "old",
                "is_new": False,
            }

        explicit_old_ward_context = _detect_explicit_old_ward_segment_context()
        if explicit_old_ward_context:
            explicit_province_info = explicit_old_ward_context["province_info"]
            explicit_district_info = explicit_old_ward_context["district_info"]
            explicit_ward_info = explicit_old_ward_context["ward_info"]
            return {
                "province": self._format_component(
                    explicit_province_info.get("name") or explicit_province_info.get("full_name"),
                    explicit_province_info.get("id") or explicit_province_info.get("code"),
                    explicit_province_info,
                ),
                "district": self._format_component(
                    explicit_district_info.get("name") or explicit_district_info.get("full_name"),
                    explicit_district_info.get("id") or explicit_district_info.get("code"),
                    explicit_district_info,
                ),
                "ward": self._format_component(
                    explicit_ward_info.get("name") or explicit_ward_info.get("full_name"),
                    explicit_ward_info.get("id") or explicit_ward_info.get("code"),
                    explicit_ward_info,
                ),
                "street_address": explicit_old_ward_context["street_address"],
                "format": "old",
                "is_new": False,
            }

        def _has_explicit_ward_only_segment() -> bool:
            for segment_std, _ in input_segments:
                if not segment_std:
                    continue
                tokens = [token for token in segment_std.split() if token]
                if not tokens:
                    return False
                first = tokens[0]
                pair = " ".join(tokens[:2]) if len(tokens) >= 2 else ""
                if first in {"phuong", "p", "xa", "x"}:
                    return True
                if first == "tt" and len(tokens) >= 2:
                    return True
                if pair in {
                    self._LIT_THI_TRAN,
                    self._LIT_DAC_KHU,
                }:
                    return True
            return False

        contextual_old_ward: ContextualOldWardResult | None = None
        if len(input_segments) >= 2 and not _has_explicit_ward_only_segment():
            contextual_old_ward = self._detect_contextual_old_ward(
                input_segments,
                input_string_basic,
                detected_prov,
            )

        if contextual_old_ward:
            promoted_new_context: PromotedContextualWardResult | None = (
                self._promote_contextual_old_ward_to_new(contextual_old_ward)
            )
            if promoted_new_context:
                promoted_province_info: AdministrativeRecord = promoted_new_context["province_info"]
                promoted_ward_info: AdministrativeRecord = promoted_new_context["ward_info"]
                return {
                    "province": self._format_component(
                        promoted_province_info.get("name")
                        or promoted_province_info.get("full_name"),
                        promoted_province_info.get("id") or promoted_province_info.get("code"),
                        promoted_province_info,
                    ),
                    "district": None,
                    "ward": self._format_component(
                        promoted_ward_info.get("full_name") or promoted_ward_info.get("name"),
                        promoted_ward_info.get("id") or promoted_ward_info.get("code"),
                        promoted_ward_info,
                    ),
                    "street_address": contextual_old_ward["street_address"],
                    "format": "new",
                    "is_new": True,
                }

            contextual_province_info: AdministrativeRecord = contextual_old_ward["province_info"]
            contextual_district_info: AdministrativeRecord = contextual_old_ward["district_info"]
            contextual_ward_info: AdministrativeRecord = contextual_old_ward["ward_info"]
            return {
                "province": self._format_component(
                    contextual_province_info.get("name")
                    or contextual_province_info.get("full_name"),
                    contextual_province_info.get("id") or contextual_province_info.get("code"),
                    contextual_province_info,
                ),
                "district": self._format_component(
                    contextual_district_info.get("name")
                    or contextual_district_info.get("full_name"),
                    contextual_district_info.get("id") or contextual_district_info.get("code"),
                    contextual_district_info,
                ),
                "ward": self._format_component(
                    contextual_ward_info.get("name") or contextual_ward_info.get("full_name"),
                    contextual_ward_info.get("id") or contextual_ward_info.get("code"),
                    contextual_ward_info,
                ),
                "street_address": contextual_old_ward["street_address"],
                "format": "old",
                "is_new": False,
            }

        if explicit_raw_ward_segment and detected_prov:
            has_explicit_district_prefix: bool = False
            for segment_std, _ in input_segments:
                if not segment_std:
                    continue
                tokens = [token for token in segment_std.split() if token]
                if not tokens:
                    continue
                if tokens[0] in {"quan", "q", "huyen", "h", "tx"}:
                    has_explicit_district_prefix = True
                    break
                if len(tokens) >= 2 and " ".join(tokens[:2]) == self._LIT_THI_XA:
                    has_explicit_district_prefix = True
                    break
                if tokens[0] == "tp" and len(tokens) >= 2:
                    has_explicit_district_prefix = True
                    break
                if len(tokens) >= 3 and " ".join(tokens[:2]) == self._LIT_THANH_PHO:
                    has_explicit_district_prefix = True
                    break

            if not has_explicit_district_prefix:
                explicit_province_name = self._resolve_detected_component(
                    "province",
                    detected_prov,
                    source_string=input_string_basic,
                )
                explicit_new_ward = (
                    self._lookup_ward_info(
                        explicit_raw_ward_segment,
                        explicit_province_name if explicit_province_name else None,
                        None,
                        preferred_format=True,
                    )
                    if explicit_province_name
                    else None
                )
                raw_segment_std = self.standardize_name(explicit_raw_ward_segment, "basic")
                explicit_new_matches = False
                if explicit_new_ward and raw_segment_std:
                    explicit_new_matches = self._entry_matches_query_name(
                        explicit_new_ward,
                        explicit_raw_ward_segment,
                    )
                if (
                    explicit_new_matches
                    and explicit_new_ward is not None
                    and explicit_new_ward.get("is_new_format") is True
                ):
                    explicit_province_id = self.normalize_address_code(
                        explicit_new_ward.get("province_id")
                    )
                    if not explicit_province_id:
                        explicit_ward_id = self.normalize_address_code(
                            explicit_new_ward.get("id") or explicit_new_ward.get("code")
                        )
                        if explicit_ward_id:
                            explicit_new_record = self.new_ward_records.get(explicit_ward_id)
                            if isinstance(explicit_new_record, dict):
                                explicit_province_id = self.normalize_address_code(
                                    explicit_new_record.get("parent_code")
                                )
                    if not explicit_province_id:
                        explicit_province_id = self._lookup_new_province_id_by_name(
                            explicit_province_name
                        )
                    explicit_new_province_info = (
                        self.new_province_records.get(explicit_province_id)
                        if explicit_province_id
                        else self._lookup_province_info(explicit_province_name)
                    )
                    province_component = self._format_component(
                        explicit_province_name,
                        explicit_province_id,
                        explicit_new_province_info,
                    )
                    explicit_new_ward_name = explicit_new_ward.get(
                        "full_name"
                    ) or explicit_new_ward.get("name")
                    ward_component = self._format_component(
                        explicit_new_ward_name,
                        explicit_new_ward.get("id") or explicit_new_ward.get("code"),
                        explicit_new_ward,
                    )
                    province_key = self._canonicalize_province_key(explicit_province_name)
                    street_parts: list[str] = []
                    for segment_std, segment_raw in input_segments:
                        raw_part = str(segment_raw).strip(self._LIT_STRIP_CHARS)
                        if not raw_part:
                            continue
                        if raw_part == explicit_raw_ward_segment:
                            continue
                        if self._canonicalize_province_key(segment_std) == province_key:
                            continue
                        if segment_std in {"viet nam", "vietnam"}:
                            continue
                        street_parts.append(raw_part)
                    street_address = " ".join(street_parts).strip()
                    if province_component:
                        province_component["aliases"] = self._gather_alias_values(
                            explicit_province_name,
                            explicit_new_province_info,
                            level="province",
                            extra_values=[detected_prov],
                        )
                    if ward_component:
                        ward_component["aliases"] = self._gather_alias_values(
                            explicit_new_ward_name,
                            explicit_new_ward,
                            level="ward",
                            extra_values=[
                                explicit_raw_ward_segment,
                                self._normalize_detected_ward_token(explicit_raw_ward_segment),
                            ],
                        )
                    return {
                        "province": province_component,
                        "district": None,
                        "ward": ward_component,
                        "street_address": street_address,
                        "format": "new",
                        "is_new": True,
                    }

        # Fallback: infer ward/district from comma-separated segments when
        # the input omits explicit prefixes (e.g. "Tứ Hạ, Hương Trà").
        compact_prefixed_ward_raw: str | None = None
        compact_prefixed_district_raw: str | None = None
        compact_prefixed_ward_info: AdministrativeRecord | None = None
        compact_prefixed_district_info: AdministrativeRecord | None = None

        def _detect_compact_prefixed_ward_district_segment() -> (
            dict[str, AdministrativeRecord | str] | None
        ):
            if not input_segments:
                return None

            province_hint = (
                self._resolve_detected_component(
                    "province",
                    detected_prov,
                    source_string=input_string_basic,
                )
                if detected_prov
                else None
            )

            for segment_std, segment_raw in input_segments:
                if not segment_std or not segment_raw:
                    continue
                tokens = [token for token in segment_std.split() if token]
                if len(tokens) < 2:
                    continue

                split_index: int | None = None
                ward_key: str | None = None
                first = tokens[0]
                if first in {"p", "phuong", "x", "xa"} and len(tokens) >= 3:
                    if not re.fullmatch(r"\d+\w*", tokens[1]):
                        continue
                    prefix = "phuong" if first in {"p", "phuong"} else "xa"
                    digits = re.match(r"\d+", tokens[1])
                    if not digits:
                        continue
                    ward_key = f"{prefix} {digits.group(0)}"
                    split_index = 2
                else:
                    compact_match = re.fullmatch(r"(p|phuong|x|xa)(\d+\w*)", first)
                    if not compact_match:
                        continue
                    prefix = "phuong" if compact_match.group(1) in {"p", "phuong"} else "xa"
                    digits = re.match(r"\d+", compact_match.group(2))
                    if not digits:
                        continue
                    ward_key = f"{prefix} {digits.group(0)}"
                    split_index = 1

                if split_index is None or not ward_key:
                    continue

                district_key: str | None = None
                district_fragment_std: str | None = None
                district_lookup_fragment_std: str | None = None
                district_entry: AdministrativeRecord | None = None
                for end_index in range(len(tokens), split_index, -1):
                    candidate = " ".join(tokens[split_index:end_index]).strip()
                    if not candidate:
                        continue
                    lookup_candidates = [candidate]
                    stripped_candidate = self._strip_generic_prefix(candidate)
                    if stripped_candidate and stripped_candidate != candidate:
                        lookup_candidates.append(stripped_candidate)

                    for lookup_candidate in lookup_candidates:
                        district_entry = self._lookup_district_info(
                            lookup_candidate,
                            province_hint if province_hint else None,
                        )
                        if not district_entry and not province_hint:
                            matched_district = self._validate_detected_value(
                                lookup_candidate,
                                self.invert_district_to_indices,
                            )
                            if matched_district:
                                district_entry = self._lookup_district_info(matched_district)
                        if district_entry:
                            district_lookup_fragment_std = lookup_candidate
                            break

                    if district_entry:
                        district_key = (
                            self.standardize_name(
                                cast(
                                    str,
                                    district_entry.get("name") or district_entry.get("full_name"),
                                ),
                                "basic",
                            )
                            or candidate
                        )
                        district_fragment_std = candidate
                        break
                if not district_key or not district_fragment_std:
                    continue

                ward_entry = self._lookup_ward_info(
                    ward_key,
                    province_hint if province_hint else None,
                    district_lookup_fragment_std or district_fragment_std,
                    preferred_format=False,
                )
                if not ward_entry or ward_entry.get("is_new_format") is not False:
                    continue
                resolved_district_entry = cast(AdministrativeRecord, district_entry)
                old_district_id = self.normalize_address_code(
                    resolved_district_entry.get("id") or resolved_district_entry.get("code")
                )
                old_district_entry = (
                    self.old_district_records.get(old_district_id) if old_district_id else None
                ) or resolved_district_entry
                old_ward_id = self.normalize_address_code(
                    ward_entry.get("id") or ward_entry.get("code")
                )
                old_ward_entry = (
                    self.old_ward_records.get(old_ward_id) if old_ward_id else None
                ) or ward_entry

                raw_text = str(segment_raw).strip(self._LIT_STRIP_CHARS)
                raw_match = re.match(
                    r"^\s*((?:p|phường|phuong|x|xã|xa)\.?\s*\d+\w*)"
                    r"[\s,;.-]+(.+?)\s*$",
                    raw_text,
                    flags=re.IGNORECASE,
                )
                if raw_match:
                    raw_ward = cast(str, raw_match.group(1)).strip(self._LIT_STRIP_CHARS)
                    raw_district = cast(str, raw_match.group(2)).strip(self._LIT_STRIP_CHARS)
                else:
                    raw_ward = self._titleize_token(ward_key)
                    raw_district = self._titleize_token(district_fragment_std)

                return {
                    "ward": ward_key,
                    "district": district_key,
                    "raw_ward": raw_ward,
                    "raw_district": raw_district,
                    "ward_info": old_ward_entry,
                    "district_info": old_district_entry,
                }

            return None

        compact_prefixed: dict[str, AdministrativeRecord | str] | None = (
            _detect_compact_prefixed_ward_district_segment()
        )
        if compact_prefixed:
            detected_ward = cast(str, compact_prefixed["ward"])
            detected_dist = cast(str, compact_prefixed["district"])
            compact_prefixed_ward_raw = cast(str, compact_prefixed["raw_ward"])
            compact_prefixed_district_raw = cast(str, compact_prefixed["raw_district"])
            compact_prefixed_ward_info = cast(AdministrativeRecord, compact_prefixed["ward_info"])
            compact_prefixed_district_info = cast(
                AdministrativeRecord, compact_prefixed["district_info"]
            )
            explicit_raw_ward_segment = compact_prefixed_ward_raw

        if (not detected_dist or not detected_ward) and len(input_segments) >= 2:
            if not detected_dist:
                for offset_from_tail, (segment_std, _) in enumerate(reversed(input_segments)):
                    if (
                        not _segment_is_candidate(segment_std)
                        or segment_std == detected_prov
                        or (offset_from_tail == 0 and _looks_like_known_province(segment_std))
                    ):
                        continue
                    matched_district: str | None = None
                    if segment_std in self.district_names_std:
                        matched_district = segment_std
                    elif offset_from_tail <= 2:
                        matched_district = self._fuzzy_match_component_key(
                            segment_std,
                            self.district_names_std,
                            cutoff=89,
                        )
                    if not matched_district:
                        continue
                    detected_dist = self._validate_detected_value(
                        matched_district, self.invert_district_to_indices
                    )
                    if detected_dist:
                        break
            if not detected_ward:
                # Prefer the ward candidate closest to the district/province tail.
                # In comma-separated VN addresses, street/sub-address often appears
                # earlier while ward is usually the last admin segment before district.
                for offset_from_tail, (segment_std, _) in enumerate(reversed(input_segments)):
                    if not _segment_is_candidate(segment_std):
                        continue
                    if offset_from_tail == 0 and _looks_like_known_province(segment_std):
                        continue
                    if detected_prov and offset_from_tail == 0:
                        continue
                    if detected_dist and offset_from_tail <= 1:
                        district_like = self._fuzzy_match_component_key(
                            segment_std,
                            self.district_names_std,
                            cutoff=89,
                        )
                        if district_like and district_like == detected_dist:
                            continue
                    matched_ward: str | None = None
                    if segment_std in self.ward_names_std:
                        matched_ward = segment_std
                    elif offset_from_tail <= 2:
                        matched_ward = self._fuzzy_match_component_key(
                            segment_std,
                            self.ward_names_std,
                            cutoff=88 if detected_dist or detected_prov else 90,
                        )
                    if not matched_ward:
                        normalized_numeric_ward = self._normalize_detected_ward_token(segment_std)
                        if normalized_numeric_ward and normalized_numeric_ward != segment_std:
                            if normalized_numeric_ward in self.ward_names_std:
                                matched_ward = normalized_numeric_ward
                            elif offset_from_tail <= 2:
                                matched_ward = self._fuzzy_match_component_key(
                                    normalized_numeric_ward,
                                    self.ward_names_std,
                                    cutoff=88 if detected_dist or detected_prov else 90,
                                )
                    if not matched_ward:
                        continue
                    if detected_dist and matched_ward == detected_dist:
                        continue
                    detected_ward = self._validate_detected_value(
                        matched_ward, self.invert_ward_to_indices
                    )
                    if detected_ward:
                        break
            if not detected_ward:
                for segment_idx, (segment_std, segment_raw) in enumerate(input_segments):
                    if not _segment_is_candidate(segment_std):
                        continue
                    suffix_match = self._infer_ward_from_segment_suffix(
                        segment_std,
                        segment_raw,
                        expected_province=detected_prov,
                        expected_district=detected_dist,
                        allow_plain_prefix=bool(
                            segment_idx > 0 and (detected_prov or detected_dist)
                        ),
                    )
                    if not suffix_match:
                        continue
                    raw_fragment, detected_fragment = suffix_match
                    if detected_dist and detected_fragment == detected_dist:
                        continue
                    detected_ward = detected_fragment
                    segment_suffix_detected_ward_raw = raw_fragment
                    break

        raw_detected_ward: str | None = detected_components_raw[2]
        if compact_prefixed_ward_raw:
            raw_detected_ward = compact_prefixed_ward_raw
        if segment_suffix_detected_ward_raw and not raw_detected_ward:
            raw_detected_ward = segment_suffix_detected_ward_raw
        if detected_ward and not raw_detected_ward:
            raw_detected_ward = self._recover_component_from_input(detected_ward, input_segments)
        raw_detected_dist: str | None = None
        normalized_detected_ward_token: str | None = (
            self._normalize_detected_ward_token(raw_detected_ward) if raw_detected_ward else None
        )
        if detected_dist:
            resolved_hint = self._resolve_detected_component(
                "district",
                detected_dist,
                expected_province=detected_prov,
                source_string=input_string_basic,
            )
            raw_detected_dist = resolved_hint
            if not raw_detected_dist:
                recovered = (
                    compact_prefixed_district_raw
                    or self._prefer_component_alias_from_segments(
                        [detected_dist],
                        input_segments,
                        require_prefix=True,
                        level="district",
                    )
                    or self._recover_component_from_input(detected_dist, input_segments)
                )
                if recovered:
                    cleaned = recovered.strip()
                    parts = [part for part in cleaned.split() if part]
                    if parts:
                        first_std = self.standardize_name(parts[0], "basic")
                        second_std = (
                            self.standardize_name(parts[1], "basic") if len(parts) >= 2 else ""
                        )
                        if first_std in {"huyen", "quan", "tp"}:
                            cleaned = " ".join(parts[1:])
                        elif first_std == "thi" and second_std == "xa":
                            cleaned = " ".join(parts[2:])
                        elif first_std == "thanh" and second_std == "pho":
                            cleaned = " ".join(parts[2:])
                    recovered = cleaned.strip()
                raw_detected_dist = recovered
        district_hint_in_input: bool = bool(raw_detected_dist)
        district_present_in_input: bool = district_hint_in_input

        def _is_province_level_city(city_name: str) -> bool:
            city_name = (city_name or "").strip()
            if not city_name:
                return False
            normalized_city_name = self._detect_special_province_token(city_name) or city_name
            province_id_new = self._lookup_new_province_id_by_name(normalized_city_name)
            if not province_id_new:
                return False
            province_record = self.external_new_province_records.get(
                province_id_new
            ) or self.new_province_records.get(province_id_new)
            return bool(
                isinstance(province_record, dict)
                and province_record.get("administrative_unit_id") == 1
            )

        def _is_known_district_alias(candidate: str | None) -> bool:
            candidate = (candidate or "").strip()
            if not candidate:
                return False
            return bool(self._validate_detected_value(candidate, self.invert_district_to_indices))

        # If any comma-separated segment explicitly starts with a district-level
        # prefix (e.g. "Huyện ...", "Quận ...", "Thị xã ..."), treat the input as
        # old format even if prefix-based name resolution fails.
        def _segment_has_valid_tinh(segment_std: str) -> bool:
            if not segment_std or not segment_std.startswith(self._LIT_TINH_PREFIX):
                return False
            fragment = segment_std[len(self._LIT_TINH_PREFIX) :].strip()
            if not fragment:
                return False
            tokens = [tok for tok in fragment.split() if tok]
            while tokens and tokens[-1] in {"viet", "nam", "vietnam"}:
                tokens.pop()
            fragment = " ".join(tokens)
            if not fragment:
                return False
            if fragment in self.province_names_std:
                return True
            special = self._detect_special_province_token(fragment)
            return bool(special and special in self.province_names_std)

        has_tinh_prefix: bool = False
        has_province_segment: bool = False
        if input_segments:
            for segment_std, _ in input_segments:
                if _segment_has_valid_tinh(segment_std or ""):
                    has_tinh_prefix = True
                    has_province_segment = True
                    break
                if (
                    segment_std
                    and segment_std in self.province_names_std
                    and not segment_std.startswith(("tp ", self._LIT_THANH_PHO_PREFIX))
                ):
                    has_province_segment = True
                    break
        district_prefix_in_input: bool = False
        if input_segments:
            for segment_std, segment_raw in input_segments:
                if not segment_std:
                    continue
                tokens = [tok for tok in segment_std.split() if tok]
                if not tokens:
                    continue
                first = tokens[0]
                if first in {"huyen", "quan", "q", "tx"}:
                    district_prefix_in_input = True
                    district_hint_in_input = True
                    district_present_in_input = True
                    if not raw_detected_dist and segment_raw:
                        raw_detected_dist = str(segment_raw).strip(self._LIT_STRIP_CHARS)
                    break
                if first == "h":
                    raw = (segment_raw or "").strip().lower()
                    if raw.startswith(("h.", "huyện", "huyen")):
                        district_prefix_in_input = True
                        district_hint_in_input = True
                        district_present_in_input = True
                        if not raw_detected_dist and segment_raw:
                            raw_detected_dist = str(segment_raw).strip(self._LIT_STRIP_CHARS)
                        break
                if len(tokens) >= 2 and f"{tokens[0]} {tokens[1]}" == self._LIT_THI_XA:
                    district_prefix_in_input = True
                    district_hint_in_input = True
                    district_present_in_input = True
                    if not raw_detected_dist and segment_raw:
                        raw_detected_dist = str(segment_raw).strip(self._LIT_STRIP_CHARS)
                    break
                if first == "tp" and len(tokens) >= 2:
                    city_name = " ".join(tokens[1:]).strip()
                    if city_name and (
                        has_tinh_prefix
                        or has_province_segment
                        or not _is_province_level_city(city_name)
                    ):
                        if (
                            has_tinh_prefix
                            or has_province_segment
                            or (
                                _is_known_district_alias(f"tp {city_name}")
                                or _is_known_district_alias(f"{self._LIT_THANH_PHO} {city_name}")
                                or _is_known_district_alias(city_name)
                            )
                        ):
                            district_prefix_in_input = True
                            district_hint_in_input = True
                            district_present_in_input = True
                            if not raw_detected_dist and segment_raw:
                                raw_detected_dist = str(segment_raw).strip(self._LIT_STRIP_CHARS)
                            break
                if len(tokens) >= 3 and f"{tokens[0]} {tokens[1]}" == self._LIT_THANH_PHO:
                    city_name = " ".join(tokens[2:]).strip()
                    if city_name and (
                        has_tinh_prefix
                        or has_province_segment
                        or not _is_province_level_city(city_name)
                    ):
                        if (
                            has_tinh_prefix
                            or has_province_segment
                            or (
                                _is_known_district_alias(f"{self._LIT_THANH_PHO} {city_name}")
                                or _is_known_district_alias(f"tp {city_name}")
                                or _is_known_district_alias(city_name)
                            )
                        ):
                            district_prefix_in_input = True
                            district_hint_in_input = True
                            district_present_in_input = True
                            if not raw_detected_dist and segment_raw:
                                raw_detected_dist = str(segment_raw).strip(self._LIT_STRIP_CHARS)
                            break

        if not district_prefix_in_input and input_segments:
            for segment_std, segment_raw in input_segments:
                if not segment_std:
                    continue
                tokens = [tok for tok in segment_std.split() if tok]
                if not tokens:
                    continue
                raw_lower = (segment_raw or "").strip().lower()

                def _matches_known_district(
                    start_index: int,
                    *,
                    min_tokens: int,
                    max_tokens: int,
                ) -> bool:
                    if start_index >= len(tokens):
                        return False
                    max_len = min(len(tokens) - start_index, max_tokens)
                    for length in range(min_tokens, max_len + 1):
                        candidate = " ".join(tokens[start_index : start_index + length])
                        if self._validate_detected_value(
                            candidate, self.invert_district_to_indices
                        ):
                            return True
                    return False

                def _matches_prefix_or_fragment(
                    prefix_start: int,
                    fragment_start: int,
                    *,
                    min_prefix_tokens: int,
                    max_prefix_tokens: int,
                ) -> bool:
                    if _matches_known_district(
                        prefix_start,
                        min_tokens=min_prefix_tokens,
                        max_tokens=max_prefix_tokens,
                    ):
                        return True
                    return _matches_known_district(
                        fragment_start,
                        min_tokens=1,
                        max_tokens=3,
                    )

                for idx, token in enumerate(tokens):
                    if token in {"quan", "q"}:
                        prev = tokens[idx - 1] if idx > 0 else ""
                        if prev in {"lo", "lot"}:
                            continue
                        if idx + 1 < len(tokens) and tokens[idx + 1].isdigit():
                            district_prefix_in_input = True
                            district_hint_in_input = True
                            district_present_in_input = True
                            break
                        if _matches_prefix_or_fragment(
                            idx,
                            idx + 1,
                            min_prefix_tokens=2,
                            max_prefix_tokens=4,
                        ):
                            district_prefix_in_input = True
                            district_hint_in_input = True
                            district_present_in_input = True
                            break
                        continue

                    if token in {"huyen", "h"}:
                        prev = tokens[idx - 1] if idx > 0 else ""
                        if prev == "duong":
                            continue
                        if token == "h":
                            if not raw_lower.startswith(("h.", "huyen")):
                                continue
                        if token == "huyen" and not (
                            "huyen" in raw_lower or re.search(r"\bhuyen\b", raw_lower)
                        ):
                            continue
                        if _matches_prefix_or_fragment(
                            idx,
                            idx + 1,
                            min_prefix_tokens=2,
                            max_prefix_tokens=4,
                        ):
                            district_prefix_in_input = True
                            district_hint_in_input = True
                            district_present_in_input = True
                            break
                        continue

                    if token == "thi" and idx + 1 < len(tokens) and tokens[idx + 1] == "xa":
                        if _matches_prefix_or_fragment(
                            idx,
                            idx + 2,
                            min_prefix_tokens=3,
                            max_prefix_tokens=5,
                        ):
                            district_prefix_in_input = True
                            district_hint_in_input = True
                            district_present_in_input = True
                            break
                        continue

                    if token == "tx":
                        if _matches_prefix_or_fragment(
                            idx,
                            idx + 1,
                            min_prefix_tokens=2,
                            max_prefix_tokens=4,
                        ):
                            district_prefix_in_input = True
                            district_hint_in_input = True
                            district_present_in_input = True
                            break
                        continue

                    if token == "tp" and idx + 1 < len(tokens):
                        city_name = " ".join(tokens[idx + 1 :]).strip()
                        if city_name and not _is_province_level_city(city_name):
                            if _matches_prefix_or_fragment(
                                idx,
                                idx + 1,
                                min_prefix_tokens=2,
                                max_prefix_tokens=5,
                            ):
                                district_prefix_in_input = True
                                district_hint_in_input = True
                                district_present_in_input = True
                                break
                        continue

                    if token == "thanh" and idx + 2 < len(tokens) and tokens[idx + 1] == "pho":
                        city_name = " ".join(tokens[idx + 2 :]).strip()
                        if city_name and not _is_province_level_city(city_name):
                            if _matches_prefix_or_fragment(
                                idx,
                                idx + 2,
                                min_prefix_tokens=3,
                                max_prefix_tokens=6,
                            ):
                                district_prefix_in_input = True
                                district_hint_in_input = True
                                district_present_in_input = True
                                break
                        continue
                if district_prefix_in_input:
                    break

        def _expected_district_for_resolution() -> str | None:
            if not district_hint_in_input:
                return None
            if district:
                return district
            for token in (detected_dist, raw_detected_dist):
                if not token:
                    continue
                token_std = (
                    token if token == detected_dist else self.standardize_name(token, "basic")
                )
                if not token_std:
                    continue
                resolved = self._resolve_detected_component(
                    "district",
                    token_std,
                    expected_province=province,
                    source_string=input_string_basic,
                )
                if resolved:
                    return resolved
            return raw_detected_dist

        detected_components: DetectedComponents = (
            detected_prov,
            detected_dist,
            detected_ward,
        )
        ngram_address_piece_list: list[NgramHit] = self.ngram_address_piece_list(
            input_string_ngram_list, self.TOPK_CANDIDATES
        )

        enforced_new_ward_entry: AdministrativeRecord | None = None

        address_candidate: list[CandidateHit] = self.address_candidate_list(
            input_string_standard,
            input_ngram_set,
            ngram_address_piece_list,
            partial_input_string,
            detected_components,
        )

        if address_candidate and not single_explicit_ward_segment:
            selected_idx: int | None = self._select_candidate_with_hints(
                address_candidate,
                detected_components,
            )
            if selected_idx is not None:
                address = self.address_node_list[selected_idx]
            else:
                address = self.address_node_list[address_candidate[0][0]]

        province: str | None = address.province_name
        district: str | None = address.district_name
        ward: str | None = address.ward_name
        province_id: str | None = address.province_id
        district_id: str | None = address.district_id
        ward_id: str | None = address.ward_id
        candidate_is_new_format: bool | None = address.is_new_format
        province_info: AdministrativeRecord | None
        district_info: AdministrativeRecord | None
        ward_info: AdministrativeRecord | None
        resolved_is_new_format: bool | None

        if not province and detected_prov:
            resolved_province = self._resolve_detected_component(
                "province", detected_prov, source_string=input_string_basic
            )
            if resolved_province:
                province = resolved_province
                province_id = None
        elif province and detected_prov:
            resolved_province = self._resolve_detected_component(
                "province", detected_prov, source_string=input_string_basic
            )
            if resolved_province:
                current_std = self.standardize_name(province, "basic")
                resolved_std = self.standardize_name(resolved_province, "basic")
                if current_std and resolved_std and current_std != resolved_std:
                    province = resolved_province
                    province_id = None

        if not district and detected_dist:
            resolved_district = self._resolve_detected_component(
                "district",
                detected_dist,
                expected_province=province,
                source_string=input_string_basic,
            )
            if resolved_district:
                district = resolved_district
                district_id = None
            elif raw_detected_dist:
                district = raw_detected_dist
                district_id = None
        elif not district and raw_detected_dist and district_prefix_in_input:
            district = raw_detected_dist
            district_id = None

        if district and detected_dist and district != detected_dist:
            district_std = self.standardize_name(district, "basic")
            if district_std and district_std.isdigit() and detected_dist.isdigit():
                resolved = self._resolve_detected_component(
                    "district",
                    detected_dist,
                    expected_province=province,
                    source_string=input_string_basic,
                )
                if resolved:
                    district = resolved
                    district_id = None

        if not ward and detected_ward and (raw_detected_dist or district_present_in_input):
            resolved_ward = self._resolve_detected_component(
                "ward",
                detected_ward,
                expected_province=province,
                expected_district=_expected_district_for_resolution(),
                source_string=input_string_basic,
            )
            if resolved_ward:
                ward = resolved_ward
                ward_id = None
            else:
                detected_ward = None
        if not ward and raw_detected_ward:
            normalized_detected_ward = (
                normalized_detected_ward_token
                if normalized_detected_ward_token is not None
                else self._normalize_detected_ward_token(raw_detected_ward)
            )
            new_entry: AdministrativeRecord | None = None
            if not district_hint_in_input:
                new_entry = self._lookup_new_format_ward_alias(
                    normalized_detected_ward,
                    expected_province=province,
                )
                if new_entry:
                    new_entry = self._prefer_hierarchical_ward_entry(
                        normalized_detected_ward,
                        new_entry,
                        expected_province=province,
                    )
            if new_entry:
                entry_is_new = new_entry.get("is_new_format")
                ward = new_entry.get("name") or raw_detected_ward
                ward_id = new_entry.get("id") or ward_id
                district_from_entry = new_entry.get("district_name")
                if district_from_entry:
                    district = district_from_entry
                    district_id = None
                    district_info = None
                elif entry_is_new is True:
                    district = ""
                    district_id = None
                    district_info = None
                detected_ward = None
                if entry_is_new is True:
                    resolved_is_new_format = True
                    candidate_is_new_format = True
                elif entry_is_new is False:
                    resolved_is_new_format = False
                    candidate_is_new_format = False
                province_from_entry = new_entry.get("province_name")
                if province_from_entry:
                    entry_matches_hint = self._entry_aligns_with_province(new_entry, province)
                    if not province or not entry_matches_hint:
                        province = province_from_entry
                        province_id = None

        enforcement_token: str | None = None
        if explicit_raw_ward_segment:
            enforcement_token = self._normalize_detected_ward_token(
                explicit_raw_ward_segment
            ) or self.standardize_name(explicit_raw_ward_segment, "basic")
        if not enforcement_token:
            enforcement_token = detected_ward or normalized_detected_ward_token
        if not enforcement_token and raw_detected_ward:
            enforcement_token = (
                self._normalize_detected_ward_token(raw_detected_ward) or raw_detected_ward
            )
        if enforcement_token and not district_prefix_in_input:
            new_format_entry = self._lookup_new_format_ward_alias(
                enforcement_token,
                expected_province=province,
            )
            if new_format_entry:
                new_format_entry = cast(
                    AdministrativeRecord,
                    self._prefer_hierarchical_ward_entry(
                        enforcement_token,
                        new_format_entry,
                        expected_province=province,
                    ),
                )
                enforced_new_ward_entry = new_format_entry
                ward_name_new = new_format_entry.get("name")
                if ward_name_new:
                    ward = ward_name_new
                elif not ward:
                    ward = detected_ward
                ward_id_new = new_format_entry.get("id")
                if ward_id_new:
                    ward_id = ward_id_new
                entry_is_new = new_format_entry.get("is_new_format")
                district_from_entry = new_format_entry.get("district_name")
                if entry_is_new is True or not district_from_entry:
                    district = ""
                    district_id = None
                    district_info = None
                else:
                    district = district_from_entry
                    district_id = None
                    district_info = None
                if entry_is_new is True:
                    resolved_is_new_format = True
                    candidate_is_new_format = True
                elif entry_is_new is False:
                    resolved_is_new_format = False
                    candidate_is_new_format = False
                province_from_entry = new_format_entry.get("province_name")
                if province_from_entry:
                    entry_matches_hint = self._entry_aligns_with_province(
                        new_format_entry, province
                    )
                    if not province or not entry_matches_hint:
                        province = province_from_entry
                        province_id = None

        if ward and detected_ward:
            current_ward_std = self.standardize_name(ward, "basic")
            should_try_override = False
            if current_ward_std and current_ward_std.isdigit() and detected_ward.isdigit():
                should_try_override = current_ward_std != detected_ward
            elif current_ward_std and current_ward_std != detected_ward:
                detected_token = normalized_detected_ward_token or detected_ward
                token_in_input = bool(detected_token and detected_token in input_string_basic)
                # Prefer the detected hint when it carries extra specificity
                if token_in_input and len(detected_ward) > len(current_ward_std):
                    should_try_override = True
                elif token_in_input and not current_ward_std.startswith(detected_ward):
                    should_try_override = True
                elif detected_ward.startswith(current_ward_std):
                    should_try_override = True

            if should_try_override:
                resolved = self._resolve_detected_component(
                    "ward",
                    detected_ward,
                    expected_province=province,
                    expected_district=_expected_district_for_resolution(),
                    source_string=input_string_basic,
                )
                if resolved:
                    ward = resolved
                    ward_id = None
                    ward_info = None

        if ward:
            current_ward_std = self.standardize_name(ward, "basic")
            if current_ward_std:
                validated_current = self._resolve_detected_component(
                    "ward",
                    current_ward_std,
                    expected_province=province,
                    expected_district=_expected_district_for_resolution(),
                    source_string=input_string_basic,
                )
                if not validated_current:
                    ward = ""
                    ward_id = None

        if province and not _appears_in_input(province):
            replacement = None
            if detected_prov:
                replacement = self._resolve_detected_component(
                    "province", detected_prov, source_string=input_string_basic
                )
                if replacement and not _appears_in_input(replacement):
                    replacement = None
            if not replacement:
                candidates = [
                    prov_std
                    for prov_std in self.province_names_std
                    if prov_std in input_string_basic
                ]
                for prov_std in sorted(candidates, key=len, reverse=True):
                    resolved = self._resolve_detected_component(
                        "province", prov_std, source_string=input_string_basic
                    )
                    if resolved:
                        replacement = resolved
                        break
            if replacement:
                province = replacement
                province_id = None
            else:
                province = ""
                province_id = None

        if district and not _appears_in_input(district):
            replacement = None
            if detected_dist:
                replacement = self._resolve_detected_component(
                    "district",
                    detected_dist,
                    expected_province=province if province else None,
                    source_string=input_string_basic,
                )
                if replacement and not _appears_in_input(replacement):
                    replacement = None
            if not replacement:
                province_std = self.standardize_name(province, "basic") if province else None
                for dist_std, entries in self.district_lookup_by_name.items():
                    if dist_std not in input_string_basic:
                        continue
                    for entry in entries:
                        if province and not self._entry_aligns_with_province(entry, province):
                            continue
                        candidate_name = entry.get("name")
                        if candidate_name:
                            replacement = candidate_name
                            break
                    if replacement:
                        break
            if replacement:
                district = replacement
                district_id = None
            else:
                district = ""
                district_id = None

        if ward and not _appears_in_input(ward):
            replacement = None
            if detected_ward:
                replacement = self._resolve_detected_component(
                    "ward",
                    detected_ward,
                    expected_province=province if province else None,
                    expected_district=_expected_district_for_resolution(),
                    source_string=input_string_basic,
                )
                if replacement and not _appears_in_input(replacement):
                    replacement = None
            if not replacement:
                recovered = self._recover_component_from_input(
                    normalized_detected_ward_token or detected_ward,
                    input_segments,
                )
                if recovered:
                    replacement = recovered
            if not replacement:
                province_std = self.standardize_name(province, "basic") if province else None
                replacement_district_std = (
                    self.standardize_name(district, "basic") if district else None
                )
                for ward_std, entries in self.ward_lookup_by_name.items():
                    if ward_std not in input_string_basic:
                        continue
                    for entry in entries:
                        if province and not self._entry_aligns_with_province(entry, province):
                            continue
                        if (
                            replacement_district_std
                            and entry.get("district_key") != replacement_district_std
                        ):
                            continue
                        candidate_name = entry.get("name")
                        if candidate_name:
                            replacement = candidate_name
                            break
                    if replacement:
                        break
            if replacement:
                ward = replacement
                ward_id = None
                ward_info = None
            else:
                ward = ""
                ward_id = None

        if not district and detected_dist:
            resolved_district = self._resolve_detected_component(
                "district",
                detected_dist,
                expected_province=province,
                source_string=input_string_basic,
            )
            if resolved_district:
                district = resolved_district
                district_id = None

        if not ward and detected_ward:
            resolved_ward = self._resolve_detected_component(
                "ward",
                detected_ward,
                expected_province=province,
                expected_district=_expected_district_for_resolution(),
                source_string=input_string_basic,
            )
            if resolved_ward:
                ward = resolved_ward
                ward_id = None
            else:
                detected_ward = None

        if detected_ward:
            resolved_detected_ward = self._resolve_detected_component(
                "ward",
                detected_ward,
                expected_province=province if province else None,
                expected_district=_expected_district_for_resolution(),
                source_string=input_string_basic,
            )
            if resolved_detected_ward and _appears_in_input(resolved_detected_ward):
                if not ward or not _appears_in_input(ward):
                    ward = resolved_detected_ward
                    ward_id = None
                    ward_info = None

        if not district and ward and candidate_is_new_format is not True and district_hint_in_input:
            inferred_district = self._infer_district_from_components(
                province,
                ward,
                source_string=input_string_basic,
            )
            if inferred_district:
                district = inferred_district
                district_id = None

        if not province:
            inferred_province = self._infer_province_from_components(district, ward)
            if inferred_province:
                province = inferred_province
                province_id = None

        province_info = self._lookup_province_info(province) if province else None
        if not province:
            province_id = None
        elif province_info and province_info.get("id") is not None:
            province_id = province_info["id"]

        if compact_prefixed_district_info and compact_prefixed_ward_info:
            district = (
                compact_prefixed_district_info.get("name")
                or compact_prefixed_district_info.get("full_name")
                or district
            )
            district_id = (
                compact_prefixed_district_info.get("id")
                or compact_prefixed_district_info.get("code")
                or district_id
            )
            ward = (
                compact_prefixed_ward_info.get("full_name")
                or compact_prefixed_ward_info.get("name")
                or ward
            )
            ward_id = (
                compact_prefixed_ward_info.get("id")
                or compact_prefixed_ward_info.get("code")
                or ward_id
            )
            district_hint_in_input = True
            district_present_in_input = True
            candidate_is_new_format = False

        province_for_lookup = province if province else None
        district_info = (
            self._lookup_district_info(district, province_for_lookup) if district else None
        )
        if not district:
            district_id = None
        elif district_info and district_info.get("id") is not None:
            district_id = district_info["id"]

        if district_hint_in_input and detected_dist:
            enforced_district = self._resolve_detected_component(
                "district",
                detected_dist,
                expected_province=province_for_lookup,
                source_string=input_string_basic,
            )
            if enforced_district:
                enforced_std = self.standardize_name(enforced_district, "basic")
                enforced_current_std = (
                    self.standardize_name(district, "basic") if district else None
                )
                if enforced_std and enforced_std != enforced_current_std:
                    district = enforced_district
                    district_info = (
                        self._lookup_district_info(district, province_for_lookup)
                        if district
                        else None
                    )
                    district_id = None
                    if district_info and district_info.get("id") is not None:
                        district_id = district_info["id"]

        district_for_lookup = district if district else None

        def _has_strict_region_hint() -> bool:
            province_hint = bool(
                province_for_lookup and (detected_prov or _appears_in_input(province))
            )
            district_hint = bool(district_for_lookup and district_hint_in_input)
            return province_hint or district_hint

        def _update_format(
            current_value: bool | None,
            info_value: AdministrativeRecord | None,
        ) -> bool | None:
            if info_value and info_value.get("is_new_format") is True:
                return True
            if info_value and info_value.get("is_new_format") is False:
                return False
            return current_value

        ward_info = (
            self._lookup_ward_info(
                ward,
                province_for_lookup,
                district_for_lookup,
                preferred_format=candidate_is_new_format,
            )
            if ward
            else None
        )
        if ward and ward_info is None and not _has_strict_region_hint():
            ward_info = self._lookup_ward_info(ward, preferred_format=candidate_is_new_format)
        enforce_locked_new_format = enforced_new_ward_entry is not None
        if enforced_new_ward_entry is not None:
            ward_info = enforced_new_ward_entry

        def _ward_prefix_from_value(value: str | None) -> str | None:
            if not value:
                return None
            std = self.standardize_name(value, "basic")
            if not std:
                return None
            if std.startswith(("phuong ", "p ")):
                return "phuong"
            if std.startswith(("xa ", "x ")):
                return "xa"
            if std.startswith(("thi tran ", "tt ")):
                return "thi tran"
            if std.startswith("dac khu "):
                return "dac khu"
            return None

        raw_ward_segment = explicit_raw_ward_segment
        if (
            raw_ward_segment is None
            and len(input_segments) == 1
            and raw_detected_ward
            and _ward_prefix_from_value(raw_detected_ward)
        ):
            raw_ward_segment = raw_detected_ward
        ward_prefix_hint = _ward_prefix_from_value(raw_ward_segment)

        def _component_keys_match(left: str | None, right: str | None) -> bool:
            if not left or not right:
                return False
            left_std = self.standardize_name(left, "basic")
            right_std = self.standardize_name(right, "basic")
            if not left_std or not right_std:
                return False
            if left_std == right_std:
                return True
            left_stripped = self._strip_generic_prefix(left_std) or left_std
            right_stripped = self._strip_generic_prefix(right_std) or right_std
            return left_stripped == right_stripped

        explicit_flat_admin_signal = len(input_segments) == 1 and bool(
            detected_prov or detected_dist or raw_detected_ward
        )
        explicit_ward_signal = bool(ward_prefix_hint) or (
            explicit_flat_admin_signal
            and raw_detected_ward
            and self._segment_has_location_prefix(self.standardize_name(raw_detected_ward, "basic"))
        )

        if detected_dist and (district_prefix_in_input or explicit_flat_admin_signal):
            resolved_prefixed_district = self._resolve_detected_component(
                "district",
                detected_dist,
                expected_province=province_for_lookup,
                source_string=input_string_basic,
            )
            if resolved_prefixed_district and not _component_keys_match(
                district, resolved_prefixed_district
            ):
                district = resolved_prefixed_district
                district_id = None
                district_info = None
                district_for_lookup = district

        prefixed_ward_token = detected_ward
        if raw_ward_segment:
            explicit_segment_token = self._validate_detected_value(
                self.standardize_name(raw_ward_segment, "basic"),
                self.invert_ward_to_indices,
            )
            if not explicit_segment_token:
                explicit_segment_token = self._validate_detected_value(
                    self._normalize_detected_ward_token(raw_ward_segment),
                    self.invert_ward_to_indices,
                )
            if explicit_segment_token:
                prefixed_ward_token = explicit_segment_token

        if prefixed_ward_token and explicit_ward_signal:
            resolved_prefixed_ward = self._resolve_detected_component(
                "ward",
                prefixed_ward_token,
                expected_province=province if province else None,
                expected_district=_expected_district_for_resolution(),
                source_string=input_string_basic,
            )
            if resolved_prefixed_ward and not _component_keys_match(ward, resolved_prefixed_ward):
                ward = resolved_prefixed_ward
                ward_id = None
                ward_info = None

        ward_lookup_hint = raw_ward_segment or normalized_detected_ward_token or raw_detected_ward

        if raw_ward_segment:
            exact_old = self._lookup_old_ward_record_by_exact_name(raw_ward_segment)
            exact_old_id = (
                self.normalize_address_code(exact_old.get("id") or exact_old.get("code"))
                if isinstance(exact_old, dict)
                else None
            )

            def _old_ward_matches_province(
                entry: AdministrativeRecord,
                expected_province_name: str | None,
            ) -> bool:
                if not expected_province_name:
                    return True
                expected_std = self.standardize_name(expected_province_name, "basic")
                if not expected_std:
                    return True
                parent_code = self.normalize_address_code(
                    entry.get("parent_code") or entry.get("district_code")
                )
                if not parent_code:
                    return True
                district_entry = self.old_district_records.get(parent_code)
                if not isinstance(district_entry, dict):
                    return True
                province_code = self.normalize_address_code(
                    district_entry.get("parent_code") or district_entry.get("province_code")
                )
                if not province_code:
                    return True
                province_entry = self.old_province_records.get(province_code)
                if not isinstance(province_entry, dict):
                    return True
                province_name = province_entry.get("name") or province_entry.get("full_name")
                province_std = (
                    self.standardize_name(province_name, "basic") if province_name else None
                )
                if not province_std:
                    return True
                return (
                    province_std == expected_std
                    or province_std.endswith(expected_std)
                    or expected_std.endswith(province_std)
                )

            # Use exact_old when it matches province; allow it even if the code exists in
            # new format when the input has a district (old-format address) so ward_code is set.
            if (
                exact_old
                and exact_old_id
                and _old_ward_matches_province(exact_old, province)
                and (
                    exact_old_id not in self.new_ward_records
                    or district_present_in_input
                    or bool(district)
                )
            ):
                ward_info = self._enrich_old_ward_with_province(
                    {**exact_old, "is_new_format": False}
                )
                ward_id = exact_old_id
                ward = exact_old.get("full_name") or exact_old.get("name") or ward
                candidate_is_new_format = False
                if not district_present_in_input:
                    district = ""
                    district_id = None
                    district_info = None
        if not ward:
            ward_id = None
        elif ward_info and ward_info.get("id") is not None:
            ward_id = ward_info["id"]

        province_confident = bool(detected_prov) or _appears_in_input(province)
        if (
            ward_info
            and province
            and province_confident
            and not self._entry_aligns_with_province(ward_info, province)
        ):

            def _enforce_ward_by_province(token: str | None) -> str | None:
                if not token:
                    return None
                return self._resolve_detected_component(
                    "ward",
                    token,
                    expected_province=province,
                    expected_district=_expected_district_for_resolution(),
                    source_string=input_string_basic,
                )

            enforced_name = None
            for ward_token_candidate in (
                detected_ward,
                normalized_detected_ward_token,
                self.standardize_name(ward, "basic") if ward else None,
            ):
                enforced_name = _enforce_ward_by_province(ward_token_candidate)
                if enforced_name:
                    break
            if enforced_name:
                ward = enforced_name
                ward_id = None
                ward_info = self._lookup_ward_info(
                    ward,
                    province_for_lookup,
                    district_for_lookup,
                    preferred_format=candidate_is_new_format,
                )
                if ward_info is None and not _has_strict_region_hint():
                    ward_info = self._lookup_ward_info(
                        ward, preferred_format=candidate_is_new_format
                    )
            if not ward_info or not self._entry_aligns_with_province(ward_info, province):
                ward_info = None
                ward_id = None
                if not enforced_name and raw_detected_ward:
                    # keep the textual hint even if we cannot map it to the requested province
                    ward = self._recover_component_from_input(
                        raw_detected_ward,
                        input_segments,
                    ) or self._titleize_token(raw_detected_ward)

        resolved_is_new_format = _update_format(candidate_is_new_format, ward_info)

        if ward_info:
            entry_aligns_province = self._entry_aligns_with_province(ward_info, province)
            ward_province = ward_info.get("province_name")
            if ward_province and (
                not province or (not province_confident and not entry_aligns_province)
            ):
                province = ward_province
                province_id = None
                province_for_lookup = province
                province_confident = bool(detected_prov) or _appears_in_input(province)
                entry_aligns_province = True
            ward_district = ward_info.get("district_name")
            if (
                ward_district
                and district_hint_in_input
                and (entry_aligns_province or not province_confident)
            ):
                district = ward_district
                district_id = None
                district_info = None
                district_for_lookup = district
            province_info = self._lookup_province_info(province) if province else None
            district_info = (
                self._lookup_district_info(district, province_for_lookup) if district else None
            )

        if ward_info and not district and district_hint_in_input:
            recovered_district_name, recovered_district_id = self._recover_district_from_ward_info(
                ward_info,
                ward,
                province,
                province_info,
            )
            if recovered_district_name:
                district = recovered_district_name
                district_id = recovered_district_id
                district_info = (
                    self._lookup_district_info(district, province_for_lookup) if district else None
                )
                if not district_id and district_info and district_info.get("id") is not None:
                    district_id = district_info["id"]
                ward_info = {
                    **ward_info,
                    "district_name": district,
                    "district_key": ward_info.get("district_key")
                    or self.standardize_name(district, "basic"),
                }

        district_for_lookup = district if district else None

        preferred_ward_from_input = self._prefer_component_alias_from_segments(
            self._gather_alias_values(
                ward,
                ward_info,
                level="ward",
                extra_values=[raw_detected_ward, normalized_detected_ward_token],
            ),
            input_segments,
            require_prefix=True,
            level="ward",
        )
        if preferred_ward_from_input:
            ward_before = ward
            ward = preferred_ward_from_input
            if self.standardize_name(ward_before or "", "basic") != self.standardize_name(
                ward, "basic"
            ):
                ward_info = self._lookup_ward_info(
                    ward,
                    province_for_lookup,
                    district_for_lookup,
                    preferred_format=candidate_is_new_format,
                )
                if ward_info is None and not _has_strict_region_hint():
                    ward_info = self._lookup_ward_info(
                        ward, preferred_format=candidate_is_new_format
                    )
                ward_id = ward_info.get("id") if ward_info else None
                resolved_is_new_format = _update_format(resolved_is_new_format, ward_info)
                ward_district = ward_info.get("district_name") if ward_info else None
                if ward_district and not district:
                    district = ward_district
                    district_id = None
                    district_info = None

        if detected_ward and district:
            district_std = self.standardize_name(district, "basic")
            detected_ward_std = self.standardize_name(ward, "basic") if ward else None
            if (
                district_std
                and district_std == detected_ward
                and detected_ward_std != detected_ward
            ):
                pass
            elif district_std and district_std == detected_ward:
                detected_ward = None

        ward_present_override = False
        # If the input only provides a district name without any district prefix,
        # but that name corresponds to a special new-format ward (e.g. "Đặc khu"),
        # treat it as a new-format ward to avoid misclassifying as old format.
        if district and not ward and not district_prefix_in_input and not district_hint_in_input:
            district_key = self.standardize_name(district, "basic")
            if district_key:
                new_entry = self._lookup_new_format_ward_alias(
                    district_key,
                    expected_province=province,
                )
                if new_entry and new_entry.get("is_new_format") is True:
                    entry_name = new_entry.get("full_name") or new_entry.get("name")
                    entry_name_std = (
                        self.standardize_name(entry_name, "basic") if entry_name else ""
                    )
                    if not entry_name_std.startswith("dac khu"):
                        new_entry = None
                if new_entry and new_entry.get("is_new_format") is True:
                    ward_info = new_entry
                    ward = new_entry.get("name") or district
                    ward_id = new_entry.get("id") or ward_id
                    province_from_entry = new_entry.get("province_name")
                    if province_from_entry:
                        province = province_from_entry
                        province_id = None
                    district = ""
                    district_id = None
                    district_info = None
                    resolved_is_new_format = True
                    candidate_is_new_format = True
                    ward_present_override = True

        ward_present_in_input = ward_present_override or _appears_in_input(ward)
        if (not ward or not ward_present_in_input) and raw_detected_ward:
            normalized_raw_token = self._normalize_detected_ward_token(raw_detected_ward)
            recovered_from_input = self._recover_component_from_input(
                normalized_raw_token or raw_detected_ward,
                input_segments,
            )
            new_entry = None
            province_from_entry = None

            if not district_present_in_input:
                new_entry = self._lookup_new_format_ward_alias(
                    normalized_raw_token,
                    expected_province=province,
                )
                if new_entry:
                    entry_name = new_entry.get("name") or new_entry.get("full_name")
                    entry_name_std = (
                        self.standardize_name(entry_name, "basic") if entry_name else ""
                    )
                    if (
                        entry_name_std
                        and normalized_raw_token
                        and entry_name_std != normalized_raw_token
                    ):
                        new_entry = None

            if not new_entry:
                matched_existing = None
                raw_lookup_district_std = (
                    self.standardize_name(district, "basic") if district else None
                )
                if raw_lookup_district_std:
                    district_entries = self.ward_lookup_by_district_key.get(
                        raw_lookup_district_std, []
                    )
                    for entry in district_entries:
                        entry_name = entry.get("name") or entry.get("full_name")
                        entry_name_std = (
                            self.standardize_name(entry_name, "basic") if entry_name else ""
                        )
                        if entry_name_std == normalized_raw_token or self._numeric_token_match(
                            entry_name_std, normalized_raw_token
                        ):
                            matched_existing = entry
                            break
                if matched_existing:
                    new_entry = matched_existing
                    province_from_entry = matched_existing.get("province_name")

            if not new_entry:
                fallback = self._resolve_detected_component(
                    "ward",
                    raw_detected_ward,
                    expected_province=province,
                    expected_district=_expected_district_for_resolution(),
                    source_string=input_string_basic,
                )
                if (
                    not fallback
                    and normalized_raw_token
                    and normalized_raw_token != raw_detected_ward
                ):
                    fallback = self._resolve_detected_component(
                        "ward",
                        normalized_raw_token,
                        expected_province=province,
                        expected_district=_expected_district_for_resolution(),
                        source_string=input_string_basic,
                    )
                if fallback and recovered_from_input and not _appears_in_input(fallback):
                    fallback = recovered_from_input
                if fallback and not district_present_in_input and not _appears_in_input(fallback):
                    fallback = None
                if fallback:
                    ward = fallback
                    ward_id = None
                    ward_present_in_input = True
                    detected_ward = None
                else:
                    fallback_name = None
                    fallback_id = ward_id
                    fallback_province_name = province
                    province_std = self.standardize_name(province, "basic") if province else None
                    ward_candidates = self.ward_lookup_by_name.get(normalized_raw_token, [])
                    for ward_candidate_entry in ward_candidates:
                        candidate_entry_name = ward_candidate_entry.get("name")
                        candidate_entry_name_std = (
                            self.standardize_name(candidate_entry_name, "basic")
                            if candidate_entry_name
                            else None
                        )
                        candidate_entry_province = ward_candidate_entry.get("province_name")
                        entry_province_std = (
                            self.standardize_name(candidate_entry_province, "basic")
                            if candidate_entry_province
                            else None
                        )
                        if (
                            candidate_entry_name
                            and candidate_entry_name_std
                            and (
                                candidate_entry_name_std == normalized_raw_token
                                or self._numeric_token_match(
                                    candidate_entry_name_std, normalized_raw_token
                                )
                            )
                            and (
                                not province_std
                                or not entry_province_std
                                or province_std == entry_province_std
                            )
                        ):
                            fallback_name = candidate_entry_name
                            fallback_id = fallback_id or ward_candidate_entry.get("id")
                            fallback_province_name = (
                                candidate_entry_province or fallback_province_name
                            )
                            matched_existing = ward_candidate_entry
                            break
                    if matched_existing:
                        new_entry = matched_existing
                        province_from_entry = matched_existing.get("province_name")
                    elif not district_present_in_input:
                        fallback_province_value = fallback_province_name or province
                        new_entry = {
                            "id": fallback_id,
                            "code": fallback_id,
                            "name": fallback_name
                            or recovered_from_input
                            or self._titleize_token(raw_detected_ward),
                            "province_name": fallback_province_value,
                            "province_key": (
                                self.standardize_name(fallback_province_value, "basic")
                                if fallback_province_value
                                else None
                            ),
                            "district_name": "",
                            "district_key": "",
                            "is_new_format": True,
                        }
                    elif district:
                        new_entry = {
                            "id": fallback_id,
                            "code": fallback_id,
                            "name": fallback_name
                            or recovered_from_input
                            or self._titleize_token(raw_detected_ward),
                            "province_name": province,
                            "province_key": (
                                self.standardize_name(province, "basic") if province else None
                            ),
                            "district_name": district,
                            "district_key": (
                                self.standardize_name(district, "basic") if district else None
                            ),
                            "is_new_format": False,
                        }

            if new_entry:
                ward_info = new_entry
                ward = new_entry.get("name") or self._titleize_token(raw_detected_ward)
                ward_id = new_entry.get("id") or ward_id
                ward_present_in_input = True
                entry_is_new = new_entry.get("is_new_format")
                if province_from_entry:
                    entry_matches_hint = self._entry_aligns_with_province(new_entry, province)
                    if entry_is_new or not province or not entry_matches_hint:
                        province = province_from_entry
                        province_id = None
                district_from_entry = new_entry.get("district_name")
                if district_from_entry and not district_present_in_input:
                    district = district_from_entry
                    district_id = None
                    district_info = None
                elif entry_is_new and not district_present_in_input:
                    district = ""
                    district_id = None
                    district_info = None
                if entry_is_new is True and not district_present_in_input:
                    resolved_is_new_format = True
                    candidate_is_new_format = True
                elif entry_is_new is False:
                    resolved_is_new_format = False
                    candidate_is_new_format = False
        if ward and not ward_present_in_input and not detected_ward and not raw_detected_ward:
            ward = ""
            ward_id = None
            ward_info = None

        if ward and district and ward_present_in_input and not district_hint_in_input:
            ward_std = self.standardize_name(ward, "basic")
            district_std = self.standardize_name(district, "basic")
            if (
                ward_std
                and district_std
                and (
                    ward_std == district_std
                    or (
                        ward_std.startswith(district_std)
                        and ward_std.split()[0] in {"phuong", "p", "xa", "thi", "tran"}
                    )
                    or (
                        district_std.startswith(ward_std)
                        and district_std.split()[0] in {"phuong", "p", "xa", "thi", "tran"}
                    )
                    or (
                        self._strip_generic_prefix(ward_std)
                        == self._strip_generic_prefix(district_std)
                        and ward_std.split()[0] in {"phuong", "p", "xa", "thi", "tran"}
                    )
                )
            ):
                district = ""
                district_id = None
                district_info = None
                resolved_is_new_format = True
                candidate_is_new_format = True

        def _std_name(value: str | None) -> str:
            return self.standardize_name(value, "basic") if value else ""

        def _canonical_region_key(value: str | None) -> str:
            key = _std_name(value)
            if not key:
                return ""
            key = re.sub(self._RE_PROVINCE_PREFIX, "", key).strip()
            return key

        def _old_ward_id_matches_province(
            ward_id_value: str | None,
            province_name_value: str | None,
        ) -> bool:
            ward_key = self.normalize_address_code(ward_id_value)
            if not ward_key or not province_name_value:
                return True
            expected_std = self.standardize_name(province_name_value, "basic")
            if not expected_std:
                return True
            ward_entry = self.old_ward_records.get(ward_key)
            if not isinstance(ward_entry, dict):
                return True
            parent_code = self.normalize_address_code(
                ward_entry.get("parent_code") or ward_entry.get("district_code")
            )
            if not parent_code:
                return True
            district_entry = self.old_district_records.get(parent_code)
            if not isinstance(district_entry, dict):
                return True
            province_code = self.normalize_address_code(
                district_entry.get("parent_code") or district_entry.get("province_code")
            )
            if not province_code:
                return True
            province_entry = self.old_province_records.get(province_code)
            if not isinstance(province_entry, dict):
                return True
            province_name = province_entry.get("name") or province_entry.get("full_name")
            province_std = self.standardize_name(province_name, "basic") if province_name else None
            if not province_std:
                return True
            return (
                province_std == expected_std
                or province_std.endswith(expected_std)
                or expected_std.endswith(province_std)
            )

        def _is_legacy_only_ward(
            ward_id_value: str | None,
            ward_name_value: str | None,
            province_name_value: str | None,
        ) -> bool:
            ward_key = self.normalize_address_code(ward_id_value)
            if not ward_key:
                return False
            if ward_key not in self.old_ward_records or ward_key in self.new_ward_records:
                return False
            if not _old_ward_id_matches_province(ward_key, province_name_value):
                return False
            if ward_name_value:
                candidate = self._lookup_ward_info(
                    ward_name_value,
                    province_name_value if province_name_value else None,
                    None,
                    preferred_format=True,
                )
                if candidate and candidate.get("is_new_format") is True:
                    return False
            return True

        def _resolve_two_level_ward_candidate(
            ward_name_value: str | None,
            province_name_value: str | None,
        ) -> tuple[AdministrativeRecord | None, bool | None]:
            if not ward_name_value:
                return None, None
            province_hint = province_name_value if province_name_value else None

            new_candidate = self._lookup_ward_info(
                ward_name_value,
                province_hint,
                None,
                preferred_format=True,
            )
            if new_candidate and new_candidate.get("is_new_format") is True:
                return new_candidate, True

            old_candidate = self._lookup_ward_info(
                ward_name_value,
                province_hint,
                None,
                preferred_format=False,
            )
            if old_candidate and old_candidate.get("is_new_format") is False:
                return old_candidate, False

            fallback = self._lookup_ward_info(
                ward_name_value,
                province_hint,
                None,
                preferred_format=None,
            )
            fallback_format = fallback.get("is_new_format") if isinstance(fallback, dict) else None
            return fallback, fallback_format

        def _apply_two_level_ward_resolution(
            *,
            drop_inferred_district_for_new: bool,
        ) -> bool | None:
            nonlocal ward, ward_id, ward_info
            nonlocal district, district_id, district_info
            nonlocal resolved_is_new_format, candidate_is_new_format

            candidate, candidate_format = _resolve_two_level_ward_candidate(ward, province)
            if candidate:
                ward_info = candidate
                candidate_id = candidate.get("id")
                if candidate_id is not None:
                    ward_id = candidate_id
                canonical = candidate.get("full_name") or candidate.get("name")
                if canonical:
                    ward = canonical

            if candidate_format is True:
                if drop_inferred_district_for_new:
                    district = ""
                    district_id = None
                    district_info = None
                resolved_is_new_format = True
                candidate_is_new_format = True
                return True

            if candidate_format is False:
                resolved_is_new_format = False
                candidate_is_new_format = False
                if not district:
                    recovered_district_name, recovered_district_id = (
                        self._recover_district_from_ward_info(
                            candidate,
                            ward,
                            province,
                            province_info,
                        )
                    )
                    if recovered_district_name:
                        district = recovered_district_name
                        district_id = recovered_district_id
                        district_info = (
                            self._lookup_district_info(
                                district,
                                province if province else None,
                            )
                            if district
                            else None
                        )
                        if (
                            not district_id
                            and district_info
                            and district_info.get("id") is not None
                        ):
                            district_id = district_info["id"]
                return False

            return None

        def _entry_matches_exact_fragment(
            entry: AdministrativeRecord | None, fragment: str | None
        ) -> bool:
            return self._entry_matches_component_fragment(entry, fragment, level="ward")

        def _rescue_flat_suffix_ward_without_district() -> bool:
            nonlocal ward, ward_id, ward_info
            nonlocal district, district_id, district_info
            nonlocal resolved_is_new_format, candidate_is_new_format

            if len(input_segments) != 1 or district_prefix_in_input:
                return False
            if raw_detected_dist or raw_detected_ward or raw_ward_segment:
                return False
            province_hint = province if province else None
            province_key = self.standardize_name(province_hint, "basic") if province_hint else ""
            if not province_key:
                return False
            tokens = [tok for tok in input_string_basic.split() if tok]
            province_tokens = [tok for tok in province_key.split() if tok]
            if not province_tokens or len(tokens) <= len(province_tokens):
                return False
            if tokens[-len(province_tokens) :] != province_tokens:
                return False

            remaining = tokens[: -len(province_tokens)]
            if len(remaining) < 2:
                return False

            max_ward_len = min(4, len(remaining))
            for ward_len in range(max_ward_len, 0, -1):
                ward_fragment_tokens = remaining[-ward_len:]
                leading_tokens = remaining[:-ward_len]
                if leading_tokens and len(leading_tokens) < 2:
                    if all(token.isdigit() for token in leading_tokens):
                        continue
                ward_fragment = " ".join(ward_fragment_tokens)
                ward_entry, candidate_format = _resolve_two_level_ward_candidate(
                    ward_fragment, province_hint
                )
                if not ward_entry or not _entry_matches_exact_fragment(ward_entry, ward_fragment):
                    continue

                ward_info = ward_entry
                ward_id = ward_entry.get("id") or ward_entry.get("code")
                ward = ward_entry.get("full_name") or ward_entry.get("name") or ward

                if candidate_format is True:
                    district = ""
                    district_id = None
                    district_info = None
                    resolved_is_new_format = True
                    candidate_is_new_format = True
                    return True

                if candidate_format is False:
                    recovered_district_name, recovered_district_id = (
                        self._recover_district_from_ward_info(
                            ward_entry,
                            ward,
                            province,
                            province_info,
                        )
                    )
                    if recovered_district_name:
                        district = recovered_district_name
                        district_id = recovered_district_id
                        district_info = (
                            self._lookup_district_info(
                                district,
                                province if province else None,
                            )
                            if district
                            else None
                        )
                    else:
                        district = ""
                        district_id = None
                        district_info = None
                    resolved_is_new_format = False
                    candidate_is_new_format = False
                    return True
            return False

        def _rescue_flat_old_suffix_components() -> bool:
            nonlocal ward, ward_id, ward_info
            nonlocal district, district_id, district_info
            nonlocal resolved_is_new_format, candidate_is_new_format

            if len(input_segments) != 1 or district_prefix_in_input:
                return False
            if raw_detected_dist or raw_detected_ward or raw_ward_segment:
                return False
            province_hint = province if province else None
            province_key = self.standardize_name(province_hint, "basic") if province_hint else ""
            if not province_key:
                return False
            tokens = [tok for tok in input_string_basic.split() if tok]
            province_tokens = [tok for tok in province_key.split() if tok]
            if not province_tokens or len(tokens) <= len(province_tokens) + 1:
                return False
            if tokens[-len(province_tokens) :] != province_tokens:
                return False

            remaining = tokens[: -len(province_tokens)]
            if len(remaining) < 2:
                return False

            max_dist_len = min(4, len(remaining) - 1)
            for dist_len in range(max_dist_len, 0, -1):
                district_fragment = " ".join(remaining[-dist_len:])
                district_entry = self._lookup_district_info(district_fragment, province_hint)
                if not district_entry:
                    continue
                ward_tokens = remaining[:-dist_len]
                if not ward_tokens:
                    continue
                district_name_value = (
                    district_entry.get("name")
                    or district_entry.get("full_name")
                    or district_fragment
                )
                max_ward_len = min(4, len(ward_tokens))
                for ward_len in range(max_ward_len, 0, -1):
                    ward_fragment = " ".join(ward_tokens[-ward_len:])
                    _, two_level_format = _resolve_two_level_ward_candidate(
                        ward_fragment, province_hint
                    )
                    if two_level_format is True:
                        continue
                    ward_entry = self._lookup_ward_info(
                        ward_fragment,
                        province_hint,
                        district_name_value,
                        preferred_format=False,
                    )
                    if not ward_entry or ward_entry.get("is_new_format") is not False:
                        continue
                    if not _entry_matches_exact_fragment(ward_entry, ward_fragment):
                        continue
                    district = district_name_value
                    district_info = district_entry
                    district_id = district_entry.get("id") or district_entry.get("code")
                    ward_info = ward_entry
                    ward_id = ward_entry.get("id") or ward_entry.get("code")
                    ward = ward_entry.get("full_name") or ward_entry.get("name") or ward
                    resolved_is_new_format = False
                    candidate_is_new_format = False
                    return True
            return False

        # If the inferred district collapses to the same region key as the province and the
        # input does not explicitly contain a district-level prefix, treat it as a 2-level
        # (ward+province) "new" address. This prevents overfitting to old-format candidates
        # where the city-of-province shares the same name as the province.
        explicit_city_district_in_input = False
        if input_segments and province:
            province_key = _canonical_region_key(province)
            if province_key:
                seen_tinh = False
                seen_tp = False
                for segment_std, _ in input_segments:
                    if not segment_std:
                        continue
                    segment_key = _canonical_region_key(segment_std)
                    if not segment_key or segment_key != province_key:
                        continue
                    if segment_std.startswith("tinh "):
                        seen_tinh = True
                    if segment_std.startswith("tp ") or segment_std.startswith(
                        self._LIT_THANH_PHO_PREFIX
                    ):
                        seen_tp = True
                explicit_city_district_in_input = seen_tinh and seen_tp

        if (
            province
            and district
            and not district_prefix_in_input
            and not explicit_city_district_in_input
            and _canonical_region_key(district)
            and _canonical_region_key(district) == _canonical_region_key(province)
        ):
            two_level_decision = _apply_two_level_ward_resolution(
                drop_inferred_district_for_new=True
            )
            if two_level_decision is True:
                district_hint_in_input = False
            elif two_level_decision is None:
                district = ""
                district_id = None
                district_info = None
                district_hint_in_input = False
                resolved_is_new_format = True
                candidate_is_new_format = True

        def _has_explicit_district_segment(component: str | None) -> bool:
            district_key = _canonical_region_key(component)
            if not district_key or not input_segments:
                return False

            province_key = _canonical_region_key(province)
            ward_key = _canonical_region_key(ward)
            has_segment = any(
                _canonical_region_key(segment_std) == district_key
                for segment_std, _ in input_segments
                if segment_std
            )
            if not has_segment:
                return False
            if (
                district_key
                and province_key
                and district_key == province_key
                and not explicit_city_district_in_input
            ):
                return False
            if (
                district_key
                and ward_key
                and district_key == ward_key
                and not district_prefix_in_input
            ):
                return False
            return True

        if (
            ward_id
            and ward_info is None
            and province
            and not district_hint_in_input
            and not _old_ward_id_matches_province(ward_id, province)
        ):
            ward_id = None

        # If there is no explicit district prefix in the input, avoid "inventing" a district
        # purely from the selected old-format candidate / ward metadata unless the district
        # is clearly present as its own comma-separated segment.
        if province and district and not district_prefix_in_input:
            district_key = _canonical_region_key(district)
            has_explicit_district_segment = False
            if district_key and input_segments:
                for segment_std, _ in input_segments:
                    if _canonical_region_key(segment_std) == district_key:
                        has_explicit_district_segment = True
                        break
            if not has_explicit_district_segment:
                two_level_decision = _apply_two_level_ward_resolution(
                    drop_inferred_district_for_new=True
                )
                if two_level_decision is True:
                    district_hint_in_input = False
                elif two_level_decision is None:
                    district = ""
                    district_id = None
                    district_info = None
                    resolved_is_new_format = True
                    candidate_is_new_format = True

        canonical_changed = False
        if ward_info:
            canonical_province = ward_info.get("province_name")
            canonical_district = ward_info.get("district_name")

            if (
                canonical_province
                and not province
                and _std_name(canonical_province)
                and _std_name(canonical_province) != _std_name(province)
            ):
                province = canonical_province
                province_id = None
                canonical_changed = True

            if (
                canonical_district
                and not district
                and district_hint_in_input
                and _std_name(canonical_district)
                and _std_name(canonical_district) != _std_name(district)
            ):
                district = canonical_district
                district_id = None
                canonical_changed = True

        if canonical_changed:
            province_info = self._lookup_province_info(province) if province else None
            if not province:
                province_id = None
            elif province_info and province_info.get("id") is not None:
                province_id = province_info["id"]

            province_for_lookup = province if province else None
            district_info = (
                self._lookup_district_info(district, province_for_lookup) if district else None
            )
            if not district:
                district_id = None
            elif district_info and district_info.get("id") is not None:
                district_id = district_info["id"]

            district_for_lookup = district if district else None
            ward_info = (
                self._lookup_ward_info(
                    ward,
                    province_for_lookup,
                    district_for_lookup,
                    preferred_format=candidate_is_new_format,
                )
                if ward
                else None
            )
            if ward and ward_info is None and not _has_strict_region_hint():
                ward_info = self._lookup_ward_info(ward, preferred_format=candidate_is_new_format)
            if not ward:
                ward_id = None
            elif ward_info and ward_info.get("id") is not None:
                ward_id = ward_info["id"]
            resolved_is_new_format = _update_format(resolved_is_new_format, ward_info)

        if ward and ward_info is None and not _has_strict_region_hint():
            ward_key = self.standardize_name(ward, "basic")
            if ward_key and self.ward_lookup_by_name.get(ward_key):
                legacy_record = None
            else:
                legacy_record = self._lookup_old_ward_record_by_name(ward)
            if legacy_record:
                ward_info = self._enrich_old_ward_with_province(
                    {**legacy_record, "is_new_format": False}
                )
                if ward_info.get("id") is not None:
                    ward_id = ward_info["id"]
                if resolved_is_new_format is not False:
                    resolved_is_new_format = False
                    candidate_is_new_format = False

        # 2-level guard: when the input contains only ward+province (no district hint/prefix),
        # prefer the new 2-level registry first; if that exact ward cannot be resolved in the
        # new registry, fall back to the legacy ward + inferred district instead of forcing
        # format="new".
        if ward and not district and not district_hint_in_input:
            two_level_decision = _apply_two_level_ward_resolution(
                drop_inferred_district_for_new=True
            )
            if two_level_decision is True:
                district_hint_in_input = False
            elif two_level_decision is None:
                is_legacy_only_ward = _is_legacy_only_ward(ward_id, ward, province)
                if not is_legacy_only_ward:
                    resolved_is_new_format = True
                    candidate_is_new_format = True

                    # If we ended up mapping the ward to an old-record entry (e.g. same name exists
                    # in both registries with different codes), attempt to upgrade to the new-format
                    # ward so IDs line up with `wards.json`.
                    if ward_info and ward_info.get("is_new_format") is False:
                        upgraded = self._lookup_ward_info(
                            ward,
                            province if province else None,
                            None,
                            preferred_format=True,
                        )
                        if upgraded and upgraded.get("is_new_format") is True:
                            ward_info = upgraded
                            if upgraded.get("id") is not None:
                                ward_id = upgraded["id"]
                            canonical = upgraded.get("full_name") or upgraded.get("name")
                            if canonical:
                                ward = canonical

        if district_prefix_in_input and resolved_is_new_format is not False:
            resolved_is_new_format = False

        if resolved_is_new_format is True and not district_hint_in_input and not district:
            district = ""
            district_id = None
            district_info = None

        preferred_lookup_format = (
            resolved_is_new_format
            if resolved_is_new_format is not None
            else candidate_is_new_format
        )
        if not district_hint_in_input and (
            raw_detected_ward or normalized_detected_ward_token or detected_ward
        ):
            if preferred_lookup_format is None:
                preferred_lookup_format = True
            elif preferred_lookup_format is False and district and not _appears_in_input(district):
                preferred_lookup_format = True

        ward_info_name_std = _std_name(ward_info.get("name")) if ward_info else ""
        ward_name_std = _std_name(ward)
        ward_name_mismatch = bool(
            ward and ward_info and ward_info_name_std and ward_info_name_std != ward_name_std
        )

        if ward and (ward_name_mismatch or ward_info is None) and not enforce_locked_new_format:
            refreshed = self._lookup_ward_info(
                ward,
                province if province else None,
                district if district else None,
                preferred_format=preferred_lookup_format,
            )
            if not refreshed and not _has_strict_region_hint():
                refreshed = self._lookup_ward_info(
                    ward,
                    preferred_format=preferred_lookup_format,
                )
            if refreshed:
                ward_info = refreshed
                if refreshed.get("id") is not None:
                    ward_id = refreshed["id"]
                resolved_is_new_format = _update_format(resolved_is_new_format, ward_info)
        elif not ward:
            ward_info = None
            ward_id = None

        if (
            preferred_lookup_format is True
            and not district_hint_in_input
            and district
            and not _appears_in_input(district)
        ):
            district = ""
            district_id = None
            district_info = None
            resolved_is_new_format = True
            candidate_is_new_format = True

        # If we have a ward_info and a district hint, prefer the canonical ward name/full_name
        # over legacy aliases that might match a district name.
        if ward_info and ward:
            canonical_name = ward_info.get("name") or ward_info.get("full_name")
            canonical_std = _std_name(canonical_name)
            ward_std = _std_name(ward)
            if (
                canonical_std
                and canonical_std != ward_std
                and (canonical_std in input_string_basic or district_hint_in_input)
            ):
                ward = canonical_name
                ward_id = ward_info.get("id") or ward_id

        # Late new-format guard: classification may be decided before ward resolution,
        # so ensure we still prefer the new 2-level registry first; if the ward only resolves
        # via legacy metadata, keep the old-format fallback instead of forcing format="new".
        if (province or ward) and not district and not district_hint_in_input:
            two_level_decision = _apply_two_level_ward_resolution(
                drop_inferred_district_for_new=False
            )
            if two_level_decision is True:
                district_hint_in_input = False
            elif two_level_decision is None:
                is_legacy_only_ward = _is_legacy_only_ward(ward_id, ward, province)
                if not is_legacy_only_ward:
                    resolved_is_new_format = True
                    candidate_is_new_format = True

        _rescue_flat_suffix_ward_without_district()
        _rescue_flat_old_suffix_components()

        # Final guard: if we only saw a ward-prefixed token (no district prefix),
        # prefer collapsing to 2-level only when the ward truly resolves in the new registry.
        # Otherwise keep the old-format fallback and its recovered district.
        if district and not district_prefix_in_input:
            district_key = _canonical_region_key(district)
            province_key = _canonical_region_key(province)
            has_explicit_district_segment = False
            if district_key and input_segments:
                for segment_std, _ in input_segments:
                    if _canonical_region_key(segment_std) == district_key:
                        has_explicit_district_segment = True
                        break
            if (
                district_key
                and province_key
                and district_key == province_key
                and not explicit_city_district_in_input
            ):
                has_explicit_district_segment = False
            if not has_explicit_district_segment:
                two_level_decision = _apply_two_level_ward_resolution(
                    drop_inferred_district_for_new=True
                )
                if two_level_decision is True:
                    district_hint_in_input = False
                elif two_level_decision is None:
                    district = ""
                    district_id = None
                    district_info = None
                    resolved_is_new_format = True
                    candidate_is_new_format = True
        # Refresh lookup metadata to reflect any late-stage overrides
        province_info = self._lookup_province_info(province) if province else None
        if not province:
            province_id = None
        elif province_info and province_info.get("id") is not None:
            province_id = province_info["id"]

        if resolved_is_new_format is True:
            province_id_new = self._lookup_new_province_id_by_name(province)
            if province_id_new:
                province_id = province_id_new
                province_info = self.new_province_records.get(province_id_new) or province_info

        province_for_lookup = province if province else None
        district_info = (
            self._lookup_district_info(district, province_for_lookup) if district else None
        )
        if not district:
            district_id = None
        elif district_info and district_info.get("id") is not None:
            district_id = district_info["id"]

        if (
            province
            and district
            and not district_prefix_in_input
            and ward_info
            and ward_info.get("is_new_format") is True
        ):
            district_key = _canonical_region_key(district)
            province_key = _canonical_region_key(province)
            has_explicit_district_segment = False
            if district_key and input_segments:
                for segment_std, _ in input_segments:
                    if _canonical_region_key(segment_std) == district_key:
                        has_explicit_district_segment = True
                        break
            if (
                district_key
                and province_key
                and district_key == province_key
                and not explicit_city_district_in_input
            ):
                has_explicit_district_segment = False
            district_core = self._strip_generic_prefix(self.standardize_name(district, "basic"))
            ward_display_name = ward_info.get("full_name") or ward
            ward_core = (
                self._strip_generic_prefix(self.standardize_name(ward_display_name, "basic"))
                if ward_display_name
                else ""
            )
            if (
                district_info is None
                or not has_explicit_district_segment
                or (district_core and ward_core and district_core == ward_core)
            ):
                district = ""
                district_id = None
                district_info = None
                district_hint_in_input = False
                resolved_is_new_format = True
                candidate_is_new_format = True

        # Final canonicalization: if we have structured ward info, trust its canonical name/id.
        if ward_info:
            canonical_ward = ward_info.get("name") or ward_info.get("full_name")
            if canonical_ward:
                ward = canonical_ward
            if ward_info.get("id") is not None:
                ward_id = ward_info["id"]
            # Rehydrate from source records to avoid legacy-only aliases overriding canonical
            # names. Prefer the registry that matches the resolved ward record (new vs old)
            # to avoid collisions where an old and new ward share the same numeric code.
            ward_record_id = cast(str, ward_info["id"])
            ward_prefers_new = ward_info.get("is_new_format")
            prefer_new_registry = None
            if ward_prefers_new is True:
                prefer_new_registry = True
            elif ward_prefers_new is False:
                prefer_new_registry = False
            elif resolved_is_new_format is True:
                prefer_new_registry = True
            elif resolved_is_new_format is False:
                prefer_new_registry = False
            if prefer_new_registry is True:
                record = self.new_ward_records.get(ward_record_id) or self.old_ward_records.get(
                    ward_record_id
                )
            elif prefer_new_registry is False:
                record = self.old_ward_records.get(ward_record_id) or self.new_ward_records.get(
                    ward_record_id
                )
            else:
                record = self.new_ward_records.get(ward_record_id) or self.old_ward_records.get(
                    ward_record_id
                )
            if isinstance(record, dict):
                canonical_from_record = record.get("full_name") or record.get("name")
                if canonical_from_record:
                    ward = canonical_from_record
                    ward_info = {
                        **ward_info,
                        "full_name": record.get("full_name") or ward_info.get("full_name"),
                        "name": record.get("name") or ward_info.get("name"),
                    }

        if ward_info and ward_prefix_hint:
            entry_label = ward_info.get("full_name") or ward_info.get("name") or ward
            entry_prefix = _ward_prefix_from_value(entry_label)
            if entry_prefix and entry_prefix != ward_prefix_hint:
                ward_info = None
                ward_id = None
                if raw_ward_segment:
                    ward = raw_ward_segment
                if not district_hint_in_input and not district_prefix_in_input:
                    resolved_is_new_format = True
                    candidate_is_new_format = True

        if ward_info and ward_prefix_hint:
            hint_source = raw_ward_segment or ward_lookup_hint or ward
            if hint_source and not self._entry_matches_component_fragment(
                ward_info, hint_source, level="ward"
            ):
                ward_info = None
                ward_id = None
                ward = raw_ward_segment or hint_source

        if (
            ward
            and raw_ward_segment
            and ward_prefix_hint
            and province
            and (ward_info is None or ward_info.get("id") is None)
            and (district or district_prefix_in_input or district_hint_in_input)
        ):
            rescued_ward_info = self._lookup_ward_info(
                ward_lookup_hint or raw_ward_segment,
                province if province else None,
                district if district else None,
                preferred_format=candidate_is_new_format,
            )
            if rescued_ward_info is None and district and candidate_is_new_format is not False:
                rescued_ward_info = self._lookup_ward_info(
                    ward_lookup_hint or raw_ward_segment,
                    province if province else None,
                    district if district else None,
                    preferred_format=False,
                )
            if rescued_ward_info is None:
                rescued_ward_info = self._lookup_ward_info(
                    ward_lookup_hint or raw_ward_segment,
                    province if province else None,
                    None,
                    preferred_format=candidate_is_new_format,
                )
            if rescued_ward_info is None and candidate_is_new_format is not False:
                rescued_ward_info = self._lookup_ward_info(
                    ward_lookup_hint or raw_ward_segment,
                    province if province else None,
                    None,
                    preferred_format=False,
                )
            if rescued_ward_info:
                ward_info = rescued_ward_info
                if rescued_ward_info.get("id") is not None:
                    ward_id = rescued_ward_info["id"]
                canonical = rescued_ward_info.get("full_name") or rescued_ward_info.get("name")
                if canonical:
                    ward = canonical
                rescued_district_name, rescued_district_id = self._recover_district_from_ward_info(
                    rescued_ward_info,
                    ward,
                    province,
                    province_info,
                )
                if rescued_district_name and not district_prefix_in_input:
                    district = rescued_district_name
                    district_id = rescued_district_id
                resolved_is_new_format = _update_format(resolved_is_new_format, ward_info)
                candidate_is_new_format = _update_format(candidate_is_new_format, ward_info)

        if raw_ward_segment and ward_prefix_hint and not district_prefix_in_input:
            explicit_province_name = None
            if detected_prov:
                explicit_province_name = self._resolve_detected_component(
                    "province",
                    detected_prov,
                    source_string=input_string_basic,
                )

            raw_segment_std = self.standardize_name(raw_ward_segment, "basic")

            def _canonical_entry_matches_raw(
                entry: AdministrativeRecord | None,
            ) -> bool:
                if not entry or not raw_segment_std:
                    return False
                return self._entry_matches_query_name(entry, raw_ward_segment)

            current_matches_raw = _canonical_entry_matches_raw(ward_info)
            if not current_matches_raw:
                canonical_raw_ward = None
                province_candidates: list[str | None] = []
                if explicit_province_name:
                    province_candidates.append(explicit_province_name)
                if province and province not in province_candidates:
                    province_candidates.append(province)
                province_candidates.append(None)
                for province_hint in province_candidates:
                    candidate = self._lookup_ward_info(
                        raw_ward_segment,
                        province_hint if province_hint else None,
                        None,
                        preferred_format=None,
                    )
                    if not _canonical_entry_matches_raw(candidate):
                        continue
                    if explicit_province_name and not self._entry_aligns_with_province(
                        candidate, explicit_province_name
                    ):
                        continue
                    canonical_raw_ward = candidate
                    break
                if canonical_raw_ward:
                    ward_info = canonical_raw_ward
                    ward_id = canonical_raw_ward.get("id")
                    ward = (
                        canonical_raw_ward.get("full_name")
                        or canonical_raw_ward.get("name")
                        or raw_ward_segment
                    )
                    candidate_is_new_format = _update_format(candidate_is_new_format, ward_info)
                    resolved_is_new_format = _update_format(resolved_is_new_format, ward_info)
                    province_from_ward = canonical_raw_ward.get("province_name")
                    if (
                        province_from_ward
                        and explicit_province_name
                        and self._entry_aligns_with_province(
                            canonical_raw_ward, explicit_province_name
                        )
                    ):
                        province = province_from_ward
                        province_id = None
                        province_info = self._lookup_province_info(province)
                    if canonical_raw_ward.get("is_new_format") is True:
                        district = ""
                        district_id = None
                        district_info = None
                        district_hint_in_input = False

        if not district and ward_id:
            ward_record_key = self.normalize_address_code(ward_id)
            if (
                ward_record_key
                and ward_record_key in self.old_ward_records
                and ward_record_key not in self.new_ward_records
            ):
                old_record: AdministrativeRecord | None = self.old_ward_records.get(ward_record_key)
                parent_code = self.normalize_address_code(
                    old_record.get("parent_code") if old_record else None
                )
                district_entry = self.old_district_records.get(parent_code) if parent_code else None
                recovered = None
                if isinstance(district_entry, dict):
                    recovered = district_entry.get("name") or district_entry.get("full_name")
                if recovered and not (district_prefix_in_input or district_hint_in_input):
                    district_key = _canonical_region_key(recovered)
                    province_key = _canonical_region_key(province)
                    has_explicit_district_segment = False
                    if district_key and input_segments:
                        for segment_std, _ in input_segments:
                            if _canonical_region_key(segment_std) == district_key:
                                has_explicit_district_segment = True
                                break
                    if (
                        district_key
                        and province_key
                        and district_key == province_key
                        and not explicit_city_district_in_input
                    ):
                        has_explicit_district_segment = False
                    if not has_explicit_district_segment:
                        recovered = None
                if recovered:
                    district = recovered
                    district_id = parent_code
                    district_info = district_entry if isinstance(district_entry, dict) else None
                    if not province and isinstance(district_entry, dict):
                        province_code_old = self.normalize_address_code(
                            district_entry.get("parent_code")
                        )
                        province_entry = (
                            self.old_province_records.get(province_code_old)
                            if province_code_old
                            else None
                        )
                        if isinstance(province_entry, dict):
                            province = province_entry.get("name") or province_entry.get("full_name")
                            province_id = province_code_old
                            province_info = province_entry
                    resolved_is_new_format = False
                    candidate_is_new_format = False

        if (
            ward
            and province
            and (ward_info is None or ward_id is None)
            and (raw_ward_segment or raw_detected_ward)
        ):
            final_ward_hint = ward_lookup_hint or raw_ward_segment or raw_detected_ward
            final_ward_info = self._lookup_ward_info(
                final_ward_hint,
                province if province else None,
                None,
                preferred_format=candidate_is_new_format,
            )
            if final_ward_info is None and candidate_is_new_format is not False:
                final_ward_info = self._lookup_ward_info(
                    final_ward_hint,
                    province if province else None,
                    None,
                    preferred_format=False,
                )
            if final_ward_info:
                ward_info = final_ward_info
                ward_id = final_ward_info.get("id") or final_ward_info.get("code")
                canonical = final_ward_info.get("full_name") or final_ward_info.get("name")
                if canonical:
                    ward = canonical
                rescued_district_name, rescued_district_id = self._recover_district_from_ward_info(
                    final_ward_info,
                    ward,
                    province,
                    province_info,
                )
                if rescued_district_name and not district_prefix_in_input:
                    district = rescued_district_name
                    district_id = rescued_district_id
                    district_info = self._lookup_district_info(
                        district,
                        province if province else None,
                    )
                resolved_is_new_format = _update_format(resolved_is_new_format, ward_info)
                candidate_is_new_format = _update_format(candidate_is_new_format, ward_info)

        ward_record_key = self.normalize_address_code(
            ward_id
            or (ward_info.get("id") if isinstance(ward_info, dict) else None)
            or (ward_info.get("code") if isinstance(ward_info, dict) else None)
        )
        ward_is_definitely_new = bool(ward_record_key and ward_record_key in self.new_ward_records)
        if district and ward and not district_prefix_in_input and ward_is_definitely_new:
            district_core = (
                self._strip_generic_prefix(self.standardize_name(district, "basic"))
                if district
                else ""
            )
            ward_core = self._strip_generic_prefix(self.standardize_name(ward, "basic"))
            if district_core and ward_core and district_core == ward_core:
                district = ""
                district_id = None
                district_info = None
                district_hint_in_input = False
                resolved_is_new_format = True
                candidate_is_new_format = True

        if not district and detected_dist:
            rescued_district = self._resolve_detected_component(
                "district",
                detected_dist,
                expected_province=province if province else None,
                source_string=input_string_basic,
            )
            if rescued_district and _appears_in_input(rescued_district):
                district = rescued_district
                district_info = (
                    self._lookup_district_info(district, province if province else None)
                    if district
                    else None
                )
                district_id = district_info.get("id") if district_info else None

        if not ward and detected_ward:
            rescued_ward = self._resolve_detected_component(
                "ward",
                detected_ward,
                expected_province=province if province else None,
                expected_district=district if district else None,
                source_string=input_string_basic,
            )
            if rescued_ward and _appears_in_input(rescued_ward):
                ward = rescued_ward
                ward_info = (
                    self._lookup_ward_info(
                        ward,
                        province if province else None,
                        district if district else None,
                        preferred_format=candidate_is_new_format,
                    )
                    if ward
                    else None
                )
                ward_id = ward_info.get("id") if ward_info else None
                resolved_is_new_format = _update_format(resolved_is_new_format, ward_info)
                candidate_is_new_format = _update_format(candidate_is_new_format, ward_info)

        if not ward and len(input_segments) >= 2:
            for offset_from_tail, (segment_std, _) in enumerate(reversed(input_segments)):
                if not _segment_is_candidate(segment_std):
                    continue
                if province and offset_from_tail == 0:
                    continue
                if district and offset_from_tail <= 1:
                    district_like = self._fuzzy_match_component_key(
                        segment_std,
                        self.district_names_std,
                        cutoff=89,
                    )
                    if district_like and district_like == self.standardize_name(district, "basic"):
                        continue
                ward_token = None
                if segment_std in self.ward_names_std:
                    ward_token = segment_std
                elif offset_from_tail <= 2:
                    ward_token = self._fuzzy_match_component_key(
                        segment_std,
                        self.ward_names_std,
                        cutoff=88 if province or district else 90,
                    )
                if not ward_token:
                    continue
                rescued_ward = self._resolve_detected_component(
                    "ward",
                    ward_token,
                    expected_province=province if province else None,
                    expected_district=district if district else None,
                    source_string=input_string_basic,
                )
                if not rescued_ward or not _appears_in_input(rescued_ward):
                    continue
                ward = rescued_ward
                ward_info = self._lookup_ward_info(
                    ward,
                    province if province else None,
                    district if district else None,
                    preferred_format=candidate_is_new_format,
                )
                ward_id = ward_info.get("id") if ward_info else None
                resolved_is_new_format = _update_format(resolved_is_new_format, ward_info)
                candidate_is_new_format = _update_format(candidate_is_new_format, ward_info)
                break

        final_detected_ward_token = (
            detected_ward
            or normalized_detected_ward_token
            or self._normalize_detected_ward_token(raw_detected_ward)
        )
        if ward and final_detected_ward_token and not _appears_in_input(ward):
            rescued_ward = self._resolve_detected_component(
                "ward",
                final_detected_ward_token,
                expected_province=province if province else None,
                expected_district=district if district else None,
                source_string=input_string_basic,
            )
            if rescued_ward and _appears_in_input(rescued_ward):
                ward = rescued_ward
                ward_info = (
                    self._lookup_ward_info(
                        ward,
                        province if province else None,
                        district if district else None,
                        preferred_format=candidate_is_new_format,
                    )
                    if ward
                    else None
                )
                ward_id = ward_info.get("id") if ward_info else None
                resolved_is_new_format = _update_format(resolved_is_new_format, ward_info)
                candidate_is_new_format = _update_format(candidate_is_new_format, ward_info)

        if ward and len(input_segments) >= 2 and not _appears_in_input(ward):
            rescued_ward = None
            for offset_from_tail, (segment_std, segment_raw) in enumerate(reversed(input_segments)):
                if not _segment_is_candidate(segment_std):
                    continue
                if province and offset_from_tail == 0 and _looks_like_known_province(segment_std):
                    continue
                if district and _component_keys_match(district, segment_std):
                    continue

                segment_candidates: list[str] = []
                if segment_std in self.ward_names_std:
                    segment_candidates.append(segment_std)
                inferred_suffix = self._infer_ward_from_segment_suffix(
                    segment_std,
                    segment_raw,
                    expected_province=province if province else None,
                    expected_district=district if district else None,
                    allow_plain_prefix=bool(offset_from_tail > 0 and (province or district)),
                )
                if inferred_suffix and inferred_suffix[1] not in segment_candidates:
                    segment_candidates.append(inferred_suffix[1])

                for ward_token in segment_candidates:
                    resolved_tail_ward = self._resolve_detected_component(
                        "ward",
                        ward_token,
                        expected_province=province if province else None,
                        expected_district=district if district else None,
                        source_string=input_string_basic,
                    )
                    if resolved_tail_ward and _appears_in_input(resolved_tail_ward):
                        rescued_ward = resolved_tail_ward
                        break
                if rescued_ward:
                    break

            if rescued_ward:
                ward = rescued_ward
                ward_info = (
                    self._lookup_ward_info(
                        ward,
                        province if province else None,
                        district if district else None,
                        preferred_format=candidate_is_new_format,
                    )
                    if ward
                    else None
                )
                ward_id = ward_info.get("id") if ward_info else None
                resolved_is_new_format = _update_format(resolved_is_new_format, ward_info)
                candidate_is_new_format = _update_format(candidate_is_new_format, ward_info)

        if self._input_looks_like_street_only_fragment(input_segments):
            if province and not _appears_in_input(province):
                province = ""
                province_id = None
                province_info = None
            if district and not _appears_in_input(district):
                district = ""
                district_id = None
                district_info = None
            if ward and not _appears_in_input(ward):
                ward = ""
                ward_id = None
                ward_info = None
            if not province and not district and not ward:
                resolved_is_new_format = False
                candidate_is_new_format = False

        if not ward and province and not district_prefix_in_input:
            ambiguous_ward_token = (
                normalized_detected_ward_token
                or self._normalize_detected_ward_token(raw_detected_dist or district)
                or detected_ward
                or detected_dist
            )
            if ambiguous_ward_token:
                ambiguous_new_entry = self._lookup_new_format_ward_alias(
                    ambiguous_ward_token,
                    expected_province=province,
                )
                if ambiguous_new_entry and ambiguous_new_entry.get("is_new_format") is True:
                    ward_info = ambiguous_new_entry
                    ward = (
                        ambiguous_new_entry.get("name")
                        or ambiguous_new_entry.get("full_name")
                        or ward
                    )
                    ward_id = ambiguous_new_entry.get("id") or ambiguous_new_entry.get("code")
                    district = ""
                    district_id = None
                    district_info = None
                    district_hint_in_input = False
                    resolved_is_new_format = True
                    candidate_is_new_format = True

        # Final 2-level normalization: once a ward is resolved in the new registry,
        # do not keep a district that was only inferred from an ambiguous bare token.
        if ward_info and ward_info.get("is_new_format") is True and not district_prefix_in_input:
            district_key = _canonical_region_key(district)
            province_key = _canonical_region_key(province)
            has_explicit_district_segment = False
            if district_key and input_segments:
                for segment_std, _ in input_segments:
                    if _canonical_region_key(segment_std) == district_key:
                        has_explicit_district_segment = True
                        break
            if (
                district_key
                and province_key
                and district_key == province_key
                and not explicit_city_district_in_input
            ):
                has_explicit_district_segment = False
            if not has_explicit_district_segment:
                district = ""
                district_id = None
                district_info = None
                district_hint_in_input = False
                resolved_is_new_format = True
                candidate_is_new_format = True

        # Final consistency guard: if the resolved ward comes from the new registry
        # and no district survives, keep the classification aligned with the 2-level
        # dataset even when earlier legacy heuristics set format="old".
        if (
            not district_prefix_in_input
            and not district
            and ward_record_key
            and ward_info
            and ward_info.get("is_new_format") is True
            and ward_record_key in self.new_ward_records
        ):
            district = ""
            district_id = None
            district_info = None
            district_hint_in_input = False
            resolved_is_new_format = True
            candidate_is_new_format = True

        if raw_ward_segment and ward_prefix_hint and not district_prefix_in_input:
            raw_segment_std = self.standardize_name(raw_ward_segment, "basic")
            explicit_province_name = (
                self._resolve_detected_component(
                    "province",
                    detected_prov,
                    source_string=input_string_basic,
                )
                if detected_prov
                else None
            )

            def _canonical_entry_matches_raw_late(
                entry: AdministrativeRecord | None,
            ) -> bool:
                if not entry or not raw_segment_std:
                    return False
                return self._entry_matches_query_name(entry, raw_ward_segment)

            if not _canonical_entry_matches_raw_late(ward_info):
                late_raw_candidate = None
                for province_hint in [explicit_province_name, None]:
                    candidate = self._lookup_ward_info(
                        raw_ward_segment,
                        province_hint if province_hint else None,
                        None,
                        preferred_format=None,
                    )
                    if not _canonical_entry_matches_raw_late(candidate):
                        continue
                    if explicit_province_name and not self._entry_aligns_with_province(
                        candidate, explicit_province_name
                    ):
                        continue
                    late_raw_candidate = candidate
                    break
                if late_raw_candidate:
                    ward_info = late_raw_candidate
                    ward_id = late_raw_candidate.get("id")
                    ward = (
                        late_raw_candidate.get("full_name")
                        or late_raw_candidate.get("name")
                        or raw_ward_segment
                    )
                    province = late_raw_candidate.get("province_name") or province
                    province_id = None if province else province_id
                    province_info = None
                    if late_raw_candidate.get("is_new_format") is True:
                        district = ""
                        district_id = None
                        district_info = None
                        district_hint_in_input = False
                        resolved_is_new_format = True
                        candidate_is_new_format = True
                    else:
                        resolved_is_new_format = _update_format(resolved_is_new_format, ward_info)
                        candidate_is_new_format = _update_format(candidate_is_new_format, ward_info)

        if not province and not district and not ward:
            resolved_is_new_format = False
            candidate_is_new_format = False

        # Hard rule: if the input explicitly includes a district-level prefix
        # (e.g. 'Huyện/Quận'), treat the address as old format regardless of any
        # ward mapping that may point to new-format records.
        if district_prefix_in_input:
            resolved_is_new_format = False
            candidate_is_new_format = False

        if raw_ward_segment and ward_prefix_hint and detected_prov:
            explicit_province_name = self._resolve_detected_component(
                "province",
                detected_prov,
                source_string=input_string_basic,
            )
            explicit_new_ward = (
                self._lookup_ward_info(
                    raw_ward_segment,
                    explicit_province_name if explicit_province_name else None,
                    None,
                    preferred_format=True,
                )
                if explicit_province_name
                else None
            )
            raw_segment_std = self.standardize_name(raw_ward_segment, "basic")
            explicit_new_matches = False
            if explicit_new_ward and raw_segment_std:
                explicit_new_matches = self._entry_matches_query_name(
                    explicit_new_ward,
                    raw_ward_segment,
                )
            if (
                explicit_new_matches
                and explicit_new_ward is not None
                and explicit_new_ward.get("is_new_format") is True
                and (not district or not _appears_in_input(district))
            ):
                ward_info = explicit_new_ward
                ward = explicit_new_ward.get("full_name") or explicit_new_ward.get("name")
                ward_id = explicit_new_ward.get("id") or explicit_new_ward.get("code")
                province = explicit_new_ward.get("province_name") or explicit_province_name
                province_id = None
                province_info = self._lookup_province_info(province) if province else None
                district = ""
                district_id = None
                district_info = None
                district_hint_in_input = False
                resolved_is_new_format = True
                candidate_is_new_format = True

        if (
            raw_ward_segment
            and ward_prefix_hint
            and not detected_prov
            and ward_info
            and ward_info.get("is_new_format") is True
            and not district
        ):
            province = ""
            province_id = None
            province_info = None

        if raw_ward_segment:
            should_preserve_unknown_province = bool(
                not province and not detected_prov and not district and not district_prefix_in_input
            )
            preferred_raw_format = False if district_prefix_in_input else candidate_is_new_format
            exact_context_ward = self._lookup_ward_info(
                raw_ward_segment,
                province if province else None,
                district if district else None,
                preferred_format=preferred_raw_format,
            )
            if exact_context_ward and self._entry_matches_query_name(
                exact_context_ward,
                raw_ward_segment,
                include_aliases=True,
            ):
                ward_info = exact_context_ward
                ward_id = exact_context_ward.get("id") or exact_context_ward.get("code")
                ward = exact_context_ward.get("full_name") or exact_context_ward.get("name") or ward
                if not should_preserve_unknown_province:
                    province = exact_context_ward.get("province_name") or province
                if exact_context_ward.get("is_new_format") is True and not district_prefix_in_input:
                    district = ""
                    district_id = None
                    district_info = None
                    resolved_is_new_format = True
                    candidate_is_new_format = True
                else:
                    district = exact_context_ward.get("district_name") or district
                    resolved_is_new_format = False
                    candidate_is_new_format = False

        if district and ward_info and ward_info.get("district_name"):
            resolved_is_new_format = False
            candidate_is_new_format = False

        if compact_prefixed_district_info and compact_prefixed_ward_info:
            district_info = compact_prefixed_district_info
            district = district_info.get("name") or district_info.get("full_name") or district
            district_id = district_info.get("id") or district_info.get("code")
            ward_info = compact_prefixed_ward_info
            ward = ward_info.get("full_name") or ward_info.get("name") or ward
            ward_id = ward_info.get("id") or ward_info.get("code")
            district_hint_in_input = True
            district_present_in_input = True
            resolved_is_new_format = False
            candidate_is_new_format = False

        if resolved_is_new_format is True and district and _has_explicit_district_segment(district):
            resolved_is_new_format = False
            candidate_is_new_format = False
            if ward:
                explicit_old_ward = self._lookup_ward_info(
                    ward,
                    province if province else None,
                    district if district else None,
                    preferred_format=False,
                )
                if explicit_old_ward and explicit_old_ward.get("is_new_format") is False:
                    ward_info = explicit_old_ward
                    ward_id = explicit_old_ward.get("id") or explicit_old_ward.get("code")
                    ward = (
                        explicit_old_ward.get("full_name") or explicit_old_ward.get("name") or ward
                    )

        def _resolve_old_context() -> tuple[
            AdministrativeRecord | None,
            AdministrativeRecord | None,
            str | None,
            str | None,
        ]:
            district_entry = district_info if isinstance(district_info, dict) else None
            district_code = self.normalize_address_code(district_id)
            if district_entry is None and district_code:
                candidate = self.old_district_records.get(district_code)
                if isinstance(candidate, dict):
                    district_entry = candidate

            ward_code = self.normalize_address_code(
                ward_id
                or (ward_info.get("id") if isinstance(ward_info, dict) else None)
                or (ward_info.get("code") if isinstance(ward_info, dict) else None)
            )
            if district_entry is None and ward_code:
                ward_entry = self.old_ward_records.get(ward_code)
                if isinstance(ward_entry, dict):
                    parent_code = self.normalize_address_code(ward_entry.get("parent_code"))
                    if parent_code:
                        candidate = self.old_district_records.get(parent_code)
                        if isinstance(candidate, dict):
                            district_entry = candidate
                            district_code = parent_code

            province_entry = province_info if isinstance(province_info, dict) else None
            province_code = self.normalize_address_code(province_id)
            if district_entry is not None:
                parent_code = self.normalize_address_code(district_entry.get("parent_code"))
                if parent_code:
                    candidate = self.old_province_records.get(parent_code)
                    if isinstance(candidate, dict):
                        province_entry = candidate
                        province_code = parent_code

            return district_entry, province_entry, district_code, province_code

        def _resolve_new_context() -> tuple[AdministrativeRecord | None, str | None]:
            ward_code = self.normalize_address_code(
                ward_id
                or (ward_info.get("id") if isinstance(ward_info, dict) else None)
                or (ward_info.get("code") if isinstance(ward_info, dict) else None)
            )
            if ward_code:
                ward_entry = self.new_ward_records.get(ward_code)
                if isinstance(ward_entry, dict):
                    parent_code = self.normalize_address_code(ward_entry.get("parent_code"))
                    if parent_code:
                        candidate = self.new_province_records.get(parent_code)
                        if isinstance(candidate, dict):
                            return candidate, parent_code

            province_code = self._lookup_new_province_id_by_name(province)
            if province_code:
                candidate = self.new_province_records.get(province_code)
                if isinstance(candidate, dict):
                    return candidate, province_code
            return None, None

        has_region_hint_from_input = bool(
            detected_prov
            or detected_dist
            or district_prefix_in_input
            or explicit_city_district_in_input
        )

        if resolved_is_new_format is True:
            if has_region_hint_from_input:
                resolved_new_province, resolved_new_province_id = _resolve_new_context()
                if resolved_new_province:
                    province_info = resolved_new_province
                    province_id = resolved_new_province_id
                    province = (
                        resolved_new_province.get("name")
                        or resolved_new_province.get("full_name")
                        or province
                    )
        elif resolved_is_new_format is False:
            if has_region_hint_from_input:
                (
                    resolved_old_district,
                    resolved_old_province,
                    resolved_old_district_id,
                    resolved_old_province_id,
                ) = _resolve_old_context()
                if resolved_old_district:
                    district_info = resolved_old_district
                    district_id = resolved_old_district_id
                    district = (
                        resolved_old_district.get("name")
                        or resolved_old_district.get("full_name")
                        or district
                    )
                if resolved_old_province:
                    province_info = resolved_old_province
                    province_id = resolved_old_province_id
                    province = (
                        resolved_old_province.get("name")
                        or resolved_old_province.get("full_name")
                        or province
                    )

        district_info_id = (
            district_info.get("id") or district_info.get("code")
            if isinstance(district_info, dict)
            else None
        )
        ward_info_id = (
            ward_info.get("id") or ward_info.get("code") if isinstance(ward_info, dict) else None
        )
        if district and not district_id and not district_info_id:
            district = ""
            district_id = None
            district_info = None
        if ward and not ward_id and not ward_info_id:
            ward = ""
            ward_id = None
            ward_info = None

        district_component = self._format_component(district, district_id, district_info)
        province_component = self._format_component(province, province_id, province_info)
        ward_component = self._format_component(ward, ward_id, ward_info)

        normalized_node = self.AddressNode(
            province or "",
            district or "",
            ward or "",
            is_new_format=resolved_is_new_format,
        )

        def _recover_tail_segment_surface(
            component: str | None, *, max_offset: int, cutoff: int = 88
        ) -> str | None:
            if not component or not input_segments:
                return None
            component_std = self.standardize_name(component, "basic")
            if not component_std:
                return None
            best_raw = None
            best_score = float("-inf")
            component_core = self._strip_generic_prefix(component_std) or component_std
            for offset_from_tail, (segment_std, segment_raw) in enumerate(reversed(input_segments)):
                if offset_from_tail > max_offset:
                    break
                if not segment_std or not segment_raw:
                    continue
                segment_core = self._strip_generic_prefix(segment_std) or segment_std
                matched = self._fuzzy_match_component_key(
                    component_std,
                    [segment_std, segment_core],
                    cutoff=cutoff,
                )
                if not matched:
                    continue
                score = ratio(component_core, segment_core)
                if score > best_score:
                    best_raw = str(segment_raw).strip(self._LIT_STRIP_CHARS)
                    best_score = score
            return best_raw

        late_locality_suffix_raw: str | None = None
        if input_segments:
            for segment_idx, (segment_std, segment_raw) in enumerate(input_segments):
                if not segment_std or segment_std in {"viet nam", "vietnam"}:
                    continue
                locality_suffix = self._infer_ward_from_segment_suffix(
                    segment_std,
                    segment_raw,
                    expected_province=province if province else None,
                    expected_district=district if district else None,
                    allow_plain_prefix=bool(segment_idx > 0 and (province or district)),
                )
                if not locality_suffix:
                    continue
                raw_fragment, fragment_std = locality_suffix
                if not self._segment_suffix_has_locality_cue(
                    segment_std,
                    fragment_std,
                ):
                    continue
                late_locality_suffix_raw = raw_fragment
                locality_preferred_format = (
                    False if not province and not district else candidate_is_new_format
                )
                locality_ward_info = self._lookup_ward_info(
                    raw_fragment,
                    province if province else None,
                    district if district else None,
                    preferred_format=locality_preferred_format,
                )
                if locality_ward_info is None and raw_fragment != fragment_std:
                    locality_ward_info = self._lookup_ward_info(
                        fragment_std,
                        province if province else None,
                        district if district else None,
                        preferred_format=locality_preferred_format,
                    )
                current_ward_std = self.standardize_name(ward, "basic") if ward else ""
                current_ward_looks_like_locality = bool(
                    current_ward_std
                    and self._segment_has_street_prefix(current_ward_std)
                    and not self._segment_has_explicit_admin_prefix(current_ward_std)
                )
                if locality_ward_info and (
                    ward_info is None
                    or current_ward_looks_like_locality
                    or (
                        not province
                        and not district
                        and locality_ward_info.get("is_new_format") is False
                    )
                ):
                    ward_info = locality_ward_info
                    ward_id = locality_ward_info.get("id") or locality_ward_info.get("code")
                    ward = (
                        locality_ward_info.get("full_name")
                        or locality_ward_info.get("name")
                        or ward
                    )
                if (
                    not province
                    and not district
                    and ward_info
                    and ward_info.get("is_new_format") is False
                ):
                    resolved_is_new_format = False
                    candidate_is_new_format = False
                break

        normalized_node = self.AddressNode(
            province or "",
            district or "",
            ward or "",
            is_new_format=resolved_is_new_format,
        )

        province_surface = _recover_tail_segment_surface(province, max_offset=1)
        district_surface = _recover_tail_segment_surface(district, max_offset=2)
        if compact_prefixed_district_raw:
            district_surface = compact_prefixed_district_raw
        ward_surface = _recover_tail_segment_surface(ward, max_offset=3)

        component_aliases: dict[AdministrativeLevel, list[str]] = {
            "province": self._gather_alias_values(
                province,
                province_info,
                level="province",
                extra_values=[
                    province_surface,
                    detected_components_raw[0] if detected_components_raw else None,
                    detected_prov,
                ],
            ),
            "district": self._gather_alias_values(
                district,
                district_info,
                level="district",
                extra_values=[district_surface, raw_detected_dist, detected_dist],
            ),
            "ward": self._gather_alias_values(
                ward,
                ward_info,
                level="ward",
                extra_values=[
                    ward_surface,
                    raw_detected_ward,
                    normalized_detected_ward_token,
                    detected_ward,
                ],
            ),
        }
        street_component_aliases: dict[AdministrativeLevel, list[str]] = {
            key: list(values) for key, values in component_aliases.items()
        }
        if compact_prefixed_ward_raw and compact_prefixed_district_raw:
            compact_admin_segment = (
                f"{compact_prefixed_ward_raw} {compact_prefixed_district_raw}".strip()
            )
            if compact_admin_segment:
                street_component_aliases.setdefault("district", []).append(compact_admin_segment)
        street_address = self._extract_street_address(
            input_string,
            normalized_node,
            street_component_aliases,
        )
        district_cleanup_aliases = self._gather_alias_values(
            district
            or (
                district_info.get("full_name") or district_info.get("name")
                if isinstance(district_info, dict)
                else None
            ),
            district_info,
            level="district",
            extra_values=[raw_detected_dist, detected_dist],
        )
        preferred_district_from_input = self._prefer_component_alias_from_segments(
            district_cleanup_aliases,
            input_segments,
            level="district",
        )
        if preferred_district_from_input and street_address:
            trimmed_street = self._strip_trailing_component_fragment(
                street_address,
                preferred_district_from_input,
            )
            if trimmed_street:
                street_address = trimmed_street
        province_cleanup_aliases = self._gather_alias_values(
            province
            or (
                province_info.get("full_name") or province_info.get("name")
                if isinstance(province_info, dict)
                else None
            ),
            province_info,
            level="province",
            extra_values=[
                detected_components_raw[0] if detected_components_raw else None,
                detected_prov,
            ],
        )
        preferred_province_from_input = self._prefer_component_alias_from_segments(
            province_cleanup_aliases,
            input_segments,
            level="province",
        )
        if preferred_province_from_input and street_address:
            trimmed_street = self._strip_trailing_component_fragment(
                street_address,
                preferred_province_from_input,
            )
            if trimmed_street:
                street_address = trimmed_street
        if segment_suffix_detected_ward_raw and street_address:
            trimmed_street = self._strip_trailing_component_fragment(
                street_address,
                segment_suffix_detected_ward_raw,
                allow_plain_prefix=True,
            )
            if trimmed_street:
                street_address = trimmed_street
        if late_locality_suffix_raw and street_address:
            trimmed_street = self._strip_trailing_component_fragment(
                street_address,
                late_locality_suffix_raw,
                allow_plain_prefix=True,
            )
            if trimmed_street:
                street_address = trimmed_street
        province_component = self._format_component(province, province_id, province_info)
        district_component = self._format_component(district, district_id, district_info)
        ward_component = self._format_component(ward, ward_id, ward_info)
        if province_component and component_aliases.get("province"):
            province_component["aliases"] = component_aliases["province"]
        if district_component and component_aliases.get("district"):
            district_component["aliases"] = component_aliases["district"]
        if ward_component and component_aliases.get("ward"):
            ward_component["aliases"] = component_aliases["ward"]
        if resolved_is_new_format is True:
            payload: ParseResult = {
                "province": province_component,
                "district": None,
                "ward": ward_component,
                "street_address": street_address,
                "format": "new",
                "is_new": True,
            }
        elif resolved_is_new_format is False:
            payload = {
                "province": province_component,
                "district": district_component,
                "ward": ward_component,
                "street_address": street_address,
                "format": "old",
                "is_new": False,
            }
        else:
            payload = {
                "province": province_component,
                "district": district_component,
                "ward": ward_component,
                "street_address": street_address,
                "format": "unknown",
                "is_new": None,
            }
        return payload

    def preprocess_address(self) -> None:
        raw_data: RawAddressDataset = self._build_raw_dataset()
        data: LegacyAddressDataset = self._normalize_address_dataset(raw_data)

        # Reset caches; the parser may be reinstantiated multiple times in tests
        self.address_node_list.clear()
        self.invert_ngrams_idx.clear()
        self.invert_province_to_indices.clear()
        self.invert_district_to_indices.clear()
        self.invert_ward_to_indices.clear()
        self.province_names_std.clear()
        self.district_names_std.clear()
        self.ward_names_std.clear()
        self.province_lookup.clear()
        self.district_lookup.clear()
        self.district_lookup_by_name.clear()
        self.ward_lookup.clear()
        self.ward_lookup_by_name.clear()
        self.ward_lookup_by_province_name.clear()
        self.ward_lookup_by_district_key.clear()

        def legacy_aliases_from(
            entry: AdministrativeRecord
            | LegacyProvinceRecord
            | LegacyDistrictRecord
            | LegacyWardRecord
            | None,
        ) -> list[str]:
            if not isinstance(entry, dict):
                return []
            raw = entry.get("legacy_names")
            aliases: list[str] = []
            if isinstance(raw, str):
                candidate = raw.strip()
                if candidate:
                    aliases.append(candidate)
            elif isinstance(raw, list):
                for alias in raw:
                    if isinstance(alias, str):
                        candidate = alias.strip()
                        if candidate and candidate not in aliases:
                            aliases.append(candidate)
            return aliases

        for province_name, province_payload in data.items():
            province_entry = province_payload or {}
            province_id = None
            province_code = None
            districts_payload: object = province_entry
            if isinstance(province_entry, dict) and "districts" in province_entry:
                province_id = province_entry.get("id")
                province_code = province_entry.get("code")
                districts_payload = province_entry.get("districts", {})
            if districts_payload is None:
                districts_payload = {}

            province_output_name = province_name
            province_output_std = self.standardize_name(province_output_name, "basic")
            reference_aliases = self._reference_aliases_for_level("province", province_code)
            province_aliases_extra = list(reference_aliases or [])
            province_aliases_extra.extend(self._get_special_province_aliases(province_output_name))
            province_aliases_extra.extend(legacy_aliases_from(province_entry))
            province_aliases = self._collect_aliases(
                province_output_name,
                province_name,
                province_aliases_extra,
            )
            province_aliases_std = self._standardize_aliases(province_aliases)
            province_info: AdministrativeRecord = {
                "id": province_id,
                "code": province_code,
                "name": province_output_name,
            }
            if isinstance(province_entry, dict):
                legacy_names = legacy_aliases_from(province_entry)
                if legacy_names:
                    province_info["legacy_names"] = legacy_names
                full_name = province_entry.get("full_name")
                if full_name:
                    province_info["full_name"] = full_name
            for alias_std in province_aliases_std:
                if not alias_std:
                    continue
                self.province_names_std.add(alias_std)
                self.province_lookup[alias_std] = province_info

            province_node = self.AddressNode(
                province_output_name,
                "",
                "",
                province_id=province_id,
            )
            std_name, ngrams = self._build_node_search_profile(
                province_aliases,
                [],
                [],
                include_province=True,
                include_district=False,
                include_ward=False,
            )

            province_node.standardized_full_name = std_name
            province_node.ngram_list = ngrams
            self.address_node_list.append(province_node)
            self._register_node_aliases(
                len(self.address_node_list) - 1,
                province_aliases_std=province_aliases_std,
            )

            for district_name, district_payload in districts_payload.items():
                district_entry = district_payload or {}
                district_id = None
                wards_payload: object = district_entry
                if isinstance(district_entry, dict) and "wards" in district_entry:
                    district_id = district_entry.get("id")
                    wards_payload = district_entry.get("wards", {})
                if wards_payload is None:
                    wards_payload = {}

                district_output_name = district_name
                district_output_std = self.standardize_name(district_output_name, "basic")
                district_key = district_output_std or ""
                district_id_value = district_id if district_output_name else None
                district_legacy_aliases = legacy_aliases_from(district_entry)

                district_code = (
                    district_entry.get("code") if isinstance(district_entry, dict) else None
                )
                district_info: AdministrativeRecord = {
                    "id": district_id_value,
                    "code": district_code,
                    "name": district_output_name,
                    "province_key": province_output_std,
                    "province_name": province_output_name,
                }
                if district_legacy_aliases:
                    district_info["legacy_names"] = district_legacy_aliases
                if isinstance(district_entry, dict):
                    full_name = district_entry.get("full_name")
                    if full_name:
                        district_info["full_name"] = full_name
                if province_output_std:
                    self.district_lookup[(province_output_std, district_key)] = district_info
                district_aliases = self._collect_aliases(
                    district_output_name,
                    district_name,
                    district_legacy_aliases,
                )
                district_aliases = self._augment_aliases(district_aliases, "district")
                district_aliases_std = self._standardize_aliases(district_aliases)
                for alias_std in district_aliases_std:
                    if not alias_std:
                        continue
                    self.district_names_std.add(alias_std)
                    if district_info not in self.district_lookup_by_name[alias_std]:
                        self.district_lookup_by_name[alias_std].append(district_info)

                if not district_output_std:
                    ward_iter = (
                        cast(dict[str, LegacyWardRecord], wards_payload).items()
                        if isinstance(wards_payload, dict)
                        else ((ward_name, None) for ward_name in cast(list[str], wards_payload))
                    )
                    for ward_name, ward_meta in ward_iter:
                        if not ward_name:
                            continue
                        ward_id_value = ward_meta.get("id") if isinstance(ward_meta, dict) else None
                        ward_code = ward_meta.get("code") if isinstance(ward_meta, dict) else None
                        ward_legacy_aliases = legacy_aliases_from(ward_meta)
                        ward_output_name, ward_lookup_name = self._derive_ward_names(
                            ward_name, ward_meta
                        )
                        ward_lookup_std = self.standardize_name(ward_lookup_name, "basic")
                        extra_aliases = list(
                            self._reference_aliases_for_level("ward", ward_code) or []
                        )
                        extra_aliases.extend(ward_legacy_aliases)
                        ward_aliases = self._collect_aliases(
                            ward_output_name,
                            ward_name,
                            extra_aliases,
                        )
                        ward_aliases = self._augment_aliases(ward_aliases, "ward")
                        ward_aliases_std = self._standardize_aliases(ward_aliases)
                        for alias_std in ward_aliases_std:
                            if alias_std:
                                self.ward_names_std.add(alias_std)

                        ward_info: AdministrativeRecord = {
                            "id": ward_id_value,
                            "code": ward_code if ward_code is not None else ward_id_value,
                            "name": ward_output_name,
                            "province_key": province_output_std,
                            "province_name": province_output_name,
                            "district_key": district_key,
                            "district_name": district_output_name,
                            "is_new_format": True,
                        }
                        if isinstance(ward_meta, dict):
                            new_ward_full_name = ward_meta.get("full_name")
                            if new_ward_full_name:
                                ward_info["full_name"] = new_ward_full_name
                        if ward_legacy_aliases:
                            ward_info["legacy_names"] = ward_legacy_aliases
                        if province_output_std and ward_lookup_std:
                            self.ward_lookup[
                                (province_output_std, district_key, ward_lookup_std)
                            ] = ward_info
                            self.ward_lookup_by_province_name[
                                (province_output_std, ward_lookup_std)
                            ].append(ward_info)
                        if ward_lookup_std:
                            self.ward_lookup_by_name[ward_lookup_std].append(ward_info)
                            self.ward_lookup_by_district_key[district_key].append(ward_info)
                        self._register_alias_lookup_entry(
                            self.ward_lookup_by_name,
                            ward_aliases_std,
                            ward_info,
                        )

                        ward_node = self.AddressNode(
                            "",
                            "",
                            ward_output_name,
                            province_id=province_id,
                            district_id=None,
                            ward_id=ward_id_value,
                            is_new_format=True,
                        )
                        std_name, ngrams = self._build_node_search_profile(
                            province_aliases,
                            [],
                            ward_aliases,
                            include_province=False,
                            include_district=False,
                            include_ward=True,
                        )
                        ward_node.standardized_full_name = std_name
                        ward_node.ngram_list = ngrams
                        self.address_node_list.append(ward_node)
                        self._register_node_aliases(
                            len(self.address_node_list) - 1,
                            ward_aliases_std=ward_aliases_std,
                        )

                        province_ward_node = self.AddressNode(
                            province_output_name,
                            "",
                            ward_output_name,
                            province_id=province_id,
                            district_id=None,
                            ward_id=ward_id_value,
                            is_new_format=True,
                        )
                        std_name, ngrams = self._build_node_search_profile(
                            province_aliases,
                            [],
                            ward_aliases,
                            include_province=True,
                            include_district=False,
                            include_ward=True,
                        )

                        province_ward_node.standardized_full_name = std_name
                        province_ward_node.ngram_list = ngrams
                        self.address_node_list.append(province_ward_node)
                        self._register_node_aliases(
                            len(self.address_node_list) - 1,
                            province_aliases_std=province_aliases_std,
                            ward_aliases_std=ward_aliases_std,
                        )
                    continue

                district_node = self.AddressNode(
                    "",
                    district_output_name,
                    "",
                    province_id=province_id,
                    district_id=district_id_value,
                    ward_id=None,
                    is_new_format=False,
                )
                std_name, ngrams = self._build_node_search_profile(
                    province_aliases,
                    district_aliases,
                    [],
                    include_province=False,
                    include_district=True,
                    include_ward=False,
                )
                district_node.standardized_full_name = std_name
                district_node.ngram_list = ngrams
                self.address_node_list.append(district_node)
                self._register_node_aliases(
                    len(self.address_node_list) - 1,
                    district_aliases_std=district_aliases_std,
                )

                province_district_node = self.AddressNode(
                    province_output_name,
                    district_output_name,
                    "",
                    province_id=province_id,
                    district_id=district_id_value,
                    ward_id=None,
                    is_new_format=False,
                )
                std_name, ngrams = self._build_node_search_profile(
                    province_aliases,
                    district_aliases,
                    [],
                    include_province=True,
                    include_district=True,
                    include_ward=False,
                )

                province_district_node.standardized_full_name = std_name
                province_district_node.ngram_list = ngrams
                self.address_node_list.append(province_district_node)
                self._register_node_aliases(
                    len(self.address_node_list) - 1,
                    province_aliases_std=province_aliases_std,
                    district_aliases_std=district_aliases_std,
                )

                ward_iter = (
                    cast(dict[str, LegacyWardRecord], wards_payload).items()
                    if isinstance(wards_payload, dict)
                    else ((ward_name, None) for ward_name in cast(list[str], wards_payload))
                )
                for ward_name, ward_meta in ward_iter:
                    if not ward_name:
                        continue
                    ward_id_value = ward_meta.get("id") if isinstance(ward_meta, dict) else None
                    ward_code = ward_meta.get("code") if isinstance(ward_meta, dict) else None
                    ward_legacy_aliases = legacy_aliases_from(ward_meta)
                    ward_output_name, ward_lookup_name = self._derive_ward_names(
                        ward_name, ward_meta
                    )
                    ward_output_std = self.standardize_name(ward_lookup_name, "basic")
                    extra_aliases = list(self._reference_aliases_for_level("ward", ward_code) or [])
                    extra_aliases.extend(ward_legacy_aliases)
                    ward_aliases = self._collect_aliases(
                        ward_output_name,
                        ward_name,
                        extra_aliases,
                    )
                    ward_aliases.extend(ward_legacy_aliases)
                    custom_aliases = CUSTOM_WARD_ALIASES_BY_CODE.get(str(ward_code))
                    if custom_aliases:
                        ward_aliases.extend(custom_aliases)
                    custom_aliases = CUSTOM_WARD_ALIASES_BY_CODE.get(str(ward_code))
                    if custom_aliases:
                        ward_aliases.extend(custom_aliases)
                    ward_aliases = self._augment_aliases(ward_aliases, "ward")
                    ward_aliases_std = self._standardize_aliases(ward_aliases)
                    for alias_std in ward_aliases_std:
                        if alias_std:
                            self.ward_names_std.add(alias_std)

                    ward_info = {
                        "id": ward_id_value,
                        "code": ward_code if ward_code is not None else ward_id_value,
                        "name": ward_output_name,
                        "province_key": province_output_std,
                        "province_name": province_output_name,
                        "district_key": district_key,
                        "district_name": district_output_name,
                        "is_new_format": False,
                    }
                    if isinstance(ward_meta, dict):
                        old_ward_full_name = ward_meta.get("full_name")
                        if old_ward_full_name:
                            ward_info["full_name"] = old_ward_full_name
                    if ward_legacy_aliases:
                        ward_info["legacy_names"] = ward_legacy_aliases
                    if province_output_std and ward_output_std:
                        self.ward_lookup[(province_output_std, district_key, ward_output_std)] = (
                            ward_info
                        )
                        self.ward_lookup_by_province_name[
                            (province_output_std, ward_output_std)
                        ].append(ward_info)
                    if ward_output_std:
                        self.ward_lookup_by_name[ward_output_std].append(ward_info)
                        self.ward_lookup_by_district_key[district_key].append(ward_info)
                    self._register_alias_lookup_entry(
                        self.ward_lookup_by_name,
                        ward_aliases_std,
                        ward_info,
                    )

                    ward_node = self.AddressNode(
                        "",
                        "",
                        ward_output_name,
                        province_id=province_id,
                        district_id=district_id_value,
                        ward_id=ward_id_value,
                        is_new_format=False,
                    )
                    std_name, ngrams = self._build_node_search_profile(
                        province_aliases,
                        district_aliases,
                        ward_aliases,
                        include_province=False,
                        include_district=False,
                        include_ward=True,
                    )
                    ward_node.standardized_full_name = std_name
                    ward_node.ngram_list = ngrams
                    self.address_node_list.append(ward_node)
                    self._register_node_aliases(
                        len(self.address_node_list) - 1,
                        ward_aliases_std=ward_aliases_std,
                    )

                    district_ward_node = self.AddressNode(
                        "",
                        district_output_name,
                        ward_output_name,
                        province_id=province_id,
                        district_id=district_id_value,
                        ward_id=ward_id_value,
                        is_new_format=False,
                    )
                    std_name, ngrams = self._build_node_search_profile(
                        province_aliases,
                        district_aliases,
                        ward_aliases,
                        include_province=False,
                        include_district=True,
                        include_ward=True,
                    )
                    district_ward_node.standardized_full_name = std_name
                    district_ward_node.ngram_list = ngrams
                    self.address_node_list.append(district_ward_node)
                    self._register_node_aliases(
                        len(self.address_node_list) - 1,
                        district_aliases_std=district_aliases_std,
                        ward_aliases_std=ward_aliases_std,
                    )

                    province_district_ward_node = self.AddressNode(
                        province_output_name,
                        district_output_name,
                        ward_output_name,
                        province_id=province_id,
                        district_id=district_id_value,
                        ward_id=ward_id_value,
                        is_new_format=False,
                    )
                    std_name, ngrams = self._build_node_search_profile(
                        province_aliases,
                        district_aliases,
                        ward_aliases,
                        include_province=True,
                        include_district=True,
                        include_ward=True,
                    )
                    province_district_ward_node.standardized_full_name = std_name
                    province_district_ward_node.ngram_list = ngrams
                    self.address_node_list.append(province_district_ward_node)
                    self._register_node_aliases(
                        len(self.address_node_list) - 1,
                        province_aliases_std=province_aliases_std,
                        district_aliases_std=district_aliases_std,
                        ward_aliases_std=ward_aliases_std,
                    )

        for index, node in enumerate(self.address_node_list, start=0):
            self.generate_ngram_inverted_index(node.ngram_list, index, self.invert_ngrams_idx)

        self._refresh_detection_choices()
        self._rebuild_packed_ngram_index()
        self._rebuild_search_engine()

    def _dataset_signature(
        self,
    ) -> DatasetSignature:
        tracked_paths = (
            self.new_format_provinces_path,
            self.new_format_wards_path,
            self.new_format_mapping_path,
            self.old_provinces_path,
            self.old_districts_path,
            self.old_wards_path,
        )
        signature: list[tuple[str, float | None, int | None]] = []
        signature.append(
            (
                "__cache_version__",
                float(self._CACHE_VERSION),
                len(self._STATEFUL_ATTRS),
            )
        )
        for path in tracked_paths:
            # Use filename only (not full path) for cross-platform compatibility
            # This allows pickle cache to be shared between Windows/Mac/Linux
            filename = os.path.basename(path)
            try:
                stat_result = os.stat(path)
                signature.append((filename, stat_result.st_mtime, stat_result.st_size))
            except OSError:
                signature.append((filename, None, None))
        return tuple(signature)

    def _hydrate_preprocessed_state(
        self,
        signature: DatasetSignature,
    ) -> bool:
        cls = self.__class__
        cache = cls._PREPROCESSED_CACHE
        if cache and cls._PREPROCESSED_SIGNATURE == signature:
            try:
                self._apply_preprocessed_state(cache)
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("Failed to restore in-memory AddressParser cache: %s", exc)
                cls._PREPROCESSED_CACHE = None
                cls._PREPROCESSED_SIGNATURE = None
                return False
            return True
        return False

    def _cache_payload(
        self,
        signature: DatasetSignature,
    ) -> PreprocessedCachePayload:
        return {
            "version": self._CACHE_VERSION,
            "signature": signature,
            "state": self._capture_preprocessed_state(),
        }

    def _persist_preprocessed_state(
        self,
        signature: DatasetSignature,
    ) -> None:
        if not self._cache_path:
            return
        payload = self._cache_payload(signature)
        try:
            with open(self._cache_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            logger.debug("Failed to persist AddressParser cache: %s", exc)

    def _hydrate_persistent_state(
        self,
        signature: DatasetSignature,
    ) -> bool:
        path = self._cache_path
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                raw_payload = cast(object, pickle.load(f))
        except Exception as exc:
            logger.debug("Failed to load AddressParser cache: %s", exc)
            return False

        payload = self._validate_preprocessed_cache_payload(raw_payload)
        if payload is None:
            return False
        if payload["version"] != self._CACHE_VERSION:
            return False
        cached_signature = payload["signature"]
        state = payload["state"]

        def _signature_equivalent_ignoring_mtime(
            cached: DatasetSignature,
            current: DatasetSignature,
        ) -> bool:
            if len(cached) != len(current) or not cached:
                return False
            for idx, (a, b) in enumerate(zip(cached, current)):
                if a[0] != b[0]:
                    return False
                # Keep strict equality for the cache header entry.
                if idx == 0:
                    if a != b:
                        return False
                    continue
                # Ignore mtime differences; size is enough to detect most dataset changes.
                if a[2] != b[2]:
                    return False
            return True

        if cached_signature != signature and not _signature_equivalent_ignoring_mtime(
            cached_signature, signature
        ):
            return False
        try:
            self._apply_preprocessed_state(state)
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Failed to restore AddressParser cache: %s", exc)
            return False
        return True

    def _cache_preprocessed_state(
        self,
        signature: DatasetSignature,
    ) -> None:
        cls = self.__class__
        cls._PREPROCESSED_CACHE = self._capture_preprocessed_state()
        cls._PREPROCESSED_SIGNATURE = signature

    def _capture_preprocessed_state(self) -> PreprocessedState:
        search_engine_state = (
            self.search_engine.get_state() if self.search_engine is not None else None
        )
        return {
            "address_node_list": self._serialize_address_nodes(self.address_node_list),
            "invert_ngrams_idx": self.invert_ngrams_idx,
            "invert_province_to_indices": self.invert_province_to_indices,
            "invert_district_to_indices": self.invert_district_to_indices,
            "invert_ward_to_indices": self.invert_ward_to_indices,
            "province_names_std": self.province_names_std,
            "district_names_std": self.district_names_std,
            "ward_names_std": self.ward_names_std,
            "province_lookup": self.province_lookup,
            "district_lookup": self.district_lookup,
            "district_lookup_by_name": self.district_lookup_by_name,
            "ward_lookup": self.ward_lookup,
            "ward_lookup_by_name": self.ward_lookup_by_name,
            "ward_lookup_by_province_name": self.ward_lookup_by_province_name,
            "ward_lookup_by_district_key": self.ward_lookup_by_district_key,
            "ward_mapping_by_old_code": self.ward_mapping_by_old_code,
            "ward_mapping_by_new_code": self.ward_mapping_by_new_code,
            "old_province_records": self.old_province_records,
            "old_district_records": self.old_district_records,
            "old_ward_records": self.old_ward_records,
            "new_province_records": self.new_province_records,
            "new_ward_records": self.new_ward_records,
            "external_new_province_records": self.external_new_province_records,
            "external_new_ward_records": self.external_new_ward_records,
            "search_engine": search_engine_state,
        }

    def _apply_preprocessed_state(self, state: PreprocessedState) -> None:
        restored_search_engine: AddressSearchEngine | None = None
        search_engine_state = state["search_engine"]
        if search_engine_state is not None:
            candidate_engine = AddressSearchEngine(
                analyzer=self._analyze_search_text,
                normalize_id=self.normalize_address_code,
            )
            candidate_engine.restore_cached_state(search_engine_state)
            if candidate_engine.document_count:
                restored_search_engine = candidate_engine

        # Apply parser fields only after the nested search state has validated.
        self.address_node_list = self._deserialize_address_nodes(state["address_node_list"])
        self.invert_ngrams_idx = state["invert_ngrams_idx"]
        self.invert_province_to_indices = state["invert_province_to_indices"]
        self.invert_district_to_indices = state["invert_district_to_indices"]
        self.invert_ward_to_indices = state["invert_ward_to_indices"]
        self.province_names_std = state["province_names_std"]
        self.district_names_std = state["district_names_std"]
        self.ward_names_std = state["ward_names_std"]
        self.province_lookup = state["province_lookup"]
        self.district_lookup = state["district_lookup"]
        self.district_lookup_by_name = state["district_lookup_by_name"]
        self.ward_lookup = state["ward_lookup"]
        self.ward_lookup_by_name = state["ward_lookup_by_name"]
        self.ward_lookup_by_province_name = state["ward_lookup_by_province_name"]
        self.ward_lookup_by_district_key = state["ward_lookup_by_district_key"]
        self.ward_mapping_by_old_code = state["ward_mapping_by_old_code"]
        self.ward_mapping_by_new_code = state["ward_mapping_by_new_code"]
        self.old_province_records = state["old_province_records"]
        self.old_district_records = state["old_district_records"]
        self.old_ward_records = state["old_ward_records"]
        self.new_province_records = state["new_province_records"]
        self.new_ward_records = state["new_ward_records"]
        self.external_new_province_records = state["external_new_province_records"]
        self.external_new_ward_records = state["external_new_ward_records"]
        self._refresh_detection_choices()
        self._rebuild_packed_ngram_index()

        self.search_engine = restored_search_engine

        # Only rebuild if search engine wasn't restored from cache
        if self.search_engine is None:
            self._rebuild_search_engine()

    def _serialize_address_nodes(
        self, nodes: list[AddressParser.AddressNode]
    ) -> list[AddressNodeState]:
        payload: list[AddressNodeState] = []
        for node in nodes:
            if not isinstance(node, AddressParser.AddressNode):
                continue
            payload.append(
                {
                    "province_name": node.province_name,
                    "district_name": node.district_name,
                    "ward_name": node.ward_name,
                    "province_id": node.province_id,
                    "district_id": node.district_id,
                    "ward_id": node.ward_id,
                    "is_new_format": node.is_new_format,
                    "standardized_full_name": node.standardized_full_name,
                    "ngram_list": list(node.ngram_list),
                }
            )
        return payload

    def _deserialize_address_nodes(
        self, payload: list[AddressNodeState]
    ) -> list[AddressParser.AddressNode]:
        nodes: list[AddressParser.AddressNode] = []
        for item in payload:
            node = self.AddressNode(
                item["province_name"],
                item["district_name"],
                item["ward_name"],
                province_id=item["province_id"],
                district_id=item["district_id"],
                ward_id=item["ward_id"],
                is_new_format=item["is_new_format"],
            )
            node.standardized_full_name = item["standardized_full_name"]
            node.ngram_list = set(item["ngram_list"])
            nodes.append(node)
        return nodes

    def _normalize_code_str(self, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise TypeError("address code must be a string or integer")
        return self.normalize_address_code(value)

    def _read_json_file(self, path: str | None) -> object | None:
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return cast(object, json.load(f))
        except Exception:
            return None

    @staticmethod
    def _object_mapping(value: object) -> dict[object, object] | None:
        if not isinstance(value, dict):
            return None
        return cast(dict[object, object], value)

    @staticmethod
    def _object_list(value: object) -> list[object] | None:
        if not isinstance(value, list):
            return None
        return cast(list[object], value)

    @staticmethod
    def _optional_string(value: object) -> str | None:
        return value if isinstance(value, str) else None

    def _validate_dataset_signature(
        self,
        value: object,
    ) -> DatasetSignature | None:
        if not isinstance(value, tuple):
            return None

        raw_signature = cast(tuple[object, ...], value)
        signature: list[tuple[str, float | None, int | None]] = []
        for raw_entry in raw_signature:
            if not isinstance(raw_entry, tuple):
                return None
            entry = cast(tuple[object, ...], raw_entry)
            if len(entry) != 3:
                return None

            filename, raw_mtime, raw_size = entry
            if not isinstance(filename, str):
                return None
            if raw_mtime is not None and not isinstance(raw_mtime, (int, float)):
                return None
            if raw_size is not None and (
                not isinstance(raw_size, int) or isinstance(raw_size, bool)
            ):
                return None
            mtime = float(raw_mtime) if raw_mtime is not None else None
            signature.append((filename, mtime, raw_size))
        return tuple(signature)

    def _validate_address_node_state(
        self,
        value: object,
    ) -> AddressNodeState | None:
        raw_node = self._object_mapping(value)
        if raw_node is None or set(raw_node) != self._ADDRESS_NODE_STATE_KEYS:
            return None

        province_name = raw_node.get("province_name")
        district_name = raw_node.get("district_name")
        ward_name = raw_node.get("ward_name")
        standardized_full_name = raw_node.get("standardized_full_name")
        if not all(
            isinstance(name, str)
            for name in (
                province_name,
                district_name,
                ward_name,
                standardized_full_name,
            )
        ):
            return None

        province_id = raw_node.get("province_id")
        district_id = raw_node.get("district_id")
        ward_id = raw_node.get("ward_id")
        if any(
            value is not None and not isinstance(value, str)
            for value in (province_id, district_id, ward_id)
        ):
            return None

        is_new_format = raw_node.get("is_new_format")
        if is_new_format is not None and not isinstance(is_new_format, bool):
            return None

        raw_ngrams = self._object_list(raw_node.get("ngram_list"))
        if raw_ngrams is None or not all(isinstance(ngram, str) for ngram in raw_ngrams):
            return None

        return cast(AddressNodeState, raw_node)

    def _validate_administrative_record_state(
        self,
        value: object,
    ) -> AdministrativeRecord | None:
        raw_record = self._object_mapping(value)
        if raw_record is None:
            return None

        keys = set(raw_record)
        if ADMINISTRATIVE_RECORD_REQUIRED_KEYS - keys or keys - ADMINISTRATIVE_RECORD_KEYS:
            return None
        for key in ADMINISTRATIVE_RECORD_STRING_KEYS:
            if key in raw_record:
                field_value = raw_record[key]
                if field_value is not None and not isinstance(field_value, str):
                    return None
        for key in ADMINISTRATIVE_RECORD_OPTIONAL_INTEGER_KEYS:
            if key in raw_record:
                field_value = raw_record[key]
                if field_value is not None and (
                    not isinstance(field_value, int) or isinstance(field_value, bool)
                ):
                    return None
        for key in ADMINISTRATIVE_RECORD_OPTIONAL_STRING_LIST_KEYS:
            if key not in raw_record:
                continue
            values = self._object_list(raw_record[key])
            if values is None or not all(isinstance(item, str) for item in values):
                return None
        if "is_new_format" in raw_record and not isinstance(raw_record["is_new_format"], bool):
            return None

        return cast(AdministrativeRecord, raw_record)

    def _validate_index_state_header(
        self,
        value: object,
        *,
        node_count: int,
    ) -> dict[str, set[int]] | None:
        raw_index = self._object_mapping(value)
        if raw_index is None:
            return None
        sample = next(iter(raw_index.items()), None)
        if sample is not None:
            token, raw_indices = sample
            if not isinstance(token, str) or not isinstance(raw_indices, set):
                return None
            indices = cast(set[object], raw_indices)
            if any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= node_count
                for index in indices
            ):
                return None
        return cast(dict[str, set[int]], raw_index)

    def _validate_search_engine_state_header(
        self,
        value: object,
    ) -> SearchEngineState | None:
        raw_state = self._object_mapping(value)
        if raw_state is None or set(raw_state) != SEARCH_ENGINE_STATE_KEYS:
            return None

        raw_lists: list[list[object]] = []
        for key in SEARCH_ENGINE_STATE_KEYS:
            raw_list = self._object_list(raw_state[key])
            if raw_list is None:
                return None
            raw_lists.append(raw_list)
        if len({len(items) for items in raw_lists}) != 1:
            return None

        metadata = self._object_list(raw_state["metadata"])
        if metadata:
            try:
                validate_search_metadata(metadata)
            except ValueError:
                return None
        return cast(SearchEngineState, raw_state)

    def _validate_preprocessed_state_header(
        self,
        value: object,
    ) -> PreprocessedState | None:
        """Check the versioned cache header before trusting its internal payload."""
        raw_state = self._object_mapping(value)
        if raw_state is None or set(raw_state) != set(self._STATEFUL_ATTRS):
            return None

        raw_nodes = self._object_list(raw_state["address_node_list"])
        if not raw_nodes:
            return None
        # This pickle is an internal, versioned performance cache. Validate a
        # representative header here; JSON source rows are validated when the
        # cache is built. A full Python walk would erase most of the cache win.
        for raw_node in (raw_nodes[0], raw_nodes[-1]):
            if self._validate_address_node_state(raw_node) is None:
                return None

        node_count = len(raw_nodes)
        for field in (
            "invert_ngrams_idx",
            "invert_province_to_indices",
            "invert_district_to_indices",
            "invert_ward_to_indices",
        ):
            if (
                self._validate_index_state_header(
                    raw_state[field],
                    node_count=node_count,
                )
                is None
            ):
                return None

        for field in (
            "invert_province_to_indices",
            "invert_district_to_indices",
            "invert_ward_to_indices",
        ):
            index = raw_state[field]
            if not isinstance(index, defaultdict) or index.default_factory is not set:
                return None

        for field in (
            "district_lookup_by_name",
            "ward_lookup_by_name",
            "ward_lookup_by_province_name",
            "ward_lookup_by_district_key",
        ):
            lookup = raw_state[field]
            if not isinstance(lookup, defaultdict) or lookup.default_factory is not list:
                return None

        mapping_fields = (
            "province_lookup",
            "district_lookup",
            "ward_lookup",
            "ward_mapping_by_old_code",
            "ward_mapping_by_new_code",
            "old_province_records",
            "old_district_records",
            "old_ward_records",
            "new_province_records",
            "new_ward_records",
            "external_new_province_records",
            "external_new_ward_records",
        )
        if any(self._object_mapping(raw_state[field]) is None for field in mapping_fields):
            return None

        set_fields = (
            "province_names_std",
            "district_names_std",
            "ward_names_std",
        )
        if any(not isinstance(raw_state[field], set) for field in set_fields):
            return None

        required_record_fields = (
            "old_province_records",
            "old_district_records",
            "old_ward_records",
            "new_province_records",
            "new_ward_records",
        )
        if any(not raw_state[field] for field in required_record_fields):
            return None

        for field in required_record_fields:
            records = self._object_mapping(raw_state[field])
            if records:
                sample_code, sample_record = next(iter(records.items()))
                if (
                    not isinstance(sample_code, str)
                    or self._validate_administrative_record_state(sample_record) is None
                ):
                    return None

        raw_search_engine = raw_state["search_engine"]
        if (
            raw_search_engine is not None
            and self._validate_search_engine_state_header(raw_search_engine) is None
        ):
            return None

        return cast(PreprocessedState, raw_state)

    def _validate_preprocessed_cache_payload(
        self,
        value: object,
    ) -> PreprocessedCachePayload | None:
        raw_payload = self._object_mapping(value)
        if raw_payload is None:
            return None

        version = raw_payload.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            return None
        signature = self._validate_dataset_signature(raw_payload.get("signature"))
        state = self._validate_preprocessed_state_header(raw_payload.get("state"))
        if signature is None or state is None:
            return None
        return {"version": version, "signature": signature, "state": state}

    def _register_external_entry(
        self,
        target: AdministrativeRecordsByCode,
        code: object,
        entry: AdministrativeRecord,
    ) -> None:
        keys: set[str] = set()
        if code is not None:
            if isinstance(code, str):
                key_str = code.strip()
            else:
                key_str = str(code)
            if key_str:
                keys.add(key_str)
        normalized = self._normalize_code_str(code)
        if normalized:
            keys.add(normalized)
        if not keys:
            return
        for key in keys:
            target[key] = entry

    def _normalize_external_record(
        self,
        value: object,
    ) -> AdministrativeRecord | None:
        raw_entry = self._object_mapping(value)
        if raw_entry is None:
            return None

        code = self._normalize_code_str(raw_entry.get("code"))
        if code is None:
            return None
        record: AdministrativeRecord = {
            "code": code,
            "id": code,
            "name": self._optional_string(raw_entry.get("name")),
        }

        full_name = self._optional_string(raw_entry.get("full_name"))
        if full_name is not None:
            record["full_name"] = full_name
        name_en = self._optional_string(raw_entry.get("name_en"))
        if name_en is not None:
            record["name_en"] = name_en
        full_name_en = self._optional_string(raw_entry.get("full_name_en"))
        if full_name_en is not None:
            record["full_name_en"] = full_name_en
        code_name = self._optional_string(raw_entry.get("code_name"))
        if code_name is not None:
            record["code_name"] = code_name
        province_code = self._normalize_code_str(raw_entry.get("province_code"))
        if province_code is not None:
            record["province_code"] = province_code

        administrative_unit_id = raw_entry.get("administrative_unit_id")
        if isinstance(administrative_unit_id, int) and not isinstance(administrative_unit_id, bool):
            record["administrative_unit_id"] = administrative_unit_id
        administrative_region_id = raw_entry.get("administrative_region_id")
        if isinstance(administrative_region_id, int) and not isinstance(
            administrative_region_id, bool
        ):
            record["administrative_region_id"] = administrative_region_id
        return record

    def _normalize_external_mapping_row(
        self,
        value: object,
    ) -> ExternalWardMappingRow | None:
        raw_row = self._object_mapping(value)
        if raw_row is None:
            return None

        old_province_code = self._normalize_code_str(raw_row.get("old_province_code"))
        old_district_code = self._normalize_code_str(raw_row.get("old_district_code"))
        old_ward_code = self._normalize_code_str(raw_row.get("old_ward_code"))
        new_province_code = self._normalize_code_str(raw_row.get("new_province_code"))
        new_ward_code = self._normalize_code_str(raw_row.get("new_ward_code"))
        if (
            old_province_code is None
            or old_district_code is None
            or old_ward_code is None
            or new_province_code is None
            or new_ward_code is None
        ):
            return None

        old_province_name = raw_row.get("old_province_name")
        old_district_name = raw_row.get("old_district_name")
        old_ward_name = raw_row.get("old_ward_name")
        new_province_name = raw_row.get("new_province_name")
        new_ward_name = raw_row.get("new_ward_name")
        if (
            not isinstance(old_province_name, str)
            or not isinstance(old_district_name, str)
            or not isinstance(new_province_name, str)
            or not isinstance(new_ward_name, str)
        ):
            return None
        if old_ward_name is not None and not isinstance(old_ward_name, str):
            return None

        mapping_row: ExternalWardMappingRow = {
            "old_province_code": old_province_code,
            "old_province_name": old_province_name,
            "old_district_code": old_district_code,
            "old_district_name": old_district_name,
            "old_ward_code": old_ward_code,
            "old_ward_name": old_ward_name,
            "new_province_code": new_province_code,
            "new_province_name": new_province_name,
            "new_ward_code": new_ward_code,
            "new_ward_name": new_ward_name,
        }
        return mapping_row

    def _load_external_new_dataset(self) -> ExternalNewDataset:
        payload: ExternalNewDataset = {
            "provinces": {},
            "wards": {},
            "ward_mappings": [],
        }
        provinces_data = self._read_json_file(self.new_format_provinces_path)
        province_entries = self._object_list(provinces_data)
        if province_entries is not None:
            for raw_entry in province_entries:
                entry = self._normalize_external_record(raw_entry)
                if entry is None:
                    continue
                self._register_external_entry(
                    payload["provinces"],
                    entry["code"],
                    entry,
                )
        wards_data = self._read_json_file(self.new_format_wards_path)
        ward_entries = self._object_list(wards_data)
        if ward_entries is not None:
            for raw_entry in ward_entries:
                entry = self._normalize_external_record(raw_entry)
                if entry is None:
                    continue
                self._register_external_entry(
                    payload["wards"],
                    entry["code"],
                    entry,
                )
        mapping_data = self._read_json_file(self.new_format_mapping_path)
        mapping_entries = self._object_list(mapping_data)
        if mapping_entries is not None:
            for raw_row in mapping_entries:
                row = self._normalize_external_mapping_row(raw_row)
                if row is not None:
                    payload["ward_mappings"].append(row)
        self.external_new_province_records = payload["provinces"]
        self.external_new_ward_records = payload["wards"]
        return payload

    def _dedupe_external_entries(
        self, records: AdministrativeRecordsByCode
    ) -> AdministrativeRecordsByCode:
        result: AdministrativeRecordsByCode = {}
        seen: set[str] = set()
        for entry in records.values():
            code = entry["code"]
            if code is None:
                continue
            code_str = code.strip()
            if not code_str or code_str in seen:
                continue
            seen.add(code_str)
            result[code_str] = entry
        return result

    def _repair_old_ward_parents(
        self,
        wards: AdministrativeRecordsByCode,
        districts: AdministrativeRecordsByCode,
        raw_source: object | None = None,
    ) -> AdministrativeRecordsByCode:
        """
        Ensure legacy ward entries point to a known district; infer the parent from
        district names when `district_code` is missing or invalid.
        """
        if not wards:
            return {}

        raw_by_code: dict[str, dict[object, object]] = {}
        raw_entries = self._object_list(raw_source)
        if raw_entries is not None:
            for raw_entry in raw_entries:
                entry = self._object_mapping(raw_entry)
                if entry is None:
                    continue
                code_str = self._normalize_code_str(entry.get("code"))
                if code_str:
                    raw_by_code[code_str] = entry

        def _province_hint(code: str) -> str | None:
            raw = raw_by_code.get(code)
            if raw is None:
                return None
            return self._normalize_code_str(raw.get("province_code"))

        district_index: dict[str, list[AdministrativeRecord]] = defaultdict(list)
        for district in districts.values():
            key = self.standardize_name(
                district.get("name") or district.get("full_name") or "", "basic"
            )
            if key:
                district_index[key].append(district)

        cleaned: AdministrativeRecordsByCode = {}
        repaired_count = 0
        dropped_count = 0

        for code, ward in wards.items():
            parent_code = ward.get("parent_code")
            if parent_code and parent_code in districts:
                cleaned[code] = ward
                continue

            ward_key = self.standardize_name(
                ward.get("name") or ward.get("full_name") or "", "basic"
            )
            if not ward_key:
                dropped_count += 1
                continue

            province_hint = _province_hint(code)
            candidates = district_index.get(ward_key, [])
            if province_hint and candidates:
                filtered = [
                    dist
                    for dist in candidates
                    if dist.get("province_code") is not None
                    and str(dist["province_code"]).strip() == province_hint
                ]
                if filtered:
                    candidates = filtered

            if len(candidates) == 1:
                inferred_parent = self.normalize_address_code(candidates[0].get("code"))
                if inferred_parent and inferred_parent in districts:
                    ward["parent_code"] = inferred_parent
                    cleaned[code] = ward
                    repaired_count += 1
                    continue

            dropped_count += 1

        if repaired_count or dropped_count:
            logger.debug(
                "Normalized legacy wards: fixed %d missing/invalid parents, dropped %d orphan entries",
                repaired_count,
                dropped_count,
            )

        return cleaned

    def _filter_new_wards_by_province(
        self,
        wards: AdministrativeRecordsByCode,
        provinces: AdministrativeRecordsByCode,
    ) -> AdministrativeRecordsByCode:
        """Drop new-format wards whose `province_code` is missing or unknown."""
        if not wards:
            return {}

        cleaned: AdministrativeRecordsByCode = {}
        dropped = 0
        for code, entry in wards.items():
            parent_code = self.normalize_address_code(entry.get("parent_code"))
            if parent_code and parent_code in provinces:
                entry["parent_code"] = parent_code
                cleaned[code] = entry
            else:
                dropped += 1

        if dropped:
            logger.debug("Dropped %d new-format wards referencing unknown provinces", dropped)

        return cleaned

    def _load_entities_by_code(
        self,
        path: str,
        *,
        parent_key: str | None = None,
    ) -> AdministrativeRecordsByCode:
        data = self._read_json_file(path)
        entries = self._object_list(data)
        if entries is None:
            return {}
        result: AdministrativeRecordsByCode = {}

        def _normalize_aliases(value: object) -> list[str] | None:
            if isinstance(value, str):
                candidate = value.strip()
                return [candidate] if candidate else None
            raw_aliases = self._object_list(value)
            if raw_aliases is None:
                return None
            aliases: list[str] = []
            for item in raw_aliases:
                if not isinstance(item, str):
                    continue
                candidate = item.strip()
                if candidate and candidate not in aliases:
                    aliases.append(candidate)
            return aliases or None

        for raw_entry in entries:
            entry = self._object_mapping(raw_entry)
            if entry is None:
                continue
            code_str = self._normalize_code_str(entry.get("code"))
            if not code_str:
                continue
            normalized: AdministrativeRecord = {
                "code": code_str,
                "id": code_str,
                "name": self._optional_string(entry.get("name")),
            }
            full_name = self._optional_string(entry.get("full_name"))
            if full_name is not None:
                normalized["full_name"] = full_name
            if parent_key:
                normalized["parent_code"] = self._normalize_code_str(entry.get(parent_key))
            name_en = self._optional_string(entry.get("name_en"))
            if name_en is not None:
                normalized["name_en"] = name_en
            full_name_en = self._optional_string(entry.get("full_name_en"))
            if full_name_en is not None:
                normalized["full_name_en"] = full_name_en
            code_name = self._optional_string(entry.get("code_name"))
            if code_name is not None:
                normalized["code_name"] = code_name
            legacy_aliases = _normalize_aliases(entry.get("legacy_names"))
            if legacy_aliases:
                normalized["legacy_names"] = legacy_aliases
            result[code_str] = normalized
        return result

    def _build_raw_dataset(self) -> RawAddressDataset:
        old_provinces = self._load_entities_by_code(self.old_provinces_path)
        old_districts = self._load_entities_by_code(
            self.old_districts_path, parent_key="province_code"
        )
        old_wards = self._load_entities_by_code(self.old_wards_path, parent_key="district_code")
        raw_old_wards = self._read_json_file(self.old_wards_path)
        old_wards = self._repair_old_ward_parents(old_wards, old_districts, raw_old_wards)

        self.old_province_records = old_provinces
        self.old_district_records = old_districts
        self.old_ward_records = old_wards

        external_payload = self._load_external_new_dataset()
        new_provinces_raw = self._dedupe_external_entries(external_payload["provinces"])
        new_wards_raw = self._dedupe_external_entries(external_payload["wards"])

        new_provinces: AdministrativeRecordsByCode = {}
        for code, entry in new_provinces_raw.items():
            province: AdministrativeRecord = {
                "code": code,
                "id": self._normalize_code_str(entry["code"] or code),
                "name": entry.get("name"),
            }
            province["name_en"] = entry.get("name_en")
            province["full_name_en"] = entry.get("full_name_en")
            province["full_name"] = entry.get("full_name")
            new_provinces[code] = province

        new_wards: AdministrativeRecordsByCode = {}
        for code, entry in new_wards_raw.items():
            parent_code = entry.get("province_code")
            normalized_parent = str(parent_code).strip() if parent_code is not None else None
            ward: AdministrativeRecord = {
                "code": code,
                "id": self._normalize_code_str(entry["code"] or code),
                "name": entry.get("name"),
                "parent_code": normalized_parent,
                "administrative_unit_id": entry.get("administrative_unit_id"),
                "is_new_format": True,
            }
            ward["full_name"] = entry.get("full_name")
            ward["name_en"] = entry.get("name_en")
            ward["full_name_en"] = entry.get("full_name_en")
            new_wards[code] = ward
        new_wards = self._filter_new_wards_by_province(new_wards, new_provinces)

        self.new_province_records = new_provinces
        self.new_ward_records = new_wards

        mapping_rows = external_payload["ward_mappings"]
        ward_old_to_new, ward_new_to_old = self._convert_external_ward_mappings(mapping_rows)

        return {
            "old": {
                "provinces": old_provinces,
                "districts": old_districts,
                "wards": old_wards,
            },
            "new": {
                "provinces": new_provinces,
                "wards": new_wards,
            },
            "mapping": {
                "ward_old_to_new": ward_old_to_new,
                "ward_new_to_old": ward_new_to_old,
            },
        }

    def _convert_external_ward_mappings(
        self,
        rows: list[ExternalWardMappingRow],
    ) -> tuple[WardMappingsByCode, WardMappingsByCode]:
        old_to_new: WardMappingsByCode = {}
        new_to_old: WardMappingsByCode = {}

        def _register(
            target: WardMappingsByCode,
            code: str | None,
            payload: WardMappingRow,
        ) -> None:
            if not code:
                return
            bucket = target.setdefault(code, [])
            bucket.append(payload)
            normalized = self.normalize_address_code(code)
            if normalized and normalized != code:
                target[normalized] = bucket

        for row in rows:
            old_code = row["old_ward_code"].strip()
            new_code = row["new_ward_code"].strip()
            if not old_code or not new_code:
                continue

            old_entry = self.old_ward_records.get(old_code)
            district_id_old = None
            city_id_old = None
            if old_entry:
                district_id_old = self.normalize_address_code(old_entry.get("parent_code"))
                if district_id_old:
                    district_entry = self.old_district_records.get(district_id_old)
                    if district_entry:
                        city_id_old = self.normalize_address_code(district_entry.get("parent_code"))

            new_entry = self.new_ward_records.get(new_code)
            city_id_new = None
            if new_entry:
                city_id_new = self.normalize_address_code(new_entry.get("parent_code"))

            old_payload: WardMappingRow = {
                "city_id_old": city_id_old,
                "district_id_old": district_id_old,
                "ward_id_old": old_code,
                "city_id_new": city_id_new,
                "ward_id_new": new_code,
                "old_ward_name": row["old_ward_name"],
                "new_ward_name": row["new_ward_name"],
                "old_province_name": row["old_province_name"],
                "new_province_name": row["new_province_name"],
                "old_district_name": row["old_district_name"],
            }
            _register(old_to_new, old_code, old_payload)

            new_payload: WardMappingRow = {
                "city_id_new": city_id_new,
                "ward_id_new": new_code,
                "city_id_old": city_id_old,
                "district_id_old": district_id_old,
                "ward_id_old": old_code,
                "old_ward_name": row["old_ward_name"],
                "new_ward_name": row["new_ward_name"],
                "old_province_name": row["old_province_name"],
                "new_province_name": row["new_province_name"],
                "old_district_name": row["old_district_name"],
            }
            _register(new_to_old, new_code, new_payload)

        return old_to_new, new_to_old

    def _normalize_address_dataset(
        self,
        raw_data: RawAddressDataset,
    ) -> LegacyAddressDataset:
        """
        Convert the data files (old/new/mapping) into a legacy nested structure
        keyed by province name for downstream processing.
        """

        old_section = raw_data["old"]
        new_section = raw_data["new"]
        mapping_section = raw_data["mapping"]

        provinces_old = old_section["provinces"]
        districts_old = old_section["districts"]
        wards_old = old_section["wards"]
        provinces_new = new_section["provinces"]
        wards_new = new_section["wards"]

        self.old_province_records = provinces_old
        self.old_district_records = districts_old
        self.old_ward_records = wards_old
        self.new_province_records = provinces_new
        self.new_ward_records = wards_new

        ward_mapping = mapping_section["ward_old_to_new"]
        ward_mapping_new = mapping_section["ward_new_to_old"]
        self.ward_mapping_by_old_code = {str(k): v for k, v in ward_mapping.items()}
        self.ward_mapping_by_new_code = {str(k): v for k, v in ward_mapping_new.items()}

        def extract_legacy_names(
            source: AdministrativeRecord | None,
        ) -> list[str]:
            if source is None:
                return []
            aliases: list[str] = []
            for alias in source.get("legacy_names", []):
                candidate = alias.strip()
                if candidate and candidate not in aliases:
                    aliases.append(candidate)
            return aliases

        legacy_view: LegacyAddressDataset = {}
        province_entries_by_code_old: dict[str, LegacyProvinceRecord] = {}
        province_entries_by_code_new: dict[str, LegacyProvinceRecord] = {}
        district_entries_by_code: dict[str, LegacyDistrictRecord] = {}

        def _preferred_name(
            entity: AdministrativeRecord | None,
            fallback: str,
        ) -> str:
            name_raw = entity.get("name") if entity is not None else None
            name = name_raw.strip() if name_raw is not None else ""
            full_name_raw = entity.get("full_name") if entity is not None else None
            extended = full_name_raw.strip() if full_name_raw is not None else ""
            if name and not name.replace(" ", "").isdigit():
                return name
            if extended:
                return extended
            if name:
                return name
            return fallback

        def ensure_province(
            code: str | None,
            payload: AdministrativeRecord | None,
            *,
            source: RegistrySource,
            prefer_name: bool = False,
        ) -> LegacyProvinceRecord:
            payload_code = payload["code"] if payload is not None else None
            normalized_code = self._normalize_code_str(code or payload_code)
            name = _preferred_name(payload, normalized_code or "Unknown Province")

            entry_by_name = legacy_view.get(name)
            if source == "new":
                entry_by_code = (
                    province_entries_by_code_new.get(normalized_code) if normalized_code else None
                )
            else:
                entry_by_code = (
                    province_entries_by_code_old.get(normalized_code) if normalized_code else None
                )

            entry: LegacyProvinceRecord | None = None
            if prefer_name and entry_by_name is not None:
                entry = entry_by_name
            if entry is None and entry_by_code is not None:
                entry = entry_by_code
            if entry is None and entry_by_name is not None:
                entry = entry_by_name

            if entry is None:
                payload_id = payload.get("id") if payload is not None else None
                payload_full_name = payload.get("full_name") if payload is not None else None
                aliases = extract_legacy_names(payload)
                entry = {
                    "id": self._normalize_code_str(payload_id or normalized_code),
                    "code": normalized_code,
                    "full_name": payload_full_name or name,
                    "districts": {},
                }
                if aliases:
                    entry["legacy_names"] = aliases
                legacy_view[name] = entry
            else:
                if entry.get("code") is None and normalized_code:
                    entry["code"] = normalized_code
                if entry.get("id") is None:
                    payload_id = payload.get("id") if payload is not None else None
                    entry["id"] = self._normalize_code_str(payload_id or normalized_code)
                payload_full_name = payload.get("full_name") if payload is not None else None
                if not entry.get("full_name") and payload_full_name:
                    entry["full_name"] = payload_full_name
                legacy_bucket = entry.setdefault("legacy_names", [])
                for alias in extract_legacy_names(payload):
                    if alias not in legacy_bucket:
                        legacy_bucket.append(alias)

            if normalized_code:
                if source == "new":
                    if prefer_name and entry_by_name is not None:
                        province_entries_by_code_new[normalized_code] = entry
                    elif normalized_code not in province_entries_by_code_new:
                        province_entries_by_code_new[normalized_code] = entry
                else:
                    if prefer_name and entry_by_name is not None:
                        province_entries_by_code_old[normalized_code] = entry
                    elif normalized_code not in province_entries_by_code_old:
                        province_entries_by_code_old[normalized_code] = entry
            return entry

        for code, info in provinces_old.items():
            ensure_province(code, info, source="old")
        for code, info in provinces_new.items():
            ensure_province(code, info, source="new", prefer_name=True)

        def merge_ward_entry(
            existing: LegacyWardRecord,
            incoming: LegacyWardRecord,
        ) -> LegacyWardRecord:
            if not existing:
                return incoming

            if existing["id"] in (None, "") and incoming["id"] not in (None, ""):
                existing["id"] = incoming["id"]
            if existing["code"] in (None, "") and incoming["code"] not in (None, ""):
                existing["code"] = incoming["code"]
            if existing["full_name"] in (None, "") and incoming["full_name"] not in (
                None,
                "",
            ):
                existing["full_name"] = incoming["full_name"]
            if (
                existing.get("administrative_unit_id") is None
                and incoming.get("administrative_unit_id") is not None
            ):
                existing["administrative_unit_id"] = incoming["administrative_unit_id"]

            # If both entries have IDs but differ, prefer the one carrying richer metadata.
            incoming_id = incoming.get("id") or incoming.get("code")
            existing_id = existing.get("id") or existing.get("code")
            has_incoming_alias = bool(incoming.get("legacy_names"))
            has_existing_alias = bool(existing.get("legacy_names"))
            if (
                incoming_id
                and existing_id
                and incoming_id != existing_id
                and has_incoming_alias
                and not has_existing_alias
            ):
                existing["id"] = incoming_id
                existing["code"] = incoming.get("code", incoming_id)

            incoming_aliases = incoming.get("legacy_names") or []
            if incoming_aliases:
                merged_aliases = list(existing.get("legacy_names") or [])
                for alias in incoming_aliases:
                    if alias not in merged_aliases:
                        merged_aliases.append(alias)
                if merged_aliases:
                    existing["legacy_names"] = merged_aliases

            return existing

        def attach_district(
            province_entry: LegacyProvinceRecord,
            code: str,
            payload: AdministrativeRecord | None,
        ) -> LegacyDistrictRecord:
            district_name = _preferred_name(payload, code)
            district_entry = province_entry["districts"].get(district_name)
            if district_entry is None:
                payload_id = payload.get("id") if payload is not None else None
                payload_code = payload.get("code") if payload is not None else None
                payload_full_name = payload.get("full_name") if payload is not None else None
                aliases = extract_legacy_names(payload)
                district_entry = {
                    "id": self._normalize_code_str(payload_id or code),
                    "code": payload_code or code,
                    "full_name": payload_full_name or district_name,
                    "wards": {},
                }
                if aliases:
                    district_entry["legacy_names"] = aliases
                province_entry["districts"][district_name] = district_entry
            else:
                legacy_bucket = district_entry.setdefault("legacy_names", [])
                for alias in extract_legacy_names(payload):
                    if alias not in legacy_bucket:
                        legacy_bucket.append(alias)
            district_entries_by_code[str(code)] = district_entry
            return district_entry

        for code, info in districts_old.items():
            province_code = info.get("parent_code")
            province_entry = province_entries_by_code_old.get(str(province_code))
            if province_entry is None:
                province_entry = ensure_province(
                    province_code,
                    provinces_old.get(str(province_code)),
                    source="old",
                )
            attach_district(province_entry, code, info)

        def new_format_bucket(
            province_entry: LegacyProvinceRecord,
        ) -> LegacyDistrictRecord:
            bucket = province_entry["districts"].get("")
            if bucket is None:
                bucket = {
                    "id": None,
                    "code": None,
                    "full_name": "",
                    "is_new_format": True,
                    "wards": {},
                }
                province_entry["districts"][""] = bucket
            bucket.setdefault("wards", {})
            return bucket

        for code, info in wards_old.items():
            parent_district = info.get("parent_code")
            district_entry = district_entries_by_code.get(str(parent_district))
            if district_entry is None:
                province_code = None
                district_payload = districts_old.get(str(parent_district))
                if district_payload is not None:
                    province_code = district_payload.get("parent_code")
                province_entry = province_entries_by_code_old.get(str(province_code))
                if province_entry is None:
                    province_entry = ensure_province(
                        province_code,
                        provinces_old.get(str(province_code)),
                        source="old",
                    )
                district_entry = attach_district(
                    province_entry, parent_district or code, district_payload
                )
            ward_name = _preferred_name(info, code)
            old_ward_entry: LegacyWardRecord = {
                "id": self._normalize_code_str(info.get("id") or code),
                "code": info.get("code") or code,
                "parent_code": info.get("parent_code"),
                "full_name": info.get("full_name") or info.get("name"),
                "administrative_unit_id": info.get("administrative_unit_id"),
                "legacy_names": extract_legacy_names(info),
            }
            if not old_ward_entry["legacy_names"]:
                del old_ward_entry["legacy_names"]
            existing_ward = district_entry["wards"].get(ward_name)
            if existing_ward is not None:
                district_entry["wards"][ward_name] = merge_ward_entry(existing_ward, old_ward_entry)
            else:
                district_entry["wards"][ward_name] = old_ward_entry

        for code, info in wards_new.items():
            province_code = info.get("parent_code")
            province_entry = ensure_province(
                province_code,
                provinces_new.get(str(province_code)),
                source="new",
                prefer_name=True,
            )
            bucket = new_format_bucket(province_entry)
            ward_name = _preferred_name(info, code)
            new_ward_entry: LegacyWardRecord = {
                "id": self._normalize_code_str(info.get("id") or code),
                "code": info.get("code") or code,
                "parent_code": info.get("parent_code"),
                "is_new_format": True,
                "full_name": info.get("full_name"),
                "administrative_unit_id": info.get("administrative_unit_id"),
                "legacy_names": extract_legacy_names(info),
            }
            if not new_ward_entry["legacy_names"]:
                del new_ward_entry["legacy_names"]
            existing_ward = bucket["wards"].get(ward_name)
            if existing_ward is not None:
                bucket["wards"][ward_name] = merge_ward_entry(existing_ward, new_ward_entry)
            else:
                bucket["wards"][ward_name] = new_ward_entry

        return legacy_view

    @staticmethod
    def normalize_address_code(
        value: AddressCodeInput | None,
    ) -> AddressCode | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise TypeError("address code must be a string or integer")
        if isinstance(value, int):
            return str(value)
        normalized = value.strip()
        return normalized or None

    def get_administrative_record(
        self,
        address_code: AddressCodeInput | None,
        *,
        level: AdministrativeLevel,
        source: RegistrySource,
    ) -> AdministrativeRecord | None:
        """Return a detached registry record for an explicit level and source."""
        if not isinstance(level, str):
            raise TypeError("level must be a string")
        if level not in {"province", "district", "ward"}:
            raise ValueError("level must be 'province', 'district', or 'ward'")
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        if source not in {"old", "new"}:
            raise ValueError("source must be 'old' or 'new'")

        normalized_code = self.normalize_address_code(address_code)
        if normalized_code is None:
            return None

        stores: tuple[AdministrativeRecordsByCode, ...]
        if source == "old":
            old_stores: dict[AdministrativeLevel, AdministrativeRecordsByCode] = {
                "province": self.old_province_records,
                "district": self.old_district_records,
                "ward": self.old_ward_records,
            }
            stores = (old_stores[level],)
        elif level == "province":
            stores = (self.new_province_records, self.external_new_province_records)
        elif level == "ward":
            stores = (self.new_ward_records, self.external_new_ward_records)
        else:
            return None

        for store in stores:
            record = store.get(normalized_code)
            if record is not None:
                return deepcopy(record)
        return None

    def map_old_ward_to_new(
        self,
        ward_identifier: AddressCodeInput | None,
    ) -> list[WardMappingRow]:
        """
        Return mapping rows (from Excel) for an old-format ward ID or name.
        """
        key = self.normalize_address_code(ward_identifier)
        if key is None:
            return []

        result = self.ward_mapping_by_old_code.get(key)
        if result:
            return result

        ward_std = self.standardize_name(key, "basic")
        if not ward_std:
            return []

        entries = self.ward_lookup_by_name.get(ward_std, [])
        for entry in entries:
            ward_id = entry.get("id")
            if ward_id is None:
                continue
            mapped = self.ward_mapping_by_old_code.get(str(ward_id))
            if mapped:
                return mapped
        return []

    def map_new_ward_to_old(
        self,
        ward_identifier: AddressCodeInput | None,
    ) -> list[WardMappingRow]:
        """
        Return mapping rows for a new-format ward ID; supports strings or ints.
        """
        key = self.normalize_address_code(ward_identifier)
        if key is None:
            return []
        return self.ward_mapping_by_new_code.get(key, [])

    def map_old_address_ids_to_new(
        self,
        *,
        province_id: AddressCodeInput | None = None,
        district_id: AddressCodeInput | None = None,
        ward_id: AddressCodeInput | None = None,
    ) -> NewAddressMappingResult | None:
        province_key = self.normalize_address_code(province_id)
        district_key = self.normalize_address_code(district_id)
        ward_key = self.normalize_address_code(ward_id)
        if not ward_key:
            return None
        rows = self.map_old_ward_to_new(ward_key)
        if not rows:
            return None

        def _match(row: WardMappingRow) -> bool:
            if province_key and row.get("city_id_old") != province_key:
                return False
            if district_key and row.get("district_id_old") != district_key:
                return False
            return True

        def _rank(row: WardMappingRow) -> tuple[int, int]:
            return (
                1 if province_key and row.get("city_id_old") == province_key else 0,
                1 if district_key and row.get("district_id_old") == district_key else 0,
            )

        ranked_rows = sorted(
            rows,
            key=_rank,
            reverse=True,
        )
        for row in ranked_rows:
            if _match(row):
                return self._build_new_mapping_response(row)
        # No strict match, fall back to best-ranked entry
        return self._build_new_mapping_response(ranked_rows[0])

    def map_new_address_ids_to_old(
        self,
        *,
        province_id: AddressCodeInput | None = None,
        ward_id: AddressCodeInput | None = None,
    ) -> OldAddressMappingResult | None:
        province_key = self.normalize_address_code(province_id)
        ward_key = self.normalize_address_code(ward_id)
        if not ward_key:
            return None
        rows = self.map_new_ward_to_old(ward_key)
        if not rows:
            return None

        def _rank(row: WardMappingRow) -> tuple[int, int]:
            city_match = int(bool(province_key and row.get("city_id_new") == province_key))
            has_old_district = 1 if row.get("district_id_old") else 0
            return (city_match, has_old_district)

        ranked_rows = sorted(rows, key=_rank, reverse=True)
        for row in ranked_rows:
            if province_key and row.get("city_id_new") != province_key:
                continue
            return self._build_old_mapping_response(row)
        return self._build_old_mapping_response(ranked_rows[0])

    def _lookup_new_province_name(
        self,
        province_id: AddressCodeInput | None,
    ) -> str | None:
        key = self.normalize_address_code(province_id)
        if not key:
            return None
        entry = self.new_province_records.get(key)
        if not isinstance(entry, dict):
            return None
        return entry.get("full_name") or entry.get("name") or entry.get("slug")

    def _lookup_new_province_id_by_name(
        self,
        province_name: str | None,
    ) -> AddressCode | None:
        if not province_name:
            return None
        target_std = self.standardize_name(province_name, "basic")
        if not target_std:
            return None
        target_stripped = self._strip_generic_prefix(target_std)
        targets = {target_std}
        if target_stripped:
            targets.add(target_stripped)
        for code, entry in self.new_province_records.items():
            if not isinstance(entry, dict):
                continue
            for key in (
                "full_name",
                "name",
                "full_name_en",
                "name_en",
                "slug",
                "code_name",
            ):
                value = cast(str | None, entry.get(key))
                value_std = self.standardize_name(value, "basic") if value else None
                if not value_std:
                    continue
                value_stripped = self._strip_generic_prefix(value_std)
                if value_std in targets or (value_stripped and value_stripped in targets):
                    return str(code)
        return None

    def _lookup_new_ward_name(
        self,
        ward_id: AddressCodeInput | None,
    ) -> str | None:
        key = self.normalize_address_code(ward_id)
        if not key:
            return None
        entry = self.new_ward_records.get(key)
        if not isinstance(entry, dict):
            return None
        return entry.get("full_name") or entry.get("name") or entry.get("slug")

    def _build_new_mapping_response(
        self,
        row: WardMappingRow,
    ) -> NewAddressMappingResult:
        province_id_new = row.get("city_id_new")
        ward_id_new = row["ward_id_new"]
        return {
            "province_id_new": province_id_new,
            "province_name_new": self._lookup_new_province_name(province_id_new),
            "ward_id_new": ward_id_new,
            "ward_name_new": self._lookup_new_ward_name(ward_id_new),
            "raw": row,
        }

    def _build_old_mapping_response(
        self,
        row: WardMappingRow,
    ) -> OldAddressMappingResult:
        return {
            "province_id_old": row.get("city_id_old"),
            "province_name_old": self._lookup_old_province_name(row.get("city_id_old")),
            "district_id_old": row.get("district_id_old"),
            "district_name_old": self._lookup_old_district_name(row.get("district_id_old")),
            "ward_id_old": row["ward_id_old"],
            "ward_name_old": self._lookup_old_ward_name(row["ward_id_old"]),
            "raw": row,
        }

    def _lookup_old_province_name(
        self,
        province_id: AddressCodeInput | None,
    ) -> str | None:
        key = self.normalize_address_code(province_id)
        if not key:
            return None
        entry = self.old_province_records.get(key)
        if not isinstance(entry, dict):
            return None
        return entry.get("full_name") or entry.get("name") or entry.get("slug")

    def _lookup_old_district_name(
        self,
        district_id: AddressCodeInput | None,
    ) -> str | None:
        key = self.normalize_address_code(district_id)
        if not key:
            return None
        entry = self.old_district_records.get(key)
        if not isinstance(entry, dict):
            return None
        return entry.get("full_name") or entry.get("name") or entry.get("slug")

    def _lookup_old_ward_name(
        self,
        ward_id: AddressCodeInput | None,
    ) -> str | None:
        key = self.normalize_address_code(ward_id)
        if not key:
            return None
        entry = self.old_ward_records.get(key)
        if not isinstance(entry, dict):
            return None
        return entry.get("full_name") or entry.get("name") or entry.get("slug")

    def _ensure_old_ward_name_index(self) -> None:
        if self._old_ward_name_index is not None:
            return
        index: dict[str, list[AddressCode]] = {}
        for code, entry in self.old_ward_records.items():
            if not isinstance(entry, dict):
                continue
            code_str = str(code).strip()
            if not code_str:
                continue
            for key in ("full_name", "name"):
                value = entry.get(key)
                if not isinstance(value, str) or not value.strip():
                    continue
                for value_std in self._standardized_name_variants(value):
                    if not value_std:
                        continue
                    bucket = index.setdefault(value_std, [])
                    if code_str not in bucket:
                        bucket.append(code_str)
        self._old_ward_name_index = index

    def _ensure_old_ward_raw_name_index(self) -> None:
        if self._old_ward_raw_name_index is not None:
            return

        def _key(value: str) -> str:
            return re.sub(r"\s+", " ", (value or "").strip().lower())

        index: dict[str, list[AddressCode]] = {}
        for code, entry in self.old_ward_records.items():
            if not isinstance(entry, dict):
                continue
            code_str = str(code).strip()
            if not code_str:
                continue
            for field in ("full_name", "name"):
                value = entry.get(field)
                if not isinstance(value, str) or not value.strip():
                    continue
                k = _key(value)
                if not k:
                    continue
                bucket = index.setdefault(k, [])
                if code_str not in bucket:
                    bucket.append(code_str)
        self._old_ward_raw_name_index = index

    def _enrich_old_ward_with_province(
        self,
        entry: AdministrativeRecord,
    ) -> AdministrativeRecord:
        """Add province_key and province_name to an old-ward record so alignment checks pass."""
        if not isinstance(entry, dict):
            return entry
        if entry.get("province_key") or entry.get("province_name"):
            return entry
        parent_code = self.normalize_address_code(
            entry.get("parent_code") or entry.get("district_code")
        )
        if not parent_code:
            return entry
        district_entry = self.old_district_records.get(parent_code)
        if not isinstance(district_entry, dict):
            return entry
        province_code = self.normalize_address_code(
            district_entry.get("parent_code") or district_entry.get("province_code")
        )
        if not province_code:
            return entry
        province_entry = self.old_province_records.get(province_code)
        if not isinstance(province_entry, dict):
            return entry
        province_name = province_entry.get("name") or province_entry.get("full_name")
        if not province_name:
            return entry
        result = entry.copy()
        result["province_name"] = province_name
        result["province_key"] = self.standardize_name(province_name, "basic")
        return result

    def _lookup_old_ward_record_by_exact_name(
        self,
        ward_name: str | None,
    ) -> AdministrativeRecord | None:
        if not ward_name:
            return None
        name_key = re.sub(r"\s+", " ", ward_name.strip().lower())
        if not name_key:
            return None
        self._ensure_old_ward_raw_name_index()
        index = self._old_ward_raw_name_index or {}
        codes = index.get(name_key) or []
        if len(codes) != 1:
            return None
        entry = self.old_ward_records.get(codes[0])
        return entry if isinstance(entry, dict) else None

    def _lookup_old_ward_record_by_name(
        self,
        ward_name: str | None,
    ) -> AdministrativeRecord | None:
        if not ward_name:
            return None
        ward_keys = self._standardized_name_variants(ward_name)
        if not ward_keys:
            return None
        self._ensure_old_ward_name_index()
        index = self._old_ward_name_index or {}
        codes: list[AddressCode] = []
        seen: set[AddressCode] = set()
        for ward_key in ward_keys:
            for code in index.get(ward_key) or []:
                if code in seen:
                    continue
                seen.add(code)
                codes.append(code)
        if len(codes) != 1:
            return None
        entry = self.old_ward_records.get(codes[0])
        return entry if isinstance(entry, dict) else None

    def _raw_name_key(self, value: str | None) -> str:
        if not isinstance(value, str):
            return ""
        return re.sub(r"\s+", " ", value.strip().lower())

    def _name_matches_query(
        self,
        candidate: str | None,
        query: str | None,
    ) -> bool:
        candidate_raw = self._raw_name_key(candidate)
        query_raw = self._raw_name_key(query)
        if candidate_raw and query_raw and candidate_raw == query_raw:
            return True
        candidate_variants = self._standardized_name_variants(candidate) if candidate else set()
        query_variants = self._standardized_name_variants(query) if query else set()
        return bool(
            candidate_variants
            and query_variants
            and candidate_variants.intersection(query_variants)
        )

    def _entry_matches_query_name(
        self,
        entry: AdministrativeRecord | None,
        query: str | None,
        *,
        include_aliases: bool = False,
    ) -> bool:
        if not isinstance(entry, dict) or not query:
            return False
        values = [entry.get("name"), entry.get("full_name")]
        if include_aliases:
            values.extend(self._entry_alias_values(entry, level="ward"))
        for value in values:
            if self._name_matches_query(value, query):
                return True
        return False

    def _entry_matches_raw_query_name(
        self,
        entry: AdministrativeRecord | None,
        query: str | None,
    ) -> bool:
        if not isinstance(entry, dict) or not query:
            return False
        query_raw = self._raw_name_key(query)
        if not query_raw:
            return False
        for value in (entry.get("name"), entry.get("full_name")):
            if self._raw_name_key(value) == query_raw:
                return True
        return False

    def _entry_alias_values(
        self,
        entry: AdministrativeRecord | None,
        *,
        level: AdministrativeLevel,
    ) -> list[str]:
        if not isinstance(entry, dict):
            return []

        aliases: list[str] = []
        seen: set[str] = set()

        def _add(value: str | None) -> None:
            if not isinstance(value, str):
                return
            candidate = value.strip()
            if not candidate or candidate in seen:
                return
            aliases.append(candidate)
            seen.add(candidate)

        legacy_names = entry.get("legacy_names")
        if isinstance(legacy_names, str):
            _add(legacy_names)
        elif isinstance(legacy_names, list):
            for alias in legacy_names:
                _add(alias)

        if level == "ward":
            code = entry.get("id") or entry.get("code")
            if code is not None:
                for alias in CUSTOM_WARD_ALIASES_BY_CODE.get(str(code), []):
                    _add(alias)

        return aliases

    def _project_component(
        self,
        entry: AdministrativeRecord | None,
        component_id: AddressCodeInput | None,
    ) -> ProjectedAddressComponent | None:
        if not entry:
            return None
        normalized_id = self.normalize_address_code(component_id)
        projected: ProjectedAddressComponent = {
            "id": normalized_id or entry.get("code"),
            "code": entry.get("code"),
            "name": entry.get("name"),
            "full_name": entry.get("full_name"),
            "slug": entry.get("slug"),
            "type": entry.get("type"),
        }
        if "path" in entry:
            projected["path"] = entry.get("path")
        if "path_with_type" in entry:
            projected["path_with_type"] = entry.get("path_with_type")
        if "parent_code" in entry:
            projected["parent_code"] = entry.get("parent_code")
        return projected

    def _format_full_address(
        self,
        province: AdministrativeRecord | None,
        district: AdministrativeRecord | None,
        ward: AdministrativeRecord | None,
    ) -> str | None:
        if ward and ward.get("path_with_type"):
            return ward.get("path_with_type")
        if district and district.get("path_with_type"):
            return district.get("path_with_type")
        pieces: list[str] = []
        for entry in (ward, district, province):
            if not entry:
                continue
            name = entry.get("full_name") or entry.get("name") or entry.get("slug")
            if name:
                pieces.append(name)
        if not pieces:
            return None
        return ", ".join(pieces)

    def _lookup_old_components(
        self,
        *,
        province_id: AddressCode | None,
        district_id: AddressCode | None,
        ward_id: AddressCode | None,
    ) -> tuple[
        AdministrativeRecord | None,
        AdministrativeRecord | None,
        AdministrativeRecord | None,
    ]:
        province = self.old_province_records.get(province_id) if province_id else None
        district = self.old_district_records.get(district_id) if district_id else None
        ward = self.old_ward_records.get(ward_id) if ward_id else None
        return province, district, ward

    def _lookup_new_components(
        self,
        *,
        province_id: AddressCode | None,
        ward_id: AddressCode | None,
    ) -> tuple[AdministrativeRecord | None, AdministrativeRecord | None]:
        province = self.new_province_records.get(province_id) if province_id else None
        ward = self.new_ward_records.get(ward_id) if ward_id else None
        return province, ward

    def get_address_components_from_ids(
        self,
        *,
        province_id: AddressCodeInput | None,
        district_id: AddressCodeInput | None = None,
        ward_id: AddressCodeInput | None = None,
        is_new_format: bool = False,
    ) -> AddressComponentsResult:
        if not isinstance(is_new_format, bool):
            raise TypeError("is_new_format must be a boolean")
        normalized_province_id = self.normalize_address_code(province_id)
        normalized_district_id = self.normalize_address_code(district_id)
        normalized_ward_id = self.normalize_address_code(ward_id)
        if is_new_format:
            province_entry, ward_entry = self._lookup_new_components(
                province_id=normalized_province_id,
                ward_id=normalized_ward_id,
            )
            district_entry = None
        else:
            province_entry, district_entry, ward_entry = self._lookup_old_components(
                province_id=normalized_province_id,
                district_id=normalized_district_id,
                ward_id=normalized_ward_id,
            )

        full_address = self._format_full_address(
            province_entry,
            district_entry,
            ward_entry,
        )

        return {
            "province": self._project_component(
                province_entry,
                normalized_province_id,
            ),
            "district": self._project_component(
                district_entry,
                normalized_district_id,
            ),
            "ward": self._project_component(ward_entry, normalized_ward_id),
            "full_address": full_address,
        }

    def map_address_ids(
        self,
        *,
        province_id: AddressCodeInput | None,
        district_id: AddressCodeInput | None,
        ward_id: AddressCodeInput | None,
        is_new_format: bool | None,
    ) -> AddressMappingSummary | None:
        if is_new_format is not None and not isinstance(is_new_format, bool):
            raise TypeError("is_new_format must be a boolean or None")
        normalized_province_id = self.normalize_address_code(province_id)
        normalized_district_id = self.normalize_address_code(district_id)
        normalized_ward_id = self.normalize_address_code(ward_id)
        if normalized_ward_id is None:
            return None

        def _summarize_old_to_new(
            mapping_payload: NewAddressMappingResult,
        ) -> OldToNewAddressMappingSummary:
            target_components = self.get_address_components_from_ids(
                province_id=mapping_payload.get("province_id_new"),
                ward_id=mapping_payload.get("ward_id_new"),
                is_new_format=True,
            )
            source_components = self.get_address_components_from_ids(
                province_id=normalized_province_id,
                district_id=normalized_district_id,
                ward_id=normalized_ward_id,
                is_new_format=False,
            )
            return {
                "direction": "old_to_new",
                "source_format_is_new": False,
                "source": source_components,
                "target": target_components,
                "mapping": mapping_payload,
            }

        def _summarize_new_to_old(
            mapping_payload: OldAddressMappingResult,
        ) -> NewToOldAddressMappingSummary:
            target_components = self.get_address_components_from_ids(
                province_id=mapping_payload.get("province_id_old"),
                district_id=mapping_payload.get("district_id_old"),
                ward_id=mapping_payload.get("ward_id_old"),
                is_new_format=False,
            )
            source_components = self.get_address_components_from_ids(
                province_id=normalized_province_id,
                district_id=normalized_district_id,
                ward_id=normalized_ward_id,
                is_new_format=True,
            )
            return {
                "direction": "new_to_old",
                "source_format_is_new": True,
                "source": source_components,
                "target": target_components,
                "mapping": mapping_payload,
            }

        if is_new_format is True:
            old_mapping_payload = self.map_new_address_ids_to_old(
                province_id=normalized_province_id,
                ward_id=normalized_ward_id,
            )
            if not old_mapping_payload:
                return None
            return _summarize_new_to_old(old_mapping_payload)

        new_mapping_payload = self.map_old_address_ids_to_new(
            province_id=normalized_province_id,
            district_id=normalized_district_id,
            ward_id=normalized_ward_id,
        )
        if new_mapping_payload:
            return _summarize_old_to_new(new_mapping_payload)

        if is_new_format is False:
            return None

        old_mapping_payload = self.map_new_address_ids_to_old(
            province_id=normalized_province_id,
            ward_id=normalized_ward_id,
        )
        if not old_mapping_payload:
            return None
        return _summarize_new_to_old(old_mapping_payload)

    @staticmethod
    def _selected_registry_sources(
        *,
        include_new: bool,
        include_old: bool,
    ) -> list[RegistrySource]:
        if not isinstance(include_new, bool) or not isinstance(include_old, bool):
            raise TypeError("include_new and include_old must be booleans")
        sources: list[RegistrySource] = []
        if include_new:
            sources.append("new")
        if include_old:
            sources.append("old")
        return sources

    def search_province(
        self,
        query: str | None,
        *,
        include_new: bool = True,
        include_old: bool = True,
        limit: int = 10,
    ) -> list[SearchResult]:
        allowed_sources = self._selected_registry_sources(
            include_new=include_new,
            include_old=include_old,
        )
        validate_search_request(
            query,
            level="province",
            allowed_sources=allowed_sources,
            limit=limit,
        )
        engine = self.search_engine
        if not engine or not allowed_sources:
            return []
        return engine.search(
            query,
            level="province",
            allowed_sources=allowed_sources,
            limit=limit,
        )

    def search_district(
        self,
        query: str | None,
        *,
        province_code: AddressCodeInput | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        validate_search_request(
            query,
            level="district",
            allowed_sources=("old",),
            limit=limit,
        )
        normalized_province_code = self.normalize_address_code(province_code)
        engine = self.search_engine
        if not engine or limit <= 0:
            return []
        return engine.search(
            query,
            level="district",
            allowed_sources=["old"],
            province_code=normalized_province_code,
            limit=limit,
        )

    def search_ward(
        self,
        query: str | None,
        *,
        province_code: AddressCodeInput | None = None,
        district_code: AddressCodeInput | None = None,
        include_new: bool = True,
        include_old: bool = True,
        limit: int = 100,
    ) -> list[SearchResult]:
        normalized_province_code = self.normalize_address_code(province_code)
        normalized_district_code = self.normalize_address_code(district_code)
        allowed_sources = self._selected_registry_sources(
            include_new=include_new,
            include_old=include_old,
        )
        validate_search_request(
            query,
            level="ward",
            allowed_sources=allowed_sources,
            limit=limit,
        )
        engine = self.search_engine
        if not engine or not allowed_sources:
            return []
        results = engine.search(
            query,
            level="ward",
            allowed_sources=allowed_sources,
            province_code=normalized_province_code,
            district_code=normalized_district_code,
            limit=limit,
        )
        return self._filter_results_by_unit(query, results, level="ward")

    def _detect_unit_token_from_query(self, query: str | None) -> str | None:
        if not query:
            return None
        raw_query = unicodedata.normalize("NFC", query).strip()
        if not raw_query:
            return None
        token_matches = list(re.finditer(self._RE_WORD_TOKEN, raw_query, flags=re.UNICODE))
        if not token_matches:
            return None

        raw_tokens = [match.group(0).strip("._").casefold() for match in token_matches[:2]]
        head = raw_tokens[0] if raw_tokens else ""
        tail = raw_tokens[1] if len(raw_tokens) > 1 else ""

        if head in ("phường", "phuong", "p", "w"):
            return "phuong"
        if head in ("xã", "xa", "x"):
            return "xa"
        if (
            head == "tt"
            or (head in {"thị", "thi"} and tail == "trấn")
            or (head == "thi" and tail == "tran")
        ):
            return self._LIT_THI_TRAN
        if (head in {"thị", "thi"} and tail == "xã") or (head == "thi" and tail == "xa"):
            return self._LIT_THI_XA
        if (head == "đặc" and tail == "khu") or (head == "dac" and tail == "khu"):
            return self._LIT_DAC_KHU
        return None

    def _unit_tokens_match(
        self,
        required: str | None,
        candidate: str | None,
    ) -> bool:
        if not required:
            return True
        required_norm = self._normalize_unit_token(required)
        candidate_norm = self._normalize_unit_token(candidate)
        if not candidate_norm:
            return False
        return required_norm == candidate_norm

    def _extract_unit_token(
        self,
        record: AdministrativeRecord | None,
        *,
        level: AdministrativeLevel,
    ) -> str | None:
        if level != "ward" or not isinstance(record, dict):
            return None
        token = self._unit_token_from_admin_id(record.get("administrative_unit_id"))
        if token:
            return token
        text = record.get("full_name") or record.get("name")
        return self._unit_token_from_text(text)

    @staticmethod
    def _unit_token_from_admin_id(unit_id: int | None) -> str | None:
        try:
            value = int(unit_id) if unit_id is not None else None
        except (TypeError, ValueError):
            return None
        mapping = {
            3: "phuong",
            4: "xa",
            5: AddressParser._LIT_THI_TRAN,
        }
        if value is None:
            return None
        return mapping.get(value)

    def _unit_token_from_text(self, name: str | None) -> str | None:
        if not name:
            return None
        normalized = self.standardize_name(name, "basic")
        if not normalized:
            return None
        tokens = normalized.split()
        if not tokens:
            return None
        if tokens[0] in ("phuong", "p", "w"):
            return "phuong"
        if tokens[0] in ("xa", "x"):
            return "xa"
        if len(tokens) >= 2 and tokens[0] == "thi" and tokens[1] == "tran":
            return self._LIT_THI_TRAN
        if len(tokens) >= 2 and tokens[0] == "thi" and tokens[1] == "xa":
            return self._LIT_THI_XA
        if tokens[0] == "tt":
            return self._LIT_THI_TRAN
        if len(tokens) >= 2 and tokens[0] == "dac" and tokens[1] == "khu":
            return self._LIT_DAC_KHU
        return None

    def _normalize_unit_token(self, token: str | None) -> str | None:
        if not token:
            return None
        normalized = self.standardize_name(str(token), "basic")
        if not normalized:
            return None
        tokens = normalized.split()
        if not tokens:
            return None
        head = tokens[0]
        tail = tokens[1] if len(tokens) > 1 else ""
        if head in ("p", "w"):
            return "phuong"
        if head == "phuong":
            return "phuong"
        if head in ("x", "xa"):
            return "xa"
        if head == "tt" or (head == "thi" and tail == "tran"):
            return self._LIT_THI_TRAN
        if head == "thi" and tail == "xa":
            return self._LIT_THI_XA
        if head == "dac" and tail == "khu":
            return self._LIT_DAC_KHU
        return head or None

    def _filter_results_by_unit(
        self,
        query: str | None,
        results: list[SearchResult],
        *,
        level: AdministrativeLevel,
    ) -> list[SearchResult]:
        if level != "ward" or not results:
            return results

        unit_token = self._detect_unit_token_from_query(query)
        if not unit_token:
            return results

        filtered: list[SearchResult] = []
        for result in results:
            if self._unit_tokens_match(unit_token, result.get("unit_token")):
                filtered.append(result)

        if filtered:
            return filtered
        return results

    @staticmethod
    def _tokens_in_order(
        needles: list[str],
        haystack: list[str],
    ) -> bool:
        if not needles:
            return True
        doc_pos = 0
        for token in needles:
            while doc_pos < len(haystack) and haystack[doc_pos] != token:
                doc_pos += 1
            if doc_pos == len(haystack):
                return False
            doc_pos += 1
        return True

    def _tokenize_with_diacritics(self, text: str | None) -> list[str]:
        if not text:
            return []
        normalized = unicodedata.normalize("NFC", str(text)).casefold()
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in normalized)
        return [token for token in cleaned.split() if token]

    def _contains_diacritics(self, text: str | None) -> bool:
        if not text:
            return False
        normalized = unicodedata.normalize("NFD", str(text))
        return any(unicodedata.category(ch) == "Mn" for ch in normalized)

    def _collect_search_text_fields(
        self,
        entry: AdministrativeRecord,
        *,
        level: AdministrativeLevel,
    ) -> list[str]:
        fields: list[str] = []
        seen: set[str] = set()

        def _add(value: str | None) -> None:
            if not isinstance(value, str):
                return
            trimmed = value.strip()
            if not trimmed or trimmed in seen:
                return
            seen.add(trimmed)
            fields.append(trimmed)

        def _add_with_variants(value: str | None) -> None:
            if not isinstance(value, str):
                return
            _add(value)
            for variant in self._standardized_name_variants(value):
                _add(variant)

        # Use one primary label (prefer full_name, fall back to name) to avoid duplicate tokens
        primary = entry.get("full_name") or entry.get("name")
        _add_with_variants(primary)

        for key in ("name", "full_name", "name_en", "full_name_en", "slug", "code_name"):
            _add_with_variants(cast(str | None, entry.get(key)))

        if level in ("province", "district"):
            legacy_names = entry.get("legacy_names")
            if isinstance(legacy_names, str):
                _add_with_variants(legacy_names)
            elif isinstance(legacy_names, (list, tuple)):
                for legacy in legacy_names:
                    _add_with_variants(legacy)

        if level == "province":
            canonical_name = entry.get("full_name") or entry.get("name")
            aliases = self._get_special_province_aliases(canonical_name)
            for alias in aliases:
                _add_with_variants(alias)

        return fields

    def _analyze_search_text(self, text: str | None) -> list[str]:
        if not text:
            return []
        normalized = self.standardize_name(text, "basic")
        tokens: list[str] = []

        def _canonicalize_token(token: str) -> str:
            if not token:
                return ""
            normalized_token = unicodedata.normalize("NFD", token.casefold())
            normalized_token = "".join(
                ch for ch in normalized_token if unicodedata.category(ch) != "Mn"
            )
            normalized_token = normalized_token.replace("đ", "d")
            normalized_token = re.sub(r"[^a-z0-9]+", "", normalized_token)
            if not normalized_token:
                return ""
            if (
                len(normalized_token) <= 3
                and "y" in normalized_token
                and not any(vowel in normalized_token for vowel in ("a", "e", "o"))
            ):
                normalized_token = normalized_token.replace("y", "i")
            return normalized_token

        if normalized:
            for token in normalized.split():
                canonical = _canonicalize_token(token)
                if canonical:
                    tokens.append(canonical)

        # Add diacritic-preserving tokens so accented queries/documents can match directly
        accented_tokens = self._tokenize_with_diacritics(text)
        for tok in accented_tokens:
            canonical = _canonicalize_token(tok)
            if canonical and canonical not in tokens:
                tokens.append(canonical)

        unit_token = self._detect_unit_token_from_query(text)
        if unit_token and unit_token not in tokens:
            tokens.append(unit_token)
        return tokens

    def _rebuild_search_engine(self) -> None:
        self.search_engine = AddressSearchEngine(
            analyzer=self._analyze_search_text,
            normalize_id=self.normalize_address_code,
        )
        engine = self.search_engine

        def _register(
            record: AdministrativeRecord | None,
            *,
            level: AdministrativeLevel,
            source: RegistrySource,
            province_code: AddressCode | None = None,
            district_code: AddressCode | None = None,
        ) -> None:
            if not isinstance(record, dict):
                return
            unit_token = self._extract_unit_token(record, level=level)
            engine.add_document(
                text_fields=self._collect_search_text_fields(record, level=level),
                metadata={
                    "level": level,
                    "source": source,
                    "record": record,
                    "province_code": province_code,
                    "district_code": district_code,
                    "unit_token": unit_token,
                },
            )

        for code, record in self.new_province_records.items():
            _register(
                record,
                level="province",
                source="new",
                province_code=self.normalize_address_code(code),
            )
        for code, record in self.old_province_records.items():
            _register(
                record,
                level="province",
                source="old",
                province_code=self.normalize_address_code(code),
            )
        for code, record in self.old_district_records.items():
            _register(
                record,
                level="district",
                source="old",
                province_code=self.normalize_address_code(record.get("parent_code")),
                district_code=self.normalize_address_code(code),
            )
        for record in self.new_ward_records.values():
            _register(
                record,
                level="ward",
                source="new",
                province_code=self.normalize_address_code(record.get("parent_code")),
            )
        for record in self.old_ward_records.values():
            district_code = self.normalize_address_code(record.get("parent_code"))
            province_code = None
            if district_code:
                district_entry = self._lookup_old_district_record(district_code)
                if district_entry:
                    province_code = self.normalize_address_code(district_entry.get("parent_code"))
            _register(
                record,
                level="ward",
                source="old",
                province_code=province_code,
                district_code=district_code,
            )

        engine.finalize()

    def _lookup_old_district_record(
        self, district_code: AddressCodeInput | None
    ) -> AdministrativeRecord | None:
        if not district_code:
            return None
        key = str(district_code).strip()
        entry = self.old_district_records.get(key)
        if entry:
            return entry
        normalized = self.normalize_address_code(district_code)
        if normalized and normalized != key:
            return self.old_district_records.get(normalized)
        return None

    def _load_reference_names(self, path: str) -> tuple[dict[str, list[str]], list[str]]:
        reference_map: dict[str, list[str]] = {}
        if not os.path.exists(path):
            return {}, []
        with open(path, encoding="utf-8") as f:
            for line in f:
                raw_name = line.strip()
                if not raw_name:
                    continue
                standardized_name = self.standardize_name(raw_name)
                if standardized_name:
                    reference_map.setdefault(standardized_name, []).append(raw_name)
        return reference_map, list(reference_map.keys())

    def _select_reference_candidate(
        self, candidates: list[str], raw_value: str | None
    ) -> tuple[str | None, float]:
        if not candidates:
            return None, 0.0
        if not raw_value:
            return candidates[0], 100.0

        normalized_raw = raw_value.casefold()
        best_candidate: str | None = None
        best_score = -1.0
        for candidate in candidates:
            score = ratio(normalized_raw, candidate.casefold())
            if score > best_score:
                best_candidate = candidate
                best_score = score
        return best_candidate, best_score

    def _match_reference(
        self,
        standardized_value: str,
        reference_map: dict[str, list[str]],
        reference_choices: list[str],
        score_cutoff: int,
        raw_value: str | None = None,
    ) -> tuple[str | None, bool]:
        if not standardized_value or not reference_choices:
            return (raw_value if raw_value else None, False)

        direct_candidates = reference_map.get(standardized_value)
        if direct_candidates:
            candidate, score = self._select_reference_candidate(direct_candidates, raw_value)
            if candidate is None:
                return (raw_value if raw_value else None, False)
            if raw_value is not None and score < self.REFERENCE_ACCEPT_RATIO:
                return raw_value, False
            return candidate, True

        result = rf_process.extractOne(
            standardized_value,
            reference_choices,
            scorer=ratio,
            score_cutoff=score_cutoff,
        )
        if result is None:
            return (raw_value if raw_value else None, False)

        match_key, _, _ = result
        candidates = reference_map.get(match_key, [])
        candidate, score = self._select_reference_candidate(candidates, raw_value)
        if candidate is None:
            return (raw_value if raw_value else None, False)
        if raw_value is not None and score < self.REFERENCE_ACCEPT_RATIO:
            return raw_value, False
        return candidate, True

    def _collect_aliases(
        self,
        primary: str | None,
        raw_value: str | None,
        extra_aliases: list[str] | None = None,
    ) -> list[str]:
        aliases: list[str] = []
        seen: set[str] = set()

        def _add(value: str | None) -> None:
            if not value:
                return
            if value in seen:
                return
            aliases.append(value)
            seen.add(value)

        _add(primary)
        _add(raw_value)
        if extra_aliases:
            for alias in extra_aliases:
                _add(alias)
        return aliases or [""]

    def _get_special_province_aliases(self, province_name: str | None) -> list[str]:
        """
        Provide legacy aliases (e.g. Thừa Thiên - Huế) for provinces that have
        been renamed in the canonical dataset so that lookup/detection still works.
        """
        if not province_name:
            return []

        def _canonicalize(value: str) -> str:
            stripped = value.strip()
            if not stripped:
                return ""
            normalized = self.standardize_name(stripped, "basic")
            if not normalized:
                return ""
            # Drop administrative prefixes so alias comparisons only rely on the core name
            normalized = re.sub(self._RE_PROVINCE_PREFIX, "", normalized).strip()
            return normalized

        province_std = _canonicalize(province_name)
        if not province_std:
            return []

        aliases: list[str] = []

        for synonyms, canonical in SPECIAL_PROVINCE_MAP.items():
            canonical_std = _canonicalize(canonical)
            if not canonical_std or canonical_std != province_std:
                continue

            if isinstance(synonyms, (list, tuple, set)):
                for alias in synonyms:
                    if alias:
                        aliases.append(alias)
            else:
                if synonyms:
                    aliases.append(synonyms)

        return aliases

    def _standardize_aliases(self, aliases: list[str]) -> set[str]:
        normalized: set[str] = set()
        for alias in aliases:
            normalized.update(self._standardized_name_variants(alias))
        return normalized

    def _titleize_token(self, token: str | None) -> str:
        if not token:
            return ""
        parts = [part.capitalize() for part in token.split() if part]
        return " ".join(parts) or token

    def _normalize_numeric_component_key(
        self,
        value: str | None,
        *,
        default_prefix: str | None = None,
    ) -> str | None:
        if not value:
            return None
        text = value.strip()
        if not text:
            return None
        match = re.match(r"^(phuong|p|xa|x|thi tran|tt)\s*(\d+)$", text)
        if match:
            prefix = cast(str, match.group(1))
            digits = cast(str, match.group(2)).lstrip("0") or "0"
            if prefix in {"p", "phuong"}:
                prefix = "phuong"
            elif prefix in {"x", "xa"}:
                prefix = "xa"
            return f"{prefix} {digits}"
        if text.isdigit() and default_prefix:
            digits = text.lstrip("0") or "0"
            return f"{default_prefix} {digits}"
        return None

    def _normalize_detected_ward_token(self, token: str | None) -> str:
        std = self.standardize_name(token, "basic") if token else ""
        if not std:
            return ""
        normalized_numeric = self._normalize_numeric_component_key(
            std,
            default_prefix="phuong",
        )
        if normalized_numeric:
            return normalized_numeric
        return std

    def _numeric_token_match(self, entry_name_std: str, detected_std: str) -> bool:
        if not entry_name_std or not detected_std:
            return False
        entry_digits = "".join(ch for ch in entry_name_std if ch.isdigit())
        detected_digits = "".join(ch for ch in detected_std if ch.isdigit())
        if not detected_digits:
            return False
        return entry_digits.lstrip("0") == detected_digits.lstrip("0")

    def _entry_matches_component_fragment(
        self,
        entry: AdministrativeRecord | None,
        fragment: str | None,
        *,
        level: AdministrativeLevel = "ward",
    ) -> bool:
        if not isinstance(entry, dict) or not fragment:
            return False
        fragment_std = self.standardize_name(fragment, "basic")
        if not fragment_std:
            return False

        fragment_keys = {fragment_std}
        stripped_fragment = self._strip_generic_prefix(fragment_std)
        if stripped_fragment:
            fragment_keys.add(stripped_fragment)

        entry_keys: set[str] = set()

        def _add(value: str | None) -> None:
            value_std = self.standardize_name(value, "basic") if value else ""
            if not value_std:
                return
            entry_keys.add(value_std)
            stripped_value = self._strip_generic_prefix(value_std)
            if stripped_value:
                entry_keys.add(stripped_value)

        for value in (entry.get("name"), entry.get("full_name")):
            _add(value)

        legacy_names = entry.get("legacy_names")
        if isinstance(legacy_names, str):
            _add(legacy_names)
        elif isinstance(legacy_names, list):
            for alias in legacy_names:
                if isinstance(alias, str):
                    _add(alias)

        if level == "ward":
            code = entry.get("id") or entry.get("code")
            if code is not None:
                for alias in CUSTOM_WARD_ALIASES_BY_CODE.get(str(code), []):
                    _add(alias)

        return bool(fragment_keys & entry_keys)

    def _detect_contextual_old_ward(
        self,
        input_segments: list[tuple[str, str]],
        input_string_basic: str,
        detected_prov: str | None,
    ) -> ContextualOldWardResult | None:
        def _province_matches_segment(segment_std: str, expected_key: str) -> bool:
            if not segment_std or not expected_key:
                return False
            if self._canonicalize_province_key(segment_std) == expected_key:
                return True
            special = self._detect_special_province_token(segment_std)
            return bool(special and self._canonicalize_province_key(special) == expected_key)

        def _find_old_province() -> tuple[AdministrativeRecord | None, int | None]:
            resolved = (
                self._resolve_detected_component(
                    "province",
                    detected_prov,
                    source_string=input_string_basic,
                )
                if detected_prov
                else None
            )
            if resolved:
                info = self._lookup_province_info(resolved)
                province_id = (
                    self.normalize_address_code(info.get("id") or info.get("code"))
                    if isinstance(info, dict)
                    else None
                )
                old_info = self.old_province_records.get(province_id) if province_id else None
                if old_info:
                    return old_info, None

            for idx, (segment_std, _) in enumerate(input_segments):
                if not segment_std:
                    continue
                resolved_segment = self._validate_detected_value(
                    segment_std,
                    self.invert_province_to_indices,
                )
                if not resolved_segment:
                    special = self._detect_special_province_token(segment_std)
                    if special:
                        resolved_segment = self._validate_detected_value(
                            special,
                            self.invert_province_to_indices,
                        )
                if not resolved_segment:
                    continue
                province_name = self._resolve_detected_component(
                    "province",
                    resolved_segment,
                    source_string=input_string_basic,
                )
                info = self._lookup_province_info(province_name)
                province_id = (
                    self.normalize_address_code(info.get("id") or info.get("code"))
                    if isinstance(info, dict)
                    else None
                )
                old_info = self.old_province_records.get(province_id) if province_id else None
                if old_info:
                    return old_info, idx
            return None, None

        province_info, province_idx = _find_old_province()
        if not province_info:
            return None

        province_name = province_info.get("name") or province_info.get("full_name")
        province_key = self._canonicalize_province_key(province_name)
        province_indices: set[int] = {
            idx
            for idx, (segment_std, _) in enumerate(input_segments)
            if _province_matches_segment(segment_std, province_key)
        }
        if province_idx is not None:
            province_indices.add(province_idx)
        if province_key == "hue":
            province_indices.update(
                idx for idx, (segment_std, _) in enumerate(input_segments) if segment_std == "tt"
            )

        district_info: AdministrativeRecord | None = None
        district_idx: int | None = None
        for idx, (segment_std, _) in enumerate(input_segments):
            if idx in province_indices or not segment_std:
                continue
            lookup_candidates = [segment_std]
            stripped = self._strip_generic_prefix(segment_std)
            if stripped and stripped != segment_std:
                lookup_candidates.append(stripped)
            for candidate in lookup_candidates:
                info = self._lookup_district_info(candidate, province_name)
                if info and info.get("province_key") == province_key:
                    district_info = info
                    district_idx = idx
                    break
            if district_info:
                break

        context_district_name = (
            district_info.get("name") or district_info.get("full_name")
            if isinstance(district_info, dict)
            else None
        )

        locality_prefix_tokens: set[str] = {
            "thon",
            "xom",
            "ap",
            "to",
            "tdp",
            "khom",
            "khu",
            "kp",
            "k",
            "nhom",
            "doi",
            "cum",
            "ban",
            "buon",
        }
        street_prefix_tokens: set[str] = {
            "duong",
            "d",
            "dg",
            "ngo",
            "ngach",
            "hem",
            "hxh",
            "ql",
            "quoclo",
            "tl",
            "tinhlo",
            "dailo",
        }
        named_street_prefix_tokens: set[str] = {
            "ong",
            "ba",
            "chu",
            "co",
            "bac",
            "anh",
            "chi",
        }

        def _old_record(
            entry: AdministrativeRecord | None,
        ) -> AdministrativeRecord | None:
            if not isinstance(entry, dict):
                return None
            code = self.normalize_address_code(entry.get("id") or entry.get("code"))
            if not code or code not in self.old_ward_records:
                return None
            return entry

        def _ward_parent_district(
            entry: AdministrativeRecord,
        ) -> AdministrativeRecord | None:
            district_code = self.normalize_address_code(
                entry.get("district_id") or entry.get("district_code") or entry.get("parent_code")
            )
            if not district_code:
                old_code = self.normalize_address_code(entry.get("id") or entry.get("code"))
                old_record = self.old_ward_records.get(old_code) if old_code else None
                district_code = (
                    self.normalize_address_code(old_record.get("parent_code"))
                    if isinstance(old_record, dict)
                    else None
                )
            return self.old_district_records.get(district_code) if district_code else None

        def _candidate_matches_entry(entry: AdministrativeRecord, query: str) -> bool:
            return self._entry_matches_component_fragment(entry, query, level="ward")

        def _candidate_matches_canonical_or_custom(
            entry: AdministrativeRecord,
            query: str,
        ) -> bool:
            for value in (entry.get("name"), entry.get("full_name")):
                if self._name_matches_query(value, query):
                    return True
            code = entry.get("id") or entry.get("code")
            if code is not None:
                for alias in CUSTOM_WARD_ALIASES_BY_CODE.get(str(code), []):
                    if self._name_matches_query(alias, query):
                        return True
            return False

        def _lookup_context_ward(query: str) -> AdministrativeRecord | None:
            candidates: list[tuple[str | None, str | None]] = []
            if context_district_name:
                candidates.append((province_name, context_district_name))
            candidates.append((province_name, None))

            for province_candidate, district_candidate in candidates:
                entry = self._lookup_ward_info(
                    query,
                    province_candidate,
                    district_candidate,
                    preferred_format=False,
                )
                entry = _old_record(entry)
                if not entry or not _candidate_matches_entry(entry, query):
                    continue

                parent_district = _ward_parent_district(entry)
                if not parent_district:
                    continue
                if district_info:
                    expected_code = self.normalize_address_code(
                        district_info.get("id") or district_info.get("code")
                    )
                    parent_code = self.normalize_address_code(
                        parent_district.get("id") or parent_district.get("code")
                    )
                    if expected_code and parent_code and expected_code != parent_code:
                        continue

                parent_province_code = self.normalize_address_code(
                    parent_district.get("parent_code")
                )
                province_code = self.normalize_address_code(
                    province_info.get("id") or province_info.get("code")
                )
                if parent_province_code and province_code and parent_province_code != province_code:
                    continue
                return entry
            return None

        def _same_ward_entry(
            left: AdministrativeRecord | None,
            right: AdministrativeRecord | None,
        ) -> bool:
            left_id = (
                self.normalize_address_code(left.get("id") or left.get("code"))
                if isinstance(left, dict)
                else None
            )
            right_id = (
                self.normalize_address_code(right.get("id") or right.get("code"))
                if isinstance(right, dict)
                else None
            )
            return bool(left_id and right_id and left_id == right_id)

        def _has_explicit_ward_unit(tokens: list[str]) -> bool:
            if not tokens:
                return False
            first = tokens[0]
            pair = " ".join(tokens[:2]) if len(tokens) >= 2 else ""
            return first in {"phuong", "p", "xa", "x", "tt"} or pair == self._LIT_THI_TRAN

        def _matching_suffix_len_for_entry(
            segment_std: str,
            entry: AdministrativeRecord,
        ) -> int | None:
            tokens = [token for token in segment_std.split() if token]
            if not tokens:
                return None
            max_suffix_len = min(5, len(tokens))
            for suffix_len in range(max_suffix_len, 0, -1):
                suffix = " ".join(tokens[-suffix_len:])
                suffix_entry = _lookup_context_ward(suffix)
                if _same_ward_entry(suffix_entry, entry):
                    return suffix_len
                province_only_entry = _old_record(
                    self._lookup_ward_info(
                        suffix,
                        province_name,
                        None,
                        preferred_format=False,
                    )
                )
                if _same_ward_entry(province_only_entry, entry):
                    return suffix_len
            return None

        def _strip_contextual_suffix(raw_value: str, suffix_len: int) -> str:
            token_matches = list(re.finditer(self._RE_WORD_TOKEN, raw_value, flags=re.UNICODE))
            if not token_matches or suffix_len <= 0 or suffix_len > len(token_matches):
                return raw_value.strip(self._LIT_STRIP_CHARS)
            cut_pos = token_matches[-suffix_len].start()
            prefix = raw_value[:cut_pos].strip(self._LIT_STRIP_CHARS)
            prefix_tokens = [
                self._normalize_token_basic(match.group(0))
                for match in re.finditer(
                    self._RE_WORD_TOKEN,
                    prefix,
                    flags=re.UNICODE,
                )
            ]
            if prefix_tokens and all(token in locality_prefix_tokens for token in prefix_tokens):
                return ""
            return prefix

        best_match: ContextualWardCandidate | None = None
        fallback_match: ContextualWardCandidate | None = None
        for idx, (segment_std, segment_raw) in enumerate(input_segments):
            if idx in province_indices or idx == district_idx or not segment_std:
                continue
            tokens = [token for token in segment_std.split() if token]
            if not tokens:
                continue
            max_suffix_len = min(5, len(tokens))
            for suffix_len in range(max_suffix_len, 0, -1):
                suffix_tokens = tokens[-suffix_len:]
                prefix_tokens = tokens[:-suffix_len]
                suffix = " ".join(suffix_tokens)
                if not suffix:
                    continue
                explicit_unit = _has_explicit_ward_unit(suffix_tokens)
                whole_segment = not prefix_tokens
                prefix_has_locality = any(
                    token in locality_prefix_tokens or re.fullmatch(r"kp\d+\w*", token)
                    for token in prefix_tokens
                )
                prefix_has_street = any(token in street_prefix_tokens for token in prefix_tokens)
                entry = _lookup_context_ward(suffix)
                fallback_entry: AdministrativeRecord | None = None
                if not entry and district_info and prefix_tokens and not explicit_unit:
                    province_only_entry = _old_record(
                        self._lookup_ward_info(
                            suffix,
                            province_name,
                            None,
                            preferred_format=False,
                        )
                    )
                    if (
                        province_only_entry
                        and _candidate_matches_entry(province_only_entry, suffix)
                        and _candidate_matches_canonical_or_custom(
                            province_only_entry,
                            suffix,
                        )
                    ):
                        stripped_raw = _strip_contextual_suffix(segment_raw, suffix_len)
                        if stripped_raw:
                            fallback_entry = province_only_entry
                if not entry:
                    if not fallback_entry:
                        continue
                    fallback_candidate: ContextualWardCandidate = {
                        "score": (
                            0,
                            1 if prefix_has_locality else 0,
                            suffix_len,
                        ),
                        "segment_idx": idx,
                        "suffix_len": suffix_len,
                        "ward_info": fallback_entry,
                        "district_info": district_info,
                    }
                    if (
                        fallback_match is None
                        or fallback_candidate["score"] > fallback_match["score"]
                    ):
                        fallback_match = fallback_candidate
                    continue
                canonical_or_custom = _candidate_matches_canonical_or_custom(
                    entry,
                    suffix,
                )
                if not canonical_or_custom and not explicit_unit and not prefix_has_locality:
                    continue
                if (
                    not whole_segment
                    and not explicit_unit
                    and not prefix_has_locality
                    and prefix_has_street
                ):
                    continue
                ward_candidate: ContextualWardCandidate = {
                    "score": (
                        1,
                        3 if explicit_unit else 0,
                        2 if whole_segment else 0,
                        1 if prefix_has_locality else 0,
                        suffix_len,
                    ),
                    "segment_idx": idx,
                    "suffix_len": suffix_len,
                    "ward_info": entry,
                    "district_info": _ward_parent_district(entry) or district_info,
                }
                if best_match is None or ward_candidate["score"] > best_match["score"]:
                    best_match = ward_candidate
                break

        if not best_match:
            best_match = fallback_match
        if not best_match:
            return None

        final_district_info = best_match["district_info"] or district_info
        if not final_district_info:
            return None
        parent_code = self.normalize_address_code(final_district_info.get("parent_code"))
        final_province_info = (
            self.old_province_records.get(parent_code) if parent_code else None
        ) or province_info

        dedicated_ward_segment_indices: set[int] = set()
        for idx, (segment_std, _) in enumerate(input_segments):
            if idx in province_indices or idx == district_idx or not segment_std:
                continue
            tokens = [token for token in segment_std.split() if token]
            if not tokens:
                continue
            whole_segment_match_len = _matching_suffix_len_for_entry(
                segment_std,
                best_match["ward_info"],
            )
            if whole_segment_match_len and whole_segment_match_len == len(tokens):
                dedicated_ward_segment_indices.add(idx)

        district_aliases = self._gather_alias_values(
            final_district_info.get("full_name") or final_district_info.get("name"),
            final_district_info,
            level="district",
        )
        dedicated_district_segment_indices: set[int] = set()
        if district_idx is not None:
            dedicated_district_segment_indices.add(district_idx)
        for idx, (segment_std, _) in enumerate(input_segments):
            if idx in province_indices or idx in dedicated_ward_segment_indices or not segment_std:
                continue
            segment_core = self._strip_generic_prefix(segment_std) or segment_std
            for alias in district_aliases:
                for alias_std in self._standardized_name_variants(alias):
                    alias_core = self._strip_generic_prefix(alias_std) or alias_std
                    if (
                        alias_std == segment_std
                        or alias_std == segment_core
                        or alias_core == segment_std
                        or alias_core == segment_core
                    ):
                        dedicated_district_segment_indices.add(idx)
                        break
                if idx in dedicated_district_segment_indices:
                    break

        has_dedicated_ward_segment = bool(dedicated_ward_segment_indices)

        street_parts: list[str] = []
        for idx, (segment_std, segment_raw) in enumerate(input_segments):
            if (
                idx in province_indices
                or idx == district_idx
                or idx in dedicated_district_segment_indices
            ):
                continue
            if segment_std in {"viet nam", "vietnam"}:
                continue
            if idx in dedicated_ward_segment_indices:
                continue
            cleaned = str(segment_raw).strip(self._LIT_STRIP_CHARS)
            if not cleaned:
                continue
            matched_suffix_len = _matching_suffix_len_for_entry(
                self.standardize_name(cleaned, "basic"),
                best_match["ward_info"],
            )
            if matched_suffix_len and not has_dedicated_ward_segment:
                cleaned_tokens_std = [
                    token for token in self.standardize_name(cleaned, "basic").split() if token
                ]
                prefix_tokens = cleaned_tokens_std[:-matched_suffix_len]
                preserve_named_street = bool(
                    prefix_tokens
                    and prefix_tokens[0] in named_street_prefix_tokens
                    and not any(token in locality_prefix_tokens for token in prefix_tokens)
                )
                if not preserve_named_street:
                    cleaned = _strip_contextual_suffix(cleaned, matched_suffix_len)
            cleaned_tokens = [
                token for token in self.standardize_name(cleaned, "basic").split() if token
            ]
            if not cleaned_tokens:
                cleaned_tokens = [
                    self._normalize_token_basic(match.group(0))
                    for match in re.finditer(
                        self._RE_WORD_TOKEN,
                        cleaned,
                        flags=re.UNICODE,
                    )
                ]
            if cleaned_tokens and all(token in locality_prefix_tokens for token in cleaned_tokens):
                cleaned = ""
            if cleaned:
                street_parts.append(cleaned)

        street_address = " ".join(street_parts).strip()
        street_address = re.sub(r"\s+[-–—]+\s+", " ", street_address)
        street_address = self._cleanup_street_address_result(street_address)

        return {
            "province_info": final_province_info,
            "district_info": final_district_info,
            "ward_info": best_match["ward_info"],
            "street_address": street_address,
            "has_dedicated_district_segment": bool(dedicated_district_segment_indices),
            "raw_ward_fragment": self._recover_raw_suffix_fragment(
                input_segments[best_match["segment_idx"]][1],
                best_match["suffix_len"],
            ),
        }

    def _segment_suffix_has_locality_cue(
        self,
        segment_std: str | None,
        suffix_std: str | None,
    ) -> bool:
        if not segment_std or not suffix_std:
            return False

        tokens = [token for token in segment_std.split() if token]
        suffix_tokens = [token for token in suffix_std.split() if token]
        if not tokens or not suffix_tokens or len(suffix_tokens) >= len(tokens):
            return False
        if tokens[-len(suffix_tokens) :] != suffix_tokens:
            return False

        prefix_tokens = tokens[: -len(suffix_tokens)]
        if not prefix_tokens:
            return False

        locality_tokens: set[str] = {
            "thon",
            "xom",
            "ap",
            "to",
            "tdp",
            "kp",
            "khu",
            "kdc",
            "kdt",
            "cum",
            "doi",
            "khom",
            "nhom",
            "ban",
            "buon",
        }
        if any(
            token in locality_tokens or re.fullmatch(r"kp\d+\w*", token) for token in prefix_tokens
        ):
            return True

        prefix_text = " ".join(prefix_tokens)
        return any(
            marker in prefix_text
            for marker in (
                "khu pho",
                "khu vuc",
                "to dan pho",
                "khu dan cu",
            )
        )

    def _promote_contextual_old_ward_to_new(
        self,
        contextual_old_ward: ContextualOldWardResult,
    ) -> PromotedContextualWardResult | None:
        if contextual_old_ward.get("has_dedicated_district_segment"):
            return None

        old_ward_info = contextual_old_ward.get("ward_info")
        province_info = contextual_old_ward.get("province_info")
        if not isinstance(old_ward_info, dict) or not isinstance(province_info, dict):
            return None

        province_name = province_info.get("name") or province_info.get("full_name")
        if not province_name:
            return None

        ward_fragment = contextual_old_ward.get("raw_ward_fragment")
        lookup_values: list[str | None] = [
            ward_fragment,
            old_ward_info.get("full_name"),
            old_ward_info.get("name"),
        ]

        new_ward_info: AdministrativeRecord | None = None
        for value in lookup_values:
            if not value:
                continue
            candidate = self._lookup_ward_info(
                value,
                province_name,
                None,
                preferred_format=True,
            )
            if candidate and candidate.get("is_new_format") is True:
                new_ward_info = candidate
                break

        if not new_ward_info:
            return None

        fragment_prefix = self._admin_prefix_from_value(ward_fragment)
        new_prefix = self._admin_prefix_from_value(
            new_ward_info.get("full_name") or new_ward_info.get("name")
        )
        if fragment_prefix and new_prefix and fragment_prefix != new_prefix:
            return None

        ward_id = self.normalize_address_code(new_ward_info.get("id") or new_ward_info.get("code"))
        province_id = self.normalize_address_code(new_ward_info.get("province_id"))
        if not province_id and ward_id:
            new_ward_record = self.new_ward_records.get(ward_id)
            if isinstance(new_ward_record, dict):
                province_id = self.normalize_address_code(new_ward_record.get("parent_code"))
        if not province_id:
            province_id = self._lookup_new_province_id_by_name(province_name)

        promoted_province_info = self.new_province_records.get(province_id) if province_id else None
        if not isinstance(promoted_province_info, dict):
            promoted_province_info = province_info

        return {
            "province_info": promoted_province_info,
            "ward_info": new_ward_info,
        }

    def _admin_prefix_from_value(self, value: str | None) -> str | None:
        if not value:
            return None
        std = self.standardize_name(value, "basic")
        if not std:
            return None
        if std.startswith(("phuong ", "p ")):
            return "phuong"
        if std.startswith(("xa ", "x ")):
            return "xa"
        if std.startswith(("thi tran ", "tt ")):
            return "thi tran"
        if std.startswith("dac khu "):
            return "dac khu"
        return None

    def _recover_raw_suffix_fragment(
        self,
        raw_segment: str | None,
        suffix_token_count: int,
    ) -> str:
        if not raw_segment or suffix_token_count <= 0:
            return ""
        token_matches = list(re.finditer(self._RE_WORD_TOKEN, raw_segment, flags=re.UNICODE))
        if len(token_matches) < suffix_token_count:
            return raw_segment.strip(self._LIT_STRIP_CHARS)
        start = token_matches[-suffix_token_count].start()
        return raw_segment[start:].strip(self._LIT_STRIP_CHARS)

    def _strip_trailing_component_fragment(
        self,
        raw_value: str | None,
        fragment: str | None,
        *,
        allow_plain_prefix: bool = False,
    ) -> str:
        if not raw_value or not fragment:
            return (raw_value or "").strip()

        fragment_std = self.standardize_name(fragment, "basic")
        fragment_tokens = [token for token in fragment_std.split() if token]
        if not fragment_tokens:
            return raw_value.strip()

        token_matches = list(re.finditer(self._RE_WORD_TOKEN, raw_value, flags=re.UNICODE))
        if len(token_matches) <= len(fragment_tokens):
            return raw_value.strip()

        raw_tokens_std = [self._normalize_token_basic(match.group(0)) for match in token_matches]
        if raw_tokens_std[-len(fragment_tokens) :] != fragment_tokens:
            return raw_value.strip()

        descriptor_tokens: set[str] = {
            "thon",
            "xom",
            "ap",
            "to",
            "kp",
            "khu",
            "kdc",
            "kdt",
            "cum",
            "doi",
            "duong",
            "d",
            "dg",
            "ngo",
            "ngach",
            "hem",
            "hxh",
            "tuyen",
            "ql",
            "quoclo",
            "tl",
            "tinhlo",
            "dailo",
            "truc",
        }

        def _is_descriptor(token: str) -> bool:
            return token in descriptor_tokens or bool(re.fullmatch(r"kp\d+\w*", token))

        prefix_tokens = raw_tokens_std[: -len(fragment_tokens)]
        custom_implicit_locality = bool(
            len(prefix_tokens) == 1
            and any(
                prefix_tokens[0] == prefix and fragment_std == ward
                for prefix, ward, _, _ in CUSTOM_LOCALITY_WARD_SUFFIXES
            )
        )
        meaningful_prefix = [
            token
            for token in prefix_tokens
            if not _is_descriptor(token) and token not in self._GENERIC_LOCATION_TOKENS
        ]
        has_descriptor_prefix = any(_is_descriptor(token) for token in prefix_tokens)
        if (
            not has_descriptor_prefix
            and not custom_implicit_locality
            and not (allow_plain_prefix and meaningful_prefix)
        ):
            return raw_value.strip()

        allow_locality_stub = bool(
            len(prefix_tokens) == 1 and re.fullmatch(r"kp\d+\w*", prefix_tokens[0])
        )
        if len(prefix_tokens) >= 2 and prefix_tokens[0] == "kp":
            allow_locality_stub = allow_locality_stub or bool(
                re.fullmatch(r"\d+\w*", prefix_tokens[1])
            )
        if len(prefix_tokens) >= 3 and " ".join(prefix_tokens[:2]) == "khu pho":
            allow_locality_stub = allow_locality_stub or bool(
                re.fullmatch(r"\d+\w*", prefix_tokens[2])
            )
        if not meaningful_prefix and not allow_locality_stub and not custom_implicit_locality:
            return raw_value.strip()

        cut_pos = token_matches[-len(fragment_tokens)].start()
        return raw_value[:cut_pos].strip(self._LIT_STRIP_CHARS)

    def _infer_ward_from_segment_suffix(
        self,
        segment_std: str | None,
        segment_raw: str | None,
        *,
        expected_province: str | None = None,
        expected_district: str | None = None,
        allow_plain_prefix: bool = False,
    ) -> tuple[str, str] | None:
        if (
            not segment_std
            or not segment_raw
            or self._segment_has_explicit_admin_prefix(segment_std)
        ):
            return None

        tokens = [token for token in segment_std.split() if token]
        if len(tokens) < 3:
            return None

        descriptor_tokens: set[str] = {
            "thon",
            "xom",
            "ap",
            "to",
            "kp",
            "khu",
            "kdc",
            "kdt",
            "cum",
            "doi",
            "duong",
            "d",
            "dg",
            "ngo",
            "ngach",
            "hem",
            "hxh",
            "tuyen",
            "ql",
            "quoclo",
            "tl",
            "tinhlo",
            "dailo",
            "truc",
        }

        def _is_descriptor(token: str) -> bool:
            return token in descriptor_tokens or bool(re.fullmatch(r"kp\d+\w*", token))

        max_ward_len = min(4, len(tokens) - 1)
        for ward_len in range(max_ward_len, 0, -1):
            prefix_tokens = tokens[:-ward_len]
            fragment = " ".join(tokens[-ward_len:])
            province_key = (
                self._canonicalize_province_key(expected_province) if expected_province else ""
            )
            district_std = (
                self.standardize_name(expected_district, "basic") if expected_district else ""
            )
            district_key = self._strip_generic_prefix(district_std) or district_std
            custom_implicit_locality = bool(
                len(prefix_tokens) == 1
                and (
                    prefix_tokens[0],
                    fragment,
                    district_key,
                    province_key,
                )
                in CUSTOM_LOCALITY_WARD_SUFFIXES
            )
            allow_locality_stub = bool(
                len(prefix_tokens) == 1 and re.fullmatch(r"kp\d+\w*", prefix_tokens[0])
            )
            if len(prefix_tokens) >= 2 and prefix_tokens[0] == "kp":
                allow_locality_stub = allow_locality_stub or bool(
                    re.fullmatch(r"\d+\w*", prefix_tokens[1])
                )
            if len(prefix_tokens) >= 3 and " ".join(prefix_tokens[:2]) == "khu pho":
                allow_locality_stub = allow_locality_stub or bool(
                    re.fullmatch(r"\d+\w*", prefix_tokens[2])
                )
            if len(prefix_tokens) < 2 and not allow_locality_stub and not custom_implicit_locality:
                continue

            meaningful_prefix = [
                token
                for token in prefix_tokens
                if not _is_descriptor(token) and token not in self._GENERIC_LOCATION_TOKENS
            ]
            has_descriptor_prefix = any(_is_descriptor(token) for token in prefix_tokens)
            if (
                not has_descriptor_prefix
                and not custom_implicit_locality
                and not (
                    allow_plain_prefix
                    and len(meaningful_prefix) >= 2
                    and (expected_province or expected_district)
                )
            ):
                continue
            if not meaningful_prefix and not allow_locality_stub and not custom_implicit_locality:
                continue

            ward_entry = self._lookup_ward_info(
                fragment,
                expected_province,
                expected_district,
            )
            if not ward_entry or not self._entry_matches_component_fragment(
                ward_entry, fragment, level="ward"
            ):
                continue

            raw_fragment = self._recover_raw_suffix_fragment(segment_raw, ward_len)
            return raw_fragment or self._titleize_token(fragment), fragment

        return None

    def _split_address_segments(self, original: str) -> list[tuple[str, str]]:
        if not original:
            return []
        segments: list[tuple[str, str]] = []
        for part in cast(list[str], re.split(r"[,;\n]+", original)):
            raw = part.strip()
            if not raw:
                continue

            # Some sources use hyphen separators instead of commas, e.g.
            # "Số ... - Quận ... - Hà Nội". Only split on dashes when they
            # behave like segment separators (i.e. at least one chunk looks
            # like an administrative segment).
            subparts = [raw]
            dash_parts = [
                part.strip()
                for part in cast(list[str], re.split(r"\s+[-–—]\s+", raw))
                if part.strip()
            ]
            if len(dash_parts) > 1:
                # Only treat dashes as separators when the right-hand side looks like
                # an admin segment (explicit prefix) or a province name. This avoids
                # splitting ward names like "Phường X - Đà Lạt" where the suffix is a
                # locality hint, not a separate component.
                should_split = False
                for chunk in dash_parts[1:]:
                    chunk_std = self.standardize_name(chunk, "basic")
                    if self._segment_has_location_prefix(chunk_std):
                        should_split = True
                        break
                    if chunk_std and chunk_std in self.province_names_std:
                        should_split = True
                        break
                if should_split:
                    subparts = dash_parts

            for cleaned in subparts:
                std = self.standardize_name(cleaned, "basic")
                if not std:
                    continue
                segments.append((std, cleaned))
        return segments

    def _segment_has_location_prefix(self, segment_std: str | None) -> bool:
        if not segment_std:
            return False
        tokens = [token for token in segment_std.split() if token]
        if not tokens:
            return False
        first = tokens[0]
        if first in self._LOCATION_PREFIX_SINGLE:
            return True
        if len(tokens) >= 2:
            combined = f"{tokens[0]} {tokens[1]}"
            if combined in self._LOCATION_PREFIX_MULTI:
                return True
        return False

    def _segment_has_explicit_admin_prefix(self, segment_std: str | None) -> bool:
        if not segment_std:
            return False
        tokens = [token for token in segment_std.split() if token]
        if not tokens:
            return False
        if tokens[0] in self._LOCATION_PREFIX_SINGLE:
            return True
        if len(tokens) >= 2 and " ".join(tokens[:2]) in {
            self._LIT_THI_TRAN,
            self._LIT_THI_XA,
            self._LIT_THANH_PHO,
            self._LIT_DAC_KHU,
        }:
            return True
        return False

    def _segment_has_street_prefix(self, segment_std: str | None) -> bool:
        if not segment_std:
            return False
        tokens = [token for token in segment_std.split() if token]
        if not tokens:
            return False
        if re.fullmatch(r"kp\d+\w*", tokens[0]):
            return True
        if tokens[0] in self._STREET_PREFIX_SINGLE:
            return True
        if len(tokens) >= 2 and " ".join(tokens[:2]) in self._STREET_PREFIX_MULTI:
            return True
        return False

    def _input_looks_like_street_only_fragment(self, input_segments: list[tuple[str, str]]) -> bool:
        has_street_prefix = False
        for segment_std, _ in input_segments:
            if not segment_std:
                continue
            if self._segment_has_explicit_admin_prefix(segment_std):
                return False
            if self._segment_has_street_prefix(segment_std):
                has_street_prefix = True
        return has_street_prefix

    def _gather_alias_values(
        self,
        current_value: str | None,
        info: AdministrativeRecord | None,
        *,
        level: AdministrativeLevel,
        extra_values: list[str | None] | None = None,
    ) -> list[str]:
        aliases: list[str] = []
        seen: set[str] = set()

        def _looks_like_locality_surface_alias(value: str | None) -> bool:
            if level != "ward" or not isinstance(value, str):
                return False
            standardized = self.standardize_name(value, "basic")
            if not standardized or self._segment_has_location_prefix(standardized):
                return False
            tokens = [token for token in standardized.split() if token]
            if not tokens:
                return False
            first = tokens[0]
            second = tokens[1] if len(tokens) >= 2 else ""
            pair = f"{first} {second}".strip() if second else ""
            if re.fullmatch(r"kp\d+\w*", first):
                return True
            if pair in {"khu pho", "khu vuc", "to dan pho", "khu dan cu"}:
                return True
            return first in {
                "kp",
                "khu",
                "thon",
                "xom",
                "ap",
                "to",
                "kdc",
                "kdt",
                "cum",
                "doi",
                "khom",
            }

        def _add(value: str | None) -> None:
            if not isinstance(value, str):
                return
            candidate = value.strip()
            if not candidate or candidate in seen:
                return
            aliases.append(candidate)
            seen.add(candidate)

        _add(current_value)

        if isinstance(info, dict):
            _add(info.get("full_name"))
            _add(info.get("name"))
            legacy = info.get("legacy_names")
            if isinstance(legacy, str):
                _add(legacy)
            elif isinstance(legacy, list):
                for alias in legacy:
                    _add(alias)

            if level == "ward":
                code = info.get("id") or info.get("code")
                if code is not None:
                    custom_aliases = CUSTOM_WARD_ALIASES_BY_CODE.get(str(code), [])
                    for alias in custom_aliases:
                        _add(alias)
            elif level == "province":
                for alias in self._get_special_province_aliases(info.get("name")):
                    _add(alias)

        if extra_values:
            for raw in extra_values:
                if _looks_like_locality_surface_alias(raw):
                    continue
                _add(raw)

        return aliases

    def _prefer_component_alias_from_segments(
        self,
        alias_values: list[str],
        segments: list[tuple[str, str]],
        *,
        require_prefix: bool = False,
        level: AdministrativeLevel | None = None,
    ) -> str | None:
        if not alias_values or not segments:
            return None

        def _segment_matches_level_prefix(segment_std: str, target_level: str) -> bool:
            tokens = [token for token in segment_std.split() if token]
            if not tokens:
                return False
            first = tokens[0]
            second = tokens[1] if len(tokens) >= 2 else ""
            pair = f"{first} {second}".strip() if second else ""

            if target_level == "province":
                if first in {"tinh", "tp"}:
                    return True
                if pair == self._LIT_THANH_PHO:
                    return True
                return False

            if target_level == "district":
                if first in {"quan", "q", "huyen", "h", "tx"}:
                    return True
                if pair in {self._LIT_THI_XA}:
                    return True
                return False

            if target_level == "ward":
                if first in {"phuong", "p", "xa", "x", "tt"}:
                    return True
                if pair in {self._LIT_THI_TRAN, self._LIT_DAC_KHU}:
                    return True
                return False

            return False

        alias_norms: list[str] = []
        seen: set[str] = set()
        for alias in alias_values:
            for std in self._standardized_name_variants(alias):
                if not std or std in seen:
                    continue
                alias_norms.append(std)
                seen.add(std)

        if not alias_norms:
            return None

        for idx in range(len(segments) - 1, -1, -1):
            segment_std, raw_value = segments[idx]
            if not segment_std:
                continue
            if require_prefix:
                if level:
                    if not _segment_matches_level_prefix(segment_std, level):
                        continue
                elif not self._segment_has_location_prefix(segment_std):
                    continue
            for alias_std in alias_norms:
                segment_core = self._strip_generic_prefix(segment_std) or segment_std
                alias_core = self._strip_generic_prefix(alias_std) or alias_std
                if (
                    alias_std == segment_std
                    or alias_std == segment_core
                    or alias_core == segment_std
                    or alias_core == segment_core
                    or alias_std in segment_std
                    or segment_std in alias_std
                ):
                    return raw_value.strip()
        return None

    def _recover_component_from_input(
        self,
        target_std: str | None,
        segments: list[tuple[str, str]],
    ) -> str | None:
        if not target_std:
            return None
        target_std = target_std.strip()
        if not target_std:
            return None
        best_match: str | None = None
        best_score: tuple[int, int] = (-1, -1)
        for segment_std, raw in segments:
            if target_std == segment_std:
                exact = 1
                length = len(segment_std)
            elif target_std in segment_std:
                exact = 0
                length = len(target_std)
            else:
                continue
            score = (exact, length)
            if score > best_score:
                best_score = score
                best_match = raw.strip()
        return best_match

    def _strip_generic_prefix(self, value: str | None) -> str:
        if not value:
            return ""
        tokens = value.split()
        if not tokens:
            return ""

        def _is_pair_generic(tok0: str, tok1: str) -> bool:
            return (tok0 == "thanh" and tok1 == "pho") or (tok0 == "thi" and tok1 in {"tran", "xa"})

        tok0 = tokens[0]
        if tok0 in {
            "phuong",
            "p",
            "xa",
            "x",
            "tt",
            "tx",
            "quan",
            "q",
            "huyen",
            "h",
            "tp",
            "tinh",
        }:
            return " ".join(tokens[1:])
        if len(tokens) >= 2 and _is_pair_generic(tokens[0], tokens[1]):
            return " ".join(tokens[2:])
        return " ".join(tokens)

    def _reference_aliases_for_level(
        self,
        level: AdministrativeLevel,
        code: AddressCodeInput | None,
    ) -> list[str]:
        if not code:
            return []
        code_str = str(code).strip()
        candidates = [code_str]
        normalized = self.normalize_address_code(code)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
        lookup = (
            self.external_new_province_records
            if level == "province"
            else self.external_new_ward_records
        )
        entry = None
        for key in candidates:
            if key and key in lookup:
                entry = lookup[key]
                break
        if not entry:
            return []
        extras: list[str] = []
        for key in ("full_name",):
            value = entry.get(key)
            if isinstance(value, str) and value not in extras:
                extras.append(value)
        return extras

    def _register_alias_lookup_entry(
        self,
        registry: dict[str, list[AdministrativeRecord]],
        alias_set: set[str],
        payload: AdministrativeRecord,
    ) -> None:
        for alias_std in alias_set:
            if not alias_std:
                continue
            bucket = registry.setdefault(alias_std, [])
            if payload not in bucket:
                bucket.append(payload)

    def _register_node_aliases(
        self,
        node_index: int,
        *,
        province_aliases_std: set[str] | None = None,
        district_aliases_std: set[str] | None = None,
        ward_aliases_std: set[str] | None = None,
    ) -> None:
        if province_aliases_std:
            for alias in province_aliases_std:
                if alias:
                    self.invert_province_to_indices[alias].add(node_index)
        if district_aliases_std:
            for alias in district_aliases_std:
                if alias:
                    self.invert_district_to_indices[alias].add(node_index)
        if ward_aliases_std:
            for alias in ward_aliases_std:
                if alias:
                    self.invert_ward_to_indices[alias].add(node_index)

    def _fuzzy_match_component_key(
        self,
        value: str | None,
        choices: set[str] | list[str] | tuple[str, ...],
        *,
        cutoff: int = 88,
    ) -> str | None:
        if not value:
            return None

        normalized = self.standardize_name(value, "basic")
        if not normalized:
            return None

        if normalized in choices:
            return normalized

        choice_list = [choice for choice in choices if choice]
        if not choice_list:
            return None

        normalized_core = self._strip_generic_prefix(normalized) or normalized
        if len(normalized_core) < 4:
            return None

        normalized_tokens = normalized_core.split()
        fragment_digits = "".join(ch for ch in normalized if ch.isdigit())
        token_count = len(normalized_tokens)

        narrowed_choices: list[str] = []
        narrowed_cores: list[str] = []
        for choice in choice_list:
            choice_core, choice_digits, choice_token_count, choice_length, _, _ = (
                self._fuzzy_choice_profile(choice)
            )
            if not choice_core:
                continue
            if fragment_digits:
                if choice_digits and choice_digits != fragment_digits:
                    continue
            token_delta = abs(choice_token_count - token_count)
            if token_delta > 1:
                continue
            len_delta = abs(choice_length - len(normalized_core))
            if len_delta > max(5, len(normalized_core) // 2):
                continue
            narrowed_choices.append(choice)
            narrowed_cores.append(choice_core)

        if not narrowed_choices:
            if fragment_digits:
                for choice in choice_list:
                    choice_core, choice_digits, _, _, _, _ = self._fuzzy_choice_profile(choice)
                    if choice_digits == fragment_digits:
                        narrowed_choices.append(choice)
                        narrowed_cores.append(choice_core)
            if not narrowed_choices:
                narrowed_choices = choice_list
                narrowed_cores = [self._fuzzy_choice_profile(choice)[0] for choice in choice_list]

        effective_cutoff = cutoff
        if len(normalized_core) <= 5:
            effective_cutoff = max(effective_cutoff, 92)
        elif len(normalized_core) <= 7:
            effective_cutoff = max(effective_cutoff, 89)
        elif len(normalized_core) <= 10:
            effective_cutoff = max(effective_cutoff, 87)

        # ``process.extract`` used to apply the prefix processor to the query as
        # well as each choice. Preserve that second query pass while feeding
        # precomputed choice cores directly.
        scorer_query = self._strip_generic_prefix(normalized_core) or normalized_core
        candidates = cast(
            list[tuple[str, float, int]],
            rf_process.extract(
                scorer_query,
                narrowed_cores,
                scorer=rf_fuzz.WRatio,
                processor=None,
                score_cutoff=max(72, effective_cutoff - 8),
                limit=8,
            ),
        )
        if not candidates:
            return None

        first_token = normalized_tokens[0] if normalized_tokens else ""
        last_token = normalized_tokens[-1] if normalized_tokens else ""

        best_choice: str | None = None
        best_score = float("-inf")
        second_score = float("-inf")

        for _, wratio_score, candidate_index in candidates:
            candidate = narrowed_choices[candidate_index]
            (
                candidate_core,
                candidate_digits,
                candidate_token_count,
                candidate_length,
                candidate_first,
                candidate_last,
            ) = self._fuzzy_choice_profile(candidate)
            if not candidate_core:
                continue

            direct_ratio = ratio(normalized_core, candidate_core)
            token_ratio = rf_fuzz.token_sort_ratio(normalized_core, candidate_core)
            partial = partial_ratio(normalized_core, candidate_core)
            len_delta = abs(candidate_length - len(normalized_core))
            token_delta = abs(candidate_token_count - token_count)

            effective_score = max(
                wratio_score,
                direct_ratio,
                token_ratio,
                min(partial, direct_ratio + 5),
            )
            effective_score -= len_delta * 1.5
            if token_delta > 1:
                effective_score -= (token_delta - 1) * 10

            if first_token and candidate_first == first_token:
                effective_score += 1.5
            if last_token and candidate_last == last_token:
                effective_score += 2.5
            if fragment_digits and candidate_digits == fragment_digits:
                effective_score += 1.5

            if effective_score > best_score:
                second_score = best_score
                best_choice = candidate
                best_score = effective_score
            elif effective_score > second_score:
                second_score = effective_score

        if best_choice is None or best_score < effective_cutoff:
            return None
        if second_score >= best_score - 1.5 and best_score < effective_cutoff + 3:
            return None
        return best_choice

    def _fuzzy_choice_profile(self, choice: str) -> FuzzyChoiceProfile:
        cached = self._fuzzy_choice_profiles.get(choice)
        if cached is not None:
            return cached
        return self._build_fuzzy_choice_profile(choice)

    def _derive_ward_names(
        self,
        ward_name: str | None,
        ward_meta: AdministrativeRecord | LegacyWardRecord | None,
    ) -> tuple[str, str]:
        """
        Return (display_name, lookup_name) for a ward. Display names keep
        special prefixes like 'Đặc khu', while lookup names stay canonical
        so dictionaries remain stable.
        """
        meta = ward_meta if isinstance(ward_meta, dict) else None
        display_name = (ward_name or "").strip()
        lookup_name = (ward_name or "").strip()

        full_name = ""
        admin_unit_id = None
        if meta:
            full_name = meta.get("full_name") or ""
            admin_unit_id = meta.get("administrative_unit_id")

        if admin_unit_id == 5 and isinstance(full_name, str) and full_name.strip():
            display_name = full_name.strip()
        elif not display_name and isinstance(full_name, str) and full_name.strip():
            display_name = full_name.strip()

        if not lookup_name and isinstance(full_name, str) and full_name.strip():
            lookup_name = full_name.strip()

        if not display_name:
            display_name = lookup_name
        if not lookup_name:
            lookup_name = display_name

        return display_name or "", lookup_name or ""

    def _augment_aliases(self, aliases: list[str], level: AdministrativeLevel) -> list[str]:
        seen: set[str] = set()
        normalized_aliases: list[str] = []
        for alias in aliases:
            if alias not in seen:
                normalized_aliases.append(alias)
                seen.add(alias)

        extras: list[str] = []
        for alias in normalized_aliases:
            std = self.standardize_name(alias, "basic")
            if not std:
                continue
            tokens = [tok for tok in std.split() if tok]
            digits = None
            if std.isdigit():
                digits = std
            elif len(tokens) == 1 and tokens[0].isdigit():
                digits = tokens[0]
            if digits:
                if level == "ward":
                    extras.extend(
                        [
                            f"phuong {digits}",
                            f"p {digits}",
                            f"ward {digits}",
                            f"w {digits}",
                        ]
                    )
                elif level == "district":
                    extras.extend(
                        [
                            f"quan {digits}",
                            f"q {digits}",
                            f"district {digits}",
                            f"d {digits}",
                        ]
                    )

            if level == "ward":
                if len(tokens) >= 3 and " ".join(tokens[:2]) == self._LIT_THI_TRAN:
                    extras.append(f"tt {' '.join(tokens[2:])}")
                elif len(tokens) >= 2 and tokens[0] == "tt":
                    extras.append(f"{self._LIT_THI_TRAN} {' '.join(tokens[1:])}")

            core_tokens = [
                tok
                for tok in tokens
                if tok not in self._GENERIC_LOCATION_TOKENS and not tok.isdigit()
            ]
            if len(core_tokens) >= 2:
                initials = "".join(
                    token[0] for token in core_tokens if token and token[0].isalpha()
                )
                if 2 <= len(initials) <= 8:
                    extras.append(initials)
                    if level == "ward":
                        extras.extend(
                            [
                                f"p {initials}",
                                f"phuong {initials}",
                                f"ward {initials}",
                            ]
                        )
                    elif level == "district":
                        extras.extend(
                            [
                                f"q {initials}",
                                f"quan {initials}",
                                f"district {initials}",
                            ]
                        )

        for extra in extras:
            if extra not in seen:
                normalized_aliases.append(extra)
                seen.add(extra)
        return normalized_aliases

    def _validate_detected_value(
        self, value: str | None, lookup: dict[str, set[int]]
    ) -> str | None:
        if not value:
            return None
        return value if value in lookup else None

    def _resolve_detected_component(
        self,
        level: AdministrativeLevel,
        detected_value: str | None,
        *,
        expected_province: str | None = None,
        expected_district: str | None = None,
        source_string: str | None = None,
    ) -> str | None:
        if not detected_value:
            return None

        invert_map: dict[AdministrativeLevel, dict[str, set[int]]] = {
            "province": self.invert_province_to_indices,
            "district": self.invert_district_to_indices,
            "ward": self.invert_ward_to_indices,
        }
        lookup = invert_map.get(level)
        if not lookup:
            return None

        lookup_keys: list[str] = [detected_value]
        if level == "province":
            canonical_detected = self._canonicalize_province_key(detected_value)
            if canonical_detected:
                lookup_keys = [canonical_detected]
                if canonical_detected != detected_value:
                    lookup_keys.append(detected_value)

        indices: set[int] = set()
        for key in lookup_keys:
            indices.update(lookup.get(key, set()))
        if not indices:
            return None

        expected_province_std = (
            self.standardize_name(expected_province, "basic") if expected_province else None
        )
        expected_district_std = (
            self.standardize_name(expected_district, "basic") if expected_district else None
        )

        fallback: str | None = None
        candidates: list[tuple[str, str]] = []

        source_norm = source_string if source_string else ""
        enforce_specificity = level == "ward" and not expected_province and not expected_district

        def _collect(relax: bool) -> list[tuple[str, str]]:
            nonlocal fallback
            local_candidates: list[tuple[str, str]] = []
            local_fallback: str | None = None
            for idx in indices:
                node = self.address_node_list[idx]
                if level == "province":
                    name = node.province_name
                    if not name:
                        continue
                    norm = self.standardize_name(name, "basic")
                    local_candidates.append((name, norm))
                    if local_fallback is None:
                        local_fallback = name
                    continue

                if level == "district":
                    name = node.district_name
                    if not name:
                        continue
                    node_prov_std = (
                        self.standardize_name(node.province_name, "basic")
                        if node.province_name
                        else None
                    )
                    if expected_province_std and not relax:
                        if not node_prov_std or (
                            node_prov_std != expected_province_std
                            and not node_prov_std.endswith(expected_province_std)
                            and not expected_province_std.endswith(node_prov_std)
                        ):
                            continue
                    norm = self.standardize_name(name, "basic")
                    local_candidates.append((name, norm))
                    if local_fallback is None:
                        local_fallback = name
                    continue

                # ward level
                name = node.ward_name
                if not name:
                    continue
                node_prov_std = (
                    self.standardize_name(node.province_name, "basic")
                    if node.province_name
                    else None
                )
                node_dist_std = (
                    self.standardize_name(node.district_name, "basic")
                    if node.district_name
                    else None
                )
                if expected_province_std and not relax:
                    if not node_prov_std or (
                        node_prov_std != expected_province_std
                        and not node_prov_std.endswith(expected_province_std)
                        and not expected_province_std.endswith(node_prov_std)
                    ):
                        continue
                if expected_district_std and not relax:
                    if not node_dist_std:
                        continue
                    if (
                        node_dist_std != expected_district_std
                        and not node_dist_std.endswith(expected_district_std)
                        and not expected_district_std.endswith(node_dist_std)
                    ):
                        continue
                norm = self.standardize_name(name, "basic")
                local_candidates.append((name, norm))
                if local_fallback is None:
                    local_fallback = name
                continue

            if fallback is None and local_fallback is not None:
                fallback = local_fallback
            return local_candidates

        candidates = _collect(relax=False)
        if not candidates:
            if expected_province_std or expected_district_std:
                return None
            candidates = _collect(relax=True)

        if not candidates:
            return fallback

        if enforce_specificity and len(candidates) > 1:
            return None

        if source_norm:
            best_name = None
            best_len = -1
            for name, norm in candidates:
                if not norm:
                    continue
                if norm in source_norm and len(norm) > best_len:
                    best_name = name
                    best_len = len(norm)
            if best_name:
                return best_name

        detected_std = self.standardize_name(detected_value, "basic")
        detected_core = self._strip_generic_prefix(detected_std) or detected_std
        if detected_core:
            best_name = None
            best_score = float("-inf")
            for name, norm in candidates:
                if not norm:
                    continue
                norm_core = self._strip_generic_prefix(norm) or norm
                if not norm_core:
                    continue
                score = max(
                    ratio(detected_core, norm_core),
                    min(
                        partial_ratio(detected_core, norm_core), ratio(detected_core, norm_core) + 5
                    ),
                )
                if score > best_score:
                    best_name = name
                    best_score = score
            if best_name and best_score >= 86:
                return best_name

        return fallback or candidates[0][0]

    def _lookup_province_info(self, province_name: str | None) -> AdministrativeRecord | None:
        if not province_name:
            return None
        key = self.standardize_name(province_name, "basic")
        if not key:
            return None
        return self.province_lookup.get(key)

    def _lookup_district_info(
        self,
        district_name: str | None,
        province_name: str | None = None,
    ) -> AdministrativeRecord | None:
        if not district_name:
            return None
        district_key = self.standardize_name(district_name, "basic")
        if not district_key:
            return None

        def _district_variant_sort_key(item: str) -> tuple[bool, int, str]:
            return item != district_key, len(item), item

        district_keys = sorted(
            self._standardized_name_variants(district_name),
            key=_district_variant_sort_key,
        )
        if district_key not in district_keys:
            district_keys.insert(0, district_key)
        for compact_variant in self._compact_district_prefix_variants(district_name):
            for variant_key in self._standardized_name_variants(compact_variant):
                if variant_key not in district_keys:
                    district_keys.append(variant_key)
        province_key = self._canonicalize_province_key(province_name) if province_name else None
        if province_key:
            for key in district_keys:
                info = self.district_lookup.get((province_key, key))
                if info:
                    return info
            candidates: list[AdministrativeRecord] = []
            seen_ids: set[str] = set()
            for key in district_keys:
                for entry in self.district_lookup_by_name.get(key, []):
                    if entry.get("province_key") != province_key:
                        continue
                    entry_id = str(entry.get("id") or entry.get("code") or id(entry))
                    if entry_id in seen_ids:
                        continue
                    seen_ids.add(entry_id)
                    candidates.append(entry)
            if len(candidates) == 1:
                return candidates[0]
            # Do not fall back to a globally-unique district from a different province.
            # OCR noise at the head of the string can otherwise invent a wrong district.
            return None
        global_candidates: list[AdministrativeRecord] = []
        global_seen_ids: set[str] = set()
        for key in district_keys:
            for entry in self.district_lookup_by_name.get(key, []):
                entry_id = str(entry.get("id") or entry.get("code") or id(entry))
                if entry_id in global_seen_ids:
                    continue
                global_seen_ids.add(entry_id)
                global_candidates.append(entry)
        if len(global_candidates) == 1:
            return global_candidates[0]
        return None

    def _compact_district_prefix_variants(self, value: str | None) -> list[str]:
        if not value:
            return []
        std = self.standardize_name(value, "basic")
        if not std:
            return []

        variants: list[str] = []
        prefix_expansions = {
            "tp": ("tp", self._LIT_THANH_PHO),
            "tx": ("tx", self._LIT_THI_XA),
            "q": ("q", "quan"),
            "h": ("h", "huyen"),
        }
        for prefix, expanded_prefixes in prefix_expansions.items():
            if not std.startswith(prefix):
                continue
            if len(std) <= len(prefix) or std[len(prefix)].isspace():
                continue
            tail = std[len(prefix) :].strip()
            if not tail:
                continue
            for expanded_prefix in expanded_prefixes:
                variant = f"{expanded_prefix} {tail}".strip()
                if variant not in variants:
                    variants.append(variant)
            if tail not in variants:
                variants.append(tail)
        return variants

    def _lookup_ward_info(
        self,
        ward_name: str | None,
        province_name: str | None = None,
        district_name: str | None = None,
        preferred_format: bool | None = None,
    ) -> AdministrativeRecord | None:
        if not ward_name:
            return None
        raw_ward_key = re.sub(r"\s+", " ", ward_name.strip().lower())
        ward_key = self.standardize_name(ward_name, "basic")
        if not ward_key:
            return None
        required_unit = self._detect_unit_token_from_query(ward_name)
        province_key = self._canonicalize_province_key(province_name) if province_name else None
        district_key = self.standardize_name(district_name, "basic") if district_name else None

        def _ward_variant_sort_key(item: str) -> tuple[bool, int, str]:
            return item != ward_key, len(item), item

        ward_keys = sorted(
            self._standardized_name_variants(ward_name),
            key=_ward_variant_sort_key,
        )
        if ward_key not in ward_keys:
            ward_keys.insert(0, ward_key)
        normalized_numeric_key = self._normalize_numeric_component_key(
            ward_key,
            default_prefix="phuong",
        )
        if normalized_numeric_key and normalized_numeric_key not in ward_keys:
            ward_keys.append(normalized_numeric_key)

        # Only broaden with stripped variants when we have a regional hint.
        if province_key or district_key:
            stripped_key = self._strip_generic_prefix(ward_key)
            if stripped_key and stripped_key not in ward_keys:
                ward_keys.append(stripped_key)
            if normalized_numeric_key:
                stripped_numeric = self._strip_generic_prefix(normalized_numeric_key)
                if stripped_numeric and stripped_numeric not in ward_keys:
                    ward_keys.append(stripped_numeric)

        def _entry_unit_matches(entry: AdministrativeRecord) -> bool:
            if not required_unit:
                return True
            if self._unit_tokens_match(
                required_unit,
                self._extract_unit_token(entry, level="ward"),
            ):
                return True

            # A current canonical ward can carry a legacy alias with a different
            # unit token, e.g. "Thị trấn Phong Điền" now represented by
            # "Phường Phong Thu". Accept the unit when the alias itself matches.
            for alias in self._entry_alias_values(entry, level="ward"):
                if self._unit_tokens_match(
                    required_unit,
                    self._unit_token_from_text(alias),
                ):
                    return True
            return False

        def _canonical_query_matches(entry: AdministrativeRecord) -> bool:
            entry_name_value = entry.get("name")
            entry_name = (
                self.standardize_name(entry_name_value, "basic") if entry_name_value else None
            )
            entry_full_value = entry.get("full_name")
            entry_full = (
                self.standardize_name(entry_full_value, "basic") if entry_full_value else None
            )
            for key in ward_keys:
                if not key:
                    continue
                if entry_name == key or entry_full == key:
                    return True
            return False

        def _raw_query_matches(entry: AdministrativeRecord) -> bool:
            if not raw_ward_key:
                return False
            for value in (entry.get("name"), entry.get("full_name")):
                if not isinstance(value, str) or not value.strip():
                    continue
                entry_raw = re.sub(r"\s+", " ", value.strip().lower())
                if entry_raw == raw_ward_key:
                    return True
            return False

        if province_key and district_key:
            for key in ward_keys:
                info = self.ward_lookup.get((province_key, district_key, key))
                if info and _entry_unit_matches(info):
                    if self._entry_matches_raw_query_name(info, ward_name):
                        return info
                    continue

        if province_key:
            for key in ward_keys:
                province_candidates = [
                    entry
                    for entry in self.ward_lookup_by_province_name.get((province_key, key), [])
                    if _entry_unit_matches(entry)
                ]
                if len(province_candidates) == 1 and self._entry_matches_raw_query_name(
                    province_candidates[0],
                    ward_name,
                ):
                    return province_candidates[0]

        if district_key:
            district_candidates: list[AdministrativeRecord] = []
            for entry in self.ward_lookup_by_district_key.get(district_key, []):
                entry_name = entry.get("name")
                if not entry_name:
                    continue
                entry_name_std = self.standardize_name(entry_name, "basic")
                if not entry_name_std:
                    continue
                for key in ward_keys:
                    if entry_name_std == key or self._numeric_token_match(entry_name_std, key):
                        if _entry_unit_matches(entry):
                            district_candidates.append(entry)
                        break
            if len(district_candidates) == 1 and self._entry_matches_raw_query_name(
                district_candidates[0],
                ward_name,
            ):
                return district_candidates[0]

        fallback_candidates: list[AdministrativeRecord] = []
        for key in ward_keys:
            bucket = self.ward_lookup_by_name.get(key, [])
            if bucket:
                for entry in bucket:
                    if _entry_unit_matches(entry):
                        fallback_candidates.append(entry)
        if not fallback_candidates:
            fallback_candidates = [
                entry
                for entry in self.ward_lookup_by_name.get(ward_key, [])
                if _entry_unit_matches(entry)
            ]
        if not fallback_candidates:
            return None

        def _std(value: str | None) -> str:
            return self.standardize_name(value, "basic") if value else ""

        candidates = fallback_candidates

        if province_key:
            province_matches = [
                c
                for c in candidates
                if (
                    c.get("province_key")
                    and self._canonicalize_province_key(c["province_key"]) == province_key
                )
                or self._canonicalize_province_key(c.get("province_name")) == province_key
            ]
            if province_matches:
                candidates = province_matches
            else:
                return None

        if district_key:
            district_key_stripped = self._strip_generic_prefix(district_key)
            district_matches = [
                c
                for c in candidates
                if (
                    (c.get("district_key") and c["district_key"] == district_key)
                    or _std(c.get("district_name")) == district_key
                    or (
                        district_key_stripped
                        and (
                            c.get("district_key") == district_key_stripped
                            or _std(c.get("district_name")) == district_key_stripped
                        )
                    )
                )
            ]
            if district_matches:
                candidates = district_matches
            else:
                return None

        raw_matches = [c for c in candidates if _raw_query_matches(c)]
        if len(raw_matches) == 1:
            return raw_matches[0]
        if raw_matches:
            candidates = raw_matches

        if preferred_format is not None:
            format_matches = [c for c in candidates if c.get("is_new_format") is preferred_format]
            if len(format_matches) == 1:
                return format_matches[0]
            if format_matches:
                candidates = format_matches

        canonical_matches = [c for c in candidates if _canonical_query_matches(c)]
        if len(canonical_matches) == 1:
            return canonical_matches[0]
        if canonical_matches:
            candidates = canonical_matches

        if len(candidates) == 1:
            return candidates[0]

        prioritized = [c for c in candidates if c.get("district_key")]
        if len(prioritized) == 1:
            return prioritized[0]

        prioritized = [c for c in candidates if c.get("province_key")]
        if len(prioritized) == 1:
            return prioritized[0]

        prioritized = [c for c in candidates if c.get("id") is not None]
        if len(prioritized) == 1:
            return prioritized[0]

        # Contextual fallback:
        # - If only province is known (no district), prefer new-format (2-level) wards.
        # - If no province/district hints, prefer old-format to avoid drifting to floating new entries.
        # - When district hint exists, prefer candidates whose canonical name/full_name matches the request
        #   (to avoid legacy-alias-only matches).
        prefer_new = bool(province_key and not district_key)
        prefer_old = bool(not province_key and not district_key)
        ward_key_set = {k for k in ward_keys if k}

        def _candidate_sort_key(
            entry: AdministrativeRecord,
        ) -> tuple[int, int, str, str, str, str]:
            is_new = entry.get("is_new_format")
            if prefer_new:
                format_rank = 0 if is_new is True else 1 if is_new is False else 2
            elif prefer_old:
                format_rank = 0 if is_new is False else 1 if is_new is True else 2
            else:
                # Default: True (0) < False (1) < unknown (2)
                format_rank = 0 if is_new is True else 1 if is_new is False else 2

            name_value = entry.get("name")
            name_std = self.standardize_name(name_value, "basic") if name_value else None
            full_value = entry.get("full_name")
            full_std = self.standardize_name(full_value, "basic") if full_value else None
            matches_canonical = 0
            if ward_key_set:
                if name_std and name_std in ward_key_set:
                    matches_canonical = -1  # prefer exact name match
                elif full_std and full_std in ward_key_set:
                    matches_canonical = -1

            return (
                matches_canonical,
                format_rank,
                entry.get("district_key") or "",
                entry.get("province_key") or "",
                entry.get("id") or entry.get("code") or "",
                entry.get("name") or "",
            )

        return sorted(candidates, key=_candidate_sort_key)[0]

    def _recover_district_from_ward_info(
        self,
        ward_info: AdministrativeRecord | None,
        ward_name: str | None,
        province_name: str | None,
        province_info: AdministrativeRecord | None,
    ) -> tuple[str | None, str | None]:
        if not ward_info:
            return None, None

        known_district = ward_info.get("district_name")
        if known_district:
            if ward_info.get("is_new_format") is False:
                return known_district, ward_info.get("district_id")
            return None, None

        ward_std = self.standardize_name(ward_name, "basic") if ward_name else ""
        if not ward_std:
            return None, None

        province_std = self.standardize_name(province_name, "basic") if province_name else None

        def _province_matches(entry: AdministrativeRecord) -> bool:
            if not province_std:
                return True
            entry_key = entry.get("province_key")
            entry_province_name = entry.get("province_name")
            if not entry_key and entry_province_name:
                entry_key = self.standardize_name(entry_province_name, "basic")
            if not entry_key:
                return True
            return entry_key == province_std

        candidates = self.ward_lookup_by_name.get(ward_std, [])
        for candidate in candidates:
            if candidate is ward_info:
                continue
            if not candidate.get("district_name"):
                continue
            if not _province_matches(candidate):
                continue
            return candidate.get("district_name"), candidate.get("district_id")

        ward_id = ward_info.get("id")
        if not ward_id:
            return None, None

        return None, None

    def _lookup_new_format_ward_alias(
        self,
        detected_token: str | None,
        expected_province: str | None = None,
    ) -> AdministrativeRecord | None:
        if not detected_token:
            return None
        token = detected_token.strip()
        if not token:
            return None

        province_std = (
            self._canonicalize_province_key(expected_province) if expected_province else None
        )

        def _filter(
            entries: list[AdministrativeRecord], *, enforce_province: bool
        ) -> AdministrativeRecord | None:
            if not entries:
                return None
            for entry in entries:
                if not entry.get("is_new_format"):
                    continue
                if (
                    enforce_province
                    and province_std
                    and not self._entry_aligns_with_province(entry, expected_province)
                ):
                    continue
                return entry
            return None

        def _new_ward_sort_key(
            entry: AdministrativeRecord,
        ) -> tuple[str, str, str]:
            return (
                entry.get("district_key") or "",
                entry.get("province_key") or "",
                entry.get("id") or "",
            )

        candidate_tokens = [token]
        token_core = self._strip_generic_prefix(token)
        if token_core and token_core not in candidate_tokens:
            candidate_tokens.append(token_core)

        if province_std:
            for candidate_token in candidate_tokens:
                province_bucket = self.ward_lookup_by_province_name.get(
                    (province_std, candidate_token), []
                )
                entry = _filter(province_bucket, enforce_province=True)
                if entry:
                    return entry

        for candidate_token in candidate_tokens:
            candidates = self.ward_lookup_by_name.get(candidate_token, [])
            filtered = [
                entry
                for entry in candidates
                if entry.get("is_new_format")
                and (not province_std or self._entry_aligns_with_province(entry, expected_province))
            ]
            if province_std:
                entry = _filter(filtered, enforce_province=True)
                if entry:
                    return entry
            # When no province hint and multiple new-format entries share the same name,
            # avoid guessing to prevent cross-province drift.
            if not province_std and len(filtered) != 1:
                continue
            if filtered:
                filtered_sorted = sorted(
                    filtered,
                    key=_new_ward_sort_key,
                )
                return filtered_sorted[0]

        if province_std:
            return None
        for candidate_token in candidate_tokens:
            fallback_candidates = self.ward_lookup_by_name.get(candidate_token, [])
            fallback = _filter(fallback_candidates, enforce_province=False)
            if fallback:
                return fallback
        return None

    def _entry_aligns_with_province(
        self,
        entry: AdministrativeRecord | None,
        expected_province: str | None,
    ) -> bool:
        if not expected_province or not isinstance(entry, dict):
            return True
        expected_std = self._canonicalize_province_key(expected_province)
        if not expected_std:
            return True
        entry_std = entry.get("province_key")
        if not entry_std:
            entry_province_name = entry.get("province_name")
            entry_std = (
                self.standardize_name(entry_province_name, "basic") if entry_province_name else None
            )
        if entry_std:
            entry_std = self._canonicalize_province_key(entry_std)
            return entry_std == expected_std
        return False

    def _canonicalize_province_key(self, value: str | None) -> str:
        if not value:
            return ""
        key = self.standardize_name(value, "basic")
        if not key:
            return ""
        key = re.sub(self._RE_PROVINCE_PREFIX, "", key).strip()
        return self._canonical_province_key_by_alias.get(key, key)

    def _build_canonical_province_key_map(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for synonyms, canonical in SPECIAL_PROVINCE_MAP.items():
            canonical_key = normalize_address_text(canonical, "basic")
            canonical_key = re.sub(self._RE_PROVINCE_PREFIX, "", canonical_key).strip()
            if not canonical_key:
                continue
            aliases[canonical_key] = canonical_key
            for alias in synonyms:
                alias_key = normalize_address_text(alias, "basic")
                alias_key = re.sub(self._RE_PROVINCE_PREFIX, "", alias_key).strip()
                if alias_key:
                    aliases[alias_key] = canonical_key
        return aliases

    def _prefer_hierarchical_ward_entry(
        self,
        normalized_token: str | None,
        entry: AdministrativeRecord | None,
        *,
        expected_province: str | None = None,
    ) -> AdministrativeRecord | None:
        if not normalized_token or not entry:
            return entry
        if entry.get("district_key"):
            return entry

        if not self._entry_aligns_with_province(entry, expected_province):
            return entry

        original_is_new = entry.get("is_new_format")
        entry_id = self.normalize_address_code(entry.get("id") or entry.get("code"))
        bucket = self.ward_lookup_by_name.get(normalized_token, [])
        if not bucket:
            return entry

        for candidate in bucket:
            if not candidate.get("district_key"):
                continue
            if not self._entry_aligns_with_province(candidate, expected_province):
                continue
            if original_is_new is True and candidate.get("is_new_format") is not True:
                continue
            if entry_id:
                candidate_id = self.normalize_address_code(
                    candidate.get("id") or candidate.get("code")
                )
                if candidate_id and candidate_id != entry_id:
                    continue
            return candidate

        return entry

    def _format_component(
        self,
        name: str | None,
        candidate_id: AddressCode | None,
        info: AdministrativeRecord | None,
    ) -> ParsedAddressComponent | None:
        if not name:
            return None
        resolved_name = name
        if info:
            alt_name = info.get("full_name") or info.get("name")
            if alt_name:
                normalized = self.standardize_name(name, "basic")
                # Some legacy datasets store numeric-only names (e.g. "5"). Prefer the descriptive
                # full name in those cases so downstream consumers see "Quận 5" instead of "5".
                if normalized and normalized.isdigit():
                    resolved_name = alt_name
                elif not resolved_name.strip():
                    resolved_name = alt_name

        payload: ParsedAddressComponent = {"name": resolved_name}
        component_id = candidate_id
        if component_id is None and info:
            component_id = info.get("id")
        if component_id is not None:
            payload["id"] = component_id
        if info:
            extended_name = info.get("full_name")
            if extended_name:
                payload["full_name"] = extended_name
            code_value = info.get("code") or info.get("id")
            if code_value is not None and code_value != "":
                payload["code"] = code_value
            legacy_aliases = info.get("legacy_names")
            if isinstance(legacy_aliases, str):
                legacy_aliases = [legacy_aliases]
            if isinstance(legacy_aliases, list):
                cleaned_aliases = []
                for alias in legacy_aliases:
                    if isinstance(alias, str):
                        candidate = alias.strip()
                        if candidate and candidate not in cleaned_aliases:
                            cleaned_aliases.append(candidate)
                if cleaned_aliases:
                    payload["legacy_names"] = cleaned_aliases
        return payload

    def _infer_province_from_components(
        self,
        district: str | None,
        ward: str | None,
    ) -> str | None:
        district_std = self.standardize_name(district, "basic") if district else ""
        ward_std = self.standardize_name(ward, "basic") if ward else ""

        candidate_sets: list[set[str]] = []

        if ward_std:
            indices = self.invert_ward_to_indices.get(ward_std, set())
            ward_candidates: set[str] = set()
            for idx in indices:
                node = self.address_node_list[idx]
                if not node.province_name:
                    continue
                if district_std:
                    node_dist_std = (
                        self.standardize_name(node.district_name, "basic")
                        if node.district_name
                        else ""
                    )
                    if not node_dist_std or node_dist_std != district_std:
                        continue
                ward_candidates.add(node.province_name)
            if ward_candidates:
                candidate_sets.append(ward_candidates)

        if district_std:
            indices = self.invert_district_to_indices.get(district_std, set())
            district_candidates: set[str] = set()
            for idx in indices:
                province_name = self.address_node_list[idx].province_name
                if province_name:
                    district_candidates.add(province_name)
            if district_candidates:
                candidate_sets.append(district_candidates)

        if not candidate_sets:
            return None

        intersection = set(candidate_sets[0])
        for candidate_set in candidate_sets[1:]:
            intersection &= candidate_set

        if len(intersection) == 1:
            return next(iter(intersection))

        if len(candidate_sets) == 1 and len(candidate_sets[0]) == 1:
            return next(iter(candidate_sets[0]))

        return None

    def _infer_district_from_components(
        self,
        province: str | None,
        ward: str | None,
        *,
        source_string: str | None = None,
    ) -> str | None:
        ward_std = self.standardize_name(ward, "basic") if ward else ""
        if not ward_std:
            return None

        province_std = self.standardize_name(province, "basic") if province else None
        indices = self.invert_ward_to_indices.get(ward_std, set())
        if not indices:
            return None

        candidate_entries: list[tuple[str, str, str | None]] = []
        for idx in indices:
            node = self.address_node_list[idx]
            district_name = node.district_name
            if not district_name:
                continue
            node_prov_std = (
                self.standardize_name(node.province_name, "basic") if node.province_name else None
            )
            if province_std and node_prov_std and node_prov_std != province_std:
                continue
            district_std = self.standardize_name(district_name, "basic")
            candidate_entries.append((district_name, district_std, node_prov_std))

        if not candidate_entries:
            return None

        normalized_source = source_string or ""
        has_hcm_candidate = (
            any(prov_std == "ho chi minh" for _, _, prov_std in candidate_entries)
            or province_std == "ho chi minh"
        )

        if normalized_source and "thu duc" in normalized_source and has_hcm_candidate:
            for name, district_std, _ in candidate_entries:
                if district_std == "thu duc":
                    return name
            return "Thủ Đức"

        best_name = None
        best_len = -1
        if normalized_source:
            for name, district_std, _ in candidate_entries:
                if district_std and district_std in normalized_source:
                    if len(district_std) > best_len:
                        best_name = name
                        best_len = len(district_std)
            if best_name:
                return best_name

        unique_names = {name for name, _, _ in candidate_entries if name}
        if len(unique_names) == 1:
            return next(iter(unique_names))

        return None

    def _build_node_search_profile(
        self,
        province_aliases: list[str],
        district_aliases: list[str],
        ward_aliases: list[str],
        *,
        include_province: bool,
        include_district: bool,
        include_ward: bool,
    ) -> tuple[str, set[str]]:
        primary_parts: list[str] = []
        if include_ward and ward_aliases:
            primary_parts.append(ward_aliases[0])
        if include_district and district_aliases:
            primary_parts.append(district_aliases[0])
        if include_province and province_aliases:
            primary_parts.append(province_aliases[0])
        primary_string = " ".join(part for part in primary_parts if part)
        primary_standardized = self.standardize_name(primary_string)

        province_candidates = province_aliases if include_province else [""]
        district_candidates = district_aliases if include_district else [""]
        ward_candidates = ward_aliases if include_ward else [""]

        ngram_set: set[str] = set()
        for ward_name in ward_candidates:
            for district_name in district_candidates:
                for province_name in province_candidates:
                    combined = " ".join(
                        part for part in [ward_name, district_name, province_name] if part
                    )
                    if not combined:
                        continue
                    standardized = self.standardize_name(combined)
                    if standardized:
                        ngram_set.update(self.generate_ngrams(standardized))

        if not ngram_set and primary_standardized:
            ngram_set.update(self.generate_ngrams(primary_standardized))

        return primary_standardized, ngram_set

    def standardize_name(self, name: str, mode: NormalizationMode = "basic") -> str:
        """Normalize an address fragment using an explicit processing mode."""
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        if mode not in {
            "basic",
            "search",
            "aggressive",
        }:
            raise ValueError("mode must be 'basic', 'search', or 'aggressive'")

        return normalize_address_text(name, mode)

    def _raw_name_variant_candidates(self, value: str | None) -> list[str]:
        if not isinstance(value, str):
            return []
        stripped = value.strip()
        if not stripped:
            return []

        candidates: list[str] = []
        seen: set[str] = set()

        def _add(candidate: str | None) -> None:
            if not isinstance(candidate, str):
                return
            item = candidate.strip()
            if not item or item in seen:
                return
            candidates.append(item)
            seen.add(item)

        _add(stripped)

        if re.search(r"[\'’`´]", stripped):
            _add(re.sub(r"[\'’`´]+", "", stripped))
            _add(re.sub(r"[\'’`´]+", " ", stripped))

        split_tokens: list[str] = []
        changed = False
        for token in stripped.split():
            match = re.match(r"^([A-Za-zĐđ])([A-ZÀ-ỸĐ].+)$", token)
            if match:
                split_tokens.extend([match.group(1), match.group(2)])
                changed = True
            else:
                split_tokens.append(token)
        if changed:
            _add(" ".join(split_tokens))

        return candidates

    def _standardized_name_variants(self, value: str | None) -> set[str]:
        variants: set[str] = set()
        queue: list[str] = []

        def _enqueue(candidate: str | None) -> None:
            if not candidate:
                return
            if candidate in variants or candidate in queue:
                return
            queue.append(candidate)

        for raw_candidate in self._raw_name_variant_candidates(value):
            standardized = self.standardize_name(raw_candidate, "basic")
            if standardized:
                _enqueue(standardized)

        while queue:
            current = queue.pop(0)
            if current in variants:
                continue
            variants.add(current)

            tokens = [token for token in current.split() if token]
            if len(tokens) < 2:
                continue

            for idx in range(len(tokens) - 1):
                if len(tokens[idx]) != 1 or not tokens[idx].isalpha():
                    continue
                if not tokens[idx + 1].isalpha():
                    continue
                merged_tokens = tokens[:idx] + [tokens[idx] + tokens[idx + 1]] + tokens[idx + 2 :]
                merged = " ".join(merged_tokens).strip()
                if merged:
                    _enqueue(merged)

        return variants

    def _normalize_token_basic(self, token: str) -> str:
        if not token:
            return ""
        token = token.lower()
        token = token.replace("đ", "d")
        token = unicodedata.normalize("NFD", token)
        token = "".join(ch for ch in token if unicodedata.category(ch) != "Mn")
        token = re.sub(r"[^a-z0-9]+", "", token)
        return token

    def _is_generic_location_token(self, raw: str | None, norm: str | None) -> bool:
        if not norm or norm not in self._GENERIC_LOCATION_TOKENS:
            return False
        if not raw:
            return True
        raw_clean = unicodedata.normalize("NFC", raw).lower().strip()
        if not raw_clean:
            return True
        if norm == "to" and ("ố" in raw_clean):
            return False
        return True

    def _build_component_signature(
        self,
        component: str | None,
        extra_aliases: list[str] | None = None,
    ) -> ComponentSignature:
        signature: ComponentSignature = {
            "sequences": [],
            "tokens": set(),
            "abbreviation_sequences": set(),
        }
        candidates: list[str] = []
        if component:
            candidates.append(component)
        if extra_aliases:
            for alias in extra_aliases:
                if alias:
                    candidates.append(alias)
        if not candidates:
            return signature

        processed: set[str] = set()

        def _register(parts: list[str], *, is_abbreviation: bool = False) -> None:
            if not parts:
                return
            normalized_parts = [part for part in parts if part]
            if not normalized_parts:
                return
            signature["sequences"].append(normalized_parts)
            for token in normalized_parts:
                signature["tokens"].add(token)
            if is_abbreviation:
                signature["abbreviation_sequences"].add(tuple(normalized_parts))

        for value in candidates:
            for standardized in self._standardized_name_variants(value):
                if not standardized or standardized in processed:
                    continue
                processed.add(standardized)
                raw_parts = [p for p in standardized.split() if p]
                parts = [self._normalize_token_basic(p) for p in raw_parts]
                parts = [p for p in parts if p]
                if not parts:
                    continue

                _register(parts)

                if len(parts) == 2 and parts[1].isdigit():
                    trimmed = parts[1].lstrip("0")
                    numeric_variants = {parts[1]}
                    if trimmed and trimmed != parts[1]:
                        numeric_variants.add(trimmed)
                    for variant in numeric_variants:
                        _register([variant])

                joined = "".join(parts)
                if joined:
                    _register([joined])

                abbr_parts: list[str] = []
                for part in parts:
                    if not part:
                        continue
                    abbr_parts.append(part if part.isdigit() else part[0])
                abbr = "".join(abbr_parts)
                if len(abbr) >= 2:
                    _register([abbr], is_abbreviation=True)
                    _register([f"tp{abbr}"], is_abbreviation=True)
                    _register(["tp", abbr], is_abbreviation=True)

                    split_abbr = cast(list[str], re.findall(r"[a-z]+|\d+", abbr))
                    if len(split_abbr) > 1 and all(split_abbr):
                        _register(split_abbr, is_abbreviation=True)

        return signature

    def _extract_street_address(
        self,
        original: str,
        node: AddressParser.AddressNode,
        component_aliases: dict[AdministrativeLevel, list[str]] | None = None,
    ) -> str:
        if not original:
            return ""

        alias_map = component_aliases or {}
        profiles: dict[AdministrativeLevel, ComponentSignature] = {
            "province": self._build_component_signature(
                node.province_name,
                alias_map.get("province"),
            ),
            "district": self._build_component_signature(
                node.district_name,
                alias_map.get("district"),
            ),
            "ward": self._build_component_signature(
                node.ward_name,
                alias_map.get("ward"),
            ),
        }

        if not any(profile["sequences"] for profile in profiles.values()):
            return self._cleanup_street_address_result(original.strip())

        token_matches = list(re.finditer(self._RE_WORD_TOKEN, original, flags=re.UNICODE))
        if not token_matches:
            return self._cleanup_street_address_result(original.strip())

        tokens: list[StreetToken] = []
        for match in token_matches:
            norm = self._normalize_token_basic(match.group(0))
            tokens.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "raw": match.group(0),
                    "norm": norm,
                }
            )

        token_count = len(tokens)
        if token_count == 0:
            return self._cleanup_street_address_result(original.strip())
        protected_generics: set[int] = set()
        for idx in range(1, token_count):
            prev_norm = tokens[idx - 1]["norm"]
            curr_norm = tokens[idx]["norm"]
            if curr_norm == "xa" and prev_norm in {"cu", "khu"}:
                protected_generics.add(idx)

        # Pre-compute comma/dash-separated segments so we can avoid crossing them
        segments: list[tuple[int, int]] = []
        segment_token_indices: list[list[int]] = []
        token_segments: list[int] = [-1] * token_count
        if token_count > 0:
            separator_matches = list(re.finditer(r"[,;\n]+|\s+[-–—]\s+", original))
            if separator_matches:
                start_char = 0
                for match in separator_matches:
                    segments.append((start_char, match.start()))
                    start_char = match.end()
                segments.append((start_char, len(original)))
            else:
                segments.append((0, len(original)))

            for seg_start, seg_end in segments:
                segment_token_indices.append([])
            for token_idx, token in enumerate(tokens):
                for seg_idx, (seg_start, seg_end) in enumerate(segments):
                    if seg_start <= token["start"] < seg_end:
                        segment_token_indices[seg_idx].append(token_idx)
                        token_segments[token_idx] = seg_idx
                        break

        indices_to_remove: set[int] = set()

        def _is_generic(idx: int) -> bool:
            if idx in protected_generics:
                return False
            token = tokens[idx]
            return self._is_generic_location_token(token.get("raw"), token.get("norm"))

        def _is_admin_generic(idx: int) -> bool:
            if idx < 0 or idx >= token_count:
                return False
            if not _is_generic(idx):
                return False
            token_norm = tokens[idx]["norm"]
            if token_norm in self._ADMIN_GENERIC_TOKENS:
                return True
            if token_norm == "pho":
                prev_idx = idx - 1
                if prev_idx >= 0 and _same_segment(idx, prev_idx):
                    prev_norm = tokens[prev_idx]["norm"]
                    if prev_norm == "thanh":
                        return True
            return False

        def _same_segment(idx_a: int, idx_b: int) -> bool:
            if idx_a < 0 or idx_b < 0 or idx_a >= token_count or idx_b >= token_count:
                return False
            seg_a = token_segments[idx_a]
            seg_b = token_segments[idx_b]
            if seg_a == -1 or seg_b == -1:
                return True
            return seg_a == seg_b

        def _sequence_has_generic_tokens(seq_tokens: list[str]) -> bool:
            for token in seq_tokens:
                if token in self._GENERIC_LOCATION_TOKENS:
                    return True
            return False

        def _adjacent_generic(idx: int, direction: int) -> bool:
            neighbor = idx + direction
            if neighbor < 0 or neighbor >= token_count:
                return False
            if not _same_segment(idx, neighbor):
                return False
            return _is_admin_generic(neighbor)

        street_descriptor_tokens: set[str] = {
            "duong",
            "d",
            "dg",
            "ngo",
            "ngach",
            "hem",
            "hxh",
            "tuyen",
            "ql",
            "quoclo",
            "tl",
            "tinhlo",
            "dailo",
            "truc",
            "khu",
            "kp",
            "kdc",
            "kdt",
            "to",
            "ap",
            "thon",
        }
        locality_descriptor_tokens: set[str] = {
            "thon",
            "xom",
            "ap",
            "to",
            "kp",
            "khu",
            "kdc",
            "kdt",
        }

        def _sequence_has_street_descriptor(seq_tokens: list[str]) -> bool:
            for token in seq_tokens:
                if token in street_descriptor_tokens:
                    return True
            return False

        def _has_street_descriptor_before(idx: int) -> bool:
            prev_idx = idx - 1
            if prev_idx < 0 or prev_idx >= token_count:
                return False
            if not _same_segment(idx, prev_idx):
                return False
            prev_norm = tokens[prev_idx]["norm"]
            return bool(prev_norm and prev_norm in street_descriptor_tokens)

        def _looks_like_street_designator(idx: int) -> bool:
            if idx < 0 or idx >= token_count:
                return False
            token_norm = tokens[idx]["norm"]
            if not token_norm:
                return False
            if _has_street_descriptor_before(idx):
                return True
            next_norm = None
            next_next_norm = None
            if idx + 1 < token_count and _same_segment(idx, idx + 1):
                next_norm = tokens[idx + 1]["norm"]
            if idx + 2 < token_count and _same_segment(idx, idx + 2):
                next_next_norm = tokens[idx + 2]["norm"]
            if next_norm and next_norm.isdigit():
                return True
            if next_norm == "so" and next_next_norm and next_next_norm.isdigit():
                return True
            return False

        def _segment_match_ratio(segment_idx: int, start_idx: int, length: int) -> float:
            if segment_idx < 0 or segment_idx >= len(segment_token_indices):
                return 0.0
            segment_tokens = segment_token_indices[segment_idx]
            if not segment_tokens:
                return 0.0
            end_idx = start_idx + length
            covered = sum(1 for token_idx in segment_tokens if start_idx <= token_idx < end_idx)
            return covered / max(1, len(segment_tokens))

        def _segment_matches_profile(segment_idx: int, profile_sequences: list[list[str]]) -> bool:
            if segment_idx < 0 or segment_idx >= len(segment_token_indices):
                return False
            seg_token_indices = segment_token_indices[segment_idx]
            if not seg_token_indices:
                return False
            seg_tokens = [tokens[token_idx]["norm"] for token_idx in seg_token_indices]
            seg_tokens = [token for token in seg_tokens if token]
            if not seg_tokens:
                return False

            seg_len = len(seg_tokens)
            for sequence in profile_sequences:
                seq = [item for item in sequence if item]
                seq_len = len(seq)
                if seq_len == 0 or seq_len > seg_len:
                    continue
                for start in range(seg_len - seq_len + 1):
                    if all(seg_tokens[start + pos] == seq[pos] for pos in range(seq_len)):
                        coverage = seq_len / max(1, seg_len)
                        if coverage >= 0.6:
                            return True
            return False

        def _has_downstream_admin_signal(segment_idx: int) -> bool:
            for next_segment_idx in range(segment_idx + 1, len(segment_token_indices)):
                segment_indices = segment_token_indices[next_segment_idx]
                if not segment_indices:
                    continue
                if any(_is_admin_generic(token_idx) for token_idx in segment_indices):
                    return True

                for profile in profiles.values():
                    profile_sequences: list[list[str]] = profile.get("sequences", [])
                    if _segment_matches_profile(next_segment_idx, profile_sequences):
                        return True
            return False

        def _segment_has_level_prefix(segment_idx: int, level: AdministrativeLevel) -> bool:
            if segment_idx < 0 or segment_idx >= len(segment_token_indices):
                return False
            seg_indices = segment_token_indices[segment_idx]
            if not seg_indices:
                return False
            seg_tokens = [tokens[token_idx]["norm"] for token_idx in seg_indices]
            seg_tokens = [token for token in seg_tokens if token]
            if not seg_tokens:
                return False
            first = seg_tokens[0]
            second = seg_tokens[1] if len(seg_tokens) >= 2 else ""
            pair = f"{first} {second}".strip() if second else ""

            if level == "province":
                return first in {"tinh", "tp"} or pair == "thanh pho"
            if level == "district":
                return first in {"quan", "q", "huyen", "h", "tx"} or pair == self._LIT_THI_XA
            if level == "ward":
                return first in {"phuong", "p", "xa", "x", "tt"} or pair in {
                    self._LIT_THI_TRAN,
                    self._LIT_DAC_KHU,
                }
            return False

        def _segment_has_street_descriptor(segment_idx: int) -> bool:
            if segment_idx < 0 or segment_idx >= len(segment_token_indices):
                return False
            seg_indices = segment_token_indices[segment_idx]
            for token_idx in seg_indices:
                token_norm = tokens[token_idx]["norm"]
                if token_norm in street_descriptor_tokens:
                    return True
            return False

        def _sequence_is_segment_suffix(
            segment_idx: int,
            start_idx: int,
            length: int,
        ) -> bool:
            if segment_idx < 0 or segment_idx >= len(segment_token_indices) or length <= 0:
                return False
            seg_indices = segment_token_indices[segment_idx]
            if length > len(seg_indices):
                return False
            return seg_indices[-length:] == list(range(start_idx, start_idx + length))

        def _segment_prefix_has_locality_descriptor(
            segment_idx: int,
            start_idx: int,
        ) -> bool:
            if segment_idx < 0 or segment_idx >= len(segment_token_indices):
                return False
            seg_indices = segment_token_indices[segment_idx]
            for token_idx in seg_indices:
                if token_idx >= start_idx:
                    break
                token_norm = tokens[token_idx]["norm"]
                if token_norm in locality_descriptor_tokens:
                    return True
            return False

        def _profile_segment_score(
            segment_idx: int,
            profile_name: AdministrativeLevel,
            profile_sequences: list[list[str]],
        ) -> float:
            if segment_idx < 0 or segment_idx >= len(segment_token_indices):
                return 0.0
            seg_indices = segment_token_indices[segment_idx]
            if not seg_indices:
                return 0.0

            segment_len = len(seg_indices)
            best_coverage = 0.0
            for sequence in profile_sequences:
                seq = [item for item in sequence if item]
                seq_len = len(seq)
                if seq_len == 0 or seq_len > segment_len:
                    continue
                for start in range(segment_len - seq_len + 1):
                    matches = True
                    for offset in range(seq_len):
                        token_idx = seg_indices[start + offset]
                        if tokens[token_idx]["norm"] != seq[offset]:
                            matches = False
                            break
                    if matches:
                        coverage = seq_len / max(1, segment_len)
                        if coverage > best_coverage:
                            best_coverage = coverage
            if best_coverage == 0.0:
                return 0.0

            score = best_coverage
            if _segment_has_level_prefix(segment_idx, profile_name):
                score += 0.35
            if any(_is_admin_generic(token_idx) for token_idx in seg_indices):
                score += 0.1
            if (
                profile_name == "ward"
                and _segment_has_street_descriptor(segment_idx)
                and not _segment_has_level_prefix(segment_idx, profile_name)
            ):
                score -= 0.25
            return score

        def mark_indices(start_idx: int, length: int) -> bool:
            if length <= 0:
                return False
            end_idx = start_idx + length
            if end_idx > token_count:
                return False
            window = tokens[start_idx:end_idx]
            if any(not token["norm"] for token in window):
                return False
            if (
                tokens[start_idx]["norm"] == "xa"
                and start_idx > 0
                and _same_segment(start_idx, start_idx - 1)
                and tokens[start_idx - 1]["norm"] in {"cu", "khu"}
            ):
                return False
            if length == 1 and window[0]["norm"].isdigit():
                prev_generic = (
                    start_idx > 0
                    and _same_segment(start_idx, start_idx - 1)
                    and _is_generic(start_idx - 1)
                )
                next_generic = (
                    end_idx < token_count
                    and _same_segment(start_idx, end_idx)
                    and _is_generic(end_idx)
                )
                if not (prev_generic or next_generic):
                    return False
            indices_to_remove.update(range(start_idx, end_idx))

            segment_id = token_segments[start_idx] if start_idx < token_count else -1
            prev_idx = start_idx - 1
            while (
                prev_idx >= 0
                and (segment_id == -1 or token_segments[prev_idx] == segment_id)
                and _is_generic(prev_idx)
            ):
                indices_to_remove.add(prev_idx)
                prev_idx -= 1

            next_idx = end_idx
            while (
                next_idx < token_count
                and (segment_id == -1 or token_segments[next_idx] == segment_id)
                and _is_generic(next_idx)
            ):
                indices_to_remove.add(next_idx)
                next_idx += 1
            return True

        best_segment_by_profile: dict[AdministrativeLevel, tuple[int, float]] = {}
        for profile_name, profile in profiles.items():
            profile_sequences = profile.get("sequences", [])
            best_segment = -1
            best_score = 0.0
            for seg_idx in range(len(segment_token_indices)):
                score = _profile_segment_score(seg_idx, profile_name, profile_sequences)
                if score <= 0.0:
                    continue
                if score > best_score or (score == best_score and seg_idx > best_segment):
                    best_segment = seg_idx
                    best_score = score
            best_segment_by_profile[profile_name] = (best_segment, best_score)

        for profile_name, profile in profiles.items():
            sequences: list[list[str]] = profile["sequences"]
            abbreviation_sequences: set[tuple[str, ...]] = profile.get(
                "abbreviation_sequences", set()
            )
            for seq in sequences:
                seq = [item for item in seq if item]
                seq_len = len(seq)
                if seq_len == 0:
                    continue
                seq_tuple = tuple(seq)
                is_abbreviation_seq = seq_tuple in abbreviation_sequences
                for idx in range(token_count - seq_len + 1):
                    window = tokens[idx : idx + seq_len]
                    if all(window[pos]["norm"] == seq[pos] for pos in range(seq_len)):
                        segment_idx = token_segments[idx]
                        allow_removal = False
                        if _sequence_has_generic_tokens(seq):
                            allow_removal = True
                        elif _adjacent_generic(idx, -1) or _adjacent_generic(idx + seq_len - 1, 1):
                            allow_removal = True
                        else:
                            is_tail_segment = segment_idx >= 0 and segment_idx >= len(segments) - 1
                            if is_tail_segment:
                                coverage = _segment_match_ratio(segment_idx, idx, seq_len)
                                if coverage >= 0.6:
                                    allow_removal = True
                            elif segment_idx == 0:
                                coverage = _segment_match_ratio(segment_idx, idx, seq_len)
                                best_segment_idx = best_segment_by_profile.get(
                                    profile_name, (-1, 0.0)
                                )[0]
                                if (
                                    coverage >= 0.95
                                    and not _sequence_has_street_descriptor(seq)
                                    and _has_downstream_admin_signal(segment_idx)
                                    and best_segment_idx == segment_idx
                                ):
                                    allow_removal = True
                                elif (
                                    profile_name == "ward"
                                    and best_segment_idx == segment_idx
                                    and _has_downstream_admin_signal(segment_idx)
                                    and _sequence_is_segment_suffix(segment_idx, idx, seq_len)
                                    and _segment_prefix_has_locality_descriptor(segment_idx, idx)
                                    and not _sequence_has_street_descriptor(seq)
                                ):
                                    # OCR often drops the comma before the ward in
                                    # patterns like "Thôn 1B Hòa Tiến, Krông Pắc,
                                    # Đắk Lắk". When a locality/sub-address
                                    # descriptor (thôn/ấp/tổ/...) appears earlier in
                                    # the first segment and the matched ward alias is
                                    # the trailing suffix of that segment, prefer
                                    # stripping the trailing ward even without an
                                    # explicit separator.
                                    allow_removal = True
                            elif segment_idx > 0:
                                coverage = _segment_match_ratio(segment_idx, idx, seq_len)
                                if coverage >= 0.6:
                                    allow_removal = True
                        if allow_removal:
                            skip_seq = False
                            if _has_street_descriptor_before(idx):
                                skip_seq = True
                            if (
                                is_abbreviation_seq
                                and seq_len == 1
                                and seq[0] in self._GENERIC_LOCATION_TOKENS
                                and seq[0] not in self._ADMIN_GENERIC_TOKENS
                            ):
                                has_prev_generic = _adjacent_generic(idx, -1)
                                has_next_generic = _adjacent_generic(idx + seq_len - 1, 1)
                                if not (has_prev_generic or has_next_generic):
                                    skip_seq = True
                                elif _looks_like_street_designator(idx):
                                    skip_seq = True
                            if skip_seq:
                                continue
                            mark_indices(idx, seq_len)

        if len(segments) > 1:
            for seg_idx, idx_list in enumerate(segment_token_indices):
                if seg_idx == 0 or not idx_list:
                    continue
                has_generic = any(_is_generic(token_idx) for token_idx in idx_list)
                has_marked = any(token_idx in indices_to_remove for token_idx in idx_list)
                if not (has_generic or has_marked):
                    continue
                should_remove = all(
                    _is_generic(token_idx) or token_idx in indices_to_remove
                    for token_idx in idx_list
                )
                if should_remove:
                    indices_to_remove.update(idx_list)

        def _strip_exact_admin_suffix(value: str) -> str:
            if not value or re.search(r"[,;\n]|\s+[-–—]\s+", original):
                return value
            suffix_profiles: list[list[str]] = []
            for profile in profiles.values():
                for sequence in profile.get("sequences", []):
                    seq = [token for token in sequence if token]
                    if seq:
                        suffix_profiles.append(seq)
            suffix_profiles.sort(key=len, reverse=True)
            if not suffix_profiles:
                return value
            current = value
            while current:
                token_matches_local = list(
                    re.finditer(self._RE_WORD_TOKEN, current, flags=re.UNICODE)
                )
                if not token_matches_local:
                    break
                token_norms_local = [
                    self._normalize_token_basic(match.group(0)) for match in token_matches_local
                ]
                removed = False
                for sequence in suffix_profiles:
                    seq_len = len(sequence)
                    if seq_len > len(token_norms_local):
                        continue
                    if token_norms_local[-seq_len:] != sequence:
                        continue
                    cut_pos = token_matches_local[-seq_len].start()
                    current = current[:cut_pos].rstrip(self._LIT_STRIP_CHARS)
                    removed = True
                    break
                if not removed:
                    break
            return current

        if not indices_to_remove:
            return self._cleanup_street_address_result(
                _strip_exact_admin_suffix(original.strip()).strip()
            )

        mask = [False] * len(original)
        for token_idx in indices_to_remove:
            start = tokens[token_idx]["start"]
            end = tokens[token_idx]["end"]
            for pos in range(start, end):
                mask[pos] = True

        filtered_chars = [ch for pos, ch in enumerate(original) if not mask[pos]]
        street = "".join(filtered_chars)
        street = re.sub(r"(?<!\w)[\'’`´]+|[\'’`´]+(?!\w)", " ", street)
        street = re.sub(r"[,\.;:]+\s*", " ", street)
        street = re.sub(r"\s+[-–—]+\s+", " ", street)
        street = re.sub(r"\s+", " ", street).strip(self._LIT_STRIP_CHARS)
        if street:
            street = re.sub(
                r"(?i)\bvi\S*t[\s-]*nam\b\.?$",
                "",
                street,
            ).strip(self._LIT_STRIP_CHARS)
            street = re.sub(
                r"(?i)(?:^|\s)(?:t|tp|q|h|x|p|tt|tx)\.?$",
                "",
                street,
            ).strip(self._LIT_STRIP_CHARS)
            street = _strip_exact_admin_suffix(street)
        return self._cleanup_street_address_result(street.strip())

    def _cleanup_street_address_result(self, street: str | None) -> str:
        if not street:
            return ""
        cleaned = street.strip(self._LIT_STRIP_CHARS)
        if not cleaned:
            return ""

        parts = [
            part.strip(self._LIT_STRIP_CHARS)
            for part in cast(list[str], re.split(r"[;,]", cleaned))
        ]
        while parts:
            tail = parts[-1]
            tail_std = self.standardize_name(tail, "basic")
            tail_compact = re.sub(r"\s+", "", tail_std or "")
            if tail_compact not in {"vietnam", "vn"}:
                break
            parts.pop()
        if parts:
            cleaned = ", ".join(part for part in parts if part)
        elif re.sub(r"\s+", "", self.standardize_name(cleaned, "basic") or "") in {
            "vietnam",
            "vn",
        }:
            cleaned = ""

        return cleaned

    def generate_ngrams(self, s: str, n: int = 4) -> list[str]:
        s = f" {s} "  # Thêm khoảng trắng ở đầu và cuối để tạo n-gram chính xác
        ngrams = [s[i : i + n] for i in range(len(s) - n + 1)]
        return ngrams

    def generate_ngram_inverted_index(
        self,
        ngram_list: Iterable[str],
        index: int,
        invert_ngram_to_index_dict: dict[str, set[int]],
    ) -> None:
        for ngram in ngram_list:
            if ngram not in invert_ngram_to_index_dict:
                invert_ngram_to_index_dict[ngram] = set()
            invert_ngram_to_index_dict[ngram].add(index)

    def ngram_address_piece_list(self, input_ngram_list: list[str], top_k: int) -> list[NgramHit]:
        packed_index = self._packed_ngram_index
        if packed_index is None:
            raise RuntimeError("packed n-gram index is not initialized")
        return packed_index.top_hits(input_ngram_list, top_k)

    def _rebuild_packed_ngram_index(self) -> None:
        self._packed_ngram_index = PackedNgramIndex(
            self.invert_ngrams_idx,
            len(self.address_node_list),
        )

    # --------------------
    # Prefix detection + prefilter
    # --------------------
    def _refresh_detection_choices(self) -> None:
        self._province_detection_choices = tuple(sorted(self.province_names_std))
        self._district_detection_choices = tuple(
            sorted(
                candidate
                for candidate in self.district_names_std
                if not candidate.isdigit() and len(candidate) >= 3
            )
        )
        self._ward_detection_choices = tuple(sorted(self.ward_names_std))
        self._fuzzy_choice_profiles = {
            choice: self._build_fuzzy_choice_profile(choice)
            for choice in (self.province_names_std | self.district_names_std | self.ward_names_std)
            if choice
        }

    def _build_fuzzy_choice_profile(self, choice: str) -> FuzzyChoiceProfile:
        core = self._strip_generic_prefix(choice) or choice
        tokens = core.split()
        return (
            core,
            "".join(character for character in choice if character.isdigit()),
            len(tokens),
            len(core),
            tokens[0] if tokens else "",
            tokens[-1] if tokens else "",
        )

    def _detect_by_prefix(self, s: str) -> DetectedComponents:
        # s should be standardized without advanced removal (to keep prefix words)
        if not s:
            return None, None, None

        # When we already have explicit segment separators (we inject `|` between
        # comma-separated parts), do not use token boundaries like `tinh`/`tp`.
        # Those tokens can legitimately appear inside names (e.g. "Sơn Tịnh"),
        # causing premature truncation (e.g. "son" -> fuzzy-match "son la").
        has_segment_separators = "|" in s
        if has_segment_separators:
            province_tinh_pref = self._RE_PREFIX_PROVINCE_SEGMENTED
            city_pref = self._RE_PREFIX_CITY_SEGMENTED
            district_pref = self._RE_PREFIX_DISTRICT_SEGMENTED
            ward_pref = self._RE_PREFIX_WARD_SEGMENTED
        else:
            province_tinh_pref = self._RE_PREFIX_PROVINCE_INLINE
            city_pref = self._RE_PREFIX_CITY_INLINE
            district_pref = self._RE_PREFIX_DISTRICT_INLINE
            ward_pref = self._RE_PREFIX_WARD_INLINE

        def _digit_key(value: str) -> str:
            return "".join(ch for ch in value if ch.isdigit())

        def _pick_best(
            fragment: str,
            choices: tuple[str, ...],
            *,
            cutoff: int = 84,
        ) -> str | None:
            fragment = fragment.strip()
            if not fragment:
                return None
            # Limit fragment to first 3 tokens to avoid swallowing next parts
            tokens = fragment.split()
            if len(tokens) <= 4:
                fragment = " ".join(tokens)
            elif len(tokens) > 3 and len(tokens[3]) == 1:
                fragment = " ".join(tokens[:4])
            else:
                fragment = " ".join(tokens[:3])
            return self._fuzzy_match_component_key(fragment, choices, cutoff=cutoff)

        def _trim_province_fragment(fragment: str) -> str:
            if not fragment:
                return fragment
            tokens = [tok for tok in fragment.split() if tok]
            if not tokens:
                return fragment
            for marker in ("tinh", "province"):
                if marker in tokens:
                    idx = tokens.index(marker)
                    if idx + 1 < len(tokens):
                        return " ".join(tokens[idx + 1 :])
            return " ".join(tokens)

        prov = dist = ward = None
        if self.province_names_std:
            province_choices = self._province_detection_choices

            def _is_central_municipality(candidate_std: str) -> bool:
                if not candidate_std:
                    return False
                province_id_new = self._lookup_new_province_id_by_name(candidate_std)
                if not province_id_new:
                    return False
                record = self.external_new_province_records.get(
                    province_id_new
                ) or self.new_province_records.get(province_id_new)
                return bool(isinstance(record, dict) and record.get("administrative_unit_id") == 1)

            def _looks_like_road_prefix(tokens: list[str]) -> bool:
                if not tokens:
                    return False
                return tokens[0] == "lo"

            for match in province_tinh_pref.finditer(s):
                fragment = cast(str, match.group(1) or "").strip()
                fragment = _trim_province_fragment(fragment)
                frag_tokens = [tok for tok in fragment.split() if tok]
                if len(frag_tokens) == 1 and len(frag_tokens[0]) <= 2:
                    fragment = ""
                    frag_tokens = []
                while frag_tokens and frag_tokens[-1] in {"viet", "nam", "vietnam"}:
                    frag_tokens.pop()
                if _looks_like_road_prefix(frag_tokens):
                    continue
                fragment = " ".join(frag_tokens)
                if fragment in {"hcm", "hcmc", "sai gon", "saigon", "sg"} or (
                    frag_tokens and frag_tokens[0] in {"hcm", "hcmc", "sg"}
                ):
                    prov = "ho chi minh"
                else:
                    prov = _pick_best(fragment, province_choices, cutoff=84)
                if prov:
                    break

            if not prov:
                m_city = city_pref.search(s)
                if m_city:
                    fragment = cast(str, m_city.group(1) or "").strip()
                    fragment = _trim_province_fragment(fragment)
                    frag_tokens = [tok for tok in fragment.split() if tok]
                    while frag_tokens and frag_tokens[-1] in {"viet", "nam", "vietnam"}:
                        frag_tokens.pop()
                    fragment = " ".join(frag_tokens)
                    if fragment in {"hcm", "hcmc", "sai gon", "saigon", "sg"} or (
                        frag_tokens and frag_tokens[0] in {"hcm", "hcmc", "sg"}
                    ):
                        prov = "ho chi minh"
                    else:
                        candidate = _pick_best(fragment, province_choices, cutoff=84)
                        if candidate and _is_central_municipality(candidate):
                            prov = candidate

        dist_num: str | None = None
        district_choices = self._district_detection_choices
        if district_choices:
            m_num = self._RE_NUMERIC_DISTRICT.search(s)
            if m_num:
                # Avoid false positives from lot codes like "Lô Q10-03" where "Q10" is not "Quận 10".
                prefix_context = s[max(0, m_num.start() - 8) : m_num.start()]
                if self._RE_LOT_PREFIX.search(prefix_context):
                    m_num = None
            if m_num:
                raw = cast(str, m_num.group(1)).strip()
                for candidate in (
                    f"quan {raw}",
                    f"district {raw}",
                    raw,
                ):
                    if candidate in self.district_names_std:
                        dist_num = candidate
                        break

            def _district_prefix_priority(prefix: str) -> int:
                priority_map = {
                    "quan": 5,
                    "q": 5,
                    "huyen": 4,
                    "h": 4,
                    self._LIT_THI_XA: 4,
                    "tx": 4,
                    self._LIT_THANH_PHO: 2,
                    "tp": 2,
                }
                return priority_map.get(prefix, 1)

            def _is_province_like(candidate: str, prefix: str) -> bool:
                if not candidate:
                    return False
                if prov and candidate == prov:
                    return True
                if prefix in {self._LIT_THANH_PHO, "tp"}:
                    return candidate in self.province_names_std
                return False

            best_dist: str | None = None
            best_priority = -1
            best_score = -1.0
            best_pos = -1
            for match in district_pref.finditer(s):
                fragment = cast(str, match.group("fragment") or "").strip()
                prefix = cast(str, match.group("prefix") or "").strip()
                if not fragment or not prefix:
                    continue
                # "TP/Thành phố" segments frequently denote the province-level municipality
                # ("TP Hà Nội", "TP Đà Nẵng", ...). When the fragment matches a known province,
                # do not treat it as a district hint.
                if prefix in {self._LIT_THANH_PHO, "tp"}:
                    if fragment in self.province_names_std:
                        continue
                    if prov and partial_ratio(fragment, prov) >= 90:
                        continue
                    if self._lookup_new_province_id_by_name(fragment):
                        continue
                # Avoid false positives from road names like "đường huyện 74":
                # a numeric fragment after "huyện/tx/..." is not a valid district name.
                if fragment.isdigit() and prefix not in {"quan", "q"}:
                    continue
                # Skip false positives where the fragment immediately starts with a ward token
                frag_tokens = fragment.split()
                frag_first_token = frag_tokens[0] if frag_tokens else ""
                frag_second_token = frag_tokens[1] if len(frag_tokens) >= 2 else ""
                if frag_first_token in {"phuong", "p", "xa", "x"}:
                    continue
                if frag_first_token == "thi" and frag_second_token == "tran":
                    continue
                candidate = _pick_best(fragment, district_choices, cutoff=84)
                if not candidate:
                    continue
                if _is_province_like(candidate, prefix):
                    continue
                priority = _district_prefix_priority(prefix)
                score = partial_ratio(fragment, candidate)
                if priority < best_priority:
                    continue
                if priority == best_priority:
                    if score < best_score:
                        continue
                    if score == best_score and best_pos >= 0 and match.start() >= best_pos:
                        continue
                best_dist = candidate
                best_priority = priority
                best_score = score
                best_pos = match.start()
            if best_dist:
                dist = best_dist
        if dist_num:
            dist = dist_num

        if self.ward_names_std:

            def _ward_prefix_priority(prefix: str) -> int:
                priority_map = {
                    self._LIT_DAC_KHU: 4,
                    "p": 3,
                    "phuong": 3,
                    "tt": 2,
                    self._LIT_THI_TRAN: 2,
                    self._LIT_THI_XA: 2,
                    "xa": 1,
                    "x": 1,
                }
                return priority_map.get(prefix, 0)

            def _preceding_token(start_index: int) -> str | None:
                prefix_slice = s[:start_index].rstrip()
                if not prefix_slice:
                    return None
                return prefix_slice.split().pop() if prefix_slice else None

            hamlet_prefix_blockers = {
                "thon",
                "xom",
                "ap",
                "to",
                "khu",
                "kp",
                "kdc",
                "ngo",
                "ngach",
                "hem",
                "thonxom",
            }

            best_priority = -1
            prefix_normalize_map = {
                "p": "phuong",
                "phuong": "phuong",
                "ward": "ward",
                "tt": self._LIT_THI_TRAN,
                self._LIT_THI_TRAN: self._LIT_THI_TRAN,
                "town": self._LIT_THI_TRAN,
                self._LIT_THI_XA: self._LIT_THI_XA,
                "xa": "xa",
                "x": "xa",
                "commune": "xa",
                self._LIT_DAC_KHU: self._LIT_DAC_KHU,
                "special administrative region": self._LIT_DAC_KHU,
            }

            def _try_prefixed_candidate(prefix: str, fragment: str) -> str | None:
                canonical = prefix_normalize_map.get(prefix)
                if not canonical:
                    return None
                normalized_fragment = fragment
                prefix_token = f"{canonical} "
                if normalized_fragment.startswith(prefix_token):
                    normalized_fragment = normalized_fragment[len(prefix_token) :].strip()
                fused = f"{canonical} {normalized_fragment}".strip()
                fused = self._RE_WHITESPACE.sub(" ", fused)
                if fused in self.ward_names_std:
                    return fused
                if normalized_fragment.isdigit():
                    digits = normalized_fragment.lstrip("0") or "0"
                    fused_digits = f"{canonical} {digits}".strip()
                    if fused_digits in self.ward_names_std:
                        return fused_digits
                return None

            for m in ward_pref.finditer(s):
                fragment = cast(str, m.group("fragment") or "").strip()
                prefix = cast(str, m.group("prefix") or "").strip()
                if not fragment or not prefix:
                    continue
                blocker = _preceding_token(m.start())
                # When we inject `|` separators between comma-separated segments, do not let
                # the previous segment's tail token (e.g. "tổ/khu/ấp") suppress a ward prefix
                # in the next segment.
                if has_segment_separators and m.start() < len(s) and s[m.start()] == "|":
                    blocker = None
                if blocker and blocker in hamlet_prefix_blockers:
                    continue
                if prefix == "xa":
                    prev_token = _preceding_token(m.start())
                    if prev_token == "cu":
                        continue
                if prefix in (self._LIT_DAC_KHU, "special administrative region"):
                    frag_tokens = fragment.split()
                    trimmed: list[str] = []
                    for token in frag_tokens:
                        if token in self._GENERIC_LOCATION_TOKENS:
                            break
                        trimmed.append(token)
                        if len(trimmed) >= 3:
                            break
                    limited = " ".join(trimmed)
                    fragment = f"{prefix} {limited}".strip() if limited else prefix
                candidate = _try_prefixed_candidate(prefix, fragment)
                if not candidate:
                    candidate = _pick_best(
                        fragment,
                        self._ward_detection_choices,
                        cutoff=84,
                    )
                if not candidate:
                    continue
                priority = _ward_prefix_priority(prefix)
                if priority < best_priority:
                    continue
                if priority == best_priority and ward is not None:
                    continue
                ward = candidate
                best_priority = priority

        return prov, dist, ward

    def _detect_special_province_token(self, standardized_basic: str) -> str | None:
        """
        Detect legacy province aliases (e.g. 'thua thien hue') directly from the
        standardized string when the user omits administrative prefixes.
        """
        if not standardized_basic:
            return None
        if re.search(r"\bhcmc?\b", standardized_basic):
            return "ho chi minh"
        for synonyms, _ in SPECIAL_PROVINCE_MAP.items():
            if isinstance(synonyms, (list, tuple, set)):
                candidates = synonyms
            else:
                candidates = (synonyms,)
            for alias in candidates:
                alias_std = self.standardize_name(alias, "basic")
                if alias_std and alias_std in standardized_basic:
                    return alias_std
        return None

    def _detect_suffix_province_token(self, standardized_basic: str) -> str | None:
        """Infer province tokens that appear without prefixes at the tail of the input."""
        if not standardized_basic or not self.province_names_std:
            return None

        tokens = standardized_basic.split()
        if not tokens:
            return None

        # Drop trailing 'Việt Nam' if present to avoid false positives.
        if len(tokens) >= 2 and tokens[-2:] == ["viet", "nam"]:
            tokens = tokens[:-2]
        if not tokens:
            return None

        # Skip trailing generic tokens such as 'city', 'province', etc.
        trimmed: list[str] = list(tokens)
        while trimmed and trimmed[-1] in self._GENERIC_LOCATION_TOKENS:
            trimmed.pop()
        if not trimmed:
            return None

        max_window = min(4, len(trimmed))
        for window in range(max_window, 1, -1):
            fragment = " ".join(trimmed[-window:])
            if fragment in self.province_names_std:
                return fragment
            fuzzy_match = self._fuzzy_match_component_key(
                fragment, self.province_names_std, cutoff=88
            )
            if fuzzy_match:
                return fuzzy_match

        last_token = trimmed[-1]
        if len(last_token) >= 4 and last_token in self.province_names_std:
            return last_token

        return None

    def _prefilter_by_prefix(self, standardized_basic: str) -> list[int]:
        prov, dist, ward = self._detect_by_prefix(standardized_basic)
        candidates: set[int] | None = None

        def _merge(current: set[int] | None, newset: set[int]) -> set[int] | None:
            if not newset:
                return current
            return set(newset) if current is None else (current & newset if current else set())

        if ward:
            candidates = _merge(candidates, self.invert_ward_to_indices.get(ward, set()))
        if dist:
            candidates = _merge(candidates, self.invert_district_to_indices.get(dist, set()))
        if prov:
            candidates = _merge(candidates, self.invert_province_to_indices.get(prov, set()))

        # If nothing detected, return empty list to signal fallback to n-gram path
        if not candidates:
            return []
        # Return stable list of indices
        return sorted(candidates)

    def _select_candidate_with_hints(
        self,
        candidates: list[CandidateHit],
        detected_components: DetectedComponents,
    ) -> int | None:
        if not candidates:
            return None
        prov_hint, dist_hint, ward_hint = (
            detected_components if detected_components else (None, None, None)
        )
        if not any((prov_hint, dist_hint, ward_hint)):
            return candidates[0][0]

        def _norm(value: str | None) -> str | None:
            if not value:
                return None
            normalized = self.standardize_name(value, "basic")
            return normalized or None

        prov_hint = _norm(prov_hint)
        dist_hint = _norm(dist_hint)
        ward_hint = _norm(ward_hint)

        best_idx: int | None = None
        best_key: tuple[int, int, int, float] | None = None

        for idx, score, _ in candidates:
            node = self.address_node_list[idx]
            node_prov = _norm(node.province_name)
            node_dist = _norm(node.district_name)
            node_ward = _norm(node.ward_name)

            ward_match = 1 if ward_hint and node_ward == ward_hint else 0
            district_match = 1 if dist_hint and node_dist == dist_hint else 0
            province_match = 1 if prov_hint and node_prov == prov_hint else 0

            ward_mismatch = 1 if ward_hint and node_ward and node_ward != ward_hint else 0
            district_mismatch = 1 if dist_hint and node_dist and node_dist != dist_hint else 0
            province_mismatch = 1 if prov_hint and node_prov and node_prov != prov_hint else 0

            ward_missing = 1 if ward_hint and not node_ward else 0
            district_missing = 1 if dist_hint and not node_dist else 0
            province_missing = 1 if prov_hint and not node_prov else 0

            match_score = (ward_match * 6) + (district_match * 3) + province_match
            mismatch_penalty = (ward_mismatch * 6) + (district_mismatch * 3) + province_mismatch
            missing_penalty = (ward_missing * 3) + (district_missing * 2) + province_missing

            key = (
                match_score,
                -mismatch_penalty,
                -missing_penalty,
                score,
            )

            if best_key is None or key > best_key:
                best_idx = idx
                best_key = key

        return best_idx if best_idx is not None else candidates[0][0]

    def address_candidate_list(
        self,
        input_string_standard: str,
        input_ngram_set: set[str],
        ngram_address_piece_list: list[NgramHit],
        partial_input_string: bool,
        detected_components: DetectedComponents,
    ) -> list[CandidateHit]:
        # Stage 1: filter by Dice; collect IDs whose Dice >= gate
        detected_prov, detected_dist, detected_ward = (
            detected_components if detected_components else (None, None, None)
        )

        input_set_length = len(input_ngram_set)
        filtered_entries: list[tuple[int, float]] = []
        dice_entries: list[tuple[int, float]] = []

        for position, (idx, intersection) in enumerate(ngram_address_piece_list, start=1):
            candidate_ngrams = self.address_node_list[idx].ngram_list

            dice_score = (2 * intersection) / (input_set_length + len(candidate_ngrams))
            dice_entries.append((idx, dice_score))

            if dice_score >= self.DICE_GATE:
                filtered_entries.append((idx, dice_score))
            elif position >= 200:
                # Counter is ordered by frequency; dice will only go down after this point
                break
        if not filtered_entries:
            if dice_entries:
                # Fall back to the best-overlapping candidates when Dice is too strict,
                # so long free-text inputs with street info still produce candidates.

                def _dice_sort_key(item: tuple[int, float]) -> float:
                    return item[1]

                filtered_entries = sorted(dice_entries, key=_dice_sort_key, reverse=True)[:80]
            if not filtered_entries:
                return []

        # Optional prefix-based filtering to favour nodes aligned with detected components
        prefix_filter: set[int] | None = None
        if detected_ward:
            prefix_filter = self.invert_ward_to_indices.get(detected_ward, set())
        if detected_dist:
            dist_set = self.invert_district_to_indices.get(detected_dist, set())
            prefix_filter = dist_set if prefix_filter is None else prefix_filter & dist_set
        if detected_prov:
            prov_set = self.invert_province_to_indices.get(detected_prov, set())
            prefix_filter = prov_set if prefix_filter is None else prefix_filter & prov_set

        if prefix_filter:
            prioritized = [entry for entry in filtered_entries if entry[0] in prefix_filter]
            if prioritized:
                nonprior = [entry for entry in filtered_entries if entry[0] not in prefix_filter]
                filtered_entries = prioritized + nonprior

        # Stage 2: richer scoring per-candidate
        scored_candidates: list[tuple[float, float, float, tuple[int, int, int], int]] = []

        def _component_boost(
            candidate_value: str | None,
            detected_value: str | None,
            exact_bonus: float,
            fuzzy_bonus: float,
            missing_penalty: float,
        ) -> float:
            if not detected_value:
                return 0.0
            if not candidate_value:
                return missing_penalty
            cand_std = self.standardize_name(candidate_value, "basic")
            if not cand_std:
                return missing_penalty
            if cand_std == detected_value:
                return exact_bonus
            similarity = ratio(cand_std, detected_value)
            if similarity >= 90:
                return fuzzy_bonus
            if similarity >= 80:
                return fuzzy_bonus / 2
            return missing_penalty

        max_candidates = 120
        input_len = max(len(input_string_standard), 1)
        for idx, dice_score in filtered_entries[:max_candidates]:
            node = self.address_node_list[idx]
            candidate_string = node.standardized_full_name
            if not candidate_string:
                continue

            base_score = ratio(input_string_standard, candidate_string)
            partial_score = partial_ratio(input_string_standard, candidate_string)
            wratio_score = rf_fuzz.WRatio(input_string_standard, candidate_string)

            length_ratio = input_len / max(len(candidate_string), 1)
            use_partial = partial_input_string or length_ratio >= 1.25

            combined = max(base_score, wratio_score)
            if use_partial:
                combined = max(combined, partial_score)
            elif base_score < 80 and partial_score >= 90:
                combined = max(combined, partial_score * 0.95)

            # Blend scores to favour balanced matches
            blended = (0.6 * base_score) + (0.4 * wratio_score)
            combined = max(combined, blended)

            boost = 0.0
            boost += _component_boost(node.ward_name, detected_ward, 18.0, 12.0, -12.0)
            boost += _component_boost(node.district_name, detected_dist, 14.0, 9.0, -10.0)
            boost += _component_boost(node.province_name, detected_prov, 6.0, 3.5, -4.0)

            comps = (
                int(bool(node.province_name))
                + int(bool(node.district_name))
                + int(bool(node.ward_name))
            )
            has_ward = 1 if node.ward_name else 0
            specificity = (comps, has_ward, len(node.standardized_full_name))

            final_score = combined + boost + (comps * 1.5) + (has_ward * 1.0) + (dice_score * 10)
            scored_candidates.append(
                (
                    final_score,
                    combined,
                    boost,
                    specificity,
                    idx,
                )
            )

        if not scored_candidates:
            return []

        def _scored_candidate_sort_key(
            item: tuple[float, float, float, tuple[int, int, int], int],
        ) -> tuple[float, float, tuple[int, int, int], int]:
            return (
                item[0],
                item[1],
                item[3],
                len(self.address_node_list[item[4]].standardized_full_name),
            )

        scored_candidates.sort(
            key=_scored_candidate_sort_key,
            reverse=True,
        )

        top_results: list[CandidateHit] = []
        for final_score, combined, boost, _, idx in scored_candidates[:25]:
            node = self.address_node_list[idx]
            top_results.append((idx, float(final_score), node.full_name))

        return top_results

    def _contains_district_token(self, s: str) -> bool:
        if not s:
            return False
        pattern = re.compile(
            r"\b(quan|q\s*\d+|huyen|thi\s*xa|thi\s*tran|thanh\s*pho|tp|district|county)\b"
        )
        return bool(pattern.search(s))
