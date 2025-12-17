import json
import logging
import os
import pickle
import re
import sys
import unicodedata
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict
from rapidfuzz.fuzz import partial_ratio, ratio
from rapidfuzz import process as rf_process
from rapidfuzz import fuzz as rf_fuzz

logger = logging.getLogger(__name__)

# Make the module import path stable for pickled cache objects.
# The cache may be created when importing as either `inexus_parser` (script usage)
# or `fuzz.inexus_parser` (package usage); alias both to avoid cache invalidation.
if __name__ == "inexus_parser":
    sys.modules.setdefault("fuzz.inexus_parser", sys.modules[__name__])
elif __name__ == "fuzz.inexus_parser":
    sys.modules.setdefault("inexus_parser", sys.modules[__name__])

try:
    from .search_engine import AddressSearchEngine
except ImportError:  # Running as a standalone script without package context
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    from search_engine import AddressSearchEngine

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
    # Legacy alias retained in historical data but removed from official dataset
    "20278": ["An Hải Tây", "Phường An Hải Tây"],
}


class AddressParser:
    _STATEFUL_ATTRS: ClassVar[Tuple[str, ...]] = (
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
    _CACHE_VERSION: ClassVar[int] = 11
    _CACHE_FILENAME: ClassVar[str] = "address_parser.preprocessed.v11.pkl"
    _PREPROCESSED_CACHE: ClassVar[Optional[Dict[str, Any]]] = None
    _PREPROCESSED_SIGNATURE: ClassVar[
        Optional[Tuple[Tuple[str, Optional[float], Optional[int]], ...]]
    ] = None
    _PREPROCESSED_LOCK: ClassVar[Lock] = Lock()

    class AddressNode:
        def __init__(
            self,
            province_name: str,
            district_name: str,
            ward_name: str,
            *,
            province_id: Optional[str] = None,
            district_id: Optional[str] = None,
            ward_id: Optional[str] = None,
            is_new_format: Optional[bool] = None,
        ):
            self.full_name = f"{ward_name} {district_name} {province_name}"
            self.full_name = re.sub(r"\s+", " ", self.full_name).strip()
            self.standardized_full_name = ""
            self.province_name = province_name
            self.district_name = district_name
            self.ward_name = ward_name
            self.ngram_list: Set[str] = set()  # List of n-grams for fuzzy matching
            # None = unknown; True = new 2-level; False = old 3-level
            self.is_new_format: Optional[bool] = is_new_format
            self.province_id: Optional[str] = province_id
            self.district_id: Optional[str] = district_id
            self.ward_id: Optional[str] = ward_id

    _GENERIC_LOCATION_TOKENS: Set[str] = {
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

    _ADMIN_GENERIC_TOKENS: Set[str] = {
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

    _LOCATION_PREFIX_SINGLE: Set[str] = {
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

    _LOCATION_PREFIX_MULTI: Set[str] = {
        "thi tran",
        "thi xa",
        "thanh pho",
        "khu pho",
        "khu vuc",
    }

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(base_dir, "data")
        self._cache_path = os.path.join(self.data_dir, self._CACHE_FILENAME)
        self.new_format_provinces_path = os.path.join(self.data_dir, "provinces.json")
        self.new_format_wards_path = os.path.join(self.data_dir, "wards.json")
        self.new_format_mapping_path = os.path.join(self.data_dir, "ward_mappings.json")
        self.old_provinces_path = os.path.join(self.data_dir, "old_provinces.json")
        self.old_districts_path = os.path.join(self.data_dir, "old_districts.json")
        self.old_wards_path = os.path.join(self.data_dir, "old_wards.json")
        self.legacy_virtual_wards_path = os.path.join(
            self.data_dir, "legacy_virtual_wards.json"
        )

        self.address_node_list: List[AddressParser.AddressNode] = []
        self.invert_ngrams_idx: dict[str, Set[int]] = {}

        # Name-level inverted indexes for fast prefiltering by known names
        self.invert_province_to_indices: Dict[str, Set[int]] = defaultdict(set)
        self.invert_district_to_indices: Dict[str, Set[int]] = defaultdict(set)
        self.invert_ward_to_indices: Dict[str, Set[int]] = defaultdict(set)

        # Flat name registries (standardized) to support prefix-based detection
        self.province_names_std: Set[str] = set()
        self.district_names_std: Set[str] = set()
        self.ward_names_std: Set[str] = set()

        # Lookup tables to attach IDs to normalized components at runtime
        self.province_lookup: Dict[str, Dict[str, Any]] = {}
        self.district_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.district_lookup_by_name = defaultdict(list)
        self.ward_lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.ward_lookup_by_name = defaultdict(list)
        self.ward_lookup_by_province_name = defaultdict(list)
        self.ward_lookup_by_district_key = defaultdict(list)

        # Tunables to cap worst-case latency
        self.TOPK_CANDIDATES = 400  # bound number of candidates from inverted index
        self.DICE_GATE = 0.4  # only compute partial ratio when Dice >= this
        self.PARTIAL_CUTOFF = 40  # minimum acceptable partial ratio
        self.REFERENCE_ACCEPT_RATIO = 90  # minimum ratio to accept a reference override

        # Dataset level metadata & mapping snapshots for downstream features
        self.ward_mapping_by_old_code: Dict[str, List[Dict[str, Any]]] = {}
        self.ward_mapping_by_new_code: Dict[str, List[Dict[str, Any]]] = {}
        self.old_province_records: Dict[str, Dict[str, Any]] = {}
        self.old_district_records: Dict[str, Dict[str, Any]] = {}
        self.old_ward_records: Dict[str, Dict[str, Any]] = {}
        self.new_province_records: Dict[str, Dict[str, Any]] = {}
        self.new_ward_records: Dict[str, Dict[str, Any]] = {}
        self.external_new_province_records: Dict[str, Dict[str, Any]] = {}
        self.external_new_ward_records: Dict[str, Dict[str, Any]] = {}

        self.search_engine: Optional[AddressSearchEngine] = None

        # Pre-process address data once when initializing the Solution object
        dataset_signature = self._dataset_signature()
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

    def process(self, input_string: str):
        # Chuẩn hóa và tạo n-gram cho input
        input_string_standard = self.standardize_name(input_string, True)
        input_string_basic = self.standardize_name(input_string, False)
        input_string_ngram_list = self.generate_ngrams(input_string_standard)
        input_segments = self._split_address_segments(input_string)
        # Keep segment boundaries to avoid prefix detectors swallowing tokens across commas
        prefix_scan_input = (
            " | ".join(seg for seg, _ in input_segments) if input_segments else ""
        ) or input_string_basic

        partial_input_string = False

        def _appears_in_input(component: Optional[str]) -> bool:
            if not component:
                return False
            component_std = self.standardize_name(component, False)
            return bool(component_std and component_std in input_string_basic)

        # Đếm tần suất xuất hiện của từng ngram
        ngram_counts = Counter(input_string_ngram_list)

        # Lấy 5 ngram phổ biến nhất
        top_5 = ngram_counts.most_common(5)
        # Nếu tổng tần suất top 5 ngram ≤ 15 → partial_input_string = True
        if top_5 and sum(count for _, count in top_5) >= 12:
            partial_input_string = True

        input_ngram_set = set(input_string_ngram_list)

        address = self.AddressNode("", "", "")

        detected_components_raw = self._detect_by_prefix(prefix_scan_input)
        detected_prov = self._validate_detected_value(
            detected_components_raw[0], self.invert_province_to_indices
        )
        detected_dist = self._validate_detected_value(
            detected_components_raw[1], self.invert_district_to_indices
        )
        detected_ward = self._validate_detected_value(
            detected_components_raw[2], self.invert_ward_to_indices
        )
        raw_detected_ward = detected_components_raw[2]
        raw_detected_dist = None
        normalized_detected_ward_token = (
            self._normalize_detected_ward_token(raw_detected_ward)
            if raw_detected_ward
            else None
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
                recovered = self._prefer_component_alias_from_segments(
                    [detected_dist],
                    input_segments,
                    require_prefix=True,
                    level="district",
                ) or self._recover_component_from_input(detected_dist, input_segments)
                if recovered:
                    cleaned = recovered.strip()
                    parts = [part for part in cleaned.split() if part]
                    if parts:
                        first_std = self.standardize_name(parts[0], False)
                        second_std = (
                            self.standardize_name(parts[1], False)
                            if len(parts) >= 2
                            else ""
                        )
                        if first_std in {"huyen", "quan", "tp"}:
                            cleaned = " ".join(parts[1:])
                        elif first_std == "thi" and second_std == "xa":
                            cleaned = " ".join(parts[2:])
                        elif first_std == "thanh" and second_std == "pho":
                            cleaned = " ".join(parts[2:])
                    recovered = cleaned.strip()
                raw_detected_dist = recovered
        district_hint_in_input = bool(raw_detected_dist)
        district_present_in_input = district_hint_in_input

        # If any comma-separated segment explicitly starts with a district-level
        # prefix (e.g. "Huyện ...", "Quận ...", "Thị xã ..."), treat the input as
        # old format even if prefix-based name resolution fails.
        district_prefix_in_input = False
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
                    break
                if first == "h":
                    raw = (segment_raw or "").strip().lower()
                    if raw.startswith(("h.", "huyện", "huyen")):
                        district_prefix_in_input = True
                        district_hint_in_input = True
                        district_present_in_input = True
                        break
                if len(tokens) >= 2 and f"{tokens[0]} {tokens[1]}" == "thi xa":
                    district_prefix_in_input = True
                    district_hint_in_input = True
                    district_present_in_input = True
                    break
                if first == "tp" and len(tokens) >= 2:
                    city_name = " ".join(tokens[1:]).strip()
                    if city_name:
                        normalized_city_name = (
                            self._detect_special_province_token(city_name) or city_name
                        )
                        province_id_new = self._lookup_new_province_id_by_name(
                            normalized_city_name
                        )
                        province_record = (
                            self.external_new_province_records.get(province_id_new)
                            or self.new_province_records.get(province_id_new)
                            if province_id_new
                            else None
                        )
                        is_central_municipality = bool(
                            isinstance(province_record, dict)
                            and province_record.get("administrative_unit_id") == 1
                        )
                        if not is_central_municipality:
                            district_prefix_in_input = True
                            district_hint_in_input = True
                            district_present_in_input = True
                            break
                if len(tokens) >= 3 and f"{tokens[0]} {tokens[1]}" == "thanh pho":
                    city_name = " ".join(tokens[2:]).strip()
                    if city_name:
                        normalized_city_name = (
                            self._detect_special_province_token(city_name) or city_name
                        )
                        province_id_new = self._lookup_new_province_id_by_name(
                            normalized_city_name
                        )
                        province_record = (
                            self.external_new_province_records.get(province_id_new)
                            or self.new_province_records.get(province_id_new)
                            if province_id_new
                            else None
                        )
                        is_central_municipality = bool(
                            isinstance(province_record, dict)
                            and province_record.get("administrative_unit_id") == 1
                        )
                        if not is_central_municipality:
                            district_prefix_in_input = True
                            district_hint_in_input = True
                            district_present_in_input = True
                            break

        def _expected_district_for_resolution() -> Optional[str]:
            if not district_hint_in_input:
                return None
            if district:
                return district
            return raw_detected_dist

        if not detected_prov:
            special_province = self._detect_special_province_token(input_string_basic)
            if special_province:
                detected_prov = self._validate_detected_value(
                    special_province, self.invert_province_to_indices
                )
        if not detected_prov:
            suffix_province = self._detect_suffix_province_token(input_string_basic)
            if suffix_province:
                detected_prov = self._validate_detected_value(
                    suffix_province, self.invert_province_to_indices
                )
        detected_components = (detected_prov, detected_dist, detected_ward)
        ngram_address_piece_list = self.ngram_address_piece_list(
            input_string_ngram_list, self.TOPK_CANDIDATES
        )

        enforced_new_ward_entry: Optional[Dict[str, Any]] = None

        address_candidate = self.address_candidate_list(
            input_string_standard,
            input_ngram_set,
            ngram_address_piece_list,
            partial_input_string,
            detected_components,
        )

        if address_candidate:
            selected_idx = self._select_candidate_with_hints(
                address_candidate,
                detected_components,
            )
            if selected_idx is not None:
                address = self.address_node_list[selected_idx]
            else:
                address = self.address_node_list[address_candidate[0][0]]

        province = address.province_name
        district = address.district_name
        ward = address.ward_name
        province_id = address.province_id
        district_id = address.district_id
        ward_id = address.ward_id
        candidate_is_new_format = address.is_new_format

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
                current_std = self.standardize_name(province, False)
                resolved_std = self.standardize_name(resolved_province, False)
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

        if district and detected_dist and district != detected_dist:
            district_std = self.standardize_name(district, False)
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

        if (
            not ward
            and detected_ward
            and (raw_detected_dist or district_present_in_input)
        ):
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
            new_entry = None
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
                    entry_matches_hint = self._entry_aligns_with_province(
                        new_entry, province
                    )
                    if not province or not entry_matches_hint:
                        province = province_from_entry
                        province_id = None

        enforcement_token = detected_ward or normalized_detected_ward_token
        if not enforcement_token and raw_detected_ward:
            enforcement_token = (
                self._normalize_detected_ward_token(raw_detected_ward)
                or raw_detected_ward
            )
        if enforcement_token and not (raw_detected_dist or district_present_in_input):
            new_format_entry = self._lookup_new_format_ward_alias(
                enforcement_token,
                expected_province=province,
            )
            if new_format_entry:
                new_format_entry = self._prefer_hierarchical_ward_entry(
                    enforcement_token,
                    new_format_entry,
                    expected_province=province,
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
            current_ward_std = self.standardize_name(ward, False)
            should_try_override = False
            if (
                current_ward_std
                and current_ward_std.isdigit()
                and detected_ward.isdigit()
            ):
                should_try_override = current_ward_std != detected_ward
            elif current_ward_std and current_ward_std != detected_ward:
                detected_token = normalized_detected_ward_token or detected_ward
                token_in_input = bool(
                    detected_token and detected_token in input_string_basic
                )
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
            current_ward_std = self.standardize_name(ward, False)
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
                province_std = (
                    self.standardize_name(province, False) if province else None
                )
                for dist_std, entries in self.district_lookup_by_name.items():
                    if dist_std not in input_string_basic:
                        continue
                    for entry in entries:
                        if province and not self._entry_aligns_with_province(
                            entry, province
                        ):
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
                province_std = (
                    self.standardize_name(province, False) if province else None
                )
                district_std = (
                    self.standardize_name(district, False) if district else None
                )
                for ward_std, entries in self.ward_lookup_by_name.items():
                    if ward_std not in input_string_basic:
                        continue
                    for entry in entries:
                        if province and not self._entry_aligns_with_province(
                            entry, province
                        ):
                            continue
                        if district_std and entry.get("district_key") != district_std:
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

        if (
            not district
            and ward
            and candidate_is_new_format is not True
            and district_hint_in_input
        ):
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

        province_for_lookup = province if province else None
        district_info = (
            self._lookup_district_info(district, province_for_lookup)
            if district
            else None
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
                enforced_std = self.standardize_name(enforced_district, False)
                current_std = (
                    self.standardize_name(district, False) if district else None
                )
                if enforced_std and enforced_std != current_std:
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

        def _update_format(
            current_value: Optional[bool], info_value: Optional[Dict[str, Any]]
        ) -> Optional[bool]:
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
        if ward and ward_info is None:
            ward_info = self._lookup_ward_info(
                ward, preferred_format=candidate_is_new_format
            )
        enforce_locked_new_format = enforced_new_ward_entry is not None
        if enforced_new_ward_entry is not None:
            ward_info = enforced_new_ward_entry
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

            def _enforce_ward_by_province(token: Optional[str]) -> Optional[str]:
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
            for token in (
                detected_ward,
                normalized_detected_ward_token,
                self.standardize_name(ward, False) if ward else None,
            ):
                enforced_name = _enforce_ward_by_province(token)
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
                if ward_info is None:
                    ward_info = self._lookup_ward_info(
                        ward, preferred_format=candidate_is_new_format
                    )
            if not ward_info or not self._entry_aligns_with_province(
                ward_info, province
            ):
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
            entry_aligns_province = self._entry_aligns_with_province(
                ward_info, province
            )
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
                self._lookup_district_info(district, province_for_lookup)
                if district
                else None
            )

        if ward_info and not district and district_hint_in_input:
            recovered_district_name, recovered_district_id = (
                self._recover_district_from_ward_info(
                    ward_info,
                    ward,
                    province,
                    province_info,
                )
            )
            if recovered_district_name:
                district = recovered_district_name
                district_id = recovered_district_id
                district_info = (
                    self._lookup_district_info(district, province_for_lookup)
                    if district
                    else None
                )
                if (
                    not district_id
                    and district_info
                    and district_info.get("id") is not None
                ):
                    district_id = district_info["id"]
                ward_info = {
                    **ward_info,
                    "district_name": district,
                    "district_key": ward_info.get("district_key")
                    or self.standardize_name(district, False),
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
            if self.standardize_name(ward_before or "", False) != self.standardize_name(
                ward, False
            ):
                ward_info = self._lookup_ward_info(
                    ward,
                    province_for_lookup,
                    district_for_lookup,
                    preferred_format=candidate_is_new_format,
                )
                if ward_info is None:
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
            district_std = self.standardize_name(district, False)
            ward_std = self.standardize_name(ward, False) if ward else None
            if (
                district_std
                and district_std == detected_ward
                and ward_std != detected_ward
            ):
                pass
            elif district_std and district_std == detected_ward:
                detected_ward = None

        ward_present_in_input = _appears_in_input(ward)
        if (not ward or not ward_present_in_input) and raw_detected_ward:
            normalized_raw_token = self._normalize_detected_ward_token(
                raw_detected_ward
            )
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
                    entry_name_std = self.standardize_name(new_entry.get("name"), False)
                    if (
                        entry_name_std
                        and normalized_raw_token
                        and entry_name_std != normalized_raw_token
                    ):
                        new_entry = None

            if not new_entry:
                matched_existing = None
                district_std = (
                    self.standardize_name(district, False) if district else None
                )
                if district_std:
                    district_entries = self.ward_lookup_by_district_key.get(
                        district_std, []
                    )
                    for entry in district_entries:
                        entry_name_std = self.standardize_name(entry.get("name"), False)
                        if (
                            entry_name_std == normalized_raw_token
                            or self._numeric_token_match(
                                entry_name_std, normalized_raw_token
                            )
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
                if (
                    fallback
                    and recovered_from_input
                    and not _appears_in_input(fallback)
                ):
                    fallback = recovered_from_input
                if (
                    fallback
                    and not district_present_in_input
                    and not _appears_in_input(fallback)
                ):
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
                    province_std = (
                        self.standardize_name(province, False) if province else None
                    )
                    candidates = self.ward_lookup_by_name.get(normalized_raw_token, [])
                    for entry in candidates:
                        entry_name = entry.get("name")
                        entry_name_std = (
                            self.standardize_name(entry_name, False)
                            if entry_name
                            else None
                        )
                        entry_province_std = (
                            self.standardize_name(entry.get("province_name"), False)
                            if entry.get("province_name")
                            else None
                        )
                        if (
                            entry_name
                            and entry_name_std
                            and (
                                entry_name_std == normalized_raw_token
                                or self._numeric_token_match(
                                    entry_name_std, normalized_raw_token
                                )
                            )
                            and (
                                not province_std
                                or not entry_province_std
                                or province_std == entry_province_std
                            )
                        ):
                            fallback_name = entry_name
                            fallback_id = fallback_id or entry.get("id")
                            fallback_province_name = (
                                entry.get("province_name") or fallback_province_name
                            )
                            matched_existing = entry
                            break
                    if matched_existing:
                        new_entry = matched_existing
                        province_from_entry = matched_existing.get("province_name")
                    elif not district_present_in_input:
                        new_entry = {
                            "id": fallback_id,
                            "name": fallback_name
                            or recovered_from_input
                            or self._titleize_token(raw_detected_ward),
                            "province_name": fallback_province_name or province,
                            "province_key": (
                                self.standardize_name(
                                    fallback_province_name or province, False
                                )
                                if (fallback_province_name or province)
                                else None
                            ),
                            "district_name": "",
                            "district_key": "",
                            "is_new_format": True,
                        }
                    elif district:
                        new_entry = {
                            "id": fallback_id,
                            "name": fallback_name
                            or recovered_from_input
                            or self._titleize_token(raw_detected_ward),
                            "province_name": province,
                            "province_key": (
                                self.standardize_name(province, False)
                                if province
                                else None
                            ),
                            "district_name": district,
                            "district_key": (
                                self.standardize_name(district, False)
                                if district
                                else None
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
                    entry_matches_hint = self._entry_aligns_with_province(
                        new_entry, province
                    )
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
        if (
            ward
            and not ward_present_in_input
            and not detected_ward
            and not raw_detected_ward
        ):
            ward = ""
            ward_id = None
            ward_info = None

        if ward and district and ward_present_in_input and not district_hint_in_input:
            ward_std = self.standardize_name(ward, False)
            district_std = self.standardize_name(district, False)
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

        def _std_name(value: Optional[str]) -> str:
            return self.standardize_name(value, False) if value else ""

        def _canonical_region_key(value: Optional[str]) -> str:
            key = _std_name(value)
            if not key:
                return ""
            key = re.sub(r"^(tinh|thanh pho|tp)\s+", "", key).strip()
            return key

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
                    if segment_std.startswith("tp ") or segment_std.startswith("thanh pho "):
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
            district = ""
            district_id = None
            district_info = None
            district_hint_in_input = False
            resolved_is_new_format = True
            candidate_is_new_format = True

        # If there is no district prefix/hint in the input, avoid "inventing" a district
        # purely from the selected old-format candidate / ward metadata unless the district
        # is clearly present as its own comma-separated segment.
        if province and district and not district_hint_in_input and not district_prefix_in_input:
            district_key = _canonical_region_key(district)
            has_explicit_district_segment = False
            if district_key and input_segments:
                for segment_std, _ in input_segments:
                    if _canonical_region_key(segment_std) == district_key:
                        has_explicit_district_segment = True
                        break
            if not has_explicit_district_segment:
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
                self._lookup_district_info(district, province_for_lookup)
                if district
                else None
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
            if ward and ward_info is None:
                ward_info = self._lookup_ward_info(
                    ward, preferred_format=candidate_is_new_format
                )
            if not ward:
                ward_id = None
            elif ward_info and ward_info.get("id") is not None:
                ward_id = ward_info["id"]
            resolved_is_new_format = _update_format(resolved_is_new_format, ward_info)

        # 2-level guard: when the input contains only ward+province (no district hint/prefix),
        # treat it as new-format regardless of whether the ward record originates from the
        # old or new registry. This avoids classifying genuine 2-level strings as "old"
        # merely because a matching ward exists in the old dataset.
        if ward and not district and not district_hint_in_input:
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
                if upgraded:
                    ward_info = upgraded
                    if upgraded.get("id") is not None:
                        ward_id = upgraded["id"]
                    canonical = upgraded.get("full_name") or upgraded.get("name")
                    if canonical:
                        ward = canonical

        if district_hint_in_input and resolved_is_new_format is not False:
            resolved_is_new_format = False

        if (
            resolved_is_new_format is True
            and not district_hint_in_input
            and not district
        ):
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
            elif (
                preferred_lookup_format is False
                and district
                and not _appears_in_input(district)
            ):
                preferred_lookup_format = True

        def _std(value: Optional[str]) -> str:
            return self.standardize_name(value, False) if value else ""

        ward_name_mismatch = bool(
            ward
            and ward_info
            and _std(ward_info.get("name"))
            and _std(ward_info.get("name")) != _std(ward)
        )

        if (
            ward
            and (ward_name_mismatch or ward_info is None)
            and not enforce_locked_new_format
        ):
            refreshed = self._lookup_ward_info(
                ward,
                province if province else None,
                district if district else None,
                preferred_format=preferred_lookup_format,
            )
            if not refreshed:
                refreshed = self._lookup_ward_info(
                    ward,
                    preferred_format=preferred_lookup_format,
                )
            if refreshed:
                ward_info = refreshed
                if refreshed.get("id") is not None:
                    ward_id = refreshed["id"]
                resolved_is_new_format = _update_format(
                    resolved_is_new_format, ward_info
                )
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
            canonical_std = _std(canonical_name)
            ward_std = _std(ward)
            if canonical_std and canonical_std != ward_std and (
                canonical_std in input_string_basic or district_hint_in_input
            ):
                ward = canonical_name
                ward_id = ward_info.get("id") or ward_id

        # Late new-format guard: classification may be decided before ward resolution,
        # so ensure we still mark province-only / ward+province inputs as "new" when there
        # is no explicit district hint in the text.
        if (province or ward) and not district and not district_hint_in_input:
            resolved_is_new_format = True
            candidate_is_new_format = True

        # Final guard: if we only saw a ward-prefixed token (no district prefix),
        # treat it as 2-level data and drop any inherited district.
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
            self._lookup_district_info(district, province_for_lookup)
            if district
            else None
        )
        if not district:
            district_id = None
        elif district_info and district_info.get("id") is not None:
            district_id = district_info["id"]

        # Final canonicalization: if we have structured ward info, trust its canonical name/id.
        if ward_info:
            canonical_ward = ward_info.get("name") or ward_info.get("full_name")
            if canonical_ward:
                ward = canonical_ward
            if ward_info.get("id") is not None:
                ward_id = ward_info["id"]
            # Rehydrate from source records to avoid legacy-only aliases overriding canonical
            # names. Prefer the registry that matches the resolved output format to avoid
            # collisions where an old and new ward share the same numeric code.
            ward_record_id = ward_info["id"]
            if resolved_is_new_format is True:
                record = self.new_ward_records.get(ward_record_id) or self.old_ward_records.get(
                    ward_record_id
                )
            else:
                record = self.old_ward_records.get(ward_record_id) or self.new_ward_records.get(
                    ward_record_id
                )
            if isinstance(record, dict):
                canonical_from_record = record.get("full_name") or record.get("name")
                if canonical_from_record:
                    ward = canonical_from_record
                    ward_info["full_name"] = record.get("full_name") or ward_info.get("full_name")
                    ward_info["name"] = record.get("name") or ward_info.get("name")

        district_component = self._format_component(
            district, district_id, district_info
        )
        province_component = self._format_component(
            province, province_id, province_info
        )
        ward_component = self._format_component(ward, ward_id, ward_info)

        # Hard rule: if the input explicitly includes a district-level prefix
        # (e.g. 'Huyện/Quận'), treat the address as old format regardless of any
        # ward mapping that may point to new-format records.
        if district_hint_in_input:
            resolved_is_new_format = False
            candidate_is_new_format = False

        fmt = (
            "new"
            if resolved_is_new_format is True
            else ("old" if resolved_is_new_format is False else "unknown")
        )
        normalized_node = self.AddressNode(
            province or "",
            district or "",
            ward or "",
            is_new_format=resolved_is_new_format,
        )
        component_aliases = {
            "province": self._gather_alias_values(
                province,
                province_info,
                level="province",
                extra_values=[
                    detected_components_raw[0] if detected_components_raw else None,
                    detected_prov,
                ],
            ),
            "district": self._gather_alias_values(
                district,
                district_info,
                level="district",
                extra_values=[raw_detected_dist, detected_dist],
            ),
            "ward": self._gather_alias_values(
                ward,
                ward_info,
                level="ward",
                extra_values=[
                    raw_detected_ward,
                    normalized_detected_ward_token,
                    detected_ward,
                ],
            ),
        }
        street_address = self._extract_street_address(
            input_string,
            normalized_node,
            component_aliases,
        )
        if province_component and component_aliases.get("province"):
            province_component["aliases"] = component_aliases["province"]
        if district_component and component_aliases.get("district"):
            district_component["aliases"] = component_aliases["district"]
        if ward_component and component_aliases.get("ward"):
            ward_component["aliases"] = component_aliases["ward"]
        return {
            "province": province_component,
            "district": district_component,
            "ward": ward_component,
            "street_address": street_address,
            "format": fmt,
            "is_new": (
                True
                if resolved_is_new_format is True
                else False if resolved_is_new_format is False else None
            ),
        }

    def preprocess_address(self):
        raw_data = self._build_raw_dataset()
        data = self._normalize_address_dataset(raw_data)

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

        def legacy_aliases_from(entry: Optional[Any]) -> List[str]:
            if not isinstance(entry, dict):
                return []
            raw = entry.get("legacy_names")
            aliases: List[str] = []
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
            districts_payload = province_entry
            if isinstance(province_entry, dict) and "districts" in province_entry:
                province_id = province_entry.get("id")
                province_code = province_entry.get("code")
                districts_payload = province_entry.get("districts", {})
            if districts_payload is None:
                districts_payload = {}

            province_output_name = province_name
            province_output_std = self.standardize_name(province_output_name, False)
            reference_aliases = self._reference_aliases_for_level(
                "province", province_code
            )
            province_aliases_extra = list(reference_aliases or [])
            province_aliases_extra.extend(
                self._get_special_province_aliases(province_output_name)
            )
            province_aliases_extra.extend(legacy_aliases_from(province_entry))
            province_aliases = self._collect_aliases(
                province_output_name,
                province_name,
                province_aliases_extra,
            )
            province_aliases_std = self._standardize_aliases(province_aliases)
            province_info = {
                "id": province_id,
                "name": province_output_name,
            }
            if isinstance(province_entry, dict):
                legacy_names = legacy_aliases_from(province_entry)
                if legacy_names:
                    province_info["legacy_names"] = legacy_names
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
                wards_payload = district_entry
                if isinstance(district_entry, dict) and "wards" in district_entry:
                    district_id = district_entry.get("id")
                    wards_payload = district_entry.get("wards", {})
                if wards_payload is None:
                    wards_payload = {}

                district_output_name = district_name
                district_output_std = self.standardize_name(district_output_name, False)
                district_key = district_output_std or ""
                district_id_value = district_id if district_output_name else None
                district_legacy_aliases = legacy_aliases_from(district_entry)

                district_info = {
                    "id": district_id_value,
                    "name": district_output_name,
                    "province_key": province_output_std,
                    "province_name": province_output_name,
                }
                if district_legacy_aliases:
                    district_info["legacy_names"] = district_legacy_aliases
                if province_output_std:
                    self.district_lookup[(province_output_std, district_key)] = (
                        district_info
                    )
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
                        wards_payload.items()
                        if isinstance(wards_payload, dict)
                        else ((ward_name, None) for ward_name in wards_payload)
                    )
                    for ward_name, ward_meta in ward_iter:
                        if not ward_name:
                            continue
                        ward_id_value = (
                            ward_meta.get("id") if isinstance(ward_meta, dict) else None
                        )
                        ward_code = (
                            ward_meta.get("code")
                            if isinstance(ward_meta, dict)
                            else None
                        )
                        ward_legacy_aliases = legacy_aliases_from(ward_meta)
                        ward_output_name, ward_lookup_name = self._derive_ward_names(
                            ward_name, ward_meta
                        )
                        ward_lookup_std = self.standardize_name(ward_lookup_name, False)
                        extra_aliases = list(
                            (self._reference_aliases_for_level("ward", ward_code) or [])
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

                        ward_info = {
                            "id": ward_id_value,
                            "name": ward_output_name,
                            "province_key": province_output_std,
                            "province_name": province_output_name,
                            "district_key": district_key,
                            "district_name": district_output_name,
                            "is_new_format": True,
                        }
                        if isinstance(ward_meta, dict):
                            full_name = ward_meta.get("full_name")
                            if full_name:
                                ward_info["full_name"] = full_name
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
                            self.ward_lookup_by_district_key[district_key].append(
                                ward_info
                            )
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
                    wards_payload.items()
                    if isinstance(wards_payload, dict)
                    else ((ward_name, None) for ward_name in wards_payload)
                )
                for ward_name, ward_meta in ward_iter:
                    if not ward_name:
                        continue
                    ward_id_value = (
                        ward_meta.get("id") if isinstance(ward_meta, dict) else None
                    )
                    ward_code = (
                        ward_meta.get("code") if isinstance(ward_meta, dict) else None
                    )
                    ward_legacy_aliases = legacy_aliases_from(ward_meta)
                    ward_output_name, ward_lookup_name = self._derive_ward_names(
                        ward_name, ward_meta
                    )
                    ward_output_std = self.standardize_name(ward_lookup_name, False)
                    extra_aliases = list(
                        (self._reference_aliases_for_level("ward", ward_code) or [])
                    )
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
                        "name": ward_output_name,
                        "province_key": province_output_std,
                        "province_name": province_output_name,
                        "district_key": district_key,
                        "district_name": district_output_name,
                        "is_new_format": False,
                    }
                    if isinstance(ward_meta, dict):
                        full_name = ward_meta.get("full_name")
                        if full_name:
                            ward_info["full_name"] = full_name
                    if ward_legacy_aliases:
                        ward_info["legacy_names"] = ward_legacy_aliases
                    if province_output_std and ward_output_std:
                        self.ward_lookup[
                            (province_output_std, district_key, ward_output_std)
                        ] = ward_info
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
            self.generate_ngram_inverted_index(
                node.ngram_list, index, self.invert_ngrams_idx
            )

        self._rebuild_search_engine()

    def _dataset_signature(
        self,
    ) -> Tuple[Tuple[str, Optional[float], Optional[int]], ...]:
        tracked_paths = (
            self.new_format_provinces_path,
            self.new_format_wards_path,
            self.new_format_mapping_path,
            self.old_provinces_path,
            self.old_districts_path,
            self.old_wards_path,
        )
        signature: List[Tuple[str, Optional[float], Optional[int]]] = []
        signature.append(
            (
                "__cache_version__",
                float(self._CACHE_VERSION),
                len(self._STATEFUL_ATTRS),
            )
        )
        for path in tracked_paths:
            try:
                stat_result = os.stat(path)
                signature.append((path, stat_result.st_mtime, stat_result.st_size))
            except OSError:
                signature.append((path, None, None))
        return tuple(signature)

    def _hydrate_preprocessed_state(
        self,
        signature: Tuple[Tuple[str, Optional[float], Optional[int]], ...],
    ) -> bool:
        cls = self.__class__
        cache = cls._PREPROCESSED_CACHE
        if cache and cls._PREPROCESSED_SIGNATURE == signature:
            self._apply_preprocessed_state(cache)
            return True
        return False

    def _cache_payload(
        self,
        signature: Tuple[Tuple[str, Optional[float], Optional[int]], ...],
    ) -> Dict[str, Any]:
        return {
            "version": self._CACHE_VERSION,
            "signature": signature,
            "state": self._capture_preprocessed_state(),
        }

    def _persist_preprocessed_state(
        self,
        signature: Tuple[Tuple[str, Optional[float], Optional[int]], ...],
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
        signature: Tuple[Tuple[str, Optional[float], Optional[int]], ...],
    ) -> bool:
        path = getattr(self, "_cache_path", None)
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as exc:
            logger.debug("Failed to load AddressParser cache: %s", exc)
            return False

        if not isinstance(payload, dict):
            return False
        if payload.get("version") != self._CACHE_VERSION:
            return False
        cached_signature = payload.get("signature")
        state = payload.get("state")

        def _signature_equivalent_ignoring_mtime(
            cached: Any,
            current: Any,
        ) -> bool:
            if not isinstance(cached, tuple) or not isinstance(current, tuple):
                return False
            if len(cached) != len(current) or not cached:
                return False
            for idx, (a, b) in enumerate(zip(cached, current)):
                if not isinstance(a, tuple) or not isinstance(b, tuple):
                    return False
                if len(a) != 3 or len(b) != 3:
                    return False
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

        if not isinstance(state, dict):
            return False
        if cached_signature != signature and not _signature_equivalent_ignoring_mtime(
            cached_signature, signature
        ):
            return False
        self._apply_preprocessed_state(state)
        return True

    def _cache_preprocessed_state(
        self,
        signature: Tuple[Tuple[str, Optional[float], Optional[int]], ...],
    ) -> None:
        cls = self.__class__
        cls._PREPROCESSED_CACHE = self._capture_preprocessed_state()
        cls._PREPROCESSED_SIGNATURE = signature

    def _capture_preprocessed_state(self) -> Dict[str, Any]:
        return {attr: getattr(self, attr) for attr in self._STATEFUL_ATTRS}

    def _apply_preprocessed_state(self, state: Dict[str, Any]) -> None:
        for attr, value in state.items():
            setattr(self, attr, value)

    def _normalize_code_str(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            value_str = str(value).strip()
        except Exception:
            return None
        return value_str or None

    def _read_json_file(self, path: Optional[str]) -> Any:
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _register_external_entry(
        self,
        target: Dict[str, Dict[str, Any]],
        code: Any,
        entry: Dict[str, Any],
    ) -> None:
        keys: Set[str] = set()
        if code is not None:
            if isinstance(code, str):
                key_str = code.strip()
            else:
                key_str = str(code)
            if key_str:
                keys.add(key_str)
        normalized = self._normalize_id_token(code)
        if normalized:
            keys.add(normalized)
        if not keys:
            return
        for key in keys:
            target[key] = entry

    def _load_external_new_dataset(self) -> Dict[str, Any]:
        payload = {
            "provinces": {},
            "wards": {},
            "ward_mappings": [],
        }
        provinces_data = self._read_json_file(self.new_format_provinces_path)
        if isinstance(provinces_data, list):
            for entry in provinces_data:
                if not isinstance(entry, dict):
                    continue
                code = entry.get("code")
                if code is None:
                    continue
                normalized_entry = dict(entry)
                normalized_entry.setdefault("code", str(code).strip())
                self._register_external_entry(
                    payload["provinces"],
                    normalized_entry.get("code"),
                    normalized_entry,
                )
        wards_data = self._read_json_file(self.new_format_wards_path)
        if isinstance(wards_data, list):
            for entry in wards_data:
                if not isinstance(entry, dict):
                    continue
                code = entry.get("code")
                if code is None:
                    continue
                normalized_entry = dict(entry)
                normalized_entry.setdefault("code", str(code).strip())
                self._register_external_entry(
                    payload["wards"],
                    normalized_entry.get("code"),
                    normalized_entry,
                )
        mapping_data = self._read_json_file(self.new_format_mapping_path)
        if isinstance(mapping_data, list):
            payload["ward_mappings"] = [
                row
                for row in mapping_data
                if isinstance(row, dict)
                and row.get("old_ward_code") is not None
                and row.get("new_ward_code") is not None
            ]
        self.external_new_province_records = payload["provinces"]
        self.external_new_ward_records = payload["wards"]
        return payload

    def _dedupe_external_entries(
        self, records: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        seen: Set[str] = set()
        for entry in records.values():
            if not isinstance(entry, dict):
                continue
            code = entry.get("code")
            if not code:
                continue
            code_str = str(code).strip()
            if not code_str or code_str in seen:
                continue
            seen.add(code_str)
            result[code_str] = entry
        return result

    def _load_entities_by_code(
        self,
        path: str,
        *,
        parent_key: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        data = self._read_json_file(path)
        if not isinstance(data, list):
            return {}
        result: Dict[str, Dict[str, Any]] = {}

        def _normalize_aliases(value: Any) -> Optional[List[str]]:
            if isinstance(value, str):
                candidate = value.strip()
                return [candidate] if candidate else None
            if isinstance(value, list):
                aliases: List[str] = []
                for item in value:
                    if not isinstance(item, str):
                        continue
                    candidate = item.strip()
                    if candidate and candidate not in aliases:
                        aliases.append(candidate)
                return aliases or None
            return None

        for entry in data:
            if not isinstance(entry, dict):
                continue
            code = entry.get("code")
            if code is None:
                continue
            code_str = str(code).strip()
            if not code_str:
                continue
            normalized: Dict[str, Any] = {
                "code": code_str,
                "id": self._normalize_code_str(code_str),
                "name": entry.get("name"),
            }
            if "full_name" in entry and isinstance(entry["full_name"], str):
                normalized["full_name"] = entry["full_name"]
            if parent_key:
                parent_value = entry.get(parent_key)
                normalized["parent_code"] = (
                    str(parent_value).strip() if parent_value is not None else None
                )
            if "name_en" in entry and isinstance(entry["name_en"], str):
                normalized["name_en"] = entry["name_en"]
            if "full_name_en" in entry and isinstance(entry["full_name_en"], str):
                normalized["full_name_en"] = entry["full_name_en"]
            if "code_name" in entry and isinstance(entry["code_name"], str):
                normalized["code_name"] = entry["code_name"]
            legacy_aliases = _normalize_aliases(entry.get("legacy_names"))
            if legacy_aliases:
                normalized["legacy_names"] = legacy_aliases
            result[code_str] = normalized
        return result

    def _build_raw_dataset(self) -> Dict[str, Any]:
        old_provinces = self._load_entities_by_code(self.old_provinces_path)
        old_districts = self._load_entities_by_code(
            self.old_districts_path, parent_key="province_code"
        )
        old_wards = self._load_entities_by_code(
            self.old_wards_path, parent_key="district_code"
        )

        self.old_province_records = old_provinces
        self.old_district_records = old_districts
        self.old_ward_records = old_wards

        external_payload = self._load_external_new_dataset()
        new_provinces_raw = self._dedupe_external_entries(
            external_payload.get("provinces") or {}
        )
        new_wards_raw = self._dedupe_external_entries(
            external_payload.get("wards") or {}
        )

        new_provinces: Dict[str, Dict[str, Any]] = {}
        for code, entry in new_provinces_raw.items():
            if not isinstance(entry, dict):
                continue
            new_provinces[code] = {
                "code": code,
                "id": self._normalize_code_str(entry.get("code") or code),
                "name": entry.get("name"),
                "name_en": entry.get("name_en"),
                "full_name_en": entry.get("full_name_en"),
                "full_name": entry.get("full_name"),
            }

        new_wards: Dict[str, Dict[str, Any]] = {}
        for code, entry in new_wards_raw.items():
            if not isinstance(entry, dict):
                continue
            parent_code = entry.get("province_code")
            normalized_parent = (
                str(parent_code).strip() if parent_code is not None else None
            )
            new_wards[code] = {
                "code": code,
                "id": self._normalize_code_str(entry.get("code") or code),
                "name": entry.get("name"),
                "full_name": entry.get("full_name"),
                "name_en": entry.get("name_en"),
                "full_name_en": entry.get("full_name_en"),
                "parent_code": normalized_parent,
                "administrative_unit_id": entry.get("administrative_unit_id"),
                "is_new_format": True,
            }

        self.new_province_records = new_provinces
        self.new_ward_records = new_wards

        mapping_rows = external_payload.get("ward_mappings") or []
        ward_old_to_new, ward_new_to_old = self._convert_external_ward_mappings(
            mapping_rows
        )

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
        rows: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
        old_to_new: Dict[str, List[Dict[str, Any]]] = {}
        new_to_old: Dict[str, List[Dict[str, Any]]] = {}

        def _register(
            target: Dict[str, List[Dict[str, Any]]],
            code: Optional[str],
            payload: Dict[str, Any],
        ):
            if not code:
                return
            bucket = target.setdefault(code, [])
            bucket.append(payload)
            normalized = self._normalize_id_token(code)
            if normalized and normalized != code:
                target[normalized] = bucket

        for row in rows:
            old_code_raw = row.get("old_ward_code") or row.get("ward_id_old")
            new_code_raw = row.get("new_ward_code") or row.get("ward_id_new")
            if old_code_raw is None or new_code_raw is None:
                continue
            old_code = str(old_code_raw).strip()
            new_code = str(new_code_raw).strip()
            if not old_code or not new_code:
                continue

            old_entry = self.old_ward_records.get(old_code)
            district_id_old = None
            city_id_old = None
            if old_entry:
                district_id_old = self._normalize_id_token(old_entry.get("parent_code"))
                if district_id_old:
                    district_entry = self.old_district_records.get(district_id_old)
                    if district_entry:
                        city_id_old = self._normalize_id_token(
                            district_entry.get("parent_code")
                        )

            new_entry = self.new_ward_records.get(new_code)
            city_id_new = None
            if new_entry:
                city_id_new = self._normalize_id_token(new_entry.get("parent_code"))

            old_payload = {
                "city_id_old": city_id_old,
                "district_id_old": district_id_old,
                "ward_id_old": old_code,
                "city_id_new": city_id_new,
                "ward_id_new": new_code,
                "old_ward_name": row.get("old_ward_name"),
                "new_ward_name": row.get("new_ward_name"),
                "old_province_name": row.get("old_province_name"),
                "new_province_name": row.get("new_province_name"),
                "old_district_name": row.get("old_district_name"),
            }
            _register(old_to_new, old_code, old_payload)

            new_payload = {
                "city_id_new": city_id_new,
                "ward_id_new": new_code,
                "city_id_old": city_id_old,
                "district_id_old": district_id_old,
                "ward_id_old": old_code,
                "old_ward_name": row.get("old_ward_name"),
                "new_ward_name": row.get("new_ward_name"),
                "old_province_name": row.get("old_province_name"),
                "new_province_name": row.get("new_province_name"),
                "old_district_name": row.get("old_district_name"),
            }
            _register(new_to_old, new_code, new_payload)

        return old_to_new, new_to_old

    def _normalize_address_dataset(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the data files (old/new/mapping) into a legacy nested structure
        keyed by province name for downstream processing.
        """

        if not isinstance(raw_data, dict):
            return {}

        old_section = raw_data.get("old") or {}
        new_section = raw_data.get("new") or {}
        mapping_section = raw_data.get("mapping") or {}

        provinces_old = old_section.get("provinces") or {}
        districts_old = old_section.get("districts") or {}
        wards_old = old_section.get("wards") or {}
        provinces_new = new_section.get("provinces") or {}
        wards_new = new_section.get("wards") or {}

        self.old_province_records = provinces_old
        self.old_district_records = districts_old
        self.old_ward_records = wards_old
        self.new_province_records = provinces_new
        self.new_ward_records = wards_new

        ward_mapping = mapping_section.get("ward_old_to_new") or {}
        ward_mapping_new = mapping_section.get("ward_new_to_old") or {}
        self.ward_mapping_by_old_code = {str(k): v for k, v in ward_mapping.items()}
        self.ward_mapping_by_new_code = {str(k): v for k, v in ward_mapping_new.items()}

        def extract_legacy_names(source: Optional[Dict[str, Any]]) -> List[str]:
            if not isinstance(source, dict):
                return []
            raw_aliases = source.get("legacy_names")
            aliases: List[str] = []
            if isinstance(raw_aliases, str):
                candidate = raw_aliases.strip()
                if candidate:
                    aliases.append(candidate)
            elif isinstance(raw_aliases, list):
                for alias in raw_aliases:
                    if isinstance(alias, str):
                        candidate = alias.strip()
                        if candidate and candidate not in aliases:
                            aliases.append(candidate)
            return aliases

        legacy_view: Dict[str, Dict[str, Any]] = {}
        province_entries_by_code: Dict[str, Dict[str, Any]] = {}
        district_entries_by_code: Dict[str, Dict[str, Any]] = {}

        def _preferred_name(entity: Dict[str, Any], fallback: str) -> str:
            name_raw = entity.get("name") if isinstance(entity, dict) else None
            name = name_raw.strip() if isinstance(name_raw, str) else ""
            full_name_raw = entity.get("full_name") if isinstance(entity, dict) else None
            extended = full_name_raw.strip() if isinstance(full_name_raw, str) else ""
            if name and not name.replace(" ", "").isdigit():
                return name
            if extended:
                return extended
            if name:
                return name
            return fallback

        def ensure_province(
            code: Optional[str],
            payload: Optional[Dict[str, Any]],
            *,
            prefer_name: bool = False,
        ) -> Dict[str, Any]:
            payload = payload or {}
            normalized_code = str(code) if code is not None else payload.get("code")
            normalized_code = str(normalized_code).strip() if normalized_code else None
            name = _preferred_name(payload, normalized_code or "Unknown Province")

            entry_by_name = legacy_view.get(name)
            entry_by_code = (
                province_entries_by_code.get(normalized_code) if normalized_code else None
            )

            entry = None
            if prefer_name and entry_by_name is not None:
                entry = entry_by_name
            if entry is None and entry_by_code is not None:
                entry = entry_by_code
            if entry is None and entry_by_name is not None:
                entry = entry_by_name

            if entry is None:
                entry = {
                    "id": self._normalize_code_str(
                        payload.get("id") or normalized_code
                    ),
                    "code": normalized_code,
                    "full_name": payload.get("full_name") or name,
                    "districts": {},
                    "legacy_names": extract_legacy_names(payload),
                }
                if not entry["legacy_names"]:
                    entry.pop("legacy_names")
                legacy_view[name] = entry
            else:
                if entry.get("code") is None and normalized_code:
                    entry["code"] = normalized_code
                if entry.get("id") is None:
                    entry["id"] = self._normalize_code_str(
                        payload.get("id") or normalized_code
                    )
                if not entry.get("full_name") and payload.get("full_name"):
                    entry["full_name"] = payload["full_name"]
                legacy_bucket = entry.setdefault("legacy_names", [])
                for alias in extract_legacy_names(payload):
                    if alias not in legacy_bucket:
                        legacy_bucket.append(alias)

            if normalized_code:
                if prefer_name and entry_by_name is not None:
                    province_entries_by_code[normalized_code] = entry
                elif normalized_code not in province_entries_by_code:
                    province_entries_by_code[normalized_code] = entry
            return entry

        for code, info in provinces_old.items():
            ensure_province(code, info)
        for code, info in provinces_new.items():
            ensure_province(code, info, prefer_name=True)

        def merge_ward_entry(
            existing: Dict[str, Any], incoming: Dict[str, Any]
        ) -> Dict[str, Any]:
            if not existing:
                return incoming

            def _set_if_missing(field: str):
                if existing.get(field) in (None, "") and incoming.get(field) not in (
                    None,
                    "",
                ):
                    existing[field] = incoming[field]

            for field in (
                "id",
                "code",
                "full_name",
                "administrative_unit_id",
            ):
                _set_if_missing(field)

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
            province_entry: Dict[str, Any], code: str, payload: Dict[str, Any]
        ) -> Dict[str, Any]:
            district_name = _preferred_name(payload, code)
            district_entry = province_entry["districts"].get(district_name)
            if district_entry is None:
                district_entry = {
                    "id": self._normalize_code_str(payload.get("id") or code),
                    "code": payload.get("code") or code,
                    "full_name": payload.get("full_name") or district_name,
                    "wards": {},
                    "legacy_names": extract_legacy_names(payload),
                }
                if not district_entry["legacy_names"]:
                    district_entry.pop("legacy_names")
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
            province_entry = province_entries_by_code.get(str(province_code))
            if province_entry is None:
                province_entry = ensure_province(
                    province_code, provinces_old.get(str(province_code))
                )
            attach_district(province_entry, code, info)

        def new_format_bucket(province_entry: Dict[str, Any]) -> Dict[str, Any]:
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
                district_payload = districts_old.get(str(parent_district)) or {}
                if district_payload:
                    province_code = district_payload.get("parent_code")
                province_entry = province_entries_by_code.get(str(province_code))
                if province_entry is None:
                    province_entry = ensure_province(
                        province_code, provinces_old.get(str(province_code))
                    )
                district_entry = attach_district(
                    province_entry, parent_district or code, district_payload
                )
            ward_name = _preferred_name(info, code)
            ward_entry = {
                "id": self._normalize_code_str(info.get("id") or code),
                "code": info.get("code") or code,
                "parent_code": info.get("parent_code"),
                "full_name": info.get("full_name") or info.get("name"),
                "administrative_unit_id": info.get("administrative_unit_id"),
                "legacy_names": extract_legacy_names(info),
            }
            if not ward_entry["legacy_names"]:
                ward_entry.pop("legacy_names")
            existing_ward = district_entry["wards"].get(ward_name)
            if existing_ward is not None:
                district_entry["wards"][ward_name] = merge_ward_entry(
                    existing_ward, ward_entry
                )
            else:
                district_entry["wards"][ward_name] = ward_entry

        for code, info in wards_new.items():
            province_code = info.get("parent_code")
            province_entry = ensure_province(
                province_code, provinces_new.get(str(province_code)), prefer_name=True
            )
            bucket = new_format_bucket(province_entry)
            ward_name = _preferred_name(info, code)
            ward_entry = {
                "id": self._normalize_code_str(info.get("id") or code),
                "code": info.get("code") or code,
                "parent_code": info.get("parent_code"),
                "is_new_format": True,
                "full_name": info.get("full_name"),
                "administrative_unit_id": info.get("administrative_unit_id"),
                "legacy_names": extract_legacy_names(info),
            }
            if not ward_entry["legacy_names"]:
                ward_entry.pop("legacy_names")
            existing_ward = bucket["wards"].get(ward_name)
            if existing_ward is not None:
                bucket["wards"][ward_name] = merge_ward_entry(
                    existing_ward, ward_entry
                )
            else:
                bucket["wards"][ward_name] = ward_entry

        return legacy_view

    def _normalize_id_token(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            try:
                return str(int(value))
            except Exception:
                return None
        value = str(value).strip()
        return value or None

    def map_old_ward_to_new(self, ward_identifier: Any) -> List[Dict[str, Any]]:
        """
        Return mapping rows (from Excel) for an old-format ward ID or name.
        """
        if ward_identifier is None:
            return []

        if isinstance(ward_identifier, (int, float)):
            key = str(int(ward_identifier))
        else:
            key = str(ward_identifier).strip()
        result = self.ward_mapping_by_old_code.get(key)
        if result:
            return result

        ward_std = self.standardize_name(key, False)
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

    def map_new_ward_to_old(self, ward_identifier: Any) -> List[Dict[str, Any]]:
        """
        Return mapping rows for a new-format ward ID; supports strings or ints.
        """
        if ward_identifier is None:
            return []
        if isinstance(ward_identifier, (int, float)):
            key = str(int(ward_identifier))
        else:
            key = str(ward_identifier).strip()
        return self.ward_mapping_by_new_code.get(key, [])

    def map_old_address_ids_to_new(
        self,
        *,
        province_id: Optional[Any] = None,
        district_id: Optional[Any] = None,
        ward_id: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        ward_key = self._normalize_id_token(ward_id)
        if not ward_key:
            return None
        rows = self.map_old_ward_to_new(ward_key)
        if not rows:
            return None
        province_key = self._normalize_id_token(province_id)
        district_key = self._normalize_id_token(district_id)

        def _match(row: Dict[str, Any]) -> bool:
            if province_key and row.get("city_id_old") != province_key:
                return False
            if district_key and row.get("district_id_old") != district_key:
                return False
            return True

        ranked_rows = sorted(
            rows,
            key=lambda r: (
                1 if province_key and r.get("city_id_old") == province_key else 0,
                1 if district_key and r.get("district_id_old") == district_key else 0,
            ),
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
        province_id: Optional[Any] = None,
        ward_id: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        ward_key = self._normalize_id_token(ward_id)
        if not ward_key:
            return None
        rows = self.map_new_ward_to_old(ward_key)
        if not rows:
            return None
        province_key = self._normalize_id_token(province_id)

        def _rank(row: Dict[str, Any]) -> Tuple[int, int]:
            city_match = int(bool(province_key and row.get("city_id_new") == province_key))
            has_old_district = 1 if row.get("district_id_old") else 0
            return (city_match, has_old_district)

        ranked_rows = sorted(rows, key=_rank, reverse=True)
        for row in ranked_rows:
            if province_key and row.get("city_id_new") != province_key:
                continue
            return self._build_old_mapping_response(row)
        return self._build_old_mapping_response(ranked_rows[0])

    def _lookup_new_province_name(self, province_id: Optional[Any]) -> Optional[str]:
        key = self._normalize_id_token(province_id)
        if not key:
            return None
        entry = self.new_province_records.get(key)
        if not isinstance(entry, dict):
            return None
        return entry.get("full_name") or entry.get("name") or entry.get("slug")

    def _lookup_new_province_id_by_name(
        self, province_name: Optional[str]
    ) -> Optional[str]:
        if not province_name:
            return None
        target_std = self.standardize_name(province_name, False)
        if not target_std:
            return None
        for code, entry in self.new_province_records.items():
            if not isinstance(entry, dict):
                continue
            for key in ("full_name", "name"):
                value = entry.get(key)
                value_std = self.standardize_name(value, False) if value else None
                if value_std and value_std == target_std:
                    return str(code)
        return None

    def _lookup_new_ward_name(self, ward_id: Optional[Any]) -> Optional[str]:
        key = self._normalize_id_token(ward_id)
        if not key:
            return None
        entry = self.new_ward_records.get(key)
        if not isinstance(entry, dict):
            return None
        return entry.get("full_name") or entry.get("name") or entry.get("slug")

    def _build_new_mapping_response(self, row: Dict[str, Any]) -> Dict[str, Any]:
        province_id_new = row.get("city_id_new")
        ward_id_new = row.get("ward_id_new")
        return {
            "province_id_new": province_id_new,
            "province_name_new": self._lookup_new_province_name(province_id_new),
            "ward_id_new": ward_id_new,
            "ward_name_new": self._lookup_new_ward_name(ward_id_new),
            "raw": row,
        }

    def _build_old_mapping_response(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "province_id_old": row.get("city_id_old"),
            "province_name_old": self._lookup_old_province_name(row.get("city_id_old")),
            "district_id_old": row.get("district_id_old"),
            "district_name_old": self._lookup_old_district_name(
                row.get("district_id_old")
            ),
            "ward_id_old": row.get("ward_id_old"),
            "ward_name_old": self._lookup_old_ward_name(row.get("ward_id_old")),
            "raw": row,
        }

    def _lookup_old_province_name(self, province_id: Optional[Any]) -> Optional[str]:
        key = self._normalize_id_token(province_id)
        if not key:
            return None
        entry = self.old_province_records.get(key)
        if not isinstance(entry, dict):
            return None
        return entry.get("full_name") or entry.get("name") or entry.get("slug")

    def _lookup_old_district_name(self, district_id: Optional[Any]) -> Optional[str]:
        key = self._normalize_id_token(district_id)
        if not key:
            return None
        entry = self.old_district_records.get(key)
        if not isinstance(entry, dict):
            return None
        return entry.get("full_name") or entry.get("name") or entry.get("slug")

    def _lookup_old_ward_name(self, ward_id: Optional[Any]) -> Optional[str]:
        key = self._normalize_id_token(ward_id)
        if not key:
            return None
        entry = self.old_ward_records.get(key)
        if not isinstance(entry, dict):
            return None
        return entry.get("full_name") or entry.get("name") or entry.get("slug")

    def _project_component(
        self, entry: Optional[Dict[str, Any]], component_id: Optional[Any]
    ) -> Optional[Dict[str, Any]]:
        if not entry:
            return None
        normalized_id = self._normalize_id_token(component_id)
        projected = {
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
        province: Optional[Dict[str, Any]],
        district: Optional[Dict[str, Any]],
        ward: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        if ward and ward.get("path_with_type"):
            return ward.get("path_with_type")
        if district and district.get("path_with_type"):
            return district.get("path_with_type")
        pieces: List[str] = []
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
        province_id: Optional[Any],
        district_id: Optional[Any],
        ward_id: Optional[Any],
    ) -> Tuple[
        Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]
    ]:
        province = self.old_province_records.get(self._normalize_id_token(province_id))
        district = self.old_district_records.get(self._normalize_id_token(district_id))
        ward = self.old_ward_records.get(self._normalize_id_token(ward_id))
        return province, district, ward

    def _lookup_new_components(
        self,
        *,
        province_id: Optional[Any],
        ward_id: Optional[Any],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        province = self.new_province_records.get(self._normalize_id_token(province_id))
        ward = self.new_ward_records.get(self._normalize_id_token(ward_id))
        return province, ward

    def get_address_components_from_ids(
        self,
        *,
        province_id: Optional[Any],
        district_id: Optional[Any] = None,
        ward_id: Optional[Any] = None,
        is_new_format: bool = False,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        if is_new_format:
            province_entry, ward_entry = self._lookup_new_components(
                province_id=province_id,
                ward_id=ward_id,
            )
            district_entry = None
        else:
            province_entry, district_entry, ward_entry = self._lookup_old_components(
                province_id=province_id,
                district_id=district_id,
                ward_id=ward_id,
            )

        full_address = self._format_full_address(
            province_entry,
            district_entry,
            ward_entry,
        )

        return {
            "province": self._project_component(province_entry, province_id),
            "district": self._project_component(district_entry, district_id),
            "ward": self._project_component(ward_entry, ward_id),
            "full_address": full_address,
        }

    def map_address_ids(
        self,
        *,
        province_id: Optional[Any],
        district_id: Optional[Any],
        ward_id: Optional[Any],
        is_new_format: Optional[bool],
    ) -> Optional[Dict[str, Any]]:
        if ward_id is None:
            return None

        def _summarize(
            direction: str,
            source_is_new: bool,
            mapping_payload: Dict[str, Any],
        ) -> Dict[str, Any]:
            if direction == "old_to_new":
                target_components = self.get_address_components_from_ids(
                    province_id=mapping_payload.get("province_id_new"),
                    ward_id=mapping_payload.get("ward_id_new"),
                    is_new_format=True,
                )
            else:
                target_components = self.get_address_components_from_ids(
                    province_id=mapping_payload.get("province_id_old"),
                    district_id=mapping_payload.get("district_id_old"),
                    ward_id=mapping_payload.get("ward_id_old"),
                    is_new_format=False,
                )

            source_components = self.get_address_components_from_ids(
                province_id=province_id,
                district_id=district_id,
                ward_id=ward_id,
                is_new_format=source_is_new,
            )

            return {
                "direction": direction,
                "source_format_is_new": source_is_new,
                "source": source_components,
                "target": target_components,
                "mapping": mapping_payload,
            }

        if is_new_format is True:
            mapping_payload = self.map_new_address_ids_to_old(
                province_id=province_id,
                ward_id=ward_id,
            )
            if not mapping_payload:
                return None
            return _summarize("new_to_old", True, mapping_payload)

        mapping_payload = self.map_old_address_ids_to_new(
            province_id=province_id,
            district_id=district_id,
            ward_id=ward_id,
        )
        if mapping_payload:
            return _summarize("old_to_new", False, mapping_payload)

        if is_new_format is False:
            return None

        mapping_payload = self.map_new_address_ids_to_old(
            province_id=province_id,
            ward_id=ward_id,
        )
        if not mapping_payload:
            return None
        return _summarize("new_to_old", True, mapping_payload)

    def search_province(
        self,
        query: Optional[str],
        *,
        include_new: bool = True,
        include_old: bool = True,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        engine = self.search_engine
        if not engine or limit <= 0:
            return []
        allowed_sources = []
        if include_new:
            allowed_sources.append("new")
        if include_old:
            allowed_sources.append("old")
        if not allowed_sources:
            return []
        return engine.search(
            query,
            level="province",
            allowed_sources=allowed_sources,
            limit=limit,
        )

    def search_district(
        self,
        query: Optional[str],
        *,
        province_code: Optional[Any] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        engine = self.search_engine
        if not engine or limit <= 0:
            return []
        return engine.search(
            query,
            level="district",
            allowed_sources=["old"],
            province_code=province_code,
            limit=limit,
        )

    def search_ward(
        self,
        query: Optional[str],
        *,
        province_code: Optional[Any] = None,
        district_code: Optional[Any] = None,
        include_new: bool = True,
        include_old: bool = True,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        engine = self.search_engine
        if not engine or limit <= 0:
            return []
        allowed_sources = []
        if include_new:
            allowed_sources.append("new")
        if include_old:
            allowed_sources.append("old")
        if not allowed_sources:
            return []
        results = engine.search(
            query,
            level="ward",
            allowed_sources=allowed_sources,
            province_code=province_code,
            district_code=district_code,
            limit=limit,
        )
        return self._filter_results_by_unit(query, results, level="ward")

    def _detect_unit_token_from_query(self, query: Optional[str]) -> Optional[str]:
        if not query:
            return None
        normalized = self.standardize_name(query, False)
        if not normalized:
            return None
        tokens = normalized.split()
        if not tokens:
            return None

        def _has_sequence(first: str, second: str) -> bool:
            try:
                i = tokens.index(first)
                j = tokens.index(second, i + 1)
                return i < j
            except ValueError:
                return False

        if "phuong" in tokens or "p" in tokens or "w" in tokens:
            return "phuong"
        if "xa" in tokens or "x" in tokens:
            return "xa"
        if _has_sequence("thi", "tran") or "tt" in tokens:
            return "thi tran"
        if _has_sequence("thi", "xa"):
            return "thi xa"
        return None

    def _unit_tokens_match(
        self, required: Optional[str], candidate: Optional[str]
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
        record: Optional[Dict[str, Any]],
        *,
        level: str,
    ) -> Optional[str]:
        if level != "ward" or not isinstance(record, dict):
            return None
        token = self._unit_token_from_admin_id(record.get("administrative_unit_id"))
        if token:
            return token
        text = record.get("full_name") or record.get("name")
        return self._unit_token_from_text(text)

    @staticmethod
    def _unit_token_from_admin_id(unit_id: Optional[Any]) -> Optional[str]:
        try:
            value = int(unit_id) if unit_id is not None else None
        except (TypeError, ValueError):
            return None
        mapping = {
            3: "phuong",
            4: "xa",
            5: "thi tran",
        }
        return mapping.get(value)

    def _unit_token_from_text(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        normalized = self.standardize_name(name, False)
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
            return "thi tran"
        if len(tokens) >= 2 and tokens[0] == "thi" and tokens[1] == "xa":
            return "thi xa"
        if tokens[0] == "tt":
            return "thi tran"
        return None

    def _normalize_unit_token(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None
        normalized = self.standardize_name(str(token), False)
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
            return "thi tran"
        if head == "thi" and tail == "xa":
            return "thi xa"
        return head or None

    def _filter_results_by_unit(
        self,
        query: Optional[str],
        results: List[Dict[str, Any]],
        *,
        level: str,
    ) -> List[Dict[str, Any]]:
        if level != "ward" or not results:
            return results

        unit_token = self._detect_unit_token_from_query(query)
        if not unit_token:
            return results

        filtered: List[Dict[str, Any]] = []
        for result in results:
            if self._unit_tokens_match(unit_token, result.get("unit_token")):
                filtered.append(result)

        if filtered:
            return filtered
        return results


    @staticmethod
    def _tokens_in_order(
        needles: List[str],
        haystack: List[str],
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

    def _tokenize_with_diacritics(self, text: Optional[str]) -> List[str]:
        if not text:
            return []
        normalized = unicodedata.normalize("NFC", str(text)).casefold()
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in normalized)
        return [token for token in cleaned.split() if token]

    def _contains_diacritics(self, text: Optional[str]) -> bool:
        if not text:
            return False
        normalized = unicodedata.normalize("NFD", str(text))
        return any(unicodedata.category(ch) == "Mn" for ch in normalized)

    def _collect_search_text_fields(
        self, entry: Dict[str, Any], *, level: str
    ) -> List[str]:
        fields: List[str] = []
        # Use one primary label (prefer full_name, fall back to name) to avoid duplicate tokens
        primary = entry.get("full_name") or entry.get("name")
        if isinstance(primary, str):
            trimmed = primary.strip()
            if trimmed:
                fields.append(trimmed)

        if level == "province":
            canonical_name = (
                entry.get("full_name")
                or entry.get("name")
            )
            aliases = self._get_special_province_aliases(canonical_name)
            fields.extend(aliases)

        return fields

    def _analyze_search_text(self, text: Optional[str]) -> List[str]:
        if not text:
            return []
        normalized = self.standardize_name(text, False)
        tokens: List[str] = []
        if normalized:
            tokens.extend(token for token in normalized.split() if token)

        # Add diacritic-preserving tokens so accented queries/documents can match directly
        accented_tokens = self._tokenize_with_diacritics(text)
        for tok in accented_tokens:
            if tok not in tokens:
                tokens.append(tok)

        unit_token = self._detect_unit_token_from_query(text)
        if unit_token and unit_token not in tokens:
            tokens.append(unit_token)
        return tokens

    def _rebuild_search_engine(self) -> None:
        self.search_engine = AddressSearchEngine(
            analyzer=self._analyze_search_text,
            normalize_id=self._normalize_id_token,
        )
        engine = self.search_engine

        def _register(
            record: Optional[Dict[str, Any]],
            *,
            level: str,
            source: str,
            province_code: Optional[str] = None,
            district_code: Optional[str] = None,
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
                province_code=self._normalize_id_token(code),
            )
        for code, record in self.old_province_records.items():
            _register(
                record,
                level="province",
                source="old",
                province_code=self._normalize_id_token(code),
            )
        for code, record in self.old_district_records.items():
            _register(
                record,
                level="district",
                source="old",
                province_code=self._normalize_id_token(record.get("parent_code")),
                district_code=self._normalize_id_token(code),
            )
        for record in self.new_ward_records.values():
            _register(
                record,
                level="ward",
                source="new",
                province_code=self._normalize_id_token(record.get("parent_code")),
            )
        for record in self.old_ward_records.values():
            district_code = self._normalize_id_token(record.get("parent_code"))
            province_code = None
            if district_code:
                district_entry = self._lookup_old_district_record(district_code)
                if district_entry:
                    province_code = self._normalize_id_token(
                        district_entry.get("parent_code")
                    )
            _register(
                record,
                level="ward",
                source="old",
                province_code=province_code,
                district_code=district_code,
            )

        engine.finalize()

    def _lookup_old_district_record(
        self, district_code: Optional[Any]
    ) -> Optional[Dict[str, Any]]:
        if not district_code:
            return None
        key = str(district_code).strip()
        entry = self.old_district_records.get(key)
        if entry:
            return entry
        normalized = self._normalize_id_token(district_code)
        if normalized and normalized != key:
            return self.old_district_records.get(normalized)
        return None

    def _load_reference_names(self, path: str):
        reference_map = {}
        if not os.path.exists(path):
            return {}, []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw_name = line.strip()
                if not raw_name:
                    continue
                standardized_name = self.standardize_name(raw_name)
                if standardized_name:
                    reference_map.setdefault(standardized_name, []).append(raw_name)
        return reference_map, list(reference_map.keys())

    def _select_reference_candidate(
        self, candidates: List[str], raw_value: Optional[str]
    ) -> Tuple[Optional[str], float]:
        if not candidates:
            return None, 0.0
        if not raw_value:
            return candidates[0], 100.0

        normalized_raw = raw_value.casefold()
        best_candidate = None
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
        reference_map: dict,
        reference_choices: list,
        score_cutoff: int,
        raw_value: Optional[str] = None,
    ) -> Tuple[Optional[str], bool]:
        if not standardized_value or not reference_choices:
            return (raw_value if raw_value else None, False)

        direct_candidates = reference_map.get(standardized_value)
        if direct_candidates:
            candidate, score = self._select_reference_candidate(
                direct_candidates, raw_value
            )
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
        primary: Optional[str],
        raw_value: Optional[str],
        extra_aliases: Optional[List[str]] = None,
    ) -> List[str]:
        aliases: List[str] = []
        seen: Set[str] = set()

        def _add(value: Optional[str]):
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

    def _get_special_province_aliases(self, province_name: Optional[str]) -> List[str]:
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
            normalized = self.standardize_name(stripped, False)
            if not normalized:
                return ""
            # Drop administrative prefixes so alias comparisons only rely on the core name
            normalized = re.sub(r"^(tinh|thanh pho|tp)\s+", "", normalized).strip()
            return normalized

        province_std = _canonicalize(province_name)
        if not province_std:
            return []

        aliases: List[str] = []

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

    def _standardize_aliases(self, aliases: List[str]) -> Set[str]:
        normalized: Set[str] = set()
        for alias in aliases:
            std = self.standardize_name(alias, False)
            if std:
                normalized.add(std)
        return normalized

    def _titleize_token(self, token: Optional[str]) -> str:
        if not token:
            return ""
        parts = [part.capitalize() for part in token.split() if part]
        return " ".join(parts) or token

    def _normalize_numeric_component_key(
        self,
        value: Optional[str],
        *,
        default_prefix: Optional[str] = None,
    ) -> Optional[str]:
        if not value:
            return None
        text = value.strip()
        if not text:
            return None
        match = re.match(r"^(phuong|p|xa|x|thi tran|tt)\s*(\d+)$", text)
        if match:
            prefix = match.group(1)
            digits = match.group(2).lstrip("0") or "0"
            if prefix in {"p", "phuong"}:
                prefix = "phuong"
            elif prefix in {"x", "xa"}:
                prefix = "xa"
            return f"{prefix} {digits}"
        if text.isdigit() and default_prefix:
            digits = text.lstrip("0") or "0"
            return f"{default_prefix} {digits}"
        return None

    def _normalize_detected_ward_token(self, token: Optional[str]) -> str:
        std = self.standardize_name(token, False) if token else ""
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

    def _split_address_segments(self, original: str) -> List[Tuple[str, str]]:
        if not original:
            return []
        segments: List[Tuple[str, str]] = []
        for part in re.split(r"[,;\n]+", original):
            cleaned = part.strip()
            if not cleaned:
                continue
            std = self.standardize_name(cleaned, False)
            if not std:
                continue
            segments.append((std, cleaned))
        return segments

    def _segment_has_location_prefix(self, segment_std: Optional[str]) -> bool:
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

    def _gather_alias_values(
        self,
        current_value: Optional[str],
        info: Optional[Dict[str, Any]],
        *,
        level: str,
        extra_values: Optional[List[Optional[str]]] = None,
    ) -> List[str]:
        aliases: List[str] = []
        seen: Set[str] = set()

        def _add(value: Optional[str]) -> None:
            if not isinstance(value, str):
                return
            candidate = value.strip()
            if not candidate or candidate in seen:
                return
            aliases.append(candidate)
            seen.add(candidate)

        _add(current_value)

        if isinstance(info, dict):
            for key in ("full_name", "name"):
                _add(info.get(key))
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
                _add(raw)

        return aliases

    def _prefer_component_alias_from_segments(
        self,
        alias_values: List[str],
        segments: List[Tuple[str, str]],
        *,
        require_prefix: bool = False,
        level: Optional[str] = None,
    ) -> Optional[str]:
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
                if pair == "thanh pho":
                    return True
                return False

            if target_level == "district":
                if first in {"quan", "q", "huyen", "h", "tx"}:
                    return True
                if pair in {"thi xa"}:
                    return True
                return False

            if target_level == "ward":
                if first in {"phuong", "p", "xa", "x", "tt"}:
                    return True
                if pair in {"thi tran", "dac khu"}:
                    return True
                return False

            return False

        alias_norms: List[str] = []
        seen: Set[str] = set()
        for alias in alias_values:
            std = self.standardize_name(alias, False)
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
                if (
                    alias_std == segment_std
                    or alias_std in segment_std
                    or segment_std in alias_std
                ):
                    return raw_value.strip()
        return None

    def _recover_component_from_input(
        self,
        target_std: Optional[str],
        segments: List[Tuple[str, str]],
    ) -> Optional[str]:
        if not target_std:
            return None
        target_std = target_std.strip()
        if not target_std:
            return None
        best_match: Optional[str] = None
        best_len = -1
        for segment_std, raw in segments:
            if target_std == segment_std:
                length = len(segment_std)
            elif target_std in segment_std:
                length = len(target_std)
            else:
                continue
            if length > best_len:
                best_len = length
                best_match = raw.strip()
        return best_match

    def _strip_generic_prefix(self, value: Optional[str]) -> str:
        if not value:
            return ""
        tokens = value.split()

        def _is_pair_generic(tok0: str, tok1: str) -> bool:
            return (tok0 == "thanh" and tok1 == "pho") or (
                tok0 == "thi" and tok1 in {"tran", "xa"}
            )

        while tokens:
            tok0 = tokens[0]
            if tok0 in {"phuong", "p", "xa", "x", "quan", "q", "huyen", "h", "tp", "tinh"}:
                tokens.pop(0)
                continue
            if len(tokens) >= 2 and _is_pair_generic(tokens[0], tokens[1]):
                tokens.pop(0)
                tokens.pop(0)
                continue
            break
        return " ".join(tokens)

    def _reference_aliases_for_level(
        self,
        level: str,
        code: Optional[Any],
    ) -> List[str]:
        if not code:
            return []
        code_str = str(code).strip()
        candidates = [code_str]
        normalized = self._normalize_id_token(code)
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
        extras: List[str] = []
        for key in ("full_name",):
            value = entry.get(key)
            if isinstance(value, str) and value not in extras:
                extras.append(value)
        return extras

    def _register_alias_lookup_entry(
        self,
        registry: Dict[str, List[Dict[str, Any]]],
        alias_set: Set[str],
        payload: Dict[str, Any],
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
        province_aliases_std: Optional[Set[str]] = None,
        district_aliases_std: Optional[Set[str]] = None,
        ward_aliases_std: Optional[Set[str]] = None,
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

    def _derive_ward_names(
        self,
        ward_name: Optional[str],
        ward_meta: Optional[Any],
    ) -> Tuple[str, str]:
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

    def _augment_aliases(self, aliases: List[str], level: str) -> List[str]:
        seen: Set[str] = set()
        normalized_aliases: List[str] = []
        for alias in aliases:
            if alias not in seen:
                normalized_aliases.append(alias)
                seen.add(alias)

        extras: List[str] = []
        for alias in normalized_aliases:
            std = self.standardize_name(alias, False)
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
        self, value: Optional[str], lookup: Dict[str, Set[int]]
    ) -> Optional[str]:
        if not value:
            return None
        return value if value in lookup else None

    def _resolve_detected_component(
        self,
        level: str,
        detected_value: Optional[str],
        *,
        expected_province: Optional[str] = None,
        expected_district: Optional[str] = None,
        source_string: Optional[str] = None,
    ) -> Optional[str]:
        if not detected_value:
            return None

        invert_map = {
            "province": self.invert_province_to_indices,
            "district": self.invert_district_to_indices,
            "ward": self.invert_ward_to_indices,
        }
        lookup = invert_map.get(level)
        if not lookup:
            return None

        indices = lookup.get(detected_value, set())
        if not indices:
            return None

        expected_province_std = (
            self.standardize_name(expected_province, False)
            if expected_province
            else None
        )
        expected_district_std = (
            self.standardize_name(expected_district, False)
            if expected_district
            else None
        )

        fallback: Optional[str] = None
        candidates: List[Tuple[str, str]] = []

        source_norm = source_string if source_string else ""
        enforce_specificity = (
            level == "ward" and not expected_province and not expected_district
        )

        def _collect(relax: bool) -> List[Tuple[str, str]]:
            nonlocal fallback
            local_candidates: List[Tuple[str, str]] = []
            local_fallback: Optional[str] = None
            for idx in indices:
                node = self.address_node_list[idx]
                if level == "province":
                    name = node.province_name
                    if not name:
                        continue
                    norm = self.standardize_name(name, False)
                    local_candidates.append((name, norm))
                    if local_fallback is None:
                        local_fallback = name
                    continue

                if level == "district":
                    name = node.district_name
                    if not name:
                        continue
                    node_prov_std = (
                        self.standardize_name(node.province_name, False)
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
                    norm = self.standardize_name(name, False)
                    local_candidates.append((name, norm))
                    if local_fallback is None:
                        local_fallback = name
                    continue

                # ward level
                name = node.ward_name
                if not name:
                    continue
                node_prov_std = (
                    self.standardize_name(node.province_name, False)
                    if node.province_name
                    else None
                )
                node_dist_std = (
                    self.standardize_name(node.district_name, False)
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
                norm = self.standardize_name(name, False)
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

        return fallback or candidates[0][0]

    def _lookup_province_info(
        self, province_name: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not province_name:
            return None
        key = self.standardize_name(province_name, False)
        if not key:
            return None
        return self.province_lookup.get(key)

    def _lookup_district_info(
        self,
        district_name: Optional[str],
        province_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not district_name:
            return None
        district_key = self.standardize_name(district_name, False)
        if not district_key:
            return None
        province_key = (
            self.standardize_name(province_name, False) if province_name else None
        )
        if province_key:
            info = self.district_lookup.get((province_key, district_key))
            if info:
                return info
        candidates = self.district_lookup_by_name.get(district_key, [])
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _lookup_ward_info(
        self,
        ward_name: Optional[str],
        province_name: Optional[str] = None,
        district_name: Optional[str] = None,
        preferred_format: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        if not ward_name:
            return None
        ward_key = self.standardize_name(ward_name, False)
        if not ward_key:
            return None
        province_key = (
            self.standardize_name(province_name, False) if province_name else None
        )
        district_key = (
            self.standardize_name(district_name, False) if district_name else None
        )

        ward_keys = [ward_key]
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

        if province_key and district_key:
            for key in ward_keys:
                info = self.ward_lookup.get((province_key, district_key, key))
                if info:
                    return info

        if province_key:
            for key in ward_keys:
                province_candidates = self.ward_lookup_by_province_name.get(
                    (province_key, key), []
                )
                if len(province_candidates) == 1:
                    return province_candidates[0]

        if district_key:
            district_candidates = []
            for entry in self.ward_lookup_by_district_key.get(district_key, []):
                entry_name_std = self.standardize_name(entry.get("name"), False)
                if not entry_name_std:
                    continue
                for key in ward_keys:
                    if entry_name_std == key or self._numeric_token_match(
                        entry_name_std, key
                    ):
                        district_candidates.append(entry)
                        break
            if len(district_candidates) == 1:
                return district_candidates[0]

        fallback_candidates: List[Dict[str, Any]] = []
        for key in ward_keys:
            bucket = self.ward_lookup_by_name.get(key, [])
            if bucket:
                fallback_candidates.extend(bucket)
        if not fallback_candidates:
            fallback_candidates = self.ward_lookup_by_name.get(ward_key, [])
        if not fallback_candidates:
            return None

        def _std(value: Optional[str]) -> str:
            return self.standardize_name(value, False) if value else ""

        candidates = fallback_candidates

        if province_key:
            province_matches = [
                c
                for c in candidates
                if (c.get("province_key") and c["province_key"] == province_key)
                or _std(c.get("province_name")) == province_key
            ]
            if province_matches:
                candidates = province_matches

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

        if preferred_format is not None:
            format_matches = [
                c for c in candidates if c.get("is_new_format") is preferred_format
            ]
            if len(format_matches) == 1:
                return format_matches[0]
            if format_matches:
                candidates = format_matches

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

        def _candidate_sort_key(entry: Dict[str, Any]):
            is_new = entry.get("is_new_format")
            if prefer_new:
                format_rank = 0 if is_new is True else 1 if is_new is False else 2
            elif prefer_old:
                format_rank = 0 if is_new is False else 1 if is_new is True else 2
            else:
                # Default: True (0) < False (1) < unknown (2)
                format_rank = 0 if is_new is True else 1 if is_new is False else 2

            name_std = self.standardize_name(entry.get("name"), False) if entry.get("name") else None
            full_std = (
                self.standardize_name(entry.get("full_name"), False)
                if entry.get("full_name")
                else None
            )
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
        ward_info: Optional[Dict[str, Any]],
        ward_name: Optional[str],
        province_name: Optional[str],
        province_info: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[str]]:
        if not ward_info:
            return None, None

        known_district = ward_info.get("district_name")
        if known_district:
            if ward_info.get("is_new_format") is False:
                return known_district, ward_info.get("district_id")
            return None, None

        ward_std = self.standardize_name(ward_name, False) if ward_name else ""
        if not ward_std:
            return None, None

        province_std = (
            self.standardize_name(province_name, False) if province_name else None
        )

        def _province_matches(entry: Dict[str, Any]) -> bool:
            if not province_std:
                return True
            entry_key = entry.get("province_key")
            if not entry_key and entry.get("province_name"):
                entry_key = self.standardize_name(entry.get("province_name"), False)
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
        detected_token: Optional[str],
        expected_province: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not detected_token:
            return None
        token = detected_token.strip()
        if not token:
            return None

        province_std = (
            self.standardize_name(expected_province, False)
            if expected_province
            else None
        )

        def _filter(
            entries: List[Dict[str, Any]], *, enforce_province: bool
        ) -> Optional[Dict[str, Any]]:
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

        if province_std:
            province_bucket = self.ward_lookup_by_province_name.get(
                (province_std, token), []
            )
            entry = _filter(province_bucket, enforce_province=True)
            if entry:
                return entry

        candidates = self.ward_lookup_by_name.get(token, [])
        filtered = [
            entry
            for entry in candidates
            if entry.get("is_new_format")
            and (
                not province_std
                or self._entry_aligns_with_province(entry, expected_province)
            )
        ]
        if province_std:
            entry = _filter(filtered, enforce_province=True)
            if entry:
                return entry
        # When no province hint and multiple new-format entries share the same name,
        # avoid guessing to prevent cross-province drift.
        if not province_std and len(filtered) != 1:
            return None
        if filtered:
            filtered_sorted = sorted(
                filtered,
                key=lambda e: (
                    e.get("district_key") or "",
                    e.get("province_key") or "",
                    e.get("id") or "",
                ),
            )
            return filtered_sorted[0]

        return _filter(candidates, enforce_province=False)

    def _entry_aligns_with_province(
        self, entry: Optional[Dict[str, Any]], expected_province: Optional[str]
    ) -> bool:
        if not expected_province or not isinstance(entry, dict):
            return True
        expected_std = self.standardize_name(expected_province, False)
        if not expected_std:
            return True
        entry_std = entry.get("province_key")
        if not entry_std:
            entry_std = (
                self.standardize_name(entry.get("province_name"), False)
                if entry.get("province_name")
                else None
            )
        if entry_std:
            return entry_std == expected_std
        return False

    def _prefer_hierarchical_ward_entry(
        self,
        normalized_token: Optional[str],
        entry: Optional[Dict[str, Any]],
        *,
        expected_province: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not normalized_token or not entry:
            return entry
        if entry.get("district_key"):
            return entry

        if not self._entry_aligns_with_province(entry, expected_province):
            return entry

        original_is_new = entry.get("is_new_format")
        entry_id = self._normalize_id_token(entry.get("id") or entry.get("code"))
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
                candidate_id = self._normalize_id_token(
                    candidate.get("id") or candidate.get("code")
                )
                if candidate_id and candidate_id != entry_id:
                    continue
            return candidate

        return entry

    def _format_component(
        self,
        name: Optional[str],
        candidate_id: Optional[str],
        info: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not name:
            return None
        resolved_name = name
        if info:
            alt_name = info.get("full_name") or info.get("name")
            if alt_name:
                normalized = self.standardize_name(name, False)
                # Some legacy datasets store numeric-only names (e.g. "5"). Prefer the descriptive
                # full name in those cases so downstream consumers see "Quận 5" instead of "5".
                if normalized and normalized.isdigit():
                    resolved_name = alt_name
                elif not resolved_name.strip():
                    resolved_name = alt_name

        payload: Dict[str, Any] = {"name": resolved_name}
        component_id = candidate_id
        if component_id is None and info:
            component_id = info.get("id")
        if component_id is not None:
            payload["id"] = component_id
        if info:
            extended_name = info.get("full_name")
            if extended_name:
                payload["full_name"] = extended_name
            code_value = info.get("code")
            if code_value:
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
        district: Optional[str],
        ward: Optional[str],
    ) -> Optional[str]:
        district_std = self.standardize_name(district, False) if district else ""
        ward_std = self.standardize_name(ward, False) if ward else ""

        candidate_sets: List[Set[str]] = []

        if ward_std:
            indices = self.invert_ward_to_indices.get(ward_std, set())
            ward_candidates: Set[str] = set()
            for idx in indices:
                node = self.address_node_list[idx]
                if not node.province_name:
                    continue
                if district_std:
                    node_dist_std = (
                        self.standardize_name(node.district_name, False)
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
            district_candidates: Set[str] = set()
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
        province: Optional[str],
        ward: Optional[str],
        *,
        source_string: Optional[str] = None,
    ) -> Optional[str]:
        ward_std = self.standardize_name(ward, False) if ward else ""
        if not ward_std:
            return None

        province_std = self.standardize_name(province, False) if province else None
        indices = self.invert_ward_to_indices.get(ward_std, set())
        if not indices:
            return None

        candidate_entries: List[Tuple[str, str, Optional[str]]] = []
        for idx in indices:
            node = self.address_node_list[idx]
            district_name = node.district_name
            if not district_name:
                continue
            node_prov_std = (
                self.standardize_name(node.province_name, False)
                if node.province_name
                else None
            )
            if province_std and node_prov_std and node_prov_std != province_std:
                continue
            district_std = self.standardize_name(district_name, False)
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
        province_aliases: List[str],
        district_aliases: List[str],
        ward_aliases: List[str],
        *,
        include_province: bool,
        include_district: bool,
        include_ward: bool,
    ) -> Tuple[str, Set[str]]:
        primary_parts: List[str] = []
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

        ngram_set: Set[str] = set()
        for ward_name in ward_candidates:
            for district_name in district_candidates:
                for province_name in province_candidates:
                    combined = " ".join(
                        part
                        for part in [ward_name, district_name, province_name]
                        if part
                    )
                    if not combined:
                        continue
                    standardized = self.standardize_name(combined)
                    if standardized:
                        ngram_set.update(self.generate_ngrams(standardized))

        if not ngram_set and primary_standardized:
            ngram_set.update(self.generate_ngrams(primary_standardized))

        return primary_standardized, ngram_set

    def standardize_name(self, name: str, advanced_process: bool = False) -> str:
        if not name:
            return ""

        # --- Bước 1: Đưa về chữ thường ---
        s = name.lower()

        # --- Bước 1.1: Loại bỏ dấu chấm và dấu phẩy ở đầu và cuối chuỗi ---
        s = re.sub(r"^[\.,]+", "", s)  # xóa tất cả . hoặc , ở đầu
        s = re.sub(r"[\.,]+$", "", s)  # xóa tất cả . hoặc , ở cuối
        # --- Bước 1.2: Xóa hẳn ký tự "/" ---
        s = s.replace("/", "")
        # # --- Bước 1.3: Thay các dấu "." và "-" bằng space ---
        # s = s.replace(".", " ").replace("-", " ")

        if advanced_process:

            s = re.sub(r"\b(t.t.h)\b", " thua thien hue ", s, flags=re.IGNORECASE)

            s = re.sub(r"\b(h.c.m|h.c.minh)\b", " ho chi minh ", s, flags=re.IGNORECASE)

            s = re.sub(r"\b(hn|h.noi|ha ni)\b", " ha noi ", s, flags=re.IGNORECASE)

            # --- Bước 2: Thay cụm từ thừa bằng space (thay chính xác 100%) ---
            redundant_phrases = [
                "thành phố",
                "thành phô",
                "thành fhố",
                "thanh fho",
                "thanh pho ",
                "thành. phố",
                "thành.phố",
                "tp.",
                "t.p",
                "tp ",
                "t.phố",
                "t. phố",
                "tỉnh",
                "tinh",
                "tt.",
                "t.",
                " t ",
                "quận",
                "qận",
                "qun",
                "q.",
                "q ",
                "huyện",
                "h.",
                " h ",
                ".h ",
                "district",
                "dist.",
                "dist ",
                "ward",
                "w.",
                "w ",
                "city",
                "province",
                "municipality",
                "town",
                "village",
                "commune",
                "thị xã",
                "thị.xã",
                "tx.",
                "t.xã",
                "tx ",
                "thị trấn",
                "thị.trấn",
                "tt ",
                "xã",
                "x.",
                "x ",
                "phường",
                "kp.",
                "p.",
                " p ",
                ".p ",
                "phường.",
                "phường ",
                "f",
                "j",
                "z",
                "w",
            ]

            for phrase in redundant_phrases:
                s = s.replace(phrase, " ")

            s = re.sub(
                r"\b("
                r"|tiểu\s*khu(\s*\d+\w*)?"  # tiểu khu 3, tiểu khu12a
                r"|khu\s*pho(\s*\d+\w*)?"  # khu phố, khu phố 3
                r"|khu\s*phố(\s*\d+\w*)?"  # khu phố, khu phố 3
                r"|khu\s*vuc(\s*\d+\w*)?"  # khu vực, khu vực 2
                r"|khu\s*vực(\s*\d+\w*)?"  # khu vực, khu vực 2
                r"|khu(\s*\d+\w*)?"  # khu, khu 3, khu12a
                r"|kp(\s*\d+\w*)?"  # kp2, kp 3
                r"|tổ\s*dân\s*phố(\s*\d+\w*)?"  # tổ dân phố 5, tổ dân phố12a
                r"|tổ(\s*\d+\w*)?"  # tổ 1
                r"|thôn(\s*\d+\w*)?"  # thôn 3
                r"|xóm(\s*\d+\w*)?"  # xóm 2
                r"|cụm(\s*\d+\w*)?"  # cụm 3
                r"|phố(\s*\d+\w*)?"  # phố 5
                r"|khóm(\s*\d+\w*)?"  # khóm 2
                r"|số\s*nhà(\s*\d+\w*)?"  # số nhà 12
                r"|số(\s*\d+\w*)?"  # số 12
                r"|nhà(\s*\d+\w*)?"  # nhà 12
                r"|ấp(\s*\d+\w*)?"  # ấp 1, ấp2
                r"|ngách\s*\d+\w*"  # ngách 12, ngách12a
                r"|ngõ\s*\d+\w*"  # ngõ 12, ngõ12a
                r"|hẻm\s*\d+\w*"
                r")\b",
                "",
                s,
                flags=re.IGNORECASE,
            )

            # --- Bước 3: Loại các cụm "tp" dính liền chữ, ví dụ "tpbao loc" → "bao loc" ---
            s = re.sub(r"\btp([a-z0-9]+)", r"\1", s)

        # --- Bước 4: Chuẩn hóa Unicode & bỏ dấu ---
        s = s.replace("đ", "d")
        s = unicodedata.normalize("NFD", s)
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

        # --- Bước 5: Giữ lại a-z, 0-9, space ---
        s = re.sub(r"[^a-z0-9\s]+", " ", s)

        if advanced_process:
            s = re.sub(
                r"\b(hochiminh|hochi\s*minh|ho\s*chiminh|hcm|hcminh)\b",
                "ho chi minh",
                s,
                flags=re.IGNORECASE,
            )
            if re.search(r"\bho chi minh\b", s, flags=re.IGNORECASE):
                mapping = {
                    "bc": "binh chanh",
                    "tb": "tan binh",
                    "bt": "binh thanh",
                    "gv": "go vap",
                    "pn": "phu nhuan",
                    "cc": "cu chi",
                    "hm": "hoc mon",
                    "nb": "nha be",
                }

                # Thay từng viết tắt bằng tên đầy đủ (chỉ thay khi là từ riêng biệt)
                for abbr, full in mapping.items():
                    s = re.sub(rf"\b{abbr}\b", full, s, flags=re.IGNORECASE)

            # --- Bước 7: Loại bỏ các chuỗi chứa từ 3 chữ số trở lên ---

            # Bỏ số 0 ở đầu của mọi cụm số
            s = re.sub(r"\b0+(\d+)\b", r"\1", s)
            # Tức là "abc123xyz" hoặc "123" đều bị loại bỏ phần chứa "123"
            s = re.sub(r"\d{3,}", "", s)

            # --- Bước 8: Bỏ 'p' hoặc 'q' trước số (vd: p1 → 1, q10 → 10) ---
            s = re.sub(r"\b[pq](\d+)\b", r"\1", s)

            # --- Bước X: Loại bỏ các cụm địa chỉ thừa ---

        # --- Bước 9: Gom space ---
        s = re.sub(r"\s+", " ", s).strip()
        # print(s)
        return s

    def _normalize_token_basic(self, token: str) -> str:
        if not token:
            return ""
        token = token.lower()
        token = token.replace("đ", "d")
        token = unicodedata.normalize("NFD", token)
        token = "".join(ch for ch in token if unicodedata.category(ch) != "Mn")
        token = re.sub(r"[^a-z0-9]+", "", token)
        return token

    def _is_generic_location_token(
        self, raw: Optional[str], norm: Optional[str]
    ) -> bool:
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
        component: Optional[str],
        extra_aliases: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, Any]:
        signature: Dict[str, Any] = {
            "sequences": [],
            "tokens": set(),
            "abbreviation_sequences": set(),
        }
        candidates: List[str] = []
        if component:
            candidates.append(component)
        if extra_aliases:
            for alias in extra_aliases:
                if alias:
                    candidates.append(alias)
        if not candidates:
            return signature

        processed: Set[str] = set()

        def _register(parts: List[str], *, is_abbreviation: bool = False):
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
            standardized = self.standardize_name(value, False)
            if not standardized or standardized in processed:
                continue
            processed.add(standardized)
            parts = [p for p in standardized.split() if p]
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

            abbr_parts: List[str] = []
            for part in parts:
                if not part:
                    continue
                abbr_parts.append(part if part.isdigit() else part[0])
            abbr = "".join(abbr_parts)
            if len(abbr) >= 2:
                _register([abbr], is_abbreviation=True)
                _register([f"tp{abbr}"], is_abbreviation=True)
                _register(["tp", abbr], is_abbreviation=True)

                split_abbr = re.findall(r"[a-z]+|\d+", abbr)
                if len(split_abbr) > 1 and all(split_abbr):
                    _register(split_abbr, is_abbreviation=True)

        return signature

    def _extract_street_address(
        self,
        original: str,
        node: "AddressParser.AddressNode",
        component_aliases: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        if not original:
            return ""

        alias_map = component_aliases or {}
        profiles = {
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
            return original.strip()

        token_matches = list(re.finditer(r"\b\w+\b", original, flags=re.UNICODE))
        if not token_matches:
            return original.strip()

        tokens = []
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
            return original.strip()
        protected_generics: Set[int] = set()
        for idx in range(1, token_count):
            prev_norm = tokens[idx - 1]["norm"]
            curr_norm = tokens[idx]["norm"]
            if curr_norm == "xa" and prev_norm in {"cu", "khu"}:
                protected_generics.add(idx)

        # Pre-compute comma-separated segments so we can avoid crossing them
        segments: List[Tuple[int, int]] = []
        segment_token_indices: List[List[int]] = []
        token_segments: List[int] = [-1] * token_count
        if token_count > 0:
            comma_positions = [m.start() for m in re.finditer(",", original)]
            if comma_positions:
                start_char = 0
                for pos in comma_positions:
                    segments.append((start_char, pos))
                    start_char = pos + 1
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

        indices_to_remove: Set[int] = set()

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

        def _sequence_has_generic_tokens(seq_tokens: List[str]) -> bool:
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

        street_descriptor_tokens = {
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

        def _segment_match_ratio(
            segment_idx: int, start_idx: int, length: int
        ) -> float:
            if segment_idx < 0 or segment_idx >= len(segment_token_indices):
                return 0.0
            segment_tokens = segment_token_indices[segment_idx]
            if not segment_tokens:
                return 0.0
            end_idx = start_idx + length
            covered = sum(
                1 for token_idx in segment_tokens if start_idx <= token_idx < end_idx
            )
            return covered / max(1, len(segment_tokens))

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

        for profile in profiles.values():
            sequences: List[List[str]] = profile["sequences"]
            abbreviation_sequences: Set[Tuple[str, ...]] = profile.get(
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
                        elif _adjacent_generic(idx, -1) or _adjacent_generic(
                            idx + seq_len - 1, 1
                        ):
                            allow_removal = True
                        else:
                            is_tail_segment = (
                                segment_idx >= 0 and segment_idx >= len(segments) - 1
                            )
                            if is_tail_segment:
                                coverage = _segment_match_ratio(
                                    segment_idx, idx, seq_len
                                )
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
                                has_next_generic = _adjacent_generic(
                                    idx + seq_len - 1, 1
                                )
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
                has_marked = any(
                    token_idx in indices_to_remove for token_idx in idx_list
                )
                if not (has_generic or has_marked):
                    continue
                should_remove = all(
                    _is_generic(token_idx) or token_idx in indices_to_remove
                    for token_idx in idx_list
                )
                if should_remove:
                    indices_to_remove.update(idx_list)

        if not indices_to_remove:
            return original.strip()

        mask = [False] * len(original)
        for token_idx in indices_to_remove:
            start = tokens[token_idx]["start"]
            end = tokens[token_idx]["end"]
            for pos in range(start, end):
                mask[pos] = True

        filtered_chars = [ch for pos, ch in enumerate(original) if not mask[pos]]
        street = "".join(filtered_chars)
        street = re.sub(r"[,\.;:]+\s*", " ", street)
        street = re.sub(r"\s+", " ", street).strip(" ,;.-")
        if street:
            street = re.sub(
                r"(?i)\bvi\S*t[\s-]*nam\b\.?$",
                "",
                street,
            ).strip(" ,;.-")
        return street.strip()

    def generate_ngrams(self, s: str, n: int = 4) -> list:
        s = f" {s} "  # Thêm khoảng trắng ở đầu và cuối để tạo n-gram chính xác
        ngrams = [s[i : i + n] for i in range(len(s) - n + 1)]
        return ngrams

    def generate_ngram_inverted_index(
        self, ngram_list: list, index: int, invert_ngram_to_index_dict: dict
    ):
        for ngram in ngram_list:
            if ngram not in invert_ngram_to_index_dict:
                invert_ngram_to_index_dict[ngram] = set()
            invert_ngram_to_index_dict[ngram].add(index)

    def ngram_address_piece_list(self, input_ngram_list: list, top_k: int) -> list:
        counter = Counter()
        invert_dict = self.invert_ngrams_idx

        # Iterate unique ngrams to avoid redundant counting
        for ngram in sorted(set(input_ngram_list)):
            if ngram in invert_dict:
                counter.update(invert_dict[ngram])  # ✅ xử lý hàng loạt

        # Return only top-K candidates to cap cost (heap-based in CPython)
        return counter.most_common(top_k)

    # --------------------
    # Prefix detection + prefilter
    # --------------------
    def _detect_by_prefix(
        self, s: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        # s should be standardized without advanced removal (to keep prefix words)
        if not s:
            return None, None, None

        # When we already have explicit segment separators (we inject `|` between
        # comma-separated parts), do not use token boundaries like `tinh`/`tp`.
        # Those tokens can legitimately appear inside names (e.g. "Sơn Tịnh"),
        # causing premature truncation (e.g. "son" -> fuzzy-match "son la").
        has_segment_separators = "|" in s
        if has_segment_separators:
            admin_boundary = r"(?:\||$)"
            sub_admin_boundary = r"(?:\||$)"
            prefix_anchor = r"(?:^|\|)\s*"
        else:
            admin_boundary = (
                r"(?:\b(?:quan|q|huyen|h|thi xa|tx|thi tran|tt|phuong|p|xa|x|tp|tinh|thanh pho)\b|\||$)"
            )
            sub_admin_boundary = (
                r"(?:\b(?:phuong|p|xa|x|thi tran|tt|quan|q|huyen|h|thi xa|tx|thanh pho|tinh|tp)\b|\||$)"
            )
            prefix_anchor = ""

        # Compile once per call; small overhead compared to overall cost
        province_tinh_pref = re.compile(
            rf"{prefix_anchor}\b(?:tinh)\b\s+([a-z0-9 ]+?)(?={admin_boundary})"
        )
        province_pref = re.compile(
            rf"{prefix_anchor}\b(?:thanh pho|tp|tinh)\b\s+([a-z0-9 ]+?)(?={admin_boundary})"
        )
        district_pref = re.compile(
            rf"{prefix_anchor}\b(?P<prefix>quan|q|huyen|thi xa|thi tran|thanh pho|tp)\b\s+(?P<fragment>[a-z0-9 ]+?)(?={sub_admin_boundary})"
        )
        ward_pref = re.compile(
            rf"{prefix_anchor}\b(?P<prefix>phuong|p|xa|thi tran|dac\s*khu)\b\s+(?P<fragment>[a-z0-9 ]+?)(?={sub_admin_boundary})"
        )

        def _digit_key(value: str) -> str:
            return "".join(ch for ch in value if ch.isdigit())

        def _pick_best(fragment: str, choices: List[str]) -> Optional[str]:
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
            exact_fragment = fragment
            if exact_fragment in choices:
                return exact_fragment

            fragment_digits = _digit_key(fragment)
            narrowed_choices = choices
            if fragment_digits:
                digit_matches = [
                    candidate
                    for candidate in choices
                    if _digit_key(candidate) == fragment_digits
                ]
                if digit_matches:
                    narrowed_choices = digit_matches

            candidates = rf_process.extract(
                fragment,
                narrowed_choices,
                scorer=partial_ratio,
                score_cutoff=70,
                limit=10,
            )
            if not candidates:
                return None

            best_choice = None
            best_score = -1.0
            best_len_delta = None

            for candidate, score, _ in candidates:
                if candidate == exact_fragment:
                    return candidate
                len_delta = abs(len(candidate) - len(fragment))
                if score > best_score:
                    best_choice = candidate
                    best_score = score
                    best_len_delta = len_delta
                    continue
                if score == best_score:
                    if len_delta < (
                        best_len_delta if best_len_delta is not None else float("inf")
                    ):
                        best_choice = candidate
                        best_len_delta = len_delta
                        continue
            return best_choice

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
        m = province_tinh_pref.search(s) or province_pref.search(s)
        if m and self.province_names_std:
            fragment = (m.group(1) or "").strip()
            fragment = _trim_province_fragment(fragment)
            frag_tokens = [tok for tok in fragment.split() if tok]
            if len(frag_tokens) == 1 and len(frag_tokens[0]) <= 2:
                fragment = ""
                frag_tokens = []
            while frag_tokens and frag_tokens[-1] in {"viet", "nam", "vietnam"}:
                frag_tokens.pop()
            fragment = " ".join(frag_tokens)
            if fragment in {"hcm", "hcmc", "sai gon", "saigon", "sg"} or (
                frag_tokens and frag_tokens[0] in {"hcm", "hcmc", "sg"}
            ):
                prov = "ho chi minh"
            else:
                prov = _pick_best(fragment, sorted(self.province_names_std))

        dist_num = None
        district_choices = (
            sorted(
                candidate
                for candidate in self.district_names_std
                if not candidate.isdigit() and len(candidate) >= 3
            )
            if self.district_names_std
            else None
        )
        if district_choices:
            m_num = re.search(r"\b(?:quan)\s*(\d{1,3})\b", s)
            if not m_num:
                m_num = re.search(r"\b(?:quan)(\d{1,3})\b", s)
            if not m_num:
                m_num = re.search(r"\bq\.?\s*(\d{1,3})\b", s)
            if not m_num:
                m_num = re.search(r"\bq(\d{1,3})\b", s)
            if m_num:
                # Avoid false positives from lot codes like "Lô Q10-03" where "Q10" is not "Quận 10".
                prefix_context = s[max(0, m_num.start() - 8) : m_num.start()]
                if re.search(r"\b(lo|lot)\s*$", prefix_context):
                    m_num = None
            if m_num:
                raw = m_num.group(1).strip()
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
                    "thi xa": 4,
                    "thi tran": 3,
                    "thanh pho": 2,
                    "tp": 2,
                }
                return priority_map.get(prefix, 1)

            def _is_province_like(candidate: str, prefix: str) -> bool:
                if not candidate:
                    return False
                if prov and candidate == prov:
                    return True
                if prefix in {"thanh pho", "tp"}:
                    return candidate in self.province_names_std
                return False

            best_dist = None
            best_priority = -1
            best_score = -1.0
            best_pos = -1
            for match in district_pref.finditer(s):
                fragment = (match.group("fragment") or "").strip()
                prefix = (match.group("prefix") or "").strip()
                if not fragment or not prefix:
                    continue
                # "TP/Thành phố" segments frequently denote the province-level municipality
                # ("TP Hà Nội", "TP Đà Nẵng", ...). When the fragment matches a known province,
                # do not treat it as a district hint.
                if prefix in {"thanh pho", "tp"} and self.province_names_std:
                    if fragment in self.province_names_std:
                        continue
                    if prov and partial_ratio(fragment, prov) >= 90:
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
                candidate = _pick_best(fragment, district_choices)
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
                    if (
                        score == best_score
                        and best_pos >= 0
                        and match.start() >= best_pos
                    ):
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
                    "dac khu": 4,
                    "p": 3,
                    "phuong": 3,
                    "thi tran": 2,
                    "thi xa": 2,
                    "xa": 1,
                }
                return priority_map.get(prefix, 0)

            def _preceding_token(start_index: int) -> Optional[str]:
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
                "thi tran": "thi tran",
                "town": "thi tran",
                "thi xa": "thi xa",
                "xa": "xa",
                "commune": "xa",
                "dac khu": "dac khu",
                "special administrative region": "dac khu",
            }

            def _try_prefixed_candidate(prefix: str, fragment: str) -> Optional[str]:
                canonical = prefix_normalize_map.get(prefix)
                if not canonical:
                    return None
                normalized_fragment = fragment
                prefix_token = f"{canonical} "
                if normalized_fragment.startswith(prefix_token):
                    normalized_fragment = normalized_fragment[
                        len(prefix_token) :
                    ].strip()
                fused = f"{canonical} {normalized_fragment}".strip()
                fused = re.sub(r"\s+", " ", fused)
                if fused in self.ward_names_std:
                    return fused
                if normalized_fragment.isdigit():
                    digits = normalized_fragment.lstrip("0") or "0"
                    fused_digits = f"{canonical} {digits}".strip()
                    if fused_digits in self.ward_names_std:
                        return fused_digits
                return None

            for m in ward_pref.finditer(s):
                fragment = (m.group("fragment") or "").strip()
                prefix = (m.group("prefix") or "").strip()
                if not fragment or not prefix:
                    continue
                blocker = _preceding_token(m.start())
                if blocker and blocker in hamlet_prefix_blockers:
                    continue
                if prefix == "xa":
                    prev_token = _preceding_token(m.start())
                    if prev_token == "cu":
                        continue
                if prefix in ("dac khu", "special administrative region"):
                    frag_tokens = fragment.split()
                    trimmed: List[str] = []
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
                    candidate = _pick_best(fragment, sorted(self.ward_names_std))
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

    def _detect_special_province_token(self, standardized_basic: str) -> Optional[str]:
        """
        Detect legacy province aliases (e.g. 'thua thien hue') directly from the
        standardized string when the user omits administrative prefixes.
        """
        if not standardized_basic:
            return None
        if re.search(r"\b(hcmc?)\b", standardized_basic):
            return "ho chi minh"
        for synonyms, _ in SPECIAL_PROVINCE_MAP.items():
            if isinstance(synonyms, (list, tuple, set)):
                candidates = synonyms
            else:
                candidates = (synonyms,)
            for alias in candidates:
                alias_std = self.standardize_name(alias, False)
                if alias_std and alias_std in standardized_basic:
                    return alias_std
        return None

    def _detect_suffix_province_token(self, standardized_basic: str) -> Optional[str]:
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
        trimmed: List[str] = list(tokens)
        while trimmed and trimmed[-1] in self._GENERIC_LOCATION_TOKENS:
            trimmed.pop()
        if not trimmed:
            return None

        max_window = min(4, len(trimmed))
        for window in range(max_window, 1, -1):
            fragment = " ".join(trimmed[-window:])
            if fragment in self.province_names_std:
                return fragment

        last_token = trimmed[-1]
        if len(last_token) >= 4 and last_token in self.province_names_std:
            return last_token

        return None

    def _prefilter_by_prefix(self, standardized_basic: str) -> List[int]:
        prov, dist, ward = self._detect_by_prefix(standardized_basic)
        candidates: Optional[Set[int]] = None

        def _merge(current: Optional[Set[int]], newset: Set[int]) -> Optional[Set[int]]:
            if not newset:
                return current
            return (
                set(newset)
                if current is None
                else (current & newset if current else set())
            )

        if ward:
            candidates = _merge(
                candidates, self.invert_ward_to_indices.get(ward, set())
            )
        if dist:
            candidates = _merge(
                candidates, self.invert_district_to_indices.get(dist, set())
            )
        if prov:
            candidates = _merge(
                candidates, self.invert_province_to_indices.get(prov, set())
            )

        # If nothing detected, return empty list to signal fallback to n-gram path
        if not candidates:
            return []
        # Return stable list of indices
        return sorted(candidates)

    def _select_candidate_with_hints(
        self,
        candidates: List[Tuple[int, float, str]],
        detected_components: Tuple[Optional[str], Optional[str], Optional[str]],
    ) -> Optional[int]:
        if not candidates:
            return None
        prov_hint, dist_hint, ward_hint = (
            detected_components if detected_components else (None, None, None)
        )
        if not any((prov_hint, dist_hint, ward_hint)):
            return candidates[0][0]

        def _norm(value: Optional[str]) -> Optional[str]:
            if not value:
                return None
            normalized = self.standardize_name(value, False)
            return normalized or None

        prov_hint = _norm(prov_hint)
        dist_hint = _norm(dist_hint)
        ward_hint = _norm(ward_hint)

        best_idx: Optional[int] = None
        best_key: Optional[Tuple[int, int, int, float]] = None

        for idx, score, _ in candidates:
            node = self.address_node_list[idx]
            node_prov = _norm(node.province_name)
            node_dist = _norm(node.district_name)
            node_ward = _norm(node.ward_name)

            ward_match = 1 if ward_hint and node_ward == ward_hint else 0
            district_match = 1 if dist_hint and node_dist == dist_hint else 0
            province_match = 1 if prov_hint and node_prov == prov_hint else 0

            ward_mismatch = (
                1 if ward_hint and node_ward and node_ward != ward_hint else 0
            )
            district_mismatch = (
                1 if dist_hint and node_dist and node_dist != dist_hint else 0
            )
            province_mismatch = (
                1 if prov_hint and node_prov and node_prov != prov_hint else 0
            )

            ward_missing = 1 if ward_hint and not node_ward else 0
            district_missing = 1 if dist_hint and not node_dist else 0
            province_missing = 1 if prov_hint and not node_prov else 0

            match_score = (ward_match * 6) + (district_match * 3) + province_match
            mismatch_penalty = (
                (ward_mismatch * 6) + (district_mismatch * 3) + province_mismatch
            )
            missing_penalty = (
                (ward_missing * 3) + (district_missing * 2) + province_missing
            )

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
        input_ngram_set: set,
        ngram_address_piece_list: list,
        partial_input_string: bool,
        detected_components: Tuple[Optional[str], Optional[str], Optional[str]],
    ) -> list:
        # Stage 1: filter by Dice; collect IDs whose Dice >= gate
        detected_prov, detected_dist, detected_ward = (
            detected_components if detected_components else (None, None, None)
        )

        input_set = input_ngram_set
        input_set_length = len(input_set)
        filtered_entries: list[Tuple[int, float]] = []
        dice_entries: list[Tuple[int, float]] = []

        index = 0
        for idx_count in ngram_address_piece_list:
            idx = idx_count[0]
            candidate_ngrams = self.address_node_list[idx].ngram_list

            # Fast overlap count without building set
            intersection = 0
            for gram in input_set:
                if gram in candidate_ngrams:
                    intersection += 1

            dice_score = (2 * intersection) / (input_set_length + len(candidate_ngrams))
            dice_entries.append((idx, dice_score))
            index += 1

            if dice_score >= self.DICE_GATE:
                filtered_entries.append((idx, dice_score))
            elif index >= 200:
                # Counter is ordered by frequency; dice will only go down after this point
                break
        if not filtered_entries:
            if dice_entries:
                # Fall back to the best-overlapping candidates when Dice is too strict,
                # so long free-text inputs with street info still produce candidates.
                filtered_entries = sorted(
                    dice_entries, key=lambda item: item[1], reverse=True
                )[:80]
            if not filtered_entries:
                return []

        # Optional prefix-based filtering to favour nodes aligned with detected components
        prefix_filter: Optional[Set[int]] = None
        if detected_ward:
            prefix_filter = set(self.invert_ward_to_indices.get(detected_ward, set()))
        if detected_dist:
            dist_set = set(self.invert_district_to_indices.get(detected_dist, set()))
            prefix_filter = (
                dist_set if prefix_filter is None else prefix_filter & dist_set
            )
        if detected_prov:
            prov_set = set(self.invert_province_to_indices.get(detected_prov, set()))
            prefix_filter = (
                prov_set if prefix_filter is None else prefix_filter & prov_set
            )

        if prefix_filter:
            prioritized = [
                entry for entry in filtered_entries if entry[0] in prefix_filter
            ]
            if prioritized:
                nonprior = [
                    entry for entry in filtered_entries if entry[0] not in prefix_filter
                ]
                filtered_entries = prioritized + nonprior

        # Stage 2: richer scoring per-candidate
        scored_candidates = []

        def _component_boost(
            candidate_value: Optional[str],
            detected_value: Optional[str],
            exact_bonus: float,
            fuzzy_bonus: float,
            missing_penalty: float,
        ) -> float:
            if not detected_value:
                return 0.0
            if not candidate_value:
                return missing_penalty
            cand_std = self.standardize_name(candidate_value, False)
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
            boost += _component_boost(
                node.district_name, detected_dist, 14.0, 9.0, -10.0
            )
            boost += _component_boost(node.province_name, detected_prov, 6.0, 3.5, -4.0)

            comps = (
                int(bool(node.province_name))
                + int(bool(node.district_name))
                + int(bool(node.ward_name))
            )
            has_ward = 1 if node.ward_name else 0
            specificity = (comps, has_ward, len(node.standardized_full_name))

            final_score = (
                combined + boost + (comps * 1.5) + (has_ward * 1.0) + (dice_score * 10)
            )
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

        scored_candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[3],
                len(self.address_node_list[item[4]].standardized_full_name),
            ),
            reverse=True,
        )

        top_results = []
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
