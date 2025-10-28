import json
import os
import unicodedata
import re
from typing import Any, List, Optional, Tuple, Set, Dict
from collections import Counter, defaultdict
from rapidfuzz.fuzz import partial_ratio, ratio
from rapidfuzz import process as rf_process


class AddressParser:
    class AddressNode:
        def __init__(
            self,
            province_name: str,
            district_name: str,
            ward_name: str,
            *,
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
    }

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.standard_address_list_path = os.path.join(base_dir, "inexus_list_address.json")

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

        # Tunables to cap worst-case latency
        self.TOPK_CANDIDATES = 400  # bound number of candidates from inverted index
        self.DICE_GATE = 0.4  # only compute partial ratio when Dice >= this
        self.PARTIAL_CUTOFF = 40  # minimum acceptable partial ratio
        self.REFERENCE_ACCEPT_RATIO = 90  # minimum ratio to accept a reference override

        # Pre-process address data once when initializing the Solution object
        self.preprocess_address()
    
    def process(self, input_string: str):
        # Chuẩn hóa và tạo n-gram cho input
        input_string_standard = self.standardize_name(input_string, True)
        input_string_basic = self.standardize_name(input_string, False)
        
        input_string_ngram_list = self.generate_ngrams(input_string_standard)
    
        partial_input_string = False

        # Đếm tần suất xuất hiện của từng ngram
        ngram_counts = Counter(input_string_ngram_list)

        # Lấy 5 ngram phổ biến nhất
        top_5 = ngram_counts.most_common(5)
        # Nếu tổng tần suất top 5 ngram ≤ 15 → partial_input_string = True
        if top_5 and sum(count for _, count in top_5) >= 12:
            partial_input_string = True

        input_ngram_set = set(input_string_ngram_list)

        address = self.AddressNode("", "", "")

        ngram_address_piece_list = self.ngram_address_piece_list(
            input_string_ngram_list, self.TOPK_CANDIDATES
        )

        address_candidate = self.address_candidate_list(
            input_string_standard,
            input_ngram_set,
            ngram_address_piece_list,
            partial_input_string,
        )

        if address_candidate:
            address = self.address_node_list[address_candidate[0][0]]

        fmt = (
            "new"
            if address.is_new_format is True
            else ("old" if address.is_new_format is False else "unknown")
        )
        street_address = self._extract_street_address(input_string, address)
        return {
            "province": address.province_name,
            "district": address.district_name,
            "ward": address.ward_name,
            "street_address": street_address,
            "format": fmt,
            "is_new": (
                True
                if address.is_new_format is True
                else False if address.is_new_format is False else None
            ),
        }

    def preprocess_address(self):
        # Đọc file JSON
        with open(self.standard_address_list_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Trust list_addresses.json; skip reference TXT filtering

        # Duyệt từng tỉnh
        for province_name, districts in data.items():
            # Trust JSON names for provinces
            province_output_name = province_name
            province_output_std = self.standardize_name(province_output_name, False)
            if province_output_std:
                self.province_names_std.add(province_output_std)
            province_aliases = self._collect_aliases(
                province_output_name, province_name
            )
            province_node = self.AddressNode(province_output_name, "", "")
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
            # Update inverted name indexes
            self.invert_province_to_indices[province_output_std].add(
                len(self.address_node_list) - 1
            )

            # Duyệt từng huyện
            for district_name, wards in districts.items():
                district_std = self.standardize_name(district_name)

                # New 2-level structure: empty district means wards belong directly to province
                if not district_std:
                    # Duyệt từng xã trực thuộc tỉnh (không có huyện)
                    for ward_name in wards:
                        ward_std = self.standardize_name(ward_name)
                        if not ward_std:
                            continue
                        # Trust JSON names for wards in 2-level structure
                        ward_output_name = ward_name
                        ward_aliases = self._collect_aliases(
                            ward_output_name, ward_name
                        )
                        ward_output_std = self.standardize_name(ward_output_name, False)
                        if ward_output_std:
                            self.ward_names_std.add(ward_output_std)

                        # Ward only node (for matching), mark as new format
                        ward_node = self.AddressNode(
                            "", "", ward_output_name, is_new_format=True
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
                        self.invert_ward_to_indices[ward_output_std].add(
                            len(self.address_node_list) - 1
                        )

                        # Province + Ward node (district empty) → new format
                        province_ward_node = self.AddressNode(
                            province_output_name,
                            "",
                            ward_output_name,
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
                        self.invert_province_to_indices[province_output_std].add(
                            len(self.address_node_list) - 1
                        )
                        self.invert_ward_to_indices[ward_output_std].add(
                            len(self.address_node_list) - 1
                        )

                    # Done with new-format wards under province
                    continue

                # Old 3-level structure: district exists
                # Trust JSON names for districts
                district_output_name = district_name
                district_output_std = self.standardize_name(district_output_name, False)
                if district_output_std:
                    self.district_names_std.add(district_output_std)
                district_aliases = self._collect_aliases(
                    district_output_name, district_name
                )

                district_node = self.AddressNode(
                    "", district_output_name, "", is_new_format=False
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
                self.invert_district_to_indices[district_output_std].add(
                    len(self.address_node_list) - 1
                )

                province_district_node = self.AddressNode(
                    province_output_name, district_output_name, "", is_new_format=False
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
                self.invert_province_to_indices[province_output_std].add(
                    len(self.address_node_list) - 1
                )
                self.invert_district_to_indices[district_output_std].add(
                    len(self.address_node_list) - 1
                )

                # Duyệt từng xã (old format)
                for ward_name in wards:
                    ward_std = self.standardize_name(ward_name)
                    if not ward_std:
                        continue

                    # Trust JSON names for wards in 3-level structure
                    ward_output_name = ward_name
                    ward_aliases = self._collect_aliases(ward_output_name, ward_name)
                    ward_output_std = self.standardize_name(ward_output_name, False)
                    if ward_output_std:
                        self.ward_names_std.add(ward_output_std)

                    ward_node = self.AddressNode(
                        "", "", ward_output_name, is_new_format=False
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
                    self.invert_ward_to_indices[ward_output_std].add(
                        len(self.address_node_list) - 1
                    )

                    district_ward_node = self.AddressNode(
                        "", district_output_name, ward_output_name, is_new_format=False
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
                    self.invert_district_to_indices[district_output_std].add(
                        len(self.address_node_list) - 1
                    )
                    self.invert_ward_to_indices[ward_output_std].add(
                        len(self.address_node_list) - 1
                    )

                    # Do NOT create province_ward_node for old format to avoid empty-district nodes

                    province_district_ward_node = self.AddressNode(
                        province_output_name,
                        district_output_name,
                        ward_output_name,
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
                    self.invert_province_to_indices[province_output_std].add(
                        len(self.address_node_list) - 1
                    )
                    self.invert_district_to_indices[district_output_std].add(
                        len(self.address_node_list) - 1
                    )
                    self.invert_ward_to_indices[ward_output_std].add(
                        len(self.address_node_list) - 1
                    )

        # Tạo các từ điển hỗ trợ tìm kiếm nhanh
        for index, node in enumerate(self.address_node_list, start=0):
            self.generate_ngram_inverted_index(
                node.ngram_list, index, self.invert_ngrams_idx
            )

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
        self, primary: Optional[str], raw_value: Optional[str]
    ) -> List[str]:
        aliases: List[str] = []
        if primary:
            aliases.append(primary)
        if raw_value and raw_value not in aliases:
            aliases.append(raw_value)
        return aliases or [""]

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

    def _build_component_signature(self, component: Optional[str]) -> Dict[str, Any]:
        signature: Dict[str, Any] = {"sequences": [], "tokens": set()}
        if not component:
            return signature

        standardized = self.standardize_name(component, False)
        if not standardized:
            return signature

        parts = [p for p in standardized.split() if p]
        tokens: Set[str] = set()
        sequences: List[List[str]] = []

        if parts:
            sequences.append(parts)
            tokens.update(parts)

        joined = "".join(parts)
        if joined:
            sequences.append([joined])
            tokens.add(joined)

        abbr = "".join(part[0] for part in parts if part)
        if len(abbr) >= 2:
            sequences.append([abbr])
            tokens.add(abbr)
            sequences.append([f"tp{abbr}"])
            tokens.add(f"tp{abbr}")
            sequences.append(["tp", abbr])

        signature["sequences"] = sequences
        signature["tokens"] = tokens
        return signature

    def _extract_street_address(
        self, original: str, node: "AddressParser.AddressNode"
    ) -> str:
        if not original:
            return ""

        profiles = {
            "province": self._build_component_signature(node.province_name),
            "district": self._build_component_signature(node.district_name),
            "ward": self._build_component_signature(node.ward_name),
        }

        if not any(profile["sequences"] for profile in profiles.values()):
            return original.strip()

        token_matches = list(
            re.finditer(r"\b\w+\b", original, flags=re.UNICODE)
        )
        if not token_matches:
            return original.strip()

        tokens = []
        for match in token_matches:
            norm = self._normalize_token_basic(match.group(0))
            tokens.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "norm": norm,
                }
            )
        token_count = len(tokens)
        if token_count == 0:
            return original.strip()

        indices_to_remove: Set[int] = set()

        def mark_indices(start_idx: int, length: int) -> bool:
            if length <= 0:
                return False
            end_idx = start_idx + length
            if end_idx > token_count:
                return False
            window = tokens[start_idx:end_idx]
            if any(not token["norm"] for token in window):
                return False
            if length == 1 and window[0]["norm"].isdigit():
                prev_generic = (
                    start_idx > 0
                    and tokens[start_idx - 1]["norm"] in self._GENERIC_LOCATION_TOKENS
                )
                next_generic = (
                    end_idx < token_count
                    and tokens[end_idx]["norm"] in self._GENERIC_LOCATION_TOKENS
                )
                if not (prev_generic or next_generic):
                    return False

            indices_to_remove.update(range(start_idx, end_idx))

            prev_idx = start_idx - 1
            while (
                prev_idx >= 0
                and tokens[prev_idx]["norm"] in self._GENERIC_LOCATION_TOKENS
            ):
                indices_to_remove.add(prev_idx)
                prev_idx -= 1

            next_idx = end_idx
            while (
                next_idx < token_count
                and tokens[next_idx]["norm"] in self._GENERIC_LOCATION_TOKENS
            ):
                indices_to_remove.add(next_idx)
                next_idx += 1
            return True

        for profile in profiles.values():
            sequences: List[List[str]] = profile["sequences"]
            for seq in sequences:
                seq = [item for item in seq if item]
                seq_len = len(seq)
                if seq_len == 0:
                    continue
                for idx in range(token_count - seq_len + 1):
                    window = tokens[idx : idx + seq_len]
                    if all(window[pos]["norm"] == seq[pos] for pos in range(seq_len)):
                        mark_indices(idx, seq_len)

        if token_count > 1:
            comma_positions = [m.start() for m in re.finditer(",", original)]
            segments: List[Tuple[int, int]] = []
            start_char = 0
            for pos in comma_positions:
                segments.append((start_char, pos))
                start_char = pos + 1
            segments.append((start_char, len(original)))

            segment_token_indices: List[List[int]] = [[] for _ in segments]
            for token_idx, token in enumerate(tokens):
                for seg_idx, (seg_start, seg_end) in enumerate(segments):
                    if seg_start <= token["start"] < seg_end:
                        segment_token_indices[seg_idx].append(token_idx)
                        break

            for seg_idx, idx_list in enumerate(segment_token_indices):
                if seg_idx == 0 or not idx_list:
                    continue
                has_generic = any(
                    tokens[token_idx]["norm"] in self._GENERIC_LOCATION_TOKENS
                    for token_idx in idx_list
                )
                if has_generic or any(
                    token_idx in indices_to_remove for token_idx in idx_list
                ):
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
        for ngram in set(input_ngram_list):
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

        # Compile once per call; small overhead compared to overall cost
        province_pref = re.compile(
            r"\b(?:thanh pho|tp|tinh)\b\s+([a-z0-9 ]+?)(?=\b(?:quan|huyen|thi xa|thi tran|phuong|xa|tp|tinh)\b|$)"
        )
        district_pref = re.compile(
            r"\b(?:quan|huyen|thi xa)\b\s+([a-z0-9 ]+?)(?=\b(?:phuong|xa|thi tran|quan|huyen|thi xa)\b|$)"
        )
        ward_pref = re.compile(
            r"\b(?:phuong|xa|thi tran)\b\s+([a-z0-9 ]+?)(?=\b(?:phuong|xa|thi tran|quan|huyen|thi xa)\b|$)"
        )

        def _pick_best(fragment: str, choices: List[str]) -> Optional[str]:
            fragment = fragment.strip()
            if not fragment:
                return None
            # Limit fragment to first 3 tokens to avoid swallowing next parts
            tokens = fragment.split()
            fragment = " ".join(tokens[:3])
            res = rf_process.extractOne(
                fragment, choices, scorer=partial_ratio, score_cutoff=75
            )
            return res[0] if res else None

        prov = dist = ward = None
        m = province_pref.search(s)
        if m and self.province_names_std:
            prov = _pick_best(m.group(1), list(self.province_names_std))

        m = district_pref.search(s)
        if m and self.district_names_std:
            dist = _pick_best(m.group(1), list(self.district_names_std))

        m = ward_pref.search(s)
        if m and self.ward_names_std:
            ward = _pick_best(m.group(1), list(self.ward_names_std))

        return prov, dist, ward

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

    def address_candidate_list(
        self,
        input_string_standard: str,
        input_ngram_set: set,
        ngram_address_piece_list: list,
        partial_input_string: bool,
    ) -> list:
        # Stage 1: filter by Dice; collect IDs whose Dice >= gate
        # Use partial scorer only when the input seems truncated
        partial_temp = False  # deprecated heuristic; keep variable for readability

        input_set = input_ngram_set
        input_set_length = len(input_set)
        filtered_ids: list[int] = []

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
            index += 1

            if dice_score >= self.DICE_GATE:
                filtered_ids.append(idx)
            elif index >= 50:
                # Counter is ordered by frequency; dice will only go down after this point
                break

        if not filtered_ids:
            return []

        # Prefer more specific nodes (more filled components, then longer string)
        def _specificity_key(idx: int):
            node = self.address_node_list[idx]
            comps = int(bool(node.province_name)) + int(bool(node.district_name)) + int(bool(node.ward_name))
            # Prefer nodes that include ward over district-only when comps equal
            has_ward = 1 if node.ward_name else 0
            return (comps, has_ward, len(node.standardized_full_name))

        filtered_ids.sort(key=_specificity_key, reverse=True)

        # Stage 2: one vectorized RapidFuzz call over filtered strings
        choices = [self.address_node_list[i].standardized_full_name for i in filtered_ids]

        result = rf_process.extractOne(
            input_string_standard,
            choices,
            scorer=partial_ratio if partial_input_string else ratio,
            score_cutoff=self.PARTIAL_CUTOFF,
        )
        if result is None:
            return []

        _, score, relative_index = result
        best_abs_idx = filtered_ids[relative_index]
        return [
            (best_abs_idx, float(score), self.address_node_list[best_abs_idx].full_name)
        ]

if __name__ == "__main__":
    parser = AddressParser()
    test_address = "50 Tôn Thất Đạm Sài Gòn TP.HCM"
    result = parser.process(test_address)
    print(result) # {'province': 'Hồ Chí Minh', 'district': '', 'ward': 'Sài Gòn', 'street_address': '50 Tôn Thất Đạm', 'format': 'new', 'is_new': True}
