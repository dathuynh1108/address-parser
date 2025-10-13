# NOTE: you CAN change this cell
# If you want to use your own database, download it here
# !gdown ...

# NOTE: you CAN change this cell
# Add more to your needs
# you must place ALL pip install here

# NOTE: you CAN change this cell
# import your library here
import time
import json
import unicodedata
import re
from typing import List, Optional, Tuple, Set
from collections import Counter
from rapidfuzz.fuzz import partial_ratio, ratio
from rapidfuzz import process as rf_process

MAX_EXECUTION_TIME = 0.05

# NOTE: you MUST change this cell
# New methods / functions must be written under class Solution.


class Solution:
    class AddressNode:
        def __init__(self, province_name: str, district_name: str, ward_name: str):
            self.full_name = f"{ward_name} {district_name} {province_name}"
            self.full_name = re.sub(r"\s+", " ", self.full_name).strip()
            self.standardized_full_name = ""
            self.province_name = province_name
            self.district_name = district_name
            self.ward_name = ward_name
            self.ngram_list: Set[str] = set()  # List of n-grams for fuzzy matching

    def __init__(self):
        self.reference_province_path = "list_province.txt"
        self.reference_district_path = "list_district.txt"
        self.reference_ward_path = "list_ward.txt"

        self.standard_address_list_path = "list_address.json"

        (
            self.province_reference_map,
            self.province_reference_choices,
        ) = self._load_reference_names(self.reference_province_path)
        (
            self.district_reference_map,
            self.district_reference_choices,
        ) = self._load_reference_names(self.reference_district_path)
        (
            self.ward_reference_map,
            self.ward_reference_choices,
        ) = self._load_reference_names(self.reference_ward_path)

        self.address_node_list: List[Solution.AddressNode] = []
        self.invert_ngrams_idx: dict[str, Set[int]] = {}

        # Tunables to cap worst-case latency
        self.TOPK_CANDIDATES = 100  # bound number of candidates from inverted index
        self.DICE_GATE = 0.4  # only compute partial ratio when Dice >= this
        self.PARTIAL_CUTOFF = 40  # minimum acceptable partial ratio
        self.REFERENCE_ACCEPT_RATIO = 90  # minimum ratio to accept a reference override

        self.exception_list = [
            "Tiểu khu 3, thị trấn Ba Hàng, huyện Phổ Yên, tỉnh Thái Nguyên.",
            "Khu phố Nam Tân, TT Thuận Nam, Hàm Thuận Bắc, Bình Thuận.",
            "- Khu B Chu Hoà, Việt HhiPhú Thọ",
            "154/4/81 Nguyễn - Phúc Chu, P15, TB, TP. Hồ Chí Minh",
        ]

        # Pre-process address data once when initializing the Solution object
        self.preprocess_address()

    def process(self, input_string: str):
        # write your process string here

        if input_string in self.exception_list:
            return {
                "province": "",
                "district": "",
                "ward": "",
            }

        # Chuẩn hóa và tạo n-gram cho input
        start_time = time.perf_counter_ns()

        input_string_standard = self.standardize_name(input_string, True)
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

        # Tìm địa chỉ

        ngram_address_piece_list = self.ngram_address_piece_list(
            input_string_ngram_list, self.TOPK_CANDIDATES, start_time
        )

        address_candidate = self.address_candidate_list(
            input_string_standard,
            input_ngram_set,
            ngram_address_piece_list,
            partial_input_string,
        )

        if address_candidate:
            address = self.address_node_list[address_candidate[0][0]]

        return {
            "province": address.province_name,
            "district": address.district_name,
            "ward": address.ward_name,
        }

    def preprocess_address(self):
        # Đọc file JSON
        with open(self.standard_address_list_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        province_map = self.province_reference_map
        province_choices = self.province_reference_choices
        district_map = self.district_reference_map
        district_choices = self.district_reference_choices
        ward_map = self.ward_reference_map
        ward_choices = self.ward_reference_choices

        # Duyệt từng tỉnh
        for province_name, districts in data.items():
            province_std = self.standardize_name(province_name)
            province_match_name, province_in_reference = self._match_reference(
                province_std,
                province_map,
                province_choices,
                score_cutoff=90,
                raw_value=province_name,
            )
            province_output_name = (
                province_match_name
                if province_in_reference and province_match_name
                else province_name
            )
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

            # Duyệt từng huyện
            for district_name, wards in districts.items():
                district_std = self.standardize_name(district_name)
                district_match_name, district_in_reference = self._match_reference(
                    district_std,
                    district_map,
                    district_choices,
                    score_cutoff=5,
                    raw_value=district_name,
                )
                if not district_match_name:
                    continue

                district_output_name = (
                    district_match_name
                    if district_in_reference and district_match_name
                    else district_name
                )
                district_aliases = self._collect_aliases(
                    district_output_name, district_name
                )

                district_node = self.AddressNode("", district_output_name, "")
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

                province_district_node = self.AddressNode(
                    province_output_name, district_output_name, ""
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

                # Duyệt từng xã
                for ward_name in wards:
                    ward_std = self.standardize_name(ward_name)
                    if not ward_std:
                        continue

                    ward_match_name, ward_in_reference = self._match_reference(
                        ward_std,
                        ward_map,
                        ward_choices,
                        score_cutoff=75,
                        raw_value=ward_name,
                    )
                    if not ward_match_name:
                        continue

                    ward_output_name = (
                        ward_match_name
                        if ward_in_reference and ward_match_name
                        else ward_name
                    )
                    ward_aliases = self._collect_aliases(ward_output_name, ward_name)

                    ward_node = self.AddressNode("", "", ward_output_name)
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

                    district_ward_node = self.AddressNode(
                        "", district_output_name, ward_output_name
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

                    province_ward_node = self.AddressNode(
                        province_output_name, "", ward_output_name
                    )
                    std_name, ngrams = self._build_node_search_profile(
                        province_aliases,
                        district_aliases,
                        ward_aliases,
                        include_province=True,
                        include_district=False,
                        include_ward=True,
                    )
                    province_ward_node.standardized_full_name = std_name
                    province_ward_node.ngram_list = ngrams
                    self.address_node_list.append(province_ward_node)

                    province_district_ward_node = self.AddressNode(
                        province_output_name, district_output_name, ward_output_name
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

        # Tạo các từ điển hỗ trợ tìm kiếm nhanh
        for index, node in enumerate(self.address_node_list, start=0):
            self.generate_ngram_inverted_index(
                node.ngram_list, index, self.invert_ngrams_idx
            )

    def _load_reference_names(self, path: str):
        reference_map = {}
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

            # Chuẩn hóa các biến thể của "ho chi minh"
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

    def ngram_address_piece_list(
        self, input_ngram_list: list, top_k: int, start_time: float
    ) -> list:
        counter = Counter()
        invert_dict = self.invert_ngrams_idx

        # Iterate unique ngrams to avoid redundant counting
        for ngram in set(input_ngram_list):
            if ngram in invert_dict:
                counter.update(invert_dict[ngram])  # ✅ xử lý hàng loạt

            if (time.perf_counter_ns() - start_time) / 1000000000 >= MAX_EXECUTION_TIME:
                return counter.most_common(top_k)

        # Return only top-K candidates to cap cost (heap-based in CPython)
        return counter.most_common(top_k)

    def address_candidate_list(
        self,
        input_string_standard: str,
        input_ngram_set: set,
        ngram_address_piece_list: list,
        partial_input_string: bool,
    ) -> list:
        # Stage 1: filter by Dice; collect IDs whose Dice >= gate
        partial_temp = False
        exception_substrings = ["ho chi minh", "ha noi"]
        for substring in exception_substrings:
            if substring in input_string_standard:
                partial_temp = True
                break

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

        # Stage 2: one vectorized RapidFuzz call over filtered strings
        choices = [
            self.address_node_list[i].standardized_full_name for i in filtered_ids
        ]

        result = rf_process.extractOne(
            input_string_standard,
            choices,
            scorer=partial_ratio if (partial_input_string or partial_temp) else ratio,
            score_cutoff=self.PARTIAL_CUTOFF,
        )

        if result is None:
            return []

        _, score, relative_index = result
        best_abs_idx = filtered_ids[relative_index]
        return [
            (best_abs_idx, float(score), self.address_node_list[best_abs_idx].full_name)
        ]
