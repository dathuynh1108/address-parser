from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from rapidfuzz.fuzz import partial_ratio
from rapidfuzz import process as rf_process


def _strip_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class AddressNode:
    province: str
    district: str
    ward: str
    standard_full_name: str
    ngrams: set[str]
    info_level: int


class Solution:
    def __init__(self) -> None:
        self.reference_province_path = Path("list_province.txt")
        self.reference_district_path = Path("list_district.txt")
        self.reference_ward_path = Path("list_ward.txt")
        self.standard_address_list_path = Path("list_address.json")

        self.address_nodes: List[AddressNode] = []
        self.inverted_index: Dict[str, set[int]] = defaultdict(set)

        self.province_to_districts: Dict[str, set[str]] = defaultdict(set)
        self.provdist_to_wards: Dict[Tuple[str, str], set[str]] = defaultdict(set)
        self.district_to_provinces: Dict[str, set[str]] = defaultdict(set)
        self.ward_to_parents: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        self.province_lookup = self._load_reference_lookup(self.reference_province_path)
        self.district_lookup = self._load_reference_lookup(self.reference_district_path)
        self.ward_lookup = self._load_reference_lookup(self.reference_ward_path)

        self.TOPK_CANDIDATES = 600
        self.DICE_GATE = 0.52
        self.PARTIAL_CUTOFF = 75.0
        self.MAX_RF_RESULTS = 12

        self._preprocess_reference_addresses()

    # ------------------------------------------------------------------
    # Standardisation & loading helpers
    # ------------------------------------------------------------------
    def standardize_name(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        redundant_tokens = [
            "thành phố",
            "thành. phố",
            "thành.phố",
            "tp.",
            "tp ",
            "t.phố",
            "t. phố",
            "tỉnh",
            "t.",
            "t ",
            "quận",
            "qận",
            "qun",
            "q.",
            "q ",
            "huyện",
            "h.",
            "h ",
            "thị xã",
            "thị.xã",
            "tx.",
            "tx ",
            "thị trấn",
            "thị.trấn",
            "tt.",
            "tt ",
            "xã",
            "x.",
            "x ",
            "phường",
            "p.",
            "p ",
            "phường.",
            "phường ",
        ]
        for token in redundant_tokens:
            text = text.replace(token, " ")
        text = re.sub(r"\btp([a-z0-9]+)", r"\1", text)
        text = text.replace("đ", "d")
        text = _strip_diacritics(text)
        text = re.sub(r"[^a-z0-9\s]+", " ", text)
        return _collapse_spaces(text)

    def generate_ngrams(self, text: str, n: int = 4) -> List[str]:
        wrapped = f" {text} "
        if len(wrapped) < n:
            return [wrapped]
        return [wrapped[i : i + n] for i in range(len(wrapped) - n + 1)]

    def _load_reference_lookup(self, path: Path) -> Dict[str, List[str]]:
        lookup: Dict[str, List[str]] = defaultdict(list)
        if not path.exists():
            return lookup
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                key = self.standardize_name(name)
                if name not in lookup[key]:
                    lookup[key].append(name)
        return lookup

    # ------------------------------------------------------------------
    # Pre-processing JSON hierarchy into address nodes + lookups
    # ------------------------------------------------------------------
    def _preprocess_reference_addresses(self) -> None:
        if not self.standard_address_list_path.exists():
            return

        with self.standard_address_list_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for province_name, districts in data.items():
            province_name = province_name.strip()
            province_std = self.standardize_name(province_name)

            self._add_node(province_name, "", "")

            if not isinstance(districts, dict):
                continue

            for district_name, wards in districts.items():
                district_name = district_name.strip()
                district_std = self.standardize_name(district_name)

                self.province_to_districts[province_std].add(district_std)
                self.district_to_provinces[district_std].add(province_name)

                self._add_node("", district_name, "")
                self._add_node(province_name, district_name, "")

                if not isinstance(wards, list):
                    continue

                for ward_name in wards:
                    ward_name = (ward_name or "").strip()
                    if not ward_name:
                        continue
                    ward_std = self.standardize_name(ward_name)

                    self.provdist_to_wards[(province_std, district_std)].add(ward_std)
                    self.ward_to_parents[ward_std].append((district_name, province_name))

                    self._add_node("", "", ward_name)
                    self._add_node("", district_name, ward_name)
                    self._add_node(province_name, "", ward_name)
                    self._add_node(province_name, district_name, ward_name)

        for idx, node in enumerate(self.address_nodes):
            for gram in node.ngrams:
                self.inverted_index[gram].add(idx)

    def _add_node(self, province: str, district: str, ward: str) -> None:
        full_name_std = self.standardize_name(f"{ward} {district} {province}")
        ngrams = set(self.generate_ngrams(full_name_std)) if full_name_std else set()
        info_level = int(bool(province)) + int(bool(district)) + int(bool(ward))
        self.address_nodes.append(
            AddressNode(
                province=province,
                district=district,
                ward=ward,
                standard_full_name=full_name_std,
                ngrams=ngrams,
                info_level=info_level,
            )
        )

    # ------------------------------------------------------------------
    # Candidate generation helpers
    # ------------------------------------------------------------------
    def _shortlist_by_ngrams(
        self, input_ngrams: Sequence[str], top_k: int
    ) -> List[Tuple[int, int]]:
        counter = Counter()
        for gram in set(input_ngrams):
            for idx in self.inverted_index.get(gram, () ):
                counter[idx] += 1
        return counter.most_common(top_k)

    def _filter_by_dice(
        self,
        input_ngrams: set[str],
        candidates: Sequence[Tuple[int, int]],
    ) -> List[int]:
        filtered: List[int] = []
        len_a = len(input_ngrams)
        for idx, _ in candidates:
            node = self.address_nodes[idx]
            grams = node.ngrams
            if not grams:
                continue
            inter = sum(1 for gram in input_ngrams if gram in grams)
            dice = (2 * inter) / (len_a + len(grams)) if (len_a + len(grams)) else 0.0
            if dice >= self.DICE_GATE:
                filtered.append(idx)
            else:
                break
        return filtered

    def _address_candidate_indices(
        self,
        normalized_query: str,
        input_ngrams: set[str],
        ngram_candidates: Sequence[Tuple[int, int]],
    ) -> List[int]:
        filtered_ids = self._filter_by_dice(input_ngrams, ngram_candidates)
        if not filtered_ids:
            filtered_ids = [idx for idx, _ in ngram_candidates[: self.MAX_RF_RESULTS]]
            if not filtered_ids:
                return []

        choices = [self.address_nodes[idx].standard_full_name for idx in filtered_ids]
        matches = rf_process.extract(
            normalized_query,
            choices,
            scorer=partial_ratio,
            score_cutoff=self.PARTIAL_CUTOFF,
            limit=min(self.MAX_RF_RESULTS, len(choices)),
        )
        if not matches:
            return []

        ranked = []
        for choice, score, rel_idx in matches:
            abs_idx = filtered_ids[rel_idx]
            info_level = self.address_nodes[abs_idx].info_level
            ranked.append((abs_idx, float(score), info_level))

        ranked.sort(key=lambda item: (item[1], item[2]), reverse=True)
        return [idx for idx, _, _ in ranked]

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------
    def _resolve_node_fields(self, node: AddressNode) -> Tuple[str, str, str]:
        province = node.province.strip()
        district = node.district.strip()
        ward = node.ward.strip()

        province_std = self.standardize_name(province)
        district_std = self.standardize_name(district)
        ward_std = self.standardize_name(ward)

        if ward and not district:
            parents = self.ward_to_parents.get(ward_std, [])
            if province:
                province_std = self.standardize_name(province)
                parents = [p for p in parents if self.standardize_name(p[1]) == province_std]
            if len(parents) == 1:
                district = parents[0][0]
                province = parents[0][1]
                district_std = self.standardize_name(district)
                province_std = self.standardize_name(province)

        if district and not province:
            provinces = self.district_to_provinces.get(district_std, set())
            if len(provinces) == 1:
                province = next(iter(provinces))
                province_std = self.standardize_name(province)

        if ward and district and not province:
            parents = self.ward_to_parents.get(ward_std, [])
            matches = [p for p in parents if self.standardize_name(p[0]) == district_std]
            if len(matches) == 1:
                province = matches[0][1]

        return province, district, ward

    def _map_to_reference(self, name: str, lookup: Dict[str, List[str]]) -> str:
        if not name:
            return ""
        key = self.standardize_name(name)
        options = lookup.get(key)
        if not options:
            return ""
        for option in options:
            if option.lower() == name.lower():
                return option
        return options[0]

    def _validate_hierarchy(
        self, province: str, district: str, ward: str
    ) -> Tuple[str, str, str]:
        pkey = self.standardize_name(province) if province else ""
        dkey = self.standardize_name(district) if district else ""
        wkey = self.standardize_name(ward) if ward else ""

        if pkey and dkey:
            dset = self.province_to_districts.get(pkey)
            if not dset or dkey not in dset:
                district = ""
                dkey = ""
                if ward:
                    ward = ""
                    wkey = ""

        if pkey and dkey and wkey:
            wset = self.provdist_to_wards.get((pkey, dkey))
            if not wset or wkey not in wset:
                ward = ""
                wkey = ""

        if pkey and not dkey and ward:
            ward = ""

        return province, district, ward

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, text: str) -> Dict[str, str]:
        normalized_query = self.standardize_name(text)
        if not normalized_query:
            return {"province": "", "district": "", "ward": ""}

        input_ngrams = self.generate_ngrams(normalized_query)
        ngram_candidates = self._shortlist_by_ngrams(input_ngrams, self.TOPK_CANDIDATES)
        candidate_indices = self._address_candidate_indices(
            normalized_query, set(input_ngrams), ngram_candidates
        )

        best_output = {"province": "", "district": "", "ward": ""}
        best_rank = (-1, -1.0)

        for position, idx in enumerate(candidate_indices):
            node = self.address_nodes[idx]
            province, district, ward = self._resolve_node_fields(node)

            province = self._map_to_reference(province, self.province_lookup)
            district = self._map_to_reference(district, self.district_lookup)
            ward = self._map_to_reference(ward, self.ward_lookup)

            province, district, ward = self._validate_hierarchy(province, district, ward)

            filled = int(bool(province)) + int(bool(district)) + int(bool(ward))
            if filled == 0:
                continue

            rank = (filled, -position)
            if rank > best_rank:
                best_rank = rank
                best_output = {"province": province, "district": district, "ward": ward}

            if filled == 3:
                break

        return best_output


if __name__ == "__main__":
    solver = Solution()
    sample = "284DBis Ng Văn Giáo, P3, Mỹ Tho, T.Giang."
    print(solver.process(sample))
