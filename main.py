from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Sequence, Set, Tuple

from rapidfuzz.fuzz import partial_ratio
from rapidfuzz import process as rf_process


_HINT_STOP_TOKENS = {
    "tinh",
    "thanh",
    "pho",
    "thanhpho",
    "tp",
    "quan",
    "q",
    "huyen",
    "h",
    "thi",
    "xa",
    "phuong",
    "p",
    "tt",
    "thi",
    "thon",
}

_SPECIAL_REPLACE = {
    # Add any special replacements if needed
    't.t.h': 'thua thien hue',  
    'tth': 'thua thien hue',
    'tphcm': 'thanh pho ho chi minh',
    'tp hcm': 'thanh pho ho chi minh',
    'hcm': 'ho chi minh',
    'hn': 'ha noi',
}

_ADMIN_PREFIX_PATTERN = re.compile(
    r"^(thành phố trực thuộc trung ương|thành phố trực thuộc tw|"
    r"thành phố thuộc tỉnh|thành phố|tp\.?|tỉnh|quận|q\.?|huyện|h\.?|"
    r"thị xã|tx\.?|thị trấn|tt\.?|phường|p\.?|xã|x\.?|thôn)\s+",
    re.IGNORECASE,
)

def _strip_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_special_key(key: str) -> str:
    base = _strip_diacritics(key.lower())
    return re.sub(r"[^a-z0-9]", "", base)


def _build_special_replace_patterns() -> List[Tuple[Pattern[str], str]]:
    patterns: List[Tuple[Pattern[str], str]] = []
    separator = r"[\s./-]*"
    for source, target in _SPECIAL_REPLACE.items():
        normalized = _normalize_special_key(source)
        if not normalized:
            continue
        pattern = r"\b" + separator.join(normalized) + r"\b"
        patterns.append((re.compile(pattern), target))
    return patterns


_SPECIAL_REPLACE_PATTERNS = _build_special_replace_patterns()


def _strip_admin_prefix(name: str) -> str:
    if not name:
        return ""
    current = name.strip()
    while True:
        match = _ADMIN_PREFIX_PATTERN.match(current)
        if not match:
            break
        current = current[match.end():].lstrip()
    return current


@dataclass
class AddressNode:
    province: str
    district: str
    ward: str
    standard_full_name: str
    ngrams: set[str]
    info_level: int


class Solution:
    def __init__(
        self,
        *,
        topk_candidates: int = 600,
        dice_gate: float = 0.52,
        partial_cutoff: float = 75.0,
        max_rf_results: int = 12,
        ngram_size: int = 4,
        max_ngrams: int = 42,
        ngram_frequency_cap: Optional[int] = None,
    ) -> None:
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
        self.inverted_index_lists: Dict[str, Tuple[int, ...]] = {}
        self.province_candidates: Dict[str, Tuple[str, ...]] = {}
        self.provdist_candidates: Dict[Tuple[str, str], Tuple[str, ...]] = {}
        self.ngram_frequencies: Dict[str, int] = {}

        self.province_lookup = self._load_reference_lookup(self.reference_province_path)
        self.district_lookup = self._load_reference_lookup(self.reference_district_path)
        self.ward_lookup = self._load_reference_lookup(self.reference_ward_path)

        self.TOPK_CANDIDATES = topk_candidates
        self.DICE_GATE = dice_gate
        self.PARTIAL_CUTOFF = partial_cutoff
        self.MAX_RF_RESULTS = max_rf_results
        self.NGRAM_SIZE = max(1, ngram_size)
        self.MAX_NGRAMS = max(1, max_ngrams)
        self.NGRAM_FREQ_CUTOFF = (
            ngram_frequency_cap if (ngram_frequency_cap or 0) > 0 else None
        )
        self.ANCHOR_NGRAMS = min(12, self.MAX_NGRAMS)

        self._preprocess_reference_addresses()
        self._finalize_indexes()

    # ------------------------------------------------------------------
    # Standardisation & loading helpers
    # ------------------------------------------------------------------
    def standardize_name(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        for pattern, replacement in _SPECIAL_REPLACE_PATTERNS:
            text = pattern.sub(replacement, text)
        
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

    def generate_ngrams(self, text: str, n: Optional[int] = None) -> List[str]:
        n = n or self.NGRAM_SIZE
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

    def _finalize_indexes(self) -> None:
        self.inverted_index_lists = {
            gram: tuple(sorted(indices))
            for gram, indices in self.inverted_index.items()
        }
        self.ngram_frequencies = {
            gram: len(indices) for gram, indices in self.inverted_index_lists.items()
        }
        self.province_candidates = {
            prov: tuple(sorted(districts))
            for prov, districts in self.province_to_districts.items()
        }
        self.provdist_candidates = {
            key: tuple(sorted(wards))
            for key, wards in self.provdist_to_wards.items()
        }

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

    def _remove_once_by_tokens(self, query_std: str, pattern_std: str) -> str:
        if not query_std or not pattern_std:
            return query_std
        q_tokens = query_std.split()
        need = Counter(pattern_std.split())
        if not need:
            return query_std
        out: List[str] = []
        for token in q_tokens:
            if need[token] > 0:
                need[token] -= 1
            else:
                out.append(token)
        return " ".join(out)

    def _extract_prefixed_number(
        self, raw_text: str, prefixes: Tuple[str, ...]
    ) -> Optional[str]:
        if not raw_text:
            return None
        raw = raw_text.lower()
        raw = _strip_diacritics(raw)
        raw = re.sub(r"[^a-z0-9\s]+", " ", raw)
        tokens = _collapse_spaces(raw).split()
        for idx, token in enumerate(tokens):
            for pref in prefixes:
                if token.startswith(pref) and len(token) > len(pref):
                    suffix = token[len(pref) :]
                    if suffix.isdigit():
                        return str(int(suffix)) if suffix.lstrip("0") else "0"
                if token.isdigit() and idx > 0 and tokens[idx - 1] == pref:
                    return str(int(token)) if token.lstrip("0") else "0"
        return None

    def _extract_phrase_after_prefix(
        self, raw_text: str, prefixes: Tuple[str, ...], max_words: int = 3
    ) -> Optional[str]:
        if not raw_text:
            return None
        raw = raw_text.lower()
        raw = _strip_diacritics(raw)
        raw = re.sub(r"[^a-z0-9\s]+", " ", raw)
        raw = _collapse_spaces(raw)
        for pref in prefixes:
            pattern_prefix = pref.replace(" ", r"\s+")
            pattern = re.compile(
                rf"\b{pattern_prefix}\s+([a-z0-9]+(?:\s+[a-z0-9]+){{0,{max_words-1}}})"
            )
            match = pattern.search(raw)
            if match:
                phrase = match.group(1).strip()
                if phrase:
                    tokens = phrase.split()
                    trimmed: List[str] = []
                    for token in tokens:
                        if token in _HINT_STOP_TOKENS:
                            break
                        trimmed.append(token)
                    if trimmed:
                        return " ".join(trimmed)
        return None

    def _hierarchical_hint(
        self, normalized_query: str, raw_text: str
    ) -> Tuple[str, str, str]:
        province = district = ward = ""

        prov_keys = list(self.province_lookup.keys())
        prov_hint = rf_process.extractOne(
            normalized_query,
            prov_keys,
            scorer=partial_ratio,
            score_cutoff=65.0,
        )
        if not prov_hint:
            return province, district, ward

        prov_std = prov_hint[0]
        province = self._map_to_reference(prov_std, self.province_lookup, raw_text)
        nq_after_province = self._remove_once_by_tokens(normalized_query, prov_std)

        d_candidates = self.province_candidates.get(prov_std, ())
        district_std = ""
        if d_candidates:
            preferred = self._extract_prefixed_number(raw_text, ("q", "quan"))
            if preferred and preferred in d_candidates:
                district_std = preferred
            else:
                phrase_hint = self._extract_phrase_after_prefix(
                    raw_text, ("quan", "huyen", "thi xa"), max_words=3
                )
                if phrase_hint:
                    phrase_std = self.standardize_name(phrase_hint)
                    if phrase_std in d_candidates:
                        district_std = phrase_std
                if not district_std:
                    dist_hint = rf_process.extractOne(
                        nq_after_province,
                        d_candidates,
                        scorer=partial_ratio,
                        score_cutoff=60.0,
                    )
                    if dist_hint:
                        district_std = dist_hint[0]

        nq_after_district = nq_after_province
        if district_std:
            district = self._map_to_reference(district_std, self.district_lookup, raw_text)
            nq_after_district = self._remove_once_by_tokens(
                nq_after_province, district_std
            )

            w_candidates = self.provdist_candidates.get((prov_std, district_std), ())
            if w_candidates:
                ward_std = ""
                preferred_ward = self._extract_prefixed_number(
                    raw_text, ("p", "phuong", "xa")
                )
                if preferred_ward and preferred_ward in w_candidates:
                    ward_std = preferred_ward
                else:
                    phrase_hint = self._extract_phrase_after_prefix(
                        raw_text,
                        ("tt", "thi tran", "phuong", "xa", "thon"),
                        max_words=4,
                    )
                    if phrase_hint:
                        phrase_std = self.standardize_name(phrase_hint)
                        if phrase_std in w_candidates:
                            ward_std = phrase_std
                        else:
                            phrase_match = rf_process.extractOne(
                                phrase_std,
                                w_candidates,
                                scorer=partial_ratio,
                                score_cutoff=58.0,
                            )
                            if phrase_match:
                                ward_std = phrase_match[0]
                if not ward_std:
                    ward_hint = rf_process.extractOne(
                        nq_after_district,
                        w_candidates,
                        scorer=partial_ratio,
                        score_cutoff=58.0,
                    )
                    ward_std = ward_hint[0] if ward_hint else ""
                if ward_std:
                    ward = self._map_to_reference(ward_std, self.ward_lookup, raw_text)

        province, district, ward = self._validate_hierarchy(province, district, ward)
        return province, district, ward

    # ------------------------------------------------------------------
    # Candidate generation helpers
    # ------------------------------------------------------------------
    def _shortlist_by_ngrams(
        self, input_ngrams: Sequence[str], top_k: int
    ) -> List[Tuple[int, int]]:
        counter: Dict[int, int] = defaultdict(int)
        seen: Set[str] = set()
        original_order: List[str] = []
        for gram in input_ngrams:
            if gram in seen:
                continue
            seen.add(gram)
            original_order.append(gram)

        ranked_ngrams = original_order
        if len(ranked_ngrams) > 1:
            ranked_ngrams = sorted(
                ranked_ngrams,
                key=lambda gram: self.ngram_frequencies.get(gram, 0),
            )

        if self.NGRAM_FREQ_CUTOFF is not None and ranked_ngrams:
            filtered: List[str] = []
            heavy: List[str] = []
            for gram in ranked_ngrams:
                freq = self.ngram_frequencies.get(gram, 0)
                if freq and freq > self.NGRAM_FREQ_CUTOFF:
                    heavy.append(gram)
                else:
                    filtered.append(gram)
            if filtered:
                ranked_ngrams = filtered
            elif heavy:
                ranked_ngrams = heavy

        final_ngrams: List[str] = []
        seen_final: Set[str] = set()
        anchor_limit = self.ANCHOR_NGRAMS
        for gram in original_order[:anchor_limit]:
            if gram not in seen_final:
                final_ngrams.append(gram)
                seen_final.add(gram)
                if len(final_ngrams) >= self.MAX_NGRAMS:
                    break

        if len(final_ngrams) < self.MAX_NGRAMS:
            for gram in ranked_ngrams:
                if gram in seen_final:
                    continue
                final_ngrams.append(gram)
                seen_final.add(gram)
                if len(final_ngrams) >= self.MAX_NGRAMS:
                    break

        if len(final_ngrams) < self.MAX_NGRAMS:
            for gram in original_order:
                if gram in seen_final:
                    continue
                final_ngrams.append(gram)
                if len(final_ngrams) >= self.MAX_NGRAMS:
                    break

        for gram in final_ngrams:
            for idx in self.inverted_index_lists.get(gram, ()):
                counter[idx] += 1
        if not counter:
            return []

        bucketed: Dict[int, List[int]] = defaultdict(list)
        for idx, count in counter.items():
            bucketed[count].append(idx)

        ordered: List[Tuple[int, int]] = []
        for count in sorted(bucketed.keys(), reverse=True):
            candidates = bucketed[count]
            candidates.sort(key=lambda i: (-self.address_nodes[i].info_level, i))
            for idx in candidates:
                ordered.append((idx, count))
                if len(ordered) == top_k:
                    return ordered
        return ordered

    def _filter_by_dice(
        self,
        input_ngrams: set[str],
        candidates: Sequence[Tuple[int, int]],
    ) -> List[int]:
        filtered: List[int] = []
        len_a = len(input_ngrams)
        if len_a == 0:
            return filtered

        for idx, _ in candidates:
            node = self.address_nodes[idx]
            grams = node.ngrams
            if not grams:
                continue
            inter = sum(1 for gram in input_ngrams if gram in grams)
            denominator = len_a + len(grams)
            dice = (2 * inter) / denominator if denominator else 0.0
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

    def _map_to_reference(
        self,
        name: str,
        lookup: Dict[str, List[str]],
        raw_text: str = "",
    ) -> str:
        if not name:
            return ""
        key = self.standardize_name(name)
        options = lookup.get(key)
        if not options:
            return ""
        lower_name = name.lower()
        for option in options:
            if option.lower() == lower_name:
                return option
        if raw_text:
            raw_lower = raw_text.lower()
            best_option = max(
                options,
                key=lambda opt: (
                    partial_ratio(opt.lower(), raw_lower),
                    opt.lower() == lower_name,
                ),
            )
            return best_option
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
        best_rank: Tuple[float, float, float, float] = (
            -1.0,
            -1.0,
            -1.0,
            float("-inf"),
        )
        text_lower = text.lower()
        text_ascii = _strip_diacritics(text_lower)

        for position, idx in enumerate(candidate_indices):
            node = self.address_nodes[idx]
            province, district, ward = self._resolve_node_fields(node)

            province = self._map_to_reference(province, self.province_lookup, text)
            district = self._map_to_reference(district, self.district_lookup, text)
            ward = self._map_to_reference(ward, self.ward_lookup, text)

            province, district, ward = self._validate_hierarchy(province, district, ward)

            filled = int(bool(province)) + int(bool(district)) + int(bool(ward))
            if filled == 0:
                continue

            candidate_raw = " ".join(
                part for part in (ward, district, province) if part
            )
            raw_score = (
                partial_ratio(candidate_raw.lower(), text.lower()) if candidate_raw else 0.0
            )
            def presence_value(part: str, weight: int) -> int:
                if not part:
                    return 0
                part_lower = part.lower()
                if part_lower in text_lower:
                    return weight * 2
                core_lower = _strip_admin_prefix(part).lower()
                if core_lower and core_lower in text_lower:
                    return weight * 2 - 1
                part_ascii = _strip_diacritics(part_lower)
                if part_ascii and part_ascii in text_ascii:
                    return weight
                core_ascii = _strip_diacritics(core_lower) if core_lower else ""
                if core_ascii and core_ascii in text_ascii:
                    return weight - 1
                return 0

            presence_score = (
                presence_value(province, 1)
                + presence_value(district, 3)
                + presence_value(ward, 5)
            )

            rank = (filled, float(presence_score), raw_score, -float(position))
            if rank > best_rank:
                best_rank = rank
                best_output = {"province": province, "district": district, "ward": ward}

            if filled == 3 and raw_score >= 100.0 and presence_score >= 8:
                break

        hint_province, hint_district, hint_ward = self._hierarchical_hint(
            normalized_query, text
        )
        hint_output = {
            "province": hint_province,
            "district": hint_district,
            "ward": hint_ward,
        }

        final_output = best_output.copy()
        if not final_output["province"] and hint_output["province"]:
            final_output["province"] = hint_output["province"]
        if not final_output["district"] and hint_output["district"]:
            final_output["district"] = hint_output["district"]
        if not final_output["ward"] and hint_output["ward"]:
            final_output["ward"] = hint_output["ward"]

        (
            final_output["province"],
            final_output["district"],
            final_output["ward"],
        ) = self._validate_hierarchy(
            final_output["province"],
            final_output["district"],
            final_output["ward"],
        )

        for key in ("province", "district", "ward"):
            final_output[key] = _strip_admin_prefix(final_output[key])

        if final_output["province"] or final_output["district"] or final_output["ward"]:
            return final_output
        for key in ("province", "district", "ward"):
            hint_output[key] = _strip_admin_prefix(hint_output[key])
        return hint_output


if __name__ == "__main__":
    solver = Solution()
    sample = ", Nam Đông,T. T.T.H"
    print(solver.process(sample))
