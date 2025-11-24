#!/usr/bin/env python
"""Synthetic NER dataset generation for Vietnamese administrative addresses."""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
DEFAULT_TYPE_WORDS = {
    "PROVINCE": "Tinh",
    "DISTRICT": "Huyen",
    "WARD": "Phuong",
}
TYPE_ABBREVIATIONS = {
    "province_city": ["TP.", "TP"],
    "province": ["Tinh"],
    "district_quan": ["Q.", "Q"],
    "district_huyen": ["H.", "H"],
    "district_thi_xa": ["TX.", "TX"],
    "district_city": ["TP.", "TP"],
    "ward_phuong": ["P.", "P"],
    "ward_xa": ["Xa", "X."],
    "ward_thi_tran": ["TT.", "TT"],
}
LABELS = [
    "O",
    "B-PROVINCE",
    "I-PROVINCE",
    "B-DISTRICT",
    "I-DISTRICT",
    "B-WARD",
    "I-WARD",
    "B-STREET",
    "I-STREET",
]

# Common Vietnamese street names and descriptors for synthetic address generation.
STREET_NAMES = [
    "Nguyễn Trãi",
    "Lê Lợi",
    "Trần Hưng Đạo",
    "Lý Thường Kiệt",
    "Phạm Văn Đồng",
    "Nguyễn Văn Cừ",
    "Cách Mạng Tháng Tám",
    "Pasteur",
    "Marry Curie",
    "Võ Thị Sáu",
    "Võ Văn Kiệt",
    "Hoàng Diệu",
    "Phạm Ngũ Lão",
    "Hai Bà Trưng",
    "Nam Kỳ Khởi Nghĩa",
    "Ngô Gia Tự",
    "Tố Hữu",
    "Điện Biên Phủ",
    "Đinh Tiên Hoàng",
    "Trần Cao Vân",
    "Đặng Văn Bi",
    "Phổ Quang",
    "Tôn Thất Đạm",
    "Trường Chinh",
    "Lê Hồng Phong",
    "Lê Duẩn",
    "Nguyễn Huệ",
    "Bạch Đằng",
    "Trần Phú",
    "Lý Thái Tổ",
    "Nguyễn Thái Học",
    "Trần Nhân Tông",
    "Phan Đình Phùng",
    "Nguyễn Đình Chiểu",
    "Trần Quang Khải",
    "Lê Quang Định",
    "Phan Xích Long",
    "Cao Thắng",
    "Nguyễn Oanh",
    "Lê Văn Sỹ",
    "Trần Xuân Soạn",
    "Nguyễn Ảnh Thủ",
    "Huỳnh Tấn Phát",
    "Lương Định Của",
    "Nguyễn Xiển",
    "Nguyễn Duy Trinh",
    "Phạm Hùng",
]

STREET_DESCRIPTORS = [
    "đường",
    "đg",
    "phố",
    "ph.",
    "ngõ",
    "ngách",
    "hẻm",
    "khu phố",
    "quốc lộ",
    "ql",
    "tỉnh lộ",
    "tl",
    "đại lộ",
    "trục",
]

RURAL_HAMLET_NAMES = [
    "An Bình", "An Lộc", "An Phú", "An Hòa", "An Thắng", "An Thành", "An Thuận",
    "An Hiệp", "An Tân", "An Nhơn", "An Đức", "An Lợi", "An Khánh", "An Trường",
    "An Quý", "Bình An", "Bình Lộc", "Bình Hưng", "Bình Mỹ", "Bình Thuận",
    "Bình Thành", "Bình Phước", "Bình Thạnh", "Bình Minh", "Bình Tiến",
    "Bình Trung", "Bình Đông", "Bình Tây", "Bình Nam", "Bình Bắc", "Bình Phú",
    "Bình Tường", "Bình Thịnh", "Bình Sơn", "Bình Long", "Bình Điền", "Tân An",
    "Tân Bình", "Tân Lộc", "Tân Phú", "Tân Hòa", "Tân Hiệp", "Tân Thuận",
    "Tân Long", "Tân Mỹ", "Tân Quý", "Tân Tiến", "Tân Thịnh", "Tân Phước",
    "Tân Khánh", "Tân Trung", "Tân Đông", "Tân Tây", "Tân Nam", "Tân Bắc",
    "Phú An", "Phú Lộc", "Phú Long", "Phú Mỹ", "Phú Hòa", "Phú Thuận",
    "Phú Thịnh", "Phú Thành", "Phú Hiệp", "Phú Đức", "Phú Tân", "Phú Khánh",
    "Phú Ninh", "Phú Quý", "Phú Sơn", "Mỹ An", "Mỹ Hòa", "Mỹ Hiệp", "Mỹ Lộc",
    "Mỹ Long", "Mỹ Đức", "Mỹ Phú", "Mỹ Thuận", "Mỹ Thạnh", "Mỹ Thành",
    "Mỹ Thắng", "Mỹ Sơn", "Mỹ Tiến", "Vĩnh An", "Vĩnh Lộc", "Vĩnh Long",
    "Vĩnh Hòa", "Vĩnh Thuận", "Vĩnh Thịnh", "Vĩnh Thành", "Vĩnh Hiệp",
    "Vĩnh Quang", "Vĩnh Sơn", "Vĩnh Phúc", "Vĩnh Tân", "Long An", "Long Hòa",
    "Long Thuận", "Long Phước", "Long Hiệp", "Long Thành", "Long Đức",
    "Long Giang", "Long Tiến", "Long Khánh", "Long Sơn", "Long Tân", "Hòa An",
    "Hòa Bình", "Hòa Lộc", "Hòa Long", "Hòa Thuận", "Hòa Thạnh", "Hòa Thành",
    "Hòa Hiệp", "Hòa Phú", "Hòa Lợi", "Hòa Tiến", "Hòa Sơn", "Hòa Đông",
    "Hòa Tây", "Hòa Trung", "Thạnh An", "Thạnh Lộc", "Thạnh Mỹ", "Thạnh Phú",
    "Thạnh Sơn", "Thạnh Hòa", "Thạnh Thuận", "Thạnh Đức", "Thạnh Đông",
    "Thạnh Tây", "Trung An", "Trung Lộc", "Trung Hòa", "Trung Sơn", "Trung Phú",
    "Trung Thuận", "Trung Thành", "Trung Hiệp", "Trung Đức", "Trung Tín",
    "Đông An", "Đông Lộc", "Đông Hòa", "Đông Phú", "Đông Thuận", "Đông Tiến",
    "Đông Thạnh", "Đông Sơn", "Đông Khánh", "Đông Thành", "Tây An", "Tây Hòa",
    "Tây Lộc", "Tây Thuận", "Tây Thạnh", "Tây Sơn", "Tây Phú", "Tây Hiệp",
    "Nam An", "Nam Lộc", "Nam Phú", "Nam Hòa", "Nam Thuận", "Nam Thạnh",
    "Nam Sơn", "Nam Hiệp", "Bắc An", "Bắc Lộc", "Bắc Hòa", "Bắc Phú",
    "Bắc Thuận", "Bắc Sơn", "Bắc Thịnh", "Bắc Thành", "Lộc An", "Lộc Điền",
    "Lộc Sơn", "Lộc Thắng", "Lộc Thịnh", "Lộc Hòa", "Lộc Thuận", "Lộc Thành",
    "Lộc Tiến", "Thuận An", "Thuận Lộc", "Thuận Hòa", "Thuận Phú", "Thuận Mỹ",
    "Thuận Sơn", "Thuận Đức", "Thuận Thành", "Thuận Tiến", "Thuận Thịnh",
    "Hiệp An", "Hiệp Lợi", "Hiệp Phú", "Hiệp Thạnh", "Hiệp Hòa", "Hiệp Thuận",
    "Hiệp Đức", "Hiệp Thành", "Hiệp Sơn", "Minh An", "Minh Lộc", "Minh Phú",
    "Minh Hòa", "Minh Thuận", "Minh Thạnh", "Minh Đức", "Minh Tiến", "Minh Tân",
    "Quang An", "Quang Lộc", "Quang Phú", "Quang Hòa", "Quang Thuận",
    "Quang Thạnh", "Quang Trung", "Quang Hiệp", "Ngọc An", "Ngọc Lộc",
    "Ngọc Phú", "Ngọc Sơn", "Ngọc Hòa", "Ngọc Thuận", "Ngọc Tiến", "Ngọc Khánh",
    "Phước An", "Phước Lộc", "Phước Long", "Phước Mỹ", "Phước Hòa",
    "Phước Thuận", "Phước Thạnh", "Phước Sơn", "Phước Tiến", "Gia An",
    "Gia Lộc", "Gia Hòa", "Gia Phú", "Gia Thuận", "Gia Thạnh", "Kim Long",
    "Kim Sơn", "Kim Phú", "Kim Thịnh", "Cẩm An", "Cẩm Lộc", "Cẩm Hòa",
    "Cẩm Phú", "Cẩm Sơn"
]

URBAN_HAMLET_NAMES = [
    "An Lạc", "Bình Trị Đông", "Bình Hưng", "Bình Phú", "Bình Tân", "Bình Chiểu",
    "Bình Thọ", "Hiệp Bình", "Hiệp Bình Chánh", "Hiệp Bình Phước", "Hiệp Thành",
    "Hiệp Phú", "Tân Phú", "Tân Sơn Nhì", "Tân Thành", "Tân Thới", "Tân Thuận",
    "Tân Kiểng", "Tân Quy", "Tân Hưng", "Tân Cảng", "Tân Định", "Phú Mỹ",
    "Phú Mỹ Hưng", "Phú Mỹ Đông", "Phú Thuận", "Phú Thuận Đông", "Phú Thuận Tây",
    "Phú Hữu", "Phú Nhuận", "Phú Lợi", "Phú Thạnh", "Phú Lâm", "Phú Tân",
    "Thạnh Mỹ Lợi", "Thạnh Lộc", "Thạnh Xuân", "Thạnh Phú", "Thạnh Hòa",
    "Thảo Điền", "Thảo Điền Pearl", "Trung Sơn", "Trung Mỹ Tây", "Trung Chánh",
    "Linh Trung", "Linh Đông", "Linh Tây", "Linh Xuân", "Linh Chiểu",
    "Vạn Phúc", "Vạn Phúc City", "Vạn Kiếp", "Vạn Lộc", "Vạn Thạnh",
    "Vĩnh Lộc", "Vĩnh Lộc A", "Vĩnh Lộc B", "Vĩnh Hiệp", "Vĩnh Thạnh",
    "Hưng Phú", "Hưng Lợi", "Đông Hòa", "Đông Hưng Thuận", "Đông Thạnh",
    "Tây Thạnh", "Tây Sơn", "Tam Bình", "Tam Phú", "Nam Long", "Nam Hưng",
    "Cityland", "Cityland Garden", "Cityland Park Hills", "Cityland Riverside",
    "Cityland Center", "Lakeview City", "EcoCity", "Riverside City",
    "Green Riverside", "Green Valley", "Sky Garden", "Sky View", "Topaz City",
    "Topaz Home", "Richland", "RichStar", "Pearl Plaza", "Pearl Garden",
    "Garden Plaza", "River Park", "River View", "Sunrise City",
    "Sunrise Riverside", "Green City", "Golden River", "Golden Park",
    "Diamond Island", "Diamond Lotus", "Saigon Mia", "Saigon Pearl",
    "The Manor", "The Vista", "Vista Verde", "Masteri", "Masteri Parkland",
    "Vinhomes Central Park", "Vinhomes Golden River", "Vinhomes Grand Park",
    "Celadon City", "Sala", "EcoPark", "Gamuda City", "Royal City",
    "Times City", "Park Hill", "Goldmark City", "Splendora", "Ciputra"
]


@dataclass
class LabelingResult:
    tokens: List[str]
    ner_tags: List[str]
    matches: Dict[str, bool]


def _normalized_token(token: str) -> str:
    return strip_accents(token or "").lower()


def _normalize_phrase(text: Optional[str]) -> List[str]:
    if not text:
        return []
    cleaned = clean_text(text, remove_slash=False)
    return tokenize(cleaned)


def tag_phrase(
    token_pairs: List[Tuple[str, str]], phrase: Optional[str], label: str
) -> bool:
    phrase_tokens = _normalize_phrase(phrase)
    if not phrase_tokens:
        return False
    normalized_phrase = [_normalized_token(tok) for tok in phrase_tokens]
    normalized_tokens = [_normalized_token(tok) for tok, _ in token_pairs]
    window = len(phrase_tokens)
    for start in range(len(token_pairs) - window + 1):
        if normalized_tokens[start : start + window] != normalized_phrase:
            continue
        token_pairs[start] = (token_pairs[start][0], f"B-{label}")
        for offset in range(1, window):
            idx = start + offset
            token_pairs[idx] = (token_pairs[idx][0], f"I-{label}")
        return True
    return False


def label_tokens(
    address: str,
    *,
    street: Optional[str] = None,
    province: Optional[str] = None,
    district: Optional[str] = None,
    ward: Optional[str] = None,
) -> LabelingResult:
    cleaned_address = clean_text(address, remove_slash=False)
    token_pairs: List[Tuple[str, str]] = [
        (tok, "O") for tok in tokenize(cleaned_address)
    ]
    matches = {
        "STREET": tag_phrase(token_pairs, street, "STREET"),
        "WARD": tag_phrase(token_pairs, ward, "WARD"),
        "DISTRICT": tag_phrase(token_pairs, district, "DISTRICT"),
        "PROVINCE": tag_phrase(token_pairs, province, "PROVINCE"),
    }
    return LabelingResult(
        tokens=[tok for tok, _ in token_pairs],
        ner_tags=[tag for _, tag in token_pairs],
        matches=matches,
    )


@dataclass(frozen=True)
class NameVariant:
    text: str
    includes_type: bool


@dataclass
class Component:
    code: str
    label: str
    names: List[NameVariant]
    type_hint: str
    type_word: str

    def pick_name(
        self, *, rng: random.Random, prefer_full: bool, prefer_short: bool
    ) -> NameVariant:
        pool = self.names
        if prefer_full:
            full = [n for n in pool if n.includes_type]
            if full:
                pool = full
        elif prefer_short:
            short = [n for n in pool if not n.includes_type]
            if short:
                pool = short
        return rng.choice(pool)

    def resolve_type_token(
        self,
        *,
        rng: random.Random,
        abbreviate: bool,
    ) -> str:
        token = self.type_word or DEFAULT_TYPE_WORDS.get(self.label, "")
        if abbreviate:
            candidates = TYPE_ABBREVIATIONS.get(
                self.type_hint
            ) or TYPE_ABBREVIATIONS.get(self.label.lower())
            if candidates:
                token = rng.choice(candidates)
        return token


@dataclass
class AddressRecord:
    ward_code: str
    ward: Component
    province: Component
    district: Optional[Component] = None
    source: str = "old"

    def components(self) -> List[Tuple[str, Component]]:
        parts: List[Tuple[str, Component]] = [("WARD", self.ward)]
        if self.district:
            parts.append(("DISTRICT", self.district))
        parts.append(("PROVINCE", self.province))
        return parts


@dataclass(frozen=True)
class VariantSpec:
    name: str
    lowercase: bool = False
    strip_accents: bool = False
    use_commas: bool = True
    abbreviate_types: bool = False
    drop_type_tokens: bool = False
    prefer_full_name: bool = False
    prefer_short_name: bool = False
    include_street: bool = True
    connectors: Dict[str, Sequence[str]] = field(default_factory=dict)
    component_order: Tuple[str, ...] = ("WARD", "DISTRICT", "PROVINCE")


VARIANT_SPECS: Tuple[VariantSpec, ...] = (
    # ===== Cơ bản =====
    VariantSpec(name="standard"),
    VariantSpec(name="standard_no_commas", use_commas=False),
    VariantSpec(name="lowercase", lowercase=True),
    VariantSpec(name="lowercase_no_commas", lowercase=True, use_commas=False),

    # Không dấu (accentless), có / không dấu phẩy
    VariantSpec(
        name="accentless",
        lowercase=True,
        strip_accents=True,
        use_commas=False,
    ),
    VariantSpec(
        name="accentless_commas",
        lowercase=True,
        strip_accents=True,
        use_commas=True,
    ),

    # Viết tắt loại (TP./Q./P./...) nhưng vẫn giữ type token
    VariantSpec(name="abbrev_commas", abbreviate_types=True),
    VariantSpec(
        name="abbrev_no_commas",
        abbreviate_types=True,
        use_commas=False,
    ),

    # ===== Compact: bỏ type token, chỉ giữ tên trơ =====
    # "compact" ở đây = không còn từ loại (Tỉnh/Quận/...) => abbreviate_types không còn ý nghĩa -> đặt False.
    VariantSpec(
        name="compact_commas",
        use_commas=True,
        abbreviate_types=False,
        drop_type_tokens=True,
    ),
    VariantSpec(
        name="compact_no_commas",
        use_commas=False,
        abbreviate_types=False,
        drop_type_tokens=True,
    ),
    VariantSpec(
        name="compact_short",
        lowercase=True,
        strip_accents=False,
        use_commas=False,
        abbreviate_types=False,
        drop_type_tokens=True,
        prefer_short_name=True,
    ),
    VariantSpec(
        name="compact_short_evil",
        lowercase=True,
        strip_accents=True,
        use_commas=False,
        abbreviate_types=False,
        drop_type_tokens=True,
        prefer_short_name=True,
    ),

    # ===== Evil nhưng còn type =====
    VariantSpec(
        name="evil_full_types",
        lowercase=True,
        strip_accents=True,
        use_commas=False,
        abbreviate_types=False,  # "phuong", "quan", "thanh pho"
        drop_type_tokens=False,  # vẫn giữ type
        prefer_full_name=True,
    ),
    VariantSpec(
        name="evil_abbrev_types",
        lowercase=True,
        strip_accents=True,
        use_commas=False,
        abbreviate_types=True,   # p., q., tp.
        drop_type_tokens=False,  # vẫn giữ type
        prefer_full_name=True,
    ),

    # ===== Có connector nghĩa ("thuộc") =====
    VariantSpec(
        name="meaningful_connectors",
        connectors={
            "street_ward": [","],
            "ward_district": ["thuộc"],
            "district_province": ["thuộc"],
            "ward_province": ["thuộc"],
        },
        prefer_full_name=True,
    ),

    # ===== Không street, nhưng vẫn đủ cấp hành chính =====
    VariantSpec(
        name="no_street_compact",
        include_street=False,
        use_commas=False,
        abbreviate_types=True,   # P./Q./TP.
        lowercase=True,
    ),
    VariantSpec(
        name="no_street_full",
        include_street=False,
        use_commas=True,
        abbreviate_types=False,
        drop_type_tokens=False,
    ),
    VariantSpec(
        name="no_street_full_abbrev",
        include_street=False,
        use_commas=True,
        abbreviate_types=True,   # P./Q./TP.
        drop_type_tokens=False,
    ),

    # ===== Subset theo cấp =====
    VariantSpec(
        name="ward_only",
        include_street=False,
        component_order=("WARD",),
    ),
    VariantSpec(
        name="ward_only_abbrev",
        include_street=False,
        component_order=("WARD",),
        abbreviate_types=True,
    ),
    VariantSpec(
        name="district_only",
        include_street=False,
        component_order=("DISTRICT",),
    ),
    VariantSpec(
        name="district_only_abbrev",
        include_street=False,
        component_order=("DISTRICT",),
        abbreviate_types=True,
    ),
    VariantSpec(
        name="province_only",
        include_street=False,
        component_order=("PROVINCE",),
    ),
    VariantSpec(
        name="province_only_abbrev",
        include_street=False,
        component_order=("PROVINCE",),
        abbreviate_types=True,
    ),

    # ===== Các tổ hợp 2 cấp =====
    VariantSpec(
        name="ward_province",
        include_street=False,
        component_order=("WARD", "PROVINCE"),
    ),
    VariantSpec(
        name="ward_province_abbrev",
        include_street=False,
        component_order=("WARD", "PROVINCE"),
        abbreviate_types=True,
    ),
    VariantSpec(
        name="province_ward",
        include_street=False,
        component_order=("PROVINCE", "WARD"),
    ),
    VariantSpec(
        name="province_ward_abbrev",
        include_street=False,
        component_order=("PROVINCE", "WARD"),
        abbreviate_types=True,
    ),
    VariantSpec(
        name="district_province",
        include_street=False,
        component_order=("DISTRICT", "PROVINCE"),
    ),
    VariantSpec(
        name="district_province_abbrev",
        include_street=False,
        component_order=("DISTRICT", "PROVINCE"),
        abbreviate_types=True,
    ),
    VariantSpec(
        name="province_district",
        include_street=False,
        component_order=("PROVINCE", "DISTRICT"),
    ),
    VariantSpec(
        name="province_district_abbrev",
        include_street=False,
        component_order=("PROVINCE", "DISTRICT"),
        abbreviate_types=True,
    ),

    # ===== Đảo order 3 cấp =====
    VariantSpec(
        name="ward_district",
        component_order=("WARD", "DISTRICT"),
    ),
    VariantSpec(
        name="ward_district_abbrev",
        component_order=("WARD", "DISTRICT"),
        abbreviate_types=True,
    ),
    VariantSpec(
        name="district_first",
        component_order=("DISTRICT", "WARD", "PROVINCE"),
    ),
    VariantSpec(
        name="district_first_abbrev",
        component_order=("DISTRICT", "WARD", "PROVINCE"),
        abbreviate_types=True,
    ),
    VariantSpec(
        name="province_first",
        component_order=("PROVINCE", "DISTRICT", "WARD"),
    ),
    VariantSpec(
        name="province_first_abbrev",
        component_order=("PROVINCE", "DISTRICT", "WARD"),
        abbreviate_types=True,
    ),
)


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    result = []
    for char in normalized:
        if unicodedata.category(char) == "Mn":
            continue
        if char == "đ":
            result.append("d")
        elif char == "Đ":
            result.append("D")
        else:
            result.append(char)
    return "".join(result)


def randomize_text_variant(
    text: str,
    rng: random.Random,
    *,
    accent_probability: float = 0.6,
    allow_title: bool = True,
) -> str:
    variant = text
    if rng.random() > accent_probability:
        variant = strip_accents(variant)
    style_roll = rng.random()
    if style_roll < 0.25:
        variant = variant.lower()
    elif style_roll < 0.32:
        variant = variant.upper()
    elif style_roll < 0.48 and allow_title:
        variant = variant.title()
    return variant


def clean_text(value: Optional[str], *, remove_slash: bool = True) -> str:
    if not value:
        return ""
    value = value.replace("\u00a0", " ")
    if remove_slash:
        value = value.replace("/", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def capitalize_first(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def infer_includes_type(text: str) -> bool:
    base = strip_accents(clean_text(text)).lower()
    prefixes = (
        "thanh pho",
        "tinh",
        "quan",
        "huyen",
        "phuong",
        "xa",
        "thi xa",
        "thi tran",
        "city",
        "district",
        "ward",
        "province",
    )
    return any(base.startswith(prefix) for prefix in prefixes)


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


def transform_tokens(tokens: List[str], spec: VariantSpec) -> List[str]:
    processed = []
    for token in tokens:
        new_token = token
        if spec.strip_accents:
            new_token = strip_accents(new_token)
        if spec.lowercase:
            new_token = new_token.lower()
        processed.append(new_token)
    return processed


def add_connector_tokens(
    tokens: List[str],
    connectors: Sequence[str],
) -> None:
    for connector in connectors:
        if connector:
            tokens.extend(tokenize(connector))


def detect_type_hint(
    level: str, full_name: str, admin_code_name: Optional[str] = None
) -> str:
    if admin_code_name:
        code = admin_code_name.lower()
        # Prefer explicit mapping from administrative unit definitions.
        admin_code_map = {
            "province": {
                "thanh_pho_truc_thuoc_trung_uong": "province_city",
                "tinh": "province",
            },
            "district": {
                "quan": "district_quan",
                "huyen": "district_huyen",
                "thi_xa": "district_thi_xa",
                "thi_tran": "district_thi_tran",
                "thanh_pho_thuoc_tinh": "district_city",
                "thanh_pho_thuoc_thanh_pho_truc_thuoc_trung_uong": "district_city",
            },
            "ward": {
                "phuong": "ward_phuong",
                "xa": "ward_xa",
                "thi_tran": "ward_thi_tran",
            },
        }
        mapped = admin_code_map.get(level, {}).get(code)
        if mapped:
            return mapped
    if admin_code_name:
        normalized = admin_code_name.lower()
        if level == "province":
            if "thanh_pho" in normalized:
                return "province_city"
            return "province"
        if level == "district":
            if "quan" in normalized:
                return "district_quan"
            if "thi_xa" in normalized:
                return "district_thi_xa"
            if "thi_tran" in normalized:
                return "district_thi_tran"
            if "thanh_pho" in normalized:
                return "district_city"
            return "district_huyen"
        if level == "ward":
            if "phuong" in normalized:
                return "ward_phuong"
            if "thi_tran" in normalized:
                return "ward_thi_tran"
            if "xa" in normalized:
                return "ward_xa"
            return "ward_phuong"
    base = strip_accents(clean_text(full_name or "")).lower()
    if level == "province":
        if base.startswith("thanh pho"):
            return "province_city"
        return "province"
    if level == "district":
        if base.startswith("quan"):
            return "district_quan"
        if base.startswith("huyen"):
            return "district_huyen"
        if base.startswith("thi xa"):
            return "district_thi_xa"
        if base.startswith("thi tran"):
            return "district_thi_tran"
        if base.startswith("thanh pho"):
            return "district_city"
        return "district_huyen"
    if level == "ward":
        if base.startswith("phuong"):
            return "ward_phuong"
        if base.startswith("xa"):
            return "ward_xa"
        if base.startswith("thi tran"):
            return "ward_thi_tran"
        return "ward_phuong"
    return level


def extract_type_word(full_name: str, fallback: str) -> str:
    if not full_name:
        return fallback
    tokens = clean_text(full_name).split()
    if not tokens:
        return fallback
    if len(tokens) >= 2:
        first_two = " ".join(tokens[:2])
        base_two = strip_accents(first_two).lower()
        if base_two in {"thanh pho", "thi xa", "thi tran"}:
            return first_two
    return tokens[0]


def _load_admin_unit_map(path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for row in _load_json(path):
        identifier = row.get("id")
        code_name = row.get("code_name")
        if identifier is None or code_name is None:
            continue
        try:
            mapping[int(identifier)] = code_name
        except (TypeError, ValueError):
            continue
    return mapping


def deduplicate_variants(variants: Iterable[NameVariant]) -> List[NameVariant]:
    seen = set()
    result: List[NameVariant] = []
    for variant in variants:
        key = strip_accents(variant.text).lower()
        if key and key not in seen:
            seen.add(key)
            result.append(variant)
    return result


def collect_variants(record: Dict[str, str]) -> List[NameVariant]:
    variants: List[NameVariant] = []
    fields = [
        ("name", False),
        ("full_name", True),
        # ("name_en", False),
        # ("full_name_en", True),
        ("code_name", False),
    ]
    for key, default_includes_type in fields:
        raw = record.get(key)
        if not raw:
            continue
        text = clean_text(raw.replace("_", " "))
        if not text:
            continue
        includes_type = default_includes_type or infer_includes_type(text)
        variants.append(NameVariant(text=text, includes_type=includes_type))
    return variants


def build_components(
    *,
    level: str,
    records: Iterable[Dict[str, str]],
    extra_records: Iterable[Dict[str, str]] = (),
    admin_units: Optional[Dict[int, str]] = None,
) -> Dict[str, Dict[str, object]]:
    data: Dict[str, Dict[str, object]] = {}
    for source in (records, extra_records):
        for record in source:
            code = record.get("code")
            if not code:
                continue
            admin_id = record.get("administrative_unit_id")
            admin_code_name = None
            if admin_units and admin_id is not None:
                try:
                    admin_code_name = admin_units.get(int(admin_id))
                except (TypeError, ValueError):
                    admin_code_name = None
            entry = data.setdefault(
                code,
                {
                    "code": code,
                    "names": [],
                    "full_name": record.get("full_name", ""),
                    "type_hint": "",
                    "type_word": "",
                    "attributes": {},
                },
            )
            entry["names"].extend(collect_variants(record))
            entry["full_name"] = entry["full_name"] or record.get("full_name", "")
            entry["attributes"]["administrative_unit_id"] = admin_id
            entry["attributes"]["administrative_unit_code_name"] = admin_code_name
            entry["type_hint"] = entry["type_hint"] or detect_type_hint(
                level, record.get("full_name", ""), admin_code_name
            )
            if not entry["type_word"]:
                entry["type_word"] = extract_type_word(
                    record.get("full_name", ""),
                    DEFAULT_TYPE_WORDS.get(level.upper(), ""),
                )
            if level == "district":
                entry["attributes"]["province_code"] = record.get("province_code")
            if level == "ward":
                entry["attributes"]["district_code"] = record.get("district_code")
                entry["attributes"]["province_code"] = record.get("province_code")
    return data


def materialize_component(raw: Dict[str, object], label: str) -> Optional[Component]:
    names = deduplicate_variants(raw["names"])
    if not names:
        return None
    return Component(
        code=raw["code"],
        label=label,
        names=names,
        type_hint=raw.get("type_hint") or label.lower(),
        type_word=raw.get("type_word") or DEFAULT_TYPE_WORDS.get(label, ""),
    )


def assemble_records(data_dir: Path) -> List[AddressRecord]:
    print("Loading administrative data in directory:", data_dir)
    old_records = _assemble_old_structure(data_dir)
    new_records = _assemble_new_structure(data_dir)
    return old_records + new_records


def _assemble_old_structure(data_dir: Path) -> List[AddressRecord]:
    admin_units = _load_admin_unit_map(data_dir / "old_administrative_units.json")
    provinces_raw = build_components(
        level="province",
        records=_load_json(data_dir / "old_provinces.json"),
        admin_units=admin_units,
    )
    districts_raw = build_components(
        level="district",
        records=_load_json(data_dir / "old_districts.json"),
        admin_units=admin_units,
    )
    wards_raw = build_components(
        level="ward",
        records=_load_json(data_dir / "old_wards.json"),
        admin_units=admin_units,
    )

    records: List[AddressRecord] = []
    for ward_code, ward_entry in wards_raw.items():
        district_code = ward_entry.get("attributes", {}).get("district_code")
        if not district_code:
            continue
        district_entry = districts_raw.get(district_code)
        if not district_entry:
            continue
        province_code = district_entry.get("attributes", {}).get("province_code")
        if not province_code:
            continue
        province_entry = provinces_raw.get(province_code)
        if not province_entry:
            continue

        ward_component = materialize_component(ward_entry, "WARD")
        district_component = materialize_component(district_entry, "DISTRICT")
        province_component = materialize_component(province_entry, "PROVINCE")
        if not ward_component or not district_component or not province_component:
            continue

        records.append(
            AddressRecord(
                ward_code=ward_code,
                ward=ward_component,
                district=district_component,
                province=province_component,
                source="old",
            )
        )
    return records


def _assemble_new_structure(data_dir: Path) -> List[AddressRecord]:
    admin_units = _load_admin_unit_map(data_dir / "administrative_units.json")
    provinces_raw = build_components(
        level="province",
        records=_load_json(data_dir / "provinces.json"),
        admin_units=admin_units,
    )
    wards_raw = build_components(
        level="ward",
        records=_load_json(data_dir / "wards.json"),
        admin_units=admin_units,
    )

    records: List[AddressRecord] = []
    for ward_code, ward_entry in wards_raw.items():
        province_code = ward_entry.get("attributes", {}).get(
            "province_code"
        ) or ward_entry.get("attributes", {}).get("parent_code")
        if not province_code:
            continue
        province_entry = provinces_raw.get(province_code)
        if not province_entry:
            continue

        ward_component = materialize_component(ward_entry, "WARD")
        province_component = materialize_component(province_entry, "PROVINCE")
        if not ward_component or not province_component:
            continue

        records.append(
            AddressRecord(
                ward_code=ward_code,
                ward=ward_component,
                province=province_component,
                district=None,
                source="new",
            )
        )
    return records


def build_street_tokens(rng: random.Random) -> List[str]:
    number = rng.randint(1, 999)
    alley = rng.randint(1, 150)
    street = randomize_text_variant(rng.choice(STREET_NAMES), rng)
    descriptor = randomize_text_variant(rng.choice(STREET_DESCRIPTORS), rng)
    number_word = randomize_text_variant(
        rng.choice(["số", "số nhà", "No.", "so"]),
        rng,
        accent_probability=0.4,
        allow_title=False,
    )
    alley_word = randomize_text_variant(
        rng.choice(["ngõ", "ngách", "hẻm", "ngo", "hem"]), rng
    )
    
    templates = [
        # Basic
        f"{number} {street}",
        f"{number} {descriptor} {street}",
        f"{descriptor} {street}",
        f"{street}",

        # Alley / hẻm
        f"{number}/{alley} {descriptor} {street}",
        f"{number}/{alley} {street}",
        f"{alley_word} {alley} {descriptor} {street}",
        f"{descriptor} {street} {alley_word} {alley}",
        f"{alley_word} {alley}/{number} {street}",

        # With 'số', 'số nhà', 'No.'
        f"{number_word} {number} {street}",
        f"{number_word} {number} {descriptor} {street}",
        f"{number_word} {number}/{alley} {street}",
        f"{number_word} {number}/{alley} {descriptor} {street}",

        # Descriptor first then number
        f"{descriptor} {street} {number_word} {number}",
    ]

    text = rng.choice(templates)
    return tokenize(text)

def build_hamlet_tokens(rng: random.Random, *, urban: bool = False) -> List[str]:
    """
    Sinh cụm thôn/ấp/khu phố/tổ dân phố.

    - urban=False (rural): Ấp, thôn, xóm, làng, buôn, bản, tổ + tên ấp/thôn/bản.
    - urban=True  (urban): Khu phố, KDC, TDP, tổ dân phố/dân cư + tên KDC/dự án.
    """

    if urban:
        # Đô thị: KP/Khu phố/KDC/TDP, tổ dân phố/dân cư
        kp_like_prefixes = [
            "khu phố",
            "kp",
            "kp.",
            "khu dân cư",
            "kdc",
        ]
        tdp_like_prefixes = [
            "tổ dân phố",
            "tổ dân cư",
            "tdp",
            "tổ",
        ]
        all_prefixes = kp_like_prefixes + tdp_like_prefixes
    else:
        # Nông thôn: Ấp/thôn/xóm/làng/buôn/bản + tổ
        village_like_prefixes = [
            "ấp",
            "ấp.",
            "thôn",
            "xóm",
            "làng",
            "buôn",
            "bản",
        ]
        tdp_like_prefixes = [
            "tổ",  # tổ 5, tổ 12...
        ]
        kp_like_prefixes = []  # rural không dùng khu phố/KDC
        all_prefixes = village_like_prefixes + tdp_like_prefixes

    raw_prefix = rng.choice(all_prefixes)
    prefix = capitalize_first(raw_prefix)


    # Chọn pattern identifier theo nhóm prefix
    if urban and raw_prefix in kp_like_prefixes:
        # Khu phố/KDC/KP: "Khu phố 5", "KP 3A", "Khu dân cư Phú Mỹ"
        identifier = rng.choice(
            [
                str(rng.randint(1, 20)),
                f"{rng.randint(1, 20)}{rng.choice(string.ascii_uppercase)}",
                rng.choice(URBAN_HAMLET_NAMES),
            ]
        )
    elif raw_prefix in tdp_like_prefixes:
        # Tổ/TDP (cả urban & rural): "Tổ 5", "Tổ 12A"
        identifier = rng.choice(
            [
                str(rng.randint(1, 40)),
                f"{rng.randint(1, 20)}{rng.choice(string.ascii_uppercase)}",
            ]
        )
    else:
        # Ấp/thôn/xóm/làng/buôn/bản: "Ấp 3A", "Thôn 5BC", "Bản Phước Lộc"
        identifier = rng.choice(
            [
                str(rng.randint(1, 30)),
                f"{rng.randint(1, 20)}{rng.choice(string.ascii_uppercase)}",
                f"{rng.randint(1, 20)}{rng.choice(string.ascii_uppercase)}{rng.choice(string.ascii_uppercase)}",
                rng.choice(RURAL_HAMLET_NAMES),
            ]
        )

    return tokenize(f"{prefix} {identifier}")

def build_location_prefix_tokens(
    rng: random.Random,
    ward_type_hint: Optional[str],
    district_type_hint: Optional[str],
) -> List[str]:
    # Emit hamlet/khu pho variants in rural/commune contexts and occasionally in urban wards,
    # while keeping a street component so mixes like "Đường ... Ấp ..." or "KDC ... Đường ..." appear.
    is_rural = (ward_type_hint in {"ward_xa", "ward_thi_tran"}) or (
        district_type_hint in {"district_huyen", "district_thi_xa", "district_thi_tran"}
    )
    is_urban = (ward_type_hint == "ward_phuong") or (
        district_type_hint in {"district_quan", "district_city"}
    )

    rural_prob = 0.45  # higher share of hamlet/ấp in rural
    urban_prob = 0.2   # smaller share of khu phố/TDP in urban
    street_tokens = build_street_tokens(rng)
    hamlet_tokens: Optional[List[str]] = None
    if is_rural and rng.random() < rural_prob:
        hamlet_tokens = build_hamlet_tokens(rng, urban=False)
    elif is_urban and rng.random() < urban_prob:
        hamlet_tokens = build_hamlet_tokens(rng, urban=True)

    if hamlet_tokens:
        # Mix order to mimic real usage: rural often "Đường ... Ấp ...", urban often "KDC ... Đường ..."
        if rng.random() < 0.6:
            return street_tokens + hamlet_tokens
        return hamlet_tokens + street_tokens

    return street_tokens

def render_component_tokens(
    component: Component,
    spec: VariantSpec,
    rng: random.Random,
) -> List[str]:
    variant = component.pick_name(
        rng=rng,
        prefer_full=spec.prefer_full_name,
        prefer_short=spec.prefer_short_name,
    )
    tokens: List[str] = []
    name_tokens = tokenize(variant.text)

    if not spec.drop_type_tokens and not variant.includes_type:
        # Lấy token loại: Quận/Huyện/Phường...
        type_token = component.resolve_type_token(
            rng=rng, abbreviate=spec.abbreviate_types
        )

        if type_token:
            first_token = name_tokens[0] if name_tokens else ""
            if spec.abbreviate_types and first_token:
                is_digit = first_token[0].isdigit()
                has_punctuation = any(not ch.isalnum() for ch in type_token)

                # -----------------------
                # 1) Compact với số: Q1 / Q.1 / Q-1 ...
                # -----------------------
                if is_digit and rng.random() < 0.5:
                    tokens.append(f"{type_token}{first_token}")
                    name_tokens = name_tokens[1:]
                # -----------------------
                # 2) Compact với chữ: chỉ khi type_token có dấu (., - ...)
                #    => sinh kiểu Q.Bình / Q-Bình, KHÔNG có QBình
                # -----------------------
                elif (not is_digit) and has_punctuation and rng.random() < 0.5:
                    tokens.append(f"{type_token}{first_token}")  # Q.Binh / Q-Binh
                    name_tokens = name_tokens[1:]
                else:
                    # Non-compact: Q Bình / Q. Bình
                    tokens.extend(tokenize(type_token))
            else:
                # Không abbreviate: "Quận Bình Thạnh", "Phường An Phú", ...
                tokens.extend(tokenize(type_token))

    tokens.extend(name_tokens)
    return tokens


def render_data_sample(
    record: AddressRecord,
    spec: VariantSpec,
    rng: random.Random,
) -> Optional[Tuple[List[str], List[str], str]]:
    tokens: List[str] = []
    component_spans: Dict[str, Tuple[int, int]] = {}

    connector_key_map = {
        ("STREET", "WARD"): "street_ward",
        ("STREET", "DISTRICT"): "street_district",
        ("WARD", "DISTRICT"): "ward_district",
        ("DISTRICT", "PROVINCE"): "district_province",
        ("WARD", "PROVINCE"): "ward_province",
        ("PROVINCE", "WARD"): "province_ward",
        ("PROVINCE", "DISTRICT"): "province_district",
    }

    def connectors_between(prev_label: Optional[str], next_label: str) -> Sequence[str]:
        if not prev_label:
            return []
        key = connector_key_map.get((prev_label, next_label))
        if not key:
            return []
        if key in spec.connectors:
            return spec.connectors[key]
        return [","] if spec.use_commas else []

    component_map = {label: component for label, component in record.components()}
    ordered_labels = [
        label for label in spec.component_order if label in component_map
    ]
    if not ordered_labels:
        return None

    if spec.include_street:
        ward_component = component_map.get("WARD")
        district_component = component_map.get("DISTRICT")
        street_tokens = build_location_prefix_tokens(
            rng,
            ward_component.type_hint if ward_component else None,
            district_component.type_hint if district_component else None,
        )
        start_idx = len(tokens)
        tokens.extend(street_tokens)
        component_spans["STREET"] = (start_idx, len(street_tokens))
        previous_label: Optional[str] = "STREET"
    else:
        previous_label = None

    for label in ordered_labels:
        component = component_map[label]
        component_tokens = render_component_tokens(component, spec, rng)
        if not component_tokens:
            return None
        connector_tokens = connectors_between(previous_label, label)
        if connector_tokens and tokens:
            add_connector_tokens(tokens, connector_tokens)
        start_idx = len(tokens)
        tokens.extend(component_tokens)
        component_spans[label] = (start_idx, len(component_tokens))
        previous_label = label

    tokens = transform_tokens(tokens, spec)
    if not component_spans:
        return None

    text = " ".join(tokens)

    def span_text(key: str) -> str:
        span = component_spans.get(key)
        if not span:
            return ""
        start, length = span
        return " ".join(tokens[start : start + length])

    labeling = label_tokens(
        text,
        street=span_text("STREET"),
        province=span_text("PROVINCE"),
        district=span_text("DISTRICT"),
        ward=span_text("WARD"),
    )
    required_labels = list(component_spans.keys())
    if not all(labeling.matches.get(key, False) for key in required_labels):
        return None
    return labeling.tokens, labeling.ner_tags, text


def _load_json(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("fuzz/data"),
        help="Directory that contains the administrative data json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ner/datasets/standard"),
        help="Where the generated dataset files will be stored.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of samples allocated to the training split.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional hard cap on the total number of generated samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    records = assemble_records(args.data_dir)
    if not records:
        raise SystemExit(
            "No address records could be assembled from the provided data directory."
        )

    all_examples: List[Dict[str, object]] = []
    seen_sequences = set()
    rng.shuffle(records)

    for record in records:
        specs = list(VARIANT_SPECS)
        rng.shuffle(specs)
        for spec in specs:
            target_unique = 12 if spec.include_street else 1
            max_attempts = target_unique * 5  # keep trying to replace dupes instead of wasting draws
            added_for_spec = 0
            attempts = 0
            while attempts < max_attempts and added_for_spec < target_unique:
                attempts += 1
                rendered = render_data_sample(record, spec, rng)
                if not rendered:
                    continue
                tokens, tags, text = rendered
                signature = tuple(tokens)
                if signature in seen_sequences:
                    continue
                seen_sequences.add(signature)
                all_examples.append(
                    {
                        "id": f"{record.ward_code}_{spec.name}_{len(all_examples)}",
                        "text": text,
                        "tokens": tokens,
                        "ner_tags": tags,
                        "source": record.source,
                    }
                )
                added_for_spec += 1
                if args.max_samples and len(all_examples) >= args.max_samples:
                    break
            if args.max_samples and len(all_examples) >= args.max_samples:
                break

    if not all_examples:
        raise SystemExit("Dataset generation produced zero examples.")

    rng.shuffle(all_examples)
    split_idx = int(len(all_examples) * args.train_ratio)
    train_rows = all_examples[:split_idx] or all_examples
    eval_rows = (
        all_examples[split_idx:] or all_examples[: max(1, len(all_examples) // 10)]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    eval_path = args.output_dir / "test.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    print(f"Generated {len(all_examples)} samples.")
    print(f"Train split: {len(train_rows)} -> {train_path}")
    print(f"Test split:  {len(eval_rows)} -> {eval_path}")


if __name__ == "__main__":
    main()
