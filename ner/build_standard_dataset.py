#!/usr/bin/env python
"""Synthetic NER dataset generation for Vietnamese administrative addresses."""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
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
LABELS = ["O", "B-PROVINCE", "I-PROVINCE", "B-DISTRICT", "I-DISTRICT", "B-WARD", "I-WARD"]
STREET_NAMES = [
    "Nguyen Trai",
    "Le Loi",
    "Tran Hung Dao",
    "Ly Thuong Kiet",
    "Pham Van Dong",
    "Nguyen Van Cu",
    "Cach Mang",
    "Pasteur",
    "Vo Thi Sau",
    "Hoang Dieu",
]
STREET_DESCRIPTORS = ["duong", "pho", "ngo", "hem", "ql", "tl", "khu pho"]


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


def tag_phrase(token_pairs: List[Tuple[str, str]], phrase: Optional[str], label: str) -> bool:
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
    province: Optional[str] = None,
    district: Optional[str] = None,
    ward: Optional[str] = None,
) -> LabelingResult:
    cleaned_address = clean_text(address, remove_slash=False)
    token_pairs: List[Tuple[str, str]] = [(tok, "O") for tok in tokenize(cleaned_address)]
    matches = {
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

    def pick_name(self, *, rng: random.Random, prefer_full: bool, prefer_short: bool) -> NameVariant:
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
            candidates = TYPE_ABBREVIATIONS.get(self.type_hint) or TYPE_ABBREVIATIONS.get(self.label.lower())
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


VARIANT_SPECS: Tuple[VariantSpec, ...] = (
    VariantSpec(name="standard"),
    VariantSpec(name="lowercase", lowercase=True),
    VariantSpec(name="accentless", lowercase=True, strip_accents=True, use_commas=False),
    VariantSpec(name="abbrev_commas", abbreviate_types=True),
    VariantSpec(
        name="compact_short",
        lowercase=True,
        strip_accents=True,
        use_commas=False,
        abbreviate_types=True,
        drop_type_tokens=True,
        prefer_short_name=True,
    ),
    VariantSpec(
        name="old_style",
        connectors={
            "street_ward": [","],
            "ward_district": ["thuoc"],
            "district_province": ["thuoc"],
            "ward_province": ["thuoc"],
        },
        prefer_full_name=True,
    ),
    VariantSpec(
        name="no_street",
        include_street=False,
        use_commas=False,
        abbreviate_types=True,
        lowercase=True,
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


def clean_text(value: Optional[str], *, remove_slash: bool = True) -> str:
    if not value:
        return ""
    value = value.replace("\u00a0", " ")
    if remove_slash:
        value = value.replace("/", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


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


def detect_type_hint(level: str, full_name: str) -> str:
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
        ("name_en", False),
        ("full_name_en", True),
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
) -> Dict[str, Dict[str, object]]:
    data: Dict[str, Dict[str, object]] = {}
    for source in (records, extra_records):
        for record in source:
            code = record.get("code")
            if not code:
                continue
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
            entry["type_hint"] = entry["type_hint"] or detect_type_hint(level, record.get("full_name", ""))
            if not entry["type_word"]:
                entry["type_word"] = extract_type_word(record.get("full_name", ""), DEFAULT_TYPE_WORDS.get(level.upper(), ""))
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
    old_records = _assemble_old_structure(data_dir)
    new_records = _assemble_new_structure(data_dir)
    return old_records + new_records


def _assemble_old_structure(data_dir: Path) -> List[AddressRecord]:
    provinces_raw = build_components(
        level="province",
        records=_load_json(data_dir / "old_provinces.json"),
    )
    districts_raw = build_components(
        level="district",
        records=_load_json(data_dir / "old_districts.json"),
    )
    wards_raw = build_components(
        level="ward",
        records=_load_json(data_dir / "old_wards.json"),
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
    provinces_raw = build_components(
        level="province",
        records=_load_json(data_dir / "provinces.json"),
    )
    wards_raw = build_components(
        level="ward",
        records=_load_json(data_dir / "wards.json"),
    )

    records: List[AddressRecord] = []
    for ward_code, ward_entry in wards_raw.items():
        province_code = ward_entry.get("attributes", {}).get("province_code") or ward_entry.get("attributes", {}).get(
            "parent_code"
        )
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
    street = rng.choice(STREET_NAMES)
    descriptor = rng.choice(STREET_DESCRIPTORS)
    templates = [
        f"{number} {descriptor} {street}",
        f"ngo {alley} {descriptor} {street}",
        f"hem {alley}/{number} {street}",
        f"{descriptor} {street} so {number}",
    ]
    text = rng.choice(templates)
    return tokenize(text)


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
    if not spec.drop_type_tokens and not variant.includes_type:
        type_token = component.resolve_type_token(rng=rng, abbreviate=spec.abbreviate_types)
        if type_token:
            tokens.extend(tokenize(type_token))
    tokens.extend(tokenize(variant.text))
    return tokens


def render_example(
    record: AddressRecord,
    spec: VariantSpec,
    rng: random.Random,
) -> Optional[Tuple[List[str], List[str], str]]:
    tokens: List[str] = []
    component_spans: Dict[str, Tuple[int, int]] = {}

    connector_key_map = {
        ("STREET", "WARD"): "street_ward",
        ("WARD", "DISTRICT"): "ward_district",
        ("DISTRICT", "PROVINCE"): "district_province",
        ("WARD", "PROVINCE"): "ward_province",
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

    if spec.include_street:
        street_tokens = build_street_tokens(rng)
        tokens.extend(street_tokens)
        previous_label: Optional[str] = "STREET"
    else:
        previous_label = None

    for label, component in record.components():
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
        default=Path("inexus/data"),
        help="Directory that contains the administrative data json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ner/datasets"),
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
        raise SystemExit("No address records could be assembled from the provided data directory.")

    all_examples: List[Dict[str, object]] = []
    seen_sequences = set()
    rng.shuffle(records)

    for record in records:
        specs = list(VARIANT_SPECS)
        rng.shuffle(specs)
        for spec in specs:
            rendered = render_example(record, spec, rng)
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
            if args.max_samples and len(all_examples) >= args.max_samples:
                break
        if args.max_samples and len(all_examples) >= args.max_samples:
            break

    if not all_examples:
        raise SystemExit("Dataset generation produced zero examples.")

    rng.shuffle(all_examples)
    split_idx = int(len(all_examples) * args.train_ratio)
    train_rows = all_examples[:split_idx] or all_examples
    eval_rows = all_examples[split_idx:] or all_examples[: max(1, len(all_examples) // 10)]

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
