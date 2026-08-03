"""Deterministic, bounded-cache normalization for parser hot paths."""

from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict
from threading import Lock

from .contracts import NormalizationMode

NORMALIZATION_CACHE_SIZE = 32_768
NORMALIZATION_CACHE_MAX_TEXT_LENGTH = 128

_NORMALIZATION_CACHE: OrderedDict[tuple[str, NormalizationMode], str] = OrderedDict()
_NORMALIZATION_CACHE_LOCK = Lock()

_LEADING_PUNCTUATION = re.compile(r"^[\.,]+")
_TRAILING_PUNCTUATION = re.compile(r"[\.,]+$")
_THUA_THIEN_HUE_ABBREVIATION = re.compile(r"\b(t.t.h)\b", re.IGNORECASE)
_HO_CHI_MINH_ABBREVIATION = re.compile(r"\b(h.c.m|h.c.minh)\b", re.IGNORECASE)
_HA_NOI_ABBREVIATION = re.compile(r"\b(hn|h.noi|ha ni)\b", re.IGNORECASE)
_AGGRESSIVE_LOCALITY = re.compile(
    r"\b("
    r"|tiểu\s*khu(\s*\d+\w*)?"
    r"|khu\s*pho(\s*\d+\w*)?"
    r"|khu\s*phố(\s*\d+\w*)?"
    r"|khu\s*vuc(\s*\d+\w*)?"
    r"|khu\s*vực(\s*\d+\w*)?"
    r"|khu(\s*\d+\w*)?"
    r"|kp(\s*\d+\w*)?"
    r"|tổ\s*dân\s*phố(\s*\d+\w*)?"
    r"|tổ(\s*\d+\w*)?"
    r"|thôn(\s*\d+\w*)?"
    r"|xóm(\s*\d+\w*)?"
    r"|cụm(\s*\d+\w*)?"
    r"|phố(\s*\d+\w*)?"
    r"|khóm(\s*\d+\w*)?"
    r"|số\s*nhà(\s*\d+\w*)?"
    r"|số(\s*\d+\w*)?"
    r"|nhà(\s*\d+\w*)?"
    r"|ấp(\s*\d+\w*)?"
    r"|ngách\s*\d+\w*"
    r"|ngõ\s*\d+\w*"
    r"|hẻm\s*\d+\w*"
    r")\b",
    re.IGNORECASE,
)
_COMPACT_CITY_PREFIX = re.compile(r"\btp([a-z0-9]+)")
_NON_ASCII_ADDRESS_CHARACTER = re.compile(r"[^a-z0-9\s]+")
_HO_CHI_MINH_VARIANT = re.compile(
    r"\b(hochi\s*minh|ho\s*chiminh|hcm|hcminh)\b",
    re.IGNORECASE,
)
_HO_CHI_MINH_NAME = re.compile(r"\bho chi minh\b", re.IGNORECASE)
_LEADING_ZEROES = re.compile(r"\b0+(\d+)\b")
_LONG_NUMBER = re.compile(r"\d{3,}")
_COMPACT_WARD_OR_DISTRICT = re.compile(r"\b[pq](\d+)\b")
_WHITESPACE = re.compile(r"\s+")

_REDUNDANT_PHRASES = (
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
)

_HO_CHI_MINH_DISTRICT_ABBREVIATIONS = tuple(
    (re.compile(rf"\b{abbreviation}\b", re.IGNORECASE), full_name)
    for abbreviation, full_name in (
        ("bc", "binh chanh"),
        ("tb", "tan binh"),
        ("bt", "binh thanh"),
        ("gv", "go vap"),
        ("pn", "phu nhuan"),
        ("cc", "cu chi"),
        ("hm", "hoc mon"),
        ("nb", "nha be"),
    )
)


def normalize_address_text(name: str, mode: NormalizationMode) -> str:
    """Normalize one immutable text/mode pair and reuse repeated results."""
    if len(name) > NORMALIZATION_CACHE_MAX_TEXT_LENGTH:
        return _normalize_address_text_uncached(name, mode)

    cache_key = (name, mode)
    try:
        return _NORMALIZATION_CACHE[cache_key]
    except KeyError:
        pass

    normalized = _normalize_address_text_uncached(name, mode)
    with _NORMALIZATION_CACHE_LOCK:
        cached = _NORMALIZATION_CACHE.get(cache_key)
        if cached is not None:
            return cached
        if len(_NORMALIZATION_CACHE) >= NORMALIZATION_CACHE_SIZE:
            _NORMALIZATION_CACHE.popitem(last=False)
        _NORMALIZATION_CACHE[cache_key] = normalized
    return normalized


def _normalize_address_text_uncached(name: str, mode: NormalizationMode) -> str:
    if not name:
        return ""

    value = name.lower()
    value = _LEADING_PUNCTUATION.sub("", value)
    value = _TRAILING_PUNCTUATION.sub("", value)
    value = value.replace("/", "")

    if mode in {"search", "aggressive"}:
        value = _THUA_THIEN_HUE_ABBREVIATION.sub(" thua thien hue ", value)
        value = _HO_CHI_MINH_ABBREVIATION.sub(" ho chi minh ", value)
        value = _HA_NOI_ABBREVIATION.sub(" ha noi ", value)

        for phrase in _REDUNDANT_PHRASES:
            value = value.replace(phrase, " ")

        if mode == "aggressive":
            value = _AGGRESSIVE_LOCALITY.sub("", value)

        value = _COMPACT_CITY_PREFIX.sub(r"\1", value)

    value = value.replace("đ", "d")
    value = unicodedata.normalize("NFD", value)
    value = "".join(character for character in value if unicodedata.category(character) != "Mn")
    value = _NON_ASCII_ADDRESS_CHARACTER.sub(" ", value)

    if mode in {"search", "aggressive"}:
        value = _HO_CHI_MINH_VARIANT.sub("ho chi minh", value)
        if _HO_CHI_MINH_NAME.search(value):
            for pattern, full_name in _HO_CHI_MINH_DISTRICT_ABBREVIATIONS:
                value = pattern.sub(full_name, value)

        if mode == "aggressive":
            value = _LEADING_ZEROES.sub(r"\1", value)
            value = _LONG_NUMBER.sub("", value)
            value = _COMPACT_WARD_OR_DISTRICT.sub(r"\1", value)

    return _WHITESPACE.sub(" ", value).strip()
