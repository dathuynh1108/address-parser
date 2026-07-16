"""Lightweight BM25-based search engine for administrative entities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

from rank_bm25 import BM25Okapi

from .contracts import (
    ADMINISTRATIVE_RECORD_KEYS,
    ADMINISTRATIVE_RECORD_OPTIONAL_INTEGER_KEYS,
    ADMINISTRATIVE_RECORD_OPTIONAL_STRING_KEYS,
    ADMINISTRATIVE_RECORD_OPTIONAL_STRING_LIST_KEYS,
    ADMINISTRATIVE_RECORD_REQUIRED_KEYS,
    SEARCH_DOCUMENT_KEYS,
    SEARCH_DOCUMENT_REQUIRED_KEYS,
    SEARCH_ENGINE_STATE_KEYS,
    AddressCode,
    AddressCodeInput,
    AdministrativeLevel,
    AdministrativeRecord,
    RegistrySource,
    SearchDocument,
    SearchDocumentInput,
    SearchEngineState,
    SearchResult,
)

Analyzer = Callable[[str | None], list[str]]
IdNormalizer = Callable[[AddressCodeInput | None], AddressCode | None]


def _invalid_state(path: str, reason: str) -> ValueError:
    return ValueError(f"Invalid search engine state at {path}: {reason}")


def _require_mapping(value: object, path: str) -> dict[object, object]:
    if not isinstance(value, dict):
        raise _invalid_state(path, "expected a dictionary")
    return cast(dict[object, object], value)


def _require_list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise _invalid_state(path, "expected a list")
    return cast(list[object], value)


def _validate_string_list(value: object, path: str) -> list[str]:
    raw_values = _require_list(value, path)
    for index, item in enumerate(raw_values):
        if not isinstance(item, str):
            raise _invalid_state(f"{path}[{index}]", "expected a string")
    return cast(list[str], raw_values)


def _validate_optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid_state(path, "expected a string or null")
    return value


def _validate_administrative_record(
    value: object,
    path: str,
) -> AdministrativeRecord:
    record = _require_mapping(value, path)
    keys = set(record)
    missing_keys = ADMINISTRATIVE_RECORD_REQUIRED_KEYS - keys
    if missing_keys:
        missing = ", ".join(sorted(cast(set[str], missing_keys)))
        raise _invalid_state(path, f"missing required keys: {missing}")
    unknown_keys = keys - ADMINISTRATIVE_RECORD_KEYS
    if unknown_keys:
        unknown = ", ".join(sorted(str(key) for key in unknown_keys))
        raise _invalid_state(path, f"unknown record keys: {unknown}")

    code = record["code"]
    if code is not None and (not isinstance(code, str) or not code):
        raise _invalid_state(
            f"{path}.code",
            "expected a non-empty string or null",
        )

    record_id = record["id"]
    _validate_optional_string(record_id, f"{path}.id")
    name = record["name"]
    _validate_optional_string(name, f"{path}.name")

    for key in ADMINISTRATIVE_RECORD_OPTIONAL_STRING_KEYS:
        if key in record:
            _validate_optional_string(record[key], f"{path}.{key}")

    for key in ADMINISTRATIVE_RECORD_OPTIONAL_INTEGER_KEYS:
        if key not in record:
            continue
        item = record[key]
        if item is not None and (not isinstance(item, int) or isinstance(item, bool)):
            raise _invalid_state(f"{path}.{key}", "expected an integer or null")

    for key in ADMINISTRATIVE_RECORD_OPTIONAL_STRING_LIST_KEYS:
        if key in record:
            _validate_string_list(record[key], f"{path}.{key}")

    if "is_new_format" in record and not isinstance(record["is_new_format"], bool):
        raise _invalid_state(f"{path}.is_new_format", "expected a boolean")

    return cast(AdministrativeRecord, record)


def _validate_search_document(value: object, path: str) -> SearchDocument:
    raw_document = _require_mapping(value, path)
    keys = set(raw_document)
    missing_keys = SEARCH_DOCUMENT_REQUIRED_KEYS - keys
    if missing_keys:
        missing = ", ".join(sorted(cast(set[str], missing_keys)))
        raise _invalid_state(path, f"missing required keys: {missing}")
    unknown_keys = keys - SEARCH_DOCUMENT_KEYS
    if unknown_keys:
        unknown = ", ".join(sorted(str(key) for key in unknown_keys))
        raise _invalid_state(path, f"unknown metadata keys: {unknown}")

    raw_level = raw_document["level"]
    if not isinstance(raw_level, str) or raw_level not in {
        "province",
        "district",
        "ward",
    }:
        raise _invalid_state(f"{path}.level", "unknown administrative level")

    raw_source = raw_document["source"]
    if not isinstance(raw_source, str) or raw_source not in {"old", "new"}:
        raise _invalid_state(f"{path}.source", "unknown registry source")

    _validate_administrative_record(
        raw_document["record"],
        f"{path}.record",
    )
    _validate_optional_string(
        raw_document["province_code"],
        f"{path}.province_code",
    )
    _validate_optional_string(
        raw_document["district_code"],
        f"{path}.district_code",
    )
    _validate_optional_string(
        raw_document["unit_token"],
        f"{path}.unit_token",
    )
    if "code" in raw_document:
        code = raw_document["code"]
        if not isinstance(code, str) or not code:
            raise _invalid_state(f"{path}.code", "expected a non-empty string")
    return cast(SearchDocument, raw_document)


def _validate_token_corpus(value: object) -> list[list[str]]:
    raw_corpus = _require_list(value, "token_corpus")
    for index, document in enumerate(raw_corpus):
        tokens = _validate_string_list(document, f"token_corpus[{index}]")
        if not tokens:
            raise _invalid_state(
                f"token_corpus[{index}]",
                "BM25 documents must not be empty",
            )
    return cast(list[list[str]], raw_corpus)


def _validate_field_tokens(value: object) -> list[list[list[str]]]:
    raw_documents = _require_list(value, "field_tokens")
    for document_index, raw_fields in enumerate(raw_documents):
        fields = _require_list(raw_fields, f"field_tokens[{document_index}]")
        for field_index, raw_tokens in enumerate(fields):
            tokens = _validate_string_list(
                raw_tokens,
                f"field_tokens[{document_index}][{field_index}]",
            )
            if not tokens:
                raise _invalid_state(
                    f"field_tokens[{document_index}][{field_index}]",
                    "field token sequences must not be empty",
                )
        if not fields:
            raise _invalid_state(
                f"field_tokens[{document_index}]",
                "documents must contain at least one field",
            )
    return cast(list[list[list[str]]], raw_documents)


def _validate_metadata(value: object) -> list[SearchDocument]:
    raw_metadata = _require_list(value, "metadata")
    for index, item in enumerate(raw_metadata):
        _validate_search_document(item, f"metadata[{index}]")
    return cast(list[SearchDocument], raw_metadata)


def _validate_token_sets(value: object) -> list[list[str]]:
    raw_token_sets = _require_list(value, "token_sets")
    for index, item in enumerate(raw_token_sets):
        _validate_string_list(item, f"token_sets[{index}]")
    return cast(list[list[str]], raw_token_sets)


def _validate_state(state: object) -> SearchEngineState:
    raw_state = _require_mapping(state, "root")
    keys = set(raw_state)
    if keys != SEARCH_ENGINE_STATE_KEYS:
        missing_keys = SEARCH_ENGINE_STATE_KEYS - keys
        unknown_keys = keys - SEARCH_ENGINE_STATE_KEYS
        details: list[str] = []
        if missing_keys:
            details.append("missing keys: " + ", ".join(sorted(cast(set[str], missing_keys))))
        if unknown_keys:
            details.append("unknown keys: " + ", ".join(sorted(str(key) for key in unknown_keys)))
        raise _invalid_state("root", "; ".join(details))

    token_corpus = _validate_token_corpus(raw_state["token_corpus"])
    field_tokens = _validate_field_tokens(raw_state["field_tokens"])
    metadata = _validate_metadata(raw_state["metadata"])
    token_sets = _validate_token_sets(raw_state["token_sets"])

    lengths = {
        len(token_corpus),
        len(field_tokens),
        len(metadata),
        len(token_sets),
    }
    if len(lengths) != 1:
        raise _invalid_state(
            "root",
            "token_corpus, field_tokens, metadata, and token_sets lengths differ",
        )

    for index, (tokens, fields, token_set) in enumerate(
        zip(token_corpus, field_tokens, token_sets, strict=True)
    ):
        flattened_fields = [token for field in fields for token in field]
        if flattened_fields != tokens:
            raise _invalid_state(
                f"field_tokens[{index}]",
                "flattened fields do not match token_corpus",
            )
        if set(token_set) != set(tokens):
            raise _invalid_state(
                f"token_sets[{index}]",
                "tokens do not match token_corpus",
            )

    return cast(SearchEngineState, raw_state)


def validate_search_request(
    query: str | None,
    *,
    level: AdministrativeLevel | None,
    allowed_sources: Sequence[RegistrySource] | None,
    limit: int,
) -> set[RegistrySource] | None:
    """Validate public search inputs before any empty-result shortcut."""
    if query is not None and not isinstance(query, str):
        raise TypeError("query must be a string or None")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if level is not None:
        if not isinstance(level, str):
            raise TypeError("level must be a string or None")
        if level not in {"province", "district", "ward"}:
            raise ValueError("level must be 'province', 'district', 'ward', or None")
    if allowed_sources is None:
        return None

    validated_sources: set[RegistrySource] = set()
    for source in allowed_sources:
        if not isinstance(source, str):
            raise TypeError("allowed_sources items must be strings")
        if source not in {"old", "new"}:
            raise ValueError("allowed_sources may contain only 'old' or 'new'")
        validated_sources.add(source)
    return validated_sources


def validate_search_metadata(value: object) -> list[SearchDocument]:
    """Validate metadata rows at the persistent-cache boundary."""
    return _validate_metadata(value)


class AddressSearchEngine:
    """BM25 index over normalized administrative records."""

    def __init__(
        self,
        *,
        analyzer: Analyzer,
        normalize_id: IdNormalizer,
    ) -> None:
        """Initialize an empty index with parser-owned normalization callbacks."""
        self._analyzer = analyzer
        self._normalize_id = normalize_id
        self._token_corpus: list[list[str]] = []
        self._field_tokens: list[list[list[str]]] = []
        self._metadata: list[SearchDocument] = []
        self._token_sets: list[set[str]] = []
        self._bm25: BM25Okapi | None = None

    def add_document(
        self,
        *,
        text_fields: Sequence[str | None],
        metadata: SearchDocumentInput,
    ) -> None:
        """Normalize and add one administrative record to the pending index."""
        validated_metadata = _validate_search_document(metadata, "metadata")
        field_tokens: list[list[str]] = []
        for index, value in enumerate(text_fields):
            if value is None:
                continue
            if not isinstance(value, str):
                raise TypeError(f"text_fields[{index}] must be a string or None")
            trimmed = value.strip()
            if not trimmed:
                continue
            analyzed_tokens = self._analyzer(trimmed)
            if analyzed_tokens:
                field_tokens.append(analyzed_tokens)

        tokens = [token for sequence in field_tokens for token in sequence]
        if not tokens:
            record = validated_metadata["record"]
            fallback = record["code"] or record["id"]
            normalized_fallback = self._normalize_id(fallback)
            if normalized_fallback:
                tokens = [normalized_fallback]
                field_tokens = [tokens.copy()]
        if not tokens:
            tokens = ["_"]
            field_tokens = [tokens.copy()]

        meta: SearchDocument = {
            "level": validated_metadata["level"],
            "source": validated_metadata["source"],
            "record": validated_metadata["record"],
            "province_code": validated_metadata["province_code"],
            "district_code": validated_metadata["district_code"],
            "unit_token": validated_metadata["unit_token"],
        }
        record = meta["record"]
        normalized_code = self._normalize_id(record["code"] or record["id"])
        if normalized_code:
            meta["code"] = normalized_code
        else:
            provided_code = validated_metadata.get("code")
            if provided_code is not None:
                normalized_code = self._normalize_id(provided_code)
                if normalized_code:
                    meta["code"] = normalized_code

        self._metadata.append(meta)
        self._token_corpus.append(tokens)
        self._field_tokens.append(field_tokens)
        self._token_sets.append(set(tokens))

    def finalize(self) -> None:
        """Build the BM25 index from all pending documents."""
        if not self._token_corpus:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(self._token_corpus)

    @property
    def document_count(self) -> int:
        """Return the number of searchable administrative records."""
        return len(self._metadata)

    def search(
        self,
        query: str | None,
        *,
        level: AdministrativeLevel | None = None,
        allowed_sources: Sequence[RegistrySource] | None = None,
        province_code: AddressCodeInput | None = None,
        district_code: AddressCodeInput | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Return ranked administrative records matching the supplied filters."""
        allowed_sources_set = validate_search_request(
            query,
            level=level,
            allowed_sources=allowed_sources,
            limit=limit,
        )
        if (
            not query
            or limit <= 0
            or not self._metadata
            or (allowed_sources_set is not None and not allowed_sources_set)
        ):
            return []

        province_filter = self._normalize_id(province_code)
        district_filter = self._normalize_id(district_code)

        numeric_query: AddressCode | None = None
        stripped_query = query.strip()
        if stripped_query.isdigit():
            numeric_query = self._normalize_id(stripped_query)

        tokenized_query = self._analyzer(query)
        query_token_set = set(tokenized_query)
        scored_indices: dict[int, float] = {}

        def _is_subsequence(needles: list[str], haystack: list[str]) -> bool:
            if not needles:
                return True
            document_position = 0
            for token in needles:
                while document_position < len(haystack) and haystack[document_position] != token:
                    document_position += 1
                if document_position == len(haystack):
                    return False
                document_position += 1
            return True

        def _register(index: int, score: float) -> None:
            previous = scored_indices.get(index)
            if previous is None or score > previous:
                scored_indices[index] = score

        def _matches_all_tokens(index: int) -> bool:
            if not tokenized_query:
                return True
            if index >= len(self._token_sets):
                return False
            if not query_token_set.issubset(self._token_sets[index]):
                return False

            ordered_fields = (
                self._field_tokens[index]
                if index < len(self._field_tokens) and self._field_tokens[index]
                else [self._token_corpus[index]]
            )
            return any(_is_subsequence(tokenized_query, sequence) for sequence in ordered_fields)

        for index, meta in enumerate(self._metadata):
            if not self._passes_filters(
                meta,
                level,
                allowed_sources_set,
                province_filter,
                district_filter,
            ):
                continue
            if numeric_query and meta.get("code") == numeric_query:
                if _matches_all_tokens(index):
                    _register(index, float("inf"))

        if self._bm25 is not None and tokenized_query:
            scores = self._bm25.get_scores(tokenized_query)
            for index, raw_score in enumerate(scores):
                if raw_score <= 0:
                    continue
                if not self._passes_filters(
                    self._metadata[index],
                    level,
                    allowed_sources_set,
                    province_filter,
                    district_filter,
                ):
                    continue
                if _matches_all_tokens(index):
                    _register(index, float(raw_score))

        if not scored_indices:
            return []

        def _sort_key(item: tuple[int, float]) -> tuple[float, int, str]:
            index, score = item
            meta = self._metadata[index]
            source_priority = 0 if meta["source"] == "new" else 1
            return (-score, source_priority, meta.get("code", ""))

        ordered_hits = sorted(scored_indices.items(), key=_sort_key)[:limit]
        results: list[SearchResult] = []
        for index, score in ordered_hits:
            meta = self._metadata[index]
            payload: SearchResult = {
                "level": meta["level"],
                "source": meta["source"],
                "record": meta["record"],
                "province_code": meta["province_code"],
                "district_code": meta["district_code"],
                "unit_token": meta["unit_token"],
                "score": float(score if score != float("inf") else 1e12),
            }
            code = meta.get("code")
            if code is not None:
                payload["code"] = code
            results.append(payload)
        return results

    def _passes_filters(
        self,
        meta: SearchDocument,
        level: AdministrativeLevel | None,
        allowed_sources: set[RegistrySource] | None,
        province_filter: AddressCode | None,
        district_filter: AddressCode | None,
    ) -> bool:
        if level and meta["level"] != level:
            return False
        if allowed_sources and meta["source"] not in allowed_sources:
            return False
        if province_filter and meta["province_code"] != province_filter:
            return False
        if district_filter and meta["district_code"] != district_filter:
            return False
        return True

    def get_state(self) -> SearchEngineState:
        """Return the serializable, callable-free state used by parser caches."""
        return {
            "token_corpus": self._token_corpus,
            "field_tokens": self._field_tokens,
            "metadata": self._metadata,
            "token_sets": [list(tokens) for tokens in self._token_sets],
        }

    def restore_state(self, state: object) -> None:
        """Validate and atomically restore a serialized search-engine state."""
        self._apply_state(_validate_state(state))

    def restore_cached_state(self, state: SearchEngineState) -> None:
        """Restore parser-owned state after its trusted cache header was checked."""
        lengths = {
            len(state["token_corpus"]),
            len(state["field_tokens"]),
            len(state["metadata"]),
            len(state["token_sets"]),
        }
        if len(lengths) != 1:
            raise _invalid_state("root", "search state array lengths differ")
        self._apply_state(state)

    def _apply_state(self, state: SearchEngineState) -> None:
        """Apply a typed state after its owning boundary has validated it."""
        token_sets = [set(tokens) for tokens in state["token_sets"]]
        self._token_corpus = state["token_corpus"]
        self._field_tokens = state["field_tokens"]
        self._metadata = state["metadata"]
        self._token_sets = token_sets
        self.finalize()
