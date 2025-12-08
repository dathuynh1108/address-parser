"""Lightweight BM25-based search engine for administrative entities."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from rank_bm25 import BM25Okapi


class AddressSearchEngine:
    def __init__(
        self,
        *,
        analyzer: Callable[[Optional[str]], List[str]],
        normalize_id: Callable[[Any], Optional[str]],
    ) -> None:
        self._analyzer = analyzer
        self._normalize_id = normalize_id
        self._token_corpus: List[List[str]] = []
        # Keep per-field token sequences to enforce in-order matching on each name field
        self._field_tokens: List[List[List[str]]] = []
        self._metadata: List[Dict[str, Any]] = []
        self._token_sets: List[Set[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    def add_document(
        self,
        *,
        text_fields: Sequence[Optional[str]],
        metadata: Dict[str, Any],
    ) -> None:
        field_tokens: List[List[str]] = []
        for value in text_fields:
            if not isinstance(value, str):
                continue
            trimmed = value.strip()
            if not trimmed:
                continue
            tokens = self._analyzer(trimmed)
            if tokens:
                field_tokens.append(tokens)

        tokens = [token for seq in field_tokens for token in seq]
        if not tokens:
            fallback = (
                metadata.get("record", {}).get("code")
                if isinstance(metadata.get("record"), dict)
                else None
            )
            normalized_fallback = self._normalize_id(fallback)
            if normalized_fallback:
                tokens = [normalized_fallback]
                field_tokens = [tokens[:]]
        if not tokens:
            tokens = ["_"]  # BM25 requires non-empty documents
            field_tokens = [tokens[:]]

        meta: Dict[str, Any] = dict(metadata)
        record = meta.get("record")
        normalized_code = None
        if isinstance(record, dict):
            normalized_code = self._normalize_id(record.get("code") or record.get("id"))
        if normalized_code:
            meta["code"] = normalized_code
        elif meta.get("code") is not None:
            normalized_code = self._normalize_id(meta["code"])
            if normalized_code:
                meta["code"] = normalized_code
            else:
                meta.pop("code", None)

        self._metadata.append(meta)
        self._token_corpus.append(tokens)
        self._field_tokens.append(field_tokens)
        self._token_sets.append(set(tokens))

    def finalize(self) -> None:
        if not self._token_corpus:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(self._token_corpus)

    def search(
        self,
        query: Optional[str],
        *,
        level: Optional[str] = None,
        allowed_sources: Optional[Sequence[str]] = None,
        province_code: Optional[Any] = None,
        district_code: Optional[Any] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        if not query or limit <= 0:
            return []
        if not self._metadata:
            return []

        allowed_sources_set = (
            {source for source in allowed_sources if source}
            if allowed_sources
            else None
        )
        province_filter = (
            self._normalize_id(province_code) if province_code is not None else None
        )
        district_filter = (
            self._normalize_id(district_code) if district_code is not None else None
        )

        numeric_query = None
        stripped_query = str(query).strip()
        if stripped_query.isdigit():
            numeric_query = self._normalize_id(stripped_query)

        tokenized_query = self._analyzer(query) if query else []
        query_token_set = set(tokenized_query)
        scored_indices: Dict[int, float] = {}

        def _is_subsequence(needles: List[str], haystack: List[str]) -> bool:
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

        def _register(idx: int, score: float) -> None:
            previous = scored_indices.get(idx)
            if previous is None or score > previous:
                scored_indices[idx] = score

        def _matches_all_tokens(idx: int) -> bool:
            if not tokenized_query:
                return True
            if idx >= len(self._token_sets):
                return False

            # Fast fail on missing tokens, then enforce in-order subsequence.
            token_set = self._token_sets[idx]
            if not query_token_set.issubset(token_set):
                return False

            ordered_fields = (
                self._field_tokens[idx]
                if idx < len(self._field_tokens) and self._field_tokens[idx]
                else [self._token_corpus[idx]]
            )
            return any(_is_subsequence(tokenized_query, seq) for seq in ordered_fields)

        for idx, meta in enumerate(self._metadata):
            if not self._passes_filters(
                meta, level, allowed_sources_set, province_filter, district_filter
            ):
                continue
            if numeric_query and meta.get("code") == numeric_query:
                if not _matches_all_tokens(idx):
                    continue
                _register(idx, float("inf"))

        if self._bm25 and tokenized_query:
            scores = self._bm25.get_scores(tokenized_query)
            for idx, score in enumerate(scores):
                if score <= 0:
                    continue
                if not self._passes_filters(
                    self._metadata[idx],
                    level,
                    allowed_sources_set,
                    province_filter,
                    district_filter,
                ):
                    continue
                if not _matches_all_tokens(idx):
                    continue
                _register(idx, float(score))

        if not scored_indices:
            return []

        def _sort_key(item: tuple[int, float]) -> tuple[float, int, str]:
            idx, score = item
            meta = self._metadata[idx]
            source_priority = 0 if meta.get("source") == "new" else 1
            code_value = meta.get("code") or ""
            return (-score, source_priority, code_value)

        ordered_hits = sorted(scored_indices.items(), key=_sort_key)[:limit]
        results: List[Dict[str, Any]] = []
        for idx, score in ordered_hits:
            payload = dict(self._metadata[idx])
            payload["score"] = float(score if score != float("inf") else 1e12)
            results.append(payload)
        return results

    def _passes_filters(
        self,
        meta: Dict[str, Any],
        level: Optional[str],
        allowed_sources: Optional[set[str]],
        province_filter: Optional[str],
        district_filter: Optional[str],
    ) -> bool:
        if level and meta.get("level") != level:
            return False
        if allowed_sources and meta.get("source") not in allowed_sources:
            return False
        if province_filter and meta.get("province_code") != province_filter:
            return False
        if district_filter and meta.get("district_code") != district_filter:
            return False
        return True
