"""Packed n-gram posting index shared by Python and native ranking kernels."""

from __future__ import annotations

from array import array
from collections import Counter
from collections.abc import Iterable, Mapping, Set

from .contracts import NgramHit
from .kernels import native_acceleration_available, rank_posting_hits

_UINT32_LIMIT = 1 << 32


class PackedNgramIndex:
    """Store posting lists in contiguous uint32 buffers for one-call ranking."""

    def __init__(self, postings_by_ngram: Mapping[str, Set[int]], node_count: int) -> None:
        if node_count < 0 or node_count >= _UINT32_LIMIT:
            raise ValueError("node_count must fit in uint32")

        self._node_count = node_count
        self._fallback_postings_by_ngram = postings_by_ngram
        self._posting_id_by_ngram: dict[str, int] = {}
        self._posting_offsets = array("I")
        self._posting_node_ids = array("I")
        if not native_acceleration_available():
            return
        self._posting_offsets.append(0)

        for posting_id, (ngram, node_ids) in enumerate(postings_by_ngram.items()):
            if posting_id >= _UINT32_LIMIT:
                raise ValueError("posting count must fit in uint32")
            self._posting_id_by_ngram[ngram] = posting_id
            self._posting_node_ids.extend(node_ids)
            if len(self._posting_node_ids) >= _UINT32_LIMIT:
                raise ValueError("flattened posting values must fit in uint32 offsets")
            self._posting_offsets.append(len(self._posting_node_ids))

        if self._posting_offsets.itemsize != 4 or self._posting_node_ids.itemsize != 4:
            raise RuntimeError("native n-gram kernels require 32-bit unsigned-int arrays")

    def top_hits(self, input_ngrams: Iterable[str], top_k: int) -> list[NgramHit]:
        unique_ngrams = sorted(set(input_ngrams))
        if not native_acceleration_available():
            counter: Counter[int] = Counter()
            for ngram in unique_ngrams:
                node_ids = self._fallback_postings_by_ngram.get(ngram)
                if node_ids:
                    counter.update(node_ids)
            return counter.most_common(top_k)

        query_posting_ids = array("I")
        for ngram in unique_ngrams:
            posting_id = self._posting_id_by_ngram.get(ngram)
            if posting_id is not None:
                query_posting_ids.append(posting_id)

        return rank_posting_hits(
            self._posting_offsets,
            self._posting_node_ids,
            query_posting_ids,
            self._node_count,
            top_k,
        )
