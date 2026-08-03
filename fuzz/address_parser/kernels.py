"""Native-first kernel dispatch with an explicit typed Python fallback."""

from __future__ import annotations

from array import array
from typing import Protocol

from ._python_kernels import (
    rank_posting_hits as _python_rank_posting_hits,
)
from ._python_kernels import (
    validate_rank_posting_inputs,
)
from .contracts import NgramHit


class RankPostingHits(Protocol):
    """Callable contract shared by the native and Python rankers."""

    def __call__(
        self,
        posting_offsets: array[int],
        posting_node_ids: array[int],
        query_posting_ids: array[int],
        node_count: int,
        top_k: int,
    ) -> list[NgramHit]: ...


_native_rank_posting_hits: RankPostingHits | None
try:
    from ._native_kernels import rank_posting_hits as _compiled_rank_posting_hits
except ModuleNotFoundError as exc:
    if exc.name != "address_parser._native_kernels":
        raise
    _native_rank_posting_hits = None
else:
    _native_rank_posting_hits = _compiled_rank_posting_hits


def native_acceleration_available() -> bool:
    """Return the active backend state without hiding runtime import failures."""
    return _native_rank_posting_hits is not None


def rank_posting_hits(
    posting_offsets: array[int],
    posting_node_ids: array[int],
    query_posting_ids: array[int],
    node_count: int,
    top_k: int,
) -> list[NgramHit]:
    """Run the compiled overlap ranker when installed, otherwise the typed reference."""
    if _native_rank_posting_hits is not None:
        validate_rank_posting_inputs(
            posting_offsets,
            posting_node_ids,
            query_posting_ids,
            node_count,
        )
        return _native_rank_posting_hits(
            posting_offsets,
            posting_node_ids,
            query_posting_ids,
            node_count,
            top_k,
        )
    return _python_rank_posting_hits(
        posting_offsets,
        posting_node_ids,
        query_posting_ids,
        node_count,
        top_k,
    )


def require_native_acceleration() -> None:
    """Fail visibly when a deployment requires, but did not build, native kernels."""
    if _native_rank_posting_hits is None:
        raise RuntimeError(
            "address_parser native kernels are unavailable; rebuild with "
            "VN_ADDRESS_PARSER_NATIVE=required"
        )
