"""Typed reference implementations for optional native kernels."""

from __future__ import annotations

from array import array

from .contracts import NgramHit

_UINT32_LIMIT = 1 << 32
_UINT32_MAX = _UINT32_LIMIT - 1


def validate_rank_posting_inputs(
    posting_offsets: array[int],
    posting_node_ids: array[int],
    query_posting_ids: array[int],
    node_count: int,
) -> None:
    """Validate the portable uint32 buffer contract shared by both backends."""
    for name, buffer in (
        ("posting_offsets", posting_offsets),
        ("posting_node_ids", posting_node_ids),
        ("query_posting_ids", query_posting_ids),
    ):
        if buffer.typecode != "I" or buffer.itemsize != 4:
            raise TypeError(f"{name} must be array('I') with 32-bit unsigned integers")
    if node_count < 0 or node_count >= _UINT32_LIMIT:
        raise ValueError("node_count must fit in uint32")


def rank_posting_hits(
    posting_offsets: array[int],
    posting_node_ids: array[int],
    query_posting_ids: array[int],
    node_count: int,
    top_k: int,
) -> list[NgramHit]:
    """Count posting overlaps and retain Counter-compatible stable tie ordering."""
    validate_rank_posting_inputs(
        posting_offsets,
        posting_node_ids,
        query_posting_ids,
        node_count,
    )
    if top_k <= 0 or node_count == 0 or not query_posting_ids:
        return []
    if not posting_offsets:
        raise ValueError("posting_offsets must contain at least one offset")

    counts: dict[int, int] = {}
    first_seen: list[int] = []
    posting_count = len(posting_offsets) - 1

    for posting_id in query_posting_ids:
        if posting_id >= posting_count:
            raise ValueError("query posting id is outside posting_offsets")
        start = posting_offsets[posting_id]
        end = posting_offsets[posting_id + 1]
        if end < start or end > len(posting_node_ids):
            raise ValueError("posting offsets are not a valid contiguous range")
        for position in range(start, end):
            node_id = posting_node_ids[position]
            if node_id >= node_count:
                raise ValueError("posting node id is outside node_count")
            current_count = counts.get(node_id)
            if current_count is None:
                counts[node_id] = 1
                first_seen.append(node_id)
            else:
                if current_count == _UINT32_MAX:
                    raise OverflowError("posting overlap count exceeds uint32")
                counts[node_id] = current_count + 1

    first_seen.sort(key=counts.__getitem__, reverse=True)
    return [(node_id, counts[node_id]) for node_id in first_seen[:top_k]]
