from array import array

from .contracts import NgramHit

def rank_posting_hits(
    posting_offsets: array[int],
    posting_node_ids: array[int],
    query_posting_ids: array[int],
    node_count: int,
    top_k: int,
) -> list[NgramHit]: ...
