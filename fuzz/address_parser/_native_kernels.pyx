# cython: language_level=3, annotation_typing=True

"""C-level kernels operating only on validated contiguous primitive buffers."""

cimport cython
from libc.stddef cimport size_t
from libc.stdint cimport SIZE_MAX, UINT32_MAX, uint32_t, uint64_t
from libc.stdlib cimport calloc, free, malloc, qsort


cdef struct RankedHit:
    uint64_t key
    uint32_t node_id


cdef int _compare_ranked_hits(const void* left, const void* right) noexcept nogil:
    cdef uint64_t left_key = (<RankedHit*>left).key
    cdef uint64_t right_key = (<RankedHit*>right).key
    if left_key < right_key:
        return 1
    if left_key > right_key:
        return -1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def rank_posting_hits(
    const uint32_t[::1] posting_offsets,
    const uint32_t[::1] posting_node_ids,
    const uint32_t[::1] query_posting_ids,
    Py_ssize_t node_count,
    Py_ssize_t top_k,
) -> list[tuple[int, int]]:
    """Rank overlap counts while preserving Counter's first-seen tie order."""
    cdef uint32_t* counts = NULL
    cdef RankedHit* hits = NULL
    cdef Py_ssize_t posting_index
    cdef Py_ssize_t position
    cdef Py_ssize_t hit_index
    cdef Py_ssize_t touched_count = 0
    cdef Py_ssize_t result_count
    cdef uint32_t posting_id
    cdef uint32_t node_id
    cdef uint32_t start
    cdef uint32_t end
    cdef int invalid_state = 0
    cdef list[tuple[int, int]] result

    if node_count < 0 or node_count >= (1 << 32):
        raise ValueError("node_count must fit in uint32")
    if top_k <= 0 or node_count == 0 or query_posting_ids.shape[0] == 0:
        return []
    if posting_offsets.shape[0] == 0:
        raise ValueError("posting_offsets must contain at least one offset")
    if <size_t>node_count > SIZE_MAX // sizeof(RankedHit):
        raise OverflowError("node_count is too large for the native ranking workspace")

    counts = <uint32_t*>calloc(<size_t>node_count, sizeof(uint32_t))
    hits = <RankedHit*>malloc(<size_t>node_count * sizeof(RankedHit))
    if counts == NULL or hits == NULL:
        free(counts)
        free(hits)
        raise MemoryError("unable to allocate native n-gram ranking workspace")

    try:
        with nogil:
            for posting_index in range(query_posting_ids.shape[0]):
                posting_id = query_posting_ids[posting_index]
                if <uint64_t>posting_id >= <uint64_t>(posting_offsets.shape[0] - 1):
                    invalid_state = 1
                    break
                start = posting_offsets[posting_id]
                end = posting_offsets[posting_id + 1]
                if end < start or <uint64_t>end > <uint64_t>posting_node_ids.shape[0]:
                    invalid_state = 2
                    break

                for position in range(start, end):
                    node_id = posting_node_ids[position]
                    if <uint64_t>node_id >= <uint64_t>node_count:
                        invalid_state = 3
                        break
                    if counts[node_id] == 0:
                        hits[touched_count].node_id = node_id
                        touched_count += 1
                    elif counts[node_id] == UINT32_MAX:
                        invalid_state = 4
                        break
                    counts[node_id] += 1
                if invalid_state != 0:
                    break

            if invalid_state == 0:
                for hit_index in range(touched_count):
                    node_id = hits[hit_index].node_id
                    hits[hit_index].key = (
                        (<uint64_t>counts[node_id] << 32)
                        | <uint64_t>(<uint32_t>0xFFFFFFFF - <uint32_t>hit_index)
                    )
                qsort(
                    hits,
                    <size_t>touched_count,
                    sizeof(RankedHit),
                    _compare_ranked_hits,
                )

        if invalid_state == 1:
            raise ValueError("query posting id is outside posting_offsets")
        if invalid_state == 2:
            raise ValueError("posting offsets are not a valid contiguous range")
        if invalid_state == 3:
            raise ValueError("posting node id is outside node_count")
        if invalid_state == 4:
            raise OverflowError("posting overlap count exceeds uint32")

        result_count = min(top_k, touched_count)
        result = []
        for hit_index in range(result_count):
            node_id = hits[hit_index].node_id
            result.append((node_id, counts[node_id]))
        return result
    finally:
        free(counts)
        free(hits)
