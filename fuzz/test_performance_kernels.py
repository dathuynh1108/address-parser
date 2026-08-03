from __future__ import annotations

import unittest
from array import array
from collections import Counter
from random import Random
from unittest.mock import patch

from address_parser import kernels, normalization
from address_parser._python_kernels import rank_posting_hits as python_rank_posting_hits
from address_parser.ngram_index import PackedNgramIndex


class PostingKernelContractTests(unittest.TestCase):
    def test_normalization_cache_does_not_retain_long_request_text(self) -> None:
        long_text = "Số 27 Nguyễn Khánh Toàn, " + ("địa chỉ rất dài " * 20)
        cache_key = (long_text, "basic")

        normalization.normalize_address_text(long_text, "basic")

        self.assertNotIn(cache_key, normalization._NORMALIZATION_CACHE)

    def test_fallback_index_does_not_materialize_unused_native_buffers(self) -> None:
        postings = {"aaaa": {1, 2}, "bbbb": {2, 3}}
        with patch(
            "address_parser.ngram_index.native_acceleration_available",
            return_value=False,
        ):
            index = PackedNgramIndex(postings, node_count=4)
            self.assertEqual(index.top_hits(["aaaa", "bbbb"], 4), [(2, 2), (1, 1), (3, 1)])

        self.assertFalse(index._posting_id_by_ngram)
        self.assertFalse(index._posting_offsets)
        self.assertFalse(index._posting_node_ids)

    def test_packed_index_matches_counter_order_and_counts(self) -> None:
        postings = {
            "aaaa": {4, 1, 7},
            "bbbb": {1, 7, 3},
            "cccc": {3, 4},
        }
        index = PackedNgramIndex(postings, node_count=8)

        expected: Counter[int] = Counter()
        for ngram in sorted({"cccc", "aaaa", "bbbb", "missing"}):
            node_ids = postings.get(ngram)
            if node_ids:
                expected.update(node_ids)

        self.assertEqual(
            index.top_hits(["cccc", "aaaa", "bbbb", "aaaa", "missing"], 8),
            expected.most_common(8),
        )

    def test_native_and_python_kernels_have_randomized_parity(self) -> None:
        random = Random(20260803)
        native_rank_posting_hits = None
        if kernels.native_acceleration_available():
            from address_parser._native_kernels import rank_posting_hits

            native_rank_posting_hits = rank_posting_hits

        for _ in range(250):
            node_count = random.randint(1, 200)
            posting_count = random.randint(1, 60)
            posting_offsets = array("I", [0])
            posting_node_ids = array("I")

            for _ in range(posting_count):
                nodes = list(range(node_count))
                random.shuffle(nodes)
                posting_node_ids.extend(nodes[: random.randint(0, node_count)])
                posting_offsets.append(len(posting_node_ids))

            posting_ids = list(range(posting_count))
            random.shuffle(posting_ids)
            query_posting_ids = array(
                "I",
                posting_ids[: random.randint(0, posting_count)],
            )
            top_k = random.randint(0, node_count + 10)
            expected = python_rank_posting_hits(
                posting_offsets,
                posting_node_ids,
                query_posting_ids,
                node_count,
                top_k,
            )
            actual = kernels.rank_posting_hits(
                posting_offsets,
                posting_node_ids,
                query_posting_ids,
                node_count,
                top_k,
            )
            self.assertEqual(actual, expected)
            if native_rank_posting_hits is not None:
                self.assertEqual(
                    native_rank_posting_hits(
                        posting_offsets,
                        posting_node_ids,
                        query_posting_ids,
                        node_count,
                        top_k,
                    ),
                    expected,
                )

    def test_kernel_rejects_out_of_range_posting_id(self) -> None:
        posting_offsets = array("I", [0, 1])
        posting_node_ids = array("I", [0])
        query_posting_ids = array("I", [1])

        with self.assertRaisesRegex(ValueError, "outside posting_offsets"):
            kernels.rank_posting_hits(
                posting_offsets,
                posting_node_ids,
                query_posting_ids,
                node_count=1,
                top_k=1,
            )

    def test_kernel_rejects_max_uint32_posting_id_without_wraparound(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside posting_offsets"):
            kernels.rank_posting_hits(
                array("I", [0, 1]),
                array("I", [0]),
                array("I", [(1 << 32) - 1]),
                node_count=1,
                top_k=1,
            )

    def test_kernel_rejects_non_uint32_buffers_consistently(self) -> None:
        with self.assertRaisesRegex(TypeError, r"array\('I'\)"):
            kernels.rank_posting_hits(
                array("i", [0, 1]),
                array("i", [0]),
                array("i", [0]),
                node_count=1,
                top_k=1,
            )

    def test_native_requirement_is_explicit(self) -> None:
        if kernels.native_acceleration_available():
            self.assertIsNone(kernels.require_native_acceleration())
        else:
            with self.assertRaisesRegex(RuntimeError, "native kernels are unavailable"):
                kernels.require_native_acceleration()


if __name__ == "__main__":
    unittest.main()
