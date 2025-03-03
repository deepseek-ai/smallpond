import itertools
import unittest

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from smallpond.common import get_nth_partition, split_into_cols, split_into_rows
from tests.test_fabric import TestFabric


class TestCommon(TestFabric, unittest.TestCase):
    def test_get_nth_partition(self):
        items = [1, 2, 3]
        # split into 1 partitions
        self.assertListEqual([1, 2, 3], get_nth_partition(items, 0, 1))
        # split into 2 partitions
        self.assertListEqual([1, 2], get_nth_partition(items, 0, 2))
        self.assertListEqual([3], get_nth_partition(items, 1, 2))
        # split into 3 partitions
        self.assertListEqual([1], get_nth_partition(items, 0, 3))
        self.assertListEqual([2], get_nth_partition(items, 1, 3))
        self.assertListEqual([3], get_nth_partition(items, 2, 3))
        # split into 5 partitions
        self.assertListEqual([1], get_nth_partition(items, 0, 5))
        self.assertListEqual([2], get_nth_partition(items, 1, 5))
        self.assertListEqual([3], get_nth_partition(items, 2, 5))
        self.assertListEqual([], get_nth_partition(items, 3, 5))
        self.assertListEqual([], get_nth_partition(items, 4, 5))

    @given(st.data())
    def test_split_into_rows(self, data: st.data):
        nelements = data.draw(st.integers(1, 100))
        npartitions = data.draw(st.integers(1, 2 * nelements))
        items = list(range(nelements))
        computed = split_into_rows(items, npartitions)
        expected = [get_nth_partition(items, n, npartitions) for n in range(npartitions)]
        self.assertEqual(expected, computed)

    @given(st.data())
    def test_split_into_cols(self, data: st.data):
        nelements = data.draw(st.integers(1, 100))
        npartitions = data.draw(st.integers(1, 2 * nelements))
        items = list(range(nelements))
        chunks = split_into_cols(items, npartitions)
        self.assertEqual(npartitions, len(chunks))
        self.assertListEqual(
            items,
            [x for row in itertools.zip_longest(*chunks) for x in row if x is not None],
        )
        chunk_sizes = set(len(chk) for chk in chunks)
        if len(chunk_sizes) > 1:
            small_size, large_size = sorted(chunk_sizes)
            self.assertEqual(small_size + 1, large_size)
        else:
            (chunk_size,) = chunk_sizes
            self.assertEqual(len(items), chunk_size * npartitions)

    def test_split_into_rows_bench(self):
        for nelements in [100000, 1000000]:
            items = np.arange(nelements)
            for npartitions in [1024, 4096, 10240, nelements, 2 * nelements]:
                chunks = split_into_rows(items, npartitions)
                self.assertEqual(npartitions, len(chunks))

    def test_split_into_cols_bench(self):
        for nelements in [100000, 1000000]:
            items = np.arange(nelements)
            for npartitions in [1024, 4096, 10240, nelements, 2 * nelements]:
                chunks = split_into_cols(items, npartitions)
                self.assertEqual(npartitions, len(chunks))
