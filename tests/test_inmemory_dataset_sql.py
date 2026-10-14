import sys
import unittest
from unittest.mock import MagicMock

if "resource" not in sys.modules:
    sys.modules["resource"] = MagicMock()

import duckdb
import pandas as pd
import pyarrow as arrow

from smallpond.logical.dataset import ArrowTableDataSet, PandasDataSet


class TestInMemoryDatasetSql(unittest.TestCase):
    def test_pandas_dataset_sql_query_fragment_none_conn(self):
        df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
        dataset = PandasDataSet(df)
        fragment = dataset.sql_query_fragment(conn=None)
        res = duckdb.sql(f"select * from {fragment}").df()
        pd.testing.assert_frame_equal(res, df)

    def test_pandas_dataset_sql_query_fragment_with_conn(self):
        conn = duckdb.connect()
        df = pd.DataFrame({"val": [10.5, 20.5]})
        dataset = PandasDataSet(df)
        fragment = dataset.sql_query_fragment(conn=conn)
        res = conn.sql(f"select * from {fragment}").df()
        pd.testing.assert_frame_equal(res, df)

    def test_arrow_dataset_sql_query_fragment_none_conn(self):
        table = arrow.table({"col_a": [100, 200], "col_b": ["alpha", "beta"]})
        dataset = ArrowTableDataSet(table)
        fragment = dataset.sql_query_fragment(conn=None)
        res = duckdb.sql(f"select * from {fragment}").arrow()
        self.assertTrue(res.equals(table))

    def test_arrow_dataset_sql_query_fragment_with_conn(self):
        conn = duckdb.connect()
        table = arrow.table({"num": [42, 99]})
        dataset = ArrowTableDataSet(table)
        fragment = dataset.sql_query_fragment(conn=conn)
        res = conn.sql(f"select * from {fragment}").arrow()
        self.assertTrue(res.equals(table))

    def test_inmemory_to_batch_reader(self):
        df = pd.DataFrame({"id": list(range(50))})
        ds = PandasDataSet(df)
        reader = ds.to_batch_reader(batch_size=10)
        batches = list(reader)
        self.assertEqual(len(batches), 5)
        self.assertEqual(sum(len(b) for b in batches), 50)


if __name__ == "__main__":
    unittest.main()
