import sys
import unittest
from unittest.mock import MagicMock

if "resource" not in sys.modules:
    sys.modules["resource"] = MagicMock()

import pandas as pd
import pyarrow as arrow

from smallpond.dataframe import DataFrame


class TestEmptyDataFrameConversion(unittest.TestCase):
    def test_to_pandas_empty_datasets(self):
        df = DataFrame.__new__(DataFrame)
        df._compute = MagicMock(return_value=[])
        res = df.to_pandas()
        self.assertIsInstance(res, pd.DataFrame)
        self.assertTrue(res.empty)

    def test_to_arrow_empty_datasets(self):
        df = DataFrame.__new__(DataFrame)
        df._compute = MagicMock(return_value=[])
        res = df.to_arrow()
        self.assertIsInstance(res, arrow.Table)
        self.assertEqual(res.num_rows, 0)

    def test_to_pandas_non_empty_datasets(self):
        mock_ds1 = MagicMock()
        mock_ds1.to_pandas.return_value = pd.DataFrame({"a": [1, 2]})
        mock_ds2 = MagicMock()
        mock_ds2.to_pandas.return_value = pd.DataFrame({"a": [3, 4]})

        df = DataFrame.__new__(DataFrame)
        df._compute = MagicMock(return_value=[mock_ds1, mock_ds2])
        res = df.to_pandas()
        self.assertIsInstance(res, pd.DataFrame)
        self.assertEqual(len(res), 4)
        self.assertEqual(list(res["a"]), [1, 2, 3, 4])

    def test_to_arrow_non_empty_datasets(self):
        t1 = arrow.table({"x": [10, 20]})
        t2 = arrow.table({"x": [30, 40]})
        mock_ds1 = MagicMock()
        mock_ds1.to_arrow_table.return_value = t1
        mock_ds2 = MagicMock()
        mock_ds2.to_arrow_table.return_value = t2

        df = DataFrame.__new__(DataFrame)
        df._compute = MagicMock(return_value=[mock_ds1, mock_ds2])
        res = df.to_arrow()
        self.assertIsInstance(res, arrow.Table)
        self.assertEqual(res.num_rows, 4)
        self.assertEqual(res.column("x").to_pylist(), [10, 20, 30, 40])


if __name__ == "__main__":
    unittest.main()
