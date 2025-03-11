import glob
import importlib
import tempfile
import unittest

from smallpond.io.arrow import cast_columns_to_large_string
from tests.test_fabric import TestFabric


@unittest.skipUnless(importlib.util.find_spec("deltalake") is not None, "cannot find deltalake")
class TestDeltaLake(TestFabric, unittest.TestCase):
    def test_read_write_deltalake(self):
        from deltalake import DeltaTable, write_deltalake

        for dataset_path in (
            "tests/data/arrow/*.parquet",
            "tests/data/large_array/*.parquet",
        ):
            parquet_files = glob.glob(dataset_path)
            expected = self._load_parquet_files(parquet_files)
            with self.subTest(dataset_path=dataset_path), tempfile.TemporaryDirectory(dir=self.output_root_abspath) as output_dir:
                write_deltalake(output_dir, expected, large_dtypes=True)
                dt = DeltaTable(output_dir)
                self._compare_arrow_tables(expected, dt.to_pyarrow_table())

    def test_load_mixed_large_dtypes(self):
        from deltalake import DeltaTable, write_deltalake

        for dataset_path in (
            "tests/data/arrow/*.parquet",
            "tests/data/large_array/*.parquet",
        ):
            parquet_files = glob.glob(dataset_path)
            with self.subTest(dataset_path=dataset_path), tempfile.TemporaryDirectory(dir=self.output_root_abspath) as output_dir:
                table = cast_columns_to_large_string(self._load_parquet_files(parquet_files))
                write_deltalake(output_dir, table, large_dtypes=True, mode="overwrite")
                write_deltalake(output_dir, table, large_dtypes=False, mode="append")
                loaded_table = DeltaTable(output_dir).to_pyarrow_table()
                print("table:\n", table.schema)
                print("loaded_table:\n", loaded_table.schema)
                self.assertEqual(table.num_rows * 2, loaded_table.num_rows)

    def test_delete_update(self):
        import pandas as pd
        from deltalake import DeltaTable, write_deltalake

        with tempfile.TemporaryDirectory(dir=self.output_root_abspath) as output_dir:
            df = pd.DataFrame({"num": [1, 2, 3], "animal": ["cat", "dog", "snake"]})
            write_deltalake(output_dir, df, mode="overwrite")
            dt = DeltaTable(output_dir)
            dt.delete("animal = 'cat'")
            dt.update(predicate="num = 3", new_values={"animal": "fish"})
