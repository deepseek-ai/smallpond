import copy
import os
import sys
import tempfile
import unittest
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

if "resource" not in sys.modules:
    sys.modules["resource"] = MagicMock()

from smallpond.execution.task import DataSinkTask
from smallpond.io.arrow import RowRange
from smallpond.logical.dataset import CsvDataSet, JsonDataSet, ParquetDataSet, PartitionedDataSet


class TestDataSinkDatasetTypes(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_dummy_file(self, filename: str) -> str:
        filepath = self.output_dir / filename
        filepath.write_text("dummy")
        return str(filepath)

    def test_collect_output_files_csv(self):
        csv_file1 = self._create_dummy_file("data1.csv")
        csv_file2 = self._create_dummy_file("data2.csv")
        schema = OrderedDict([("a", "INTEGER")])
        csv_ds = CsvDataSet([csv_file1, csv_file2], schema=schema, delim=";")

        task = DataSinkTask.__new__(DataSinkTask)
        task.type = "manifest"
        task.output_root = str(self.output_dir)
        task.input_datasets = [csv_ds]

        with ThreadPoolExecutor(2) as pool:
            task.collect_output_files(pool)

        self.assertIsInstance(task.dataset, CsvDataSet)
        self.assertEqual(task.dataset.delim, ";")
        self.assertEqual(task.dataset.schema, schema)
        self.assertEqual(len(task.dataset.resolved_paths), 2)

    def test_collect_output_files_json(self):
        json_file = self._create_dummy_file("data.json")
        json_ds = JsonDataSet([json_file], format="array", schema={"name": "VARCHAR"})

        task = DataSinkTask.__new__(DataSinkTask)
        task.type = "manifest"
        task.output_root = str(self.output_dir)
        task.input_datasets = [json_ds]

        with ThreadPoolExecutor(1) as pool:
            task.collect_output_files(pool)

        self.assertIsInstance(task.dataset, JsonDataSet)
        self.assertEqual(task.dataset.format, "array")
        self.assertEqual(task.dataset.schema, {"name": "VARCHAR"})

    def test_collect_output_files_parquet_preserves_row_ranges_manifest(self):
        pq_file1 = self._create_dummy_file("part1.parquet")
        pq_file2 = self._create_dummy_file("part2.parquet")

        ds1 = ParquetDataSet([pq_file1])
        rr1 = RowRange(pq_file1, 1000, 100, 0, 50)
        ds1._resolved_row_ranges = [rr1]

        ds2 = ParquetDataSet([pq_file2])
        rr2 = RowRange(pq_file2, 1000, 100, 50, 100)
        ds2._resolved_row_ranges = [rr2]

        task = DataSinkTask.__new__(DataSinkTask)
        task.type = "manifest"
        task.output_root = str(self.output_dir)
        task.input_datasets = [ds1, ds2]

        with ThreadPoolExecutor(2) as pool:
            task.collect_output_files(pool)

        self.assertIsInstance(task.dataset, ParquetDataSet)
        self.assertIsNotNone(task.dataset._resolved_row_ranges)
        self.assertEqual(len(task.dataset._resolved_row_ranges), 2)
        self.assertEqual(task.dataset._resolved_row_ranges[0].begin, 0)
        self.assertEqual(task.dataset._resolved_row_ranges[0].end, 50)
        self.assertEqual(task.dataset._resolved_row_ranges[1].begin, 50)
        self.assertEqual(task.dataset._resolved_row_ranges[1].end, 100)

    def test_collect_output_files_parquet_preserves_row_ranges_copy(self):
        in_dir = self.output_dir / "inputs"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir = self.output_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        pq_file = str(in_dir / "part_copy.parquet")
        Path(pq_file).write_text("dummy")

        ds = ParquetDataSet([pq_file])
        rr = RowRange(pq_file, 500, 50, 10, 40)
        ds._resolved_row_ranges = [rr]

        task = DataSinkTask.__new__(DataSinkTask)
        task.type = "link_manifest"
        task.output_root = str(out_dir)
        task.input_datasets = [ds]

        with ThreadPoolExecutor(1) as pool:
            task.collect_output_files(pool)

        self.assertIsInstance(task.dataset, ParquetDataSet)
        self.assertIsNotNone(task.dataset._resolved_row_ranges)
        self.assertEqual(len(task.dataset._resolved_row_ranges), 1)
        self.assertEqual(task.dataset._resolved_row_ranges[0].begin, 10)
        self.assertEqual(task.dataset._resolved_row_ranges[0].end, 40)
        self.assertEqual(task.dataset._resolved_row_ranges[0].path, task.dataset.resolved_paths[0])

    def test_collect_output_files_partitioned_dataset_unpacking(self):
        csv_file = self._create_dummy_file("part.csv")
        csv_ds = CsvDataSet([csv_file], schema=OrderedDict(), delim="\t")
        partitioned = PartitionedDataSet([csv_ds])

        task = DataSinkTask.__new__(DataSinkTask)
        task.type = "manifest"
        task.output_root = str(self.output_dir)
        task.input_datasets = [partitioned]

        with ThreadPoolExecutor(1) as pool:
            task.collect_output_files(pool)

        self.assertIsInstance(task.dataset, CsvDataSet)
        self.assertEqual(task.dataset.delim, "\t")

    def test_collect_output_files_extension_fallback(self):
        csv_file = self._create_dummy_file("fallback.csv")
        mock_ds = MagicMock()
        mock_ds.resolved_paths = [csv_file]

        task = DataSinkTask.__new__(DataSinkTask)
        task.type = "manifest"
        task.output_root = str(self.output_dir)
        task.input_datasets = [mock_ds]

        with ThreadPoolExecutor(1) as pool:
            task.collect_output_files(pool)

        self.assertIsInstance(task.dataset, CsvDataSet)


if __name__ == "__main__":
    unittest.main()
