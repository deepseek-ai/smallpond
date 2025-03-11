import glob
import os.path
import tempfile
import unittest

import pyarrow.parquet as parquet
from loguru import logger

from smallpond.io.arrow import (
    RowRange,
    build_batch_reader_from_files,
    cast_columns_to_large_string,
    dump_to_parquet_files,
    load_from_parquet_files,
)
from smallpond.utility import ConcurrentIter
from tests.test_fabric import TestFabric


class TestArrow(TestFabric, unittest.TestCase):
    def test_load_from_parquet_files(self):
        for dataset_path in (
            "tests/data/arrow/*.parquet",
            "tests/data/large_array/*.parquet",
        ):
            with self.subTest(dataset_path=dataset_path):
                parquet_files = glob.glob(dataset_path)
                expected = self._load_parquet_files(parquet_files)
                actual = load_from_parquet_files(parquet_files)
                self._compare_arrow_tables(expected, actual)

    def test_load_parquet_row_ranges(self):
        for dataset_path in (
            "tests/data/arrow/data0.parquet",
            "tests/data/large_array/large_array.parquet",
        ):
            with self.subTest(dataset_path=dataset_path):
                metadata = parquet.read_metadata(dataset_path)
                file_num_rows = metadata.num_rows
                data_size = sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))
                row_range = RowRange(
                    path=dataset_path,
                    begin=100,
                    end=200,
                    data_size=data_size,
                    file_num_rows=file_num_rows,
                )
                expected = self._load_parquet_files([dataset_path]).slice(offset=100, length=100)
                actual = load_from_parquet_files([row_range])
                self._compare_arrow_tables(expected, actual)

    def test_dump_to_parquet_files(self):
        for dataset_path in (
            "tests/data/arrow/*.parquet",
            "tests/data/large_array/*.parquet",
        ):
            with self.subTest(dataset_path=dataset_path):
                parquet_files = glob.glob(dataset_path)
                expected = self._load_parquet_files(parquet_files)
                with tempfile.TemporaryDirectory(dir=self.output_root_abspath) as output_dir:
                    ok = dump_to_parquet_files(expected, output_dir)
                    self.assertTrue(ok)
                    actual = self._load_parquet_files(glob.glob(f"{output_dir}/*.parquet"))
                    self._compare_arrow_tables(expected, actual)

    def test_dump_load_empty_table(self):
        # create empty table
        empty_table = self._load_parquet_files(["tests/data/arrow/data0.parquet"]).slice(length=0)
        self.assertEqual(empty_table.num_rows, 0)
        # dump empty table
        with tempfile.TemporaryDirectory(dir=self.output_root_abspath) as output_dir:
            ok = dump_to_parquet_files(empty_table, output_dir)
            self.assertTrue(ok)
            parquet_files = glob.glob(f"{output_dir}/*.parquet")
            # load empty table from file
            actual_table = load_from_parquet_files(parquet_files)
            self._compare_arrow_tables(empty_table, actual_table)

    def test_parquet_batch_reader(self):
        for dataset_path in (
            "tests/data/arrow/*.parquet",
            "tests/data/large_array/*.parquet",
        ):
            with self.subTest(dataset_path=dataset_path):
                parquet_files = glob.glob(dataset_path)
                expected_num_rows = sum(parquet.read_metadata(file).num_rows for file in parquet_files)
                with build_batch_reader_from_files(
                    parquet_files,
                    batch_size=expected_num_rows,
                    max_batch_byte_size=None,
                ) as batch_reader, ConcurrentIter(batch_reader) as concurrent_iter:
                    total_num_rows = 0
                    for batch in concurrent_iter:
                        print(f"batch.num_rows {batch.num_rows}, max_batch_row_size {expected_num_rows}")
                        self.assertLessEqual(batch.num_rows, expected_num_rows)
                        total_num_rows += batch.num_rows
                    self.assertEqual(total_num_rows, expected_num_rows)

    def test_table_to_batches(self):
        for dataset_path in (
            "tests/data/arrow/*.parquet",
            "tests/data/large_array/*.parquet",
        ):
            with self.subTest(dataset_path=dataset_path):
                parquet_files = glob.glob(dataset_path)
                table = self._load_parquet_files(parquet_files)
                total_num_rows = 0
                for batch in table.to_batches(max_chunksize=table.num_rows):
                    print(f"batch.num_rows {batch.num_rows}, max_batch_row_size {table.num_rows}")
                    self.assertLessEqual(batch.num_rows, table.num_rows)
                    total_num_rows += batch.num_rows
                self.assertEqual(total_num_rows, table.num_rows)

    def test_arrow_schema_metadata(self):
        table = self._load_parquet_files(glob.glob("tests/data/arrow/*.parquet"))
        metadata = {b"a": b"1", b"b": b"2"}
        table_with_meta = table.replace_schema_metadata(metadata)
        print(f"table_with_meta.schema.metadata {table_with_meta.schema.metadata}")

        with tempfile.TemporaryDirectory(dir=self.output_root_abspath) as output_dir:
            self.assertTrue(dump_to_parquet_files(table_with_meta, output_dir, "arrow_schema_metadata", max_workers=2))
            parquet_files = glob.glob(os.path.join(output_dir, "arrow_schema_metadata*.parquet"))
            loaded_table = load_from_parquet_files(parquet_files, table.column_names[:1])
            print(f"loaded_table.schema.metadata {loaded_table.schema.metadata}")
            self.assertEqual(table_with_meta.schema.metadata, loaded_table.schema.metadata)
            with parquet.ParquetFile(parquet_files[0]) as file:
                print(f"file.schema_arrow.metadata {file.schema_arrow.metadata}")
                self.assertEqual(table_with_meta.schema.metadata, file.schema_arrow.metadata)

    def test_load_mixed_string_types(self):
        parquet_paths = glob.glob("tests/data/arrow/*.parquet")
        table = self._load_parquet_files(parquet_paths)

        with tempfile.TemporaryDirectory(dir=self.output_root_abspath) as output_dir:
            dump_to_parquet_files(cast_columns_to_large_string(table), output_dir)
            parquet_paths += glob.glob(os.path.join(output_dir, "*.parquet"))
            loaded_table = load_from_parquet_files(parquet_paths)
            self.assertEqual(table.num_rows * 2, loaded_table.num_rows)
            batch_reader = build_batch_reader_from_files(parquet_paths)
            self.assertEqual(table.num_rows * 2, sum(batch.num_rows for batch in batch_reader))

    @logger.catch(reraise=True, message="failed to load parquet files")
    def _load_from_parquet_files_with_log(self, paths, columns):
        load_from_parquet_files(paths, columns)

    def test_load_not_exist_column(self):
        parquet_files = glob.glob("tests/data/arrow/*.parquet")
        with self.assertRaises(AssertionError) as context:
            self._load_from_parquet_files_with_log(parquet_files, ["not_exist_column"])

    def test_change_ordering_of_columns(self):
        parquet_files = glob.glob("tests/data/arrow/*.parquet")
        loaded_table = load_from_parquet_files(parquet_files)
        reversed_cols = list(reversed(loaded_table.column_names))
        loaded_table = load_from_parquet_files(parquet_files, reversed_cols)
        self.assertEqual(loaded_table.column_names, reversed_cols)
