import glob
import os.path
import unittest
from pathlib import PurePath

import duckdb
import pandas
import pyarrow as arrow
import pytest
from loguru import logger

from smallpond.common import DEFAULT_ROW_GROUP_SIZE, MB
from smallpond.logical.dataset import ParquetDataSet
from smallpond.utility import ConcurrentIter
from tests.test_fabric import TestFabric


class TestDataSet(TestFabric, unittest.TestCase):
    def test_parquet_file_created_by_pandas(self):
        num_urls = 0
        for txt_file in glob.glob("tests/data/mock_urls/*.tsv"):
            urls = pandas.read_csv(txt_file, delimiter="\t", names=["url"])
            urls.to_parquet(
                os.path.join(
                    self.output_root_abspath,
                    PurePath(os.path.basename(txt_file)).with_suffix(".parquet"),
                )
            )
            num_urls += urls.size
        dataset = ParquetDataSet([os.path.join(self.output_root_abspath, "*.parquet")])
        self.assertEqual(num_urls, dataset.num_rows)

    def _generate_parquet_dataset(self, output_path, npartitions, num_rows, row_group_size):
        duckdb.sql(
            f"""copy (
               select range as i, range % {npartitions} as partition from range(0, {num_rows}) )
               to '{output_path}'
               (FORMAT PARQUET, ROW_GROUP_SIZE {row_group_size}, PARTITION_BY partition, OVERWRITE_OR_IGNORE true)"""
        )
        return ParquetDataSet([f"{output_path}/**/*.parquet"])

    def _check_partition_datasets(self, orig_dataset: ParquetDataSet, partition_func, npartition):
        # build partitioned datasets
        partitioned_datasets = partition_func(npartition)
        self.assertEqual(npartition, len(partitioned_datasets))
        self.assertEqual(
            orig_dataset.num_rows,
            sum(dataset.num_rows for dataset in partitioned_datasets),
        )
        # load as arrow table
        loaded_table = arrow.concat_tables([dataset.to_arrow_table(max_workers=1) for dataset in partitioned_datasets])
        self.assertEqual(orig_dataset.num_rows, loaded_table.num_rows)
        # compare arrow tables
        orig_table = orig_dataset.to_arrow_table(max_workers=1)
        self.assertEqual(orig_table.shape, loaded_table.shape)
        self.assertTrue(orig_table.sort_by("i").equals(loaded_table.sort_by("i")))
        # compare sql query results
        join_query = f"""
      select count(a.i) as num_rows
      from {orig_dataset.sql_query_fragment()} as a
      join ( {' union all '.join([dataset.sql_query_fragment() for dataset in partitioned_datasets])} ) as b on a.i = b.i"""
        results = duckdb.sql(join_query).fetchall()
        self.assertEqual(orig_dataset.num_rows, results[0][0])

    def test_num_rows(self):
        dataset = ParquetDataSet(["tests/data/arrow/*.parquet"])
        self.assertEqual(dataset.num_rows, 1000)

    def test_partition_by_files(self):
        output_path = os.path.join(self.output_root_abspath, "test_partition_by_files")
        orig_dataset = self._generate_parquet_dataset(output_path, npartitions=11, num_rows=170 * 1000, row_group_size=10 * 1000)
        num_files = len(orig_dataset.resolved_paths)
        for npartition in range(1, num_files + 1):
            for random_shuffle in (False, True):
                with self.subTest(npartition=npartition, random_shuffle=random_shuffle):
                    orig_dataset.reset(orig_dataset.paths, orig_dataset.root_dir)
                    self._check_partition_datasets(
                        orig_dataset,
                        lambda n: orig_dataset.partition_by_files(n, random_shuffle=random_shuffle),
                        npartition,
                    )

    def test_partition_by_rows(self):
        output_path = os.path.join(self.output_root_abspath, "test_partition_by_rows")
        orig_dataset = self._generate_parquet_dataset(output_path, npartitions=11, num_rows=170 * 1000, row_group_size=10 * 1000)
        num_files = len(orig_dataset.resolved_paths)
        for npartition in range(1, 2 * num_files + 1):
            for random_shuffle in (False, True):
                with self.subTest(npartition=npartition, random_shuffle=random_shuffle):
                    orig_dataset.reset(orig_dataset.paths, orig_dataset.root_dir)
                    self._check_partition_datasets(
                        orig_dataset,
                        lambda n: orig_dataset.partition_by_rows(n, random_shuffle=random_shuffle),
                        npartition,
                    )

    def test_resolved_many_paths(self):
        with open("tests/data/long_path_list.txt", buffering=16 * MB) as fin:
            filenames = list(map(os.path.basename, map(str.strip, fin.readlines())))
            logger.info(f"loaded {len(filenames)} filenames")
        dataset = ParquetDataSet(filenames)
        self.assertEqual(len(dataset.resolved_paths), len(filenames))

    def test_paths_with_char_ranges(self):
        dataset_with_char_ranges = ParquetDataSet(["tests/data/arrow/data[0-9].parquet"])
        dataset_with_wildcards = ParquetDataSet(["tests/data/arrow/*.parquet"])
        self.assertEqual(
            len(dataset_with_char_ranges.resolved_paths),
            len(dataset_with_wildcards.resolved_paths),
        )

    def test_to_arrow_table_batch_reader(self):
        memdb = duckdb.connect(database=":memory:", config={"arrow_large_buffer_size": "true"})
        for dataset_path in (
            "tests/data/arrow/*.parquet",
            "tests/data/large_array/*.parquet",
        ):
            for conn in (None, memdb):
                print(f"dataset_path: {dataset_path}, conn: {conn}")
                with self.subTest(dataset_path=dataset_path, conn=conn):
                    dataset = ParquetDataSet([dataset_path])
                    to_batches = dataset.to_arrow_table(max_workers=1, conn=conn).to_batches(max_chunksize=DEFAULT_ROW_GROUP_SIZE * 2)
                    batch_reader = dataset.to_batch_reader(batch_size=DEFAULT_ROW_GROUP_SIZE * 2, conn=conn)
                    with ConcurrentIter(batch_reader, max_buffer_size=2) as batch_reader:
                        for batch_iter in (to_batches, batch_reader):
                            total_num_rows = 0
                            for batch in batch_iter:
                                print(f"batch.num_rows {batch.num_rows}, max_batch_row_size {DEFAULT_ROW_GROUP_SIZE*2}")
                                self.assertLessEqual(batch.num_rows, DEFAULT_ROW_GROUP_SIZE * 2)
                                total_num_rows += batch.num_rows
                            print(f"{dataset_path}: total_num_rows {total_num_rows}")
                            self.assertEqual(total_num_rows, dataset.num_rows)


@pytest.mark.parametrize("reader", ["arrow", "duckdb"])
@pytest.mark.parametrize("dataset_path", ["tests/data/arrow/*.parquet"])
# @pytest.mark.parametrize("dataset_path", ["tests/data/arrow/*.parquet", "tests/data/large_array/*.parquet"])
def test_arrow_reader(benchmark, reader: str, dataset_path: str):
    dataset = ParquetDataSet([dataset_path])
    conn = None
    if reader == "duckdb":
        conn = duckdb.connect(database=":memory:", config={"arrow_large_buffer_size": "true"})
    benchmark(dataset.to_arrow_table, conn=conn)
    # result: arrow reader is 4x faster than duckdb reader in small dataset, 1.4x faster in large dataset
