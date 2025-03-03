import shutil
import unittest

from benchmarks.file_io_benchmark import file_io_benchmark
from benchmarks.gray_sort_benchmark import generate_random_records, gray_sort_benchmark
from benchmarks.hash_partition_benchmark import hash_partition_benchmark
from benchmarks.urls_sort_benchmark import urls_sort_benchmark
from smallpond.common import MB
from smallpond.logical.node import Context, LogicalPlan
from tests.test_fabric import TestFabric


class TestBench(TestFabric, unittest.TestCase):

    fault_inject_prob = 0.05

    def test_file_io_benchmark(self):
        for io_engine in ("duckdb", "arrow", "stream"):
            with self.subTest(io_engine=io_engine):
                plan = file_io_benchmark(
                    ["tests/data/mock_urls/*.parquet"],
                    npartitions=3,
                    io_engine=io_engine,
                )
                self.execute_plan(plan, enable_profiling=True)

    def test_urls_sort_benchmark(self):
        for engine_type in ("duckdb", "arrow"):
            with self.subTest(engine_type=engine_type):
                plan = urls_sort_benchmark(
                    ["tests/data/mock_urls/*.tsv"],
                    num_data_partitions=3,
                    num_hash_partitions=3,
                    engine_type=engine_type,
                )
                self.execute_plan(plan, enable_profiling=True)

    @unittest.skipIf(shutil.which("gensort") is None, "gensort not found")
    def test_gray_sort_benchmark(self):
        record_nbytes = 100
        key_nbytes = 10
        total_data_nbytes = 100 * MB
        gensort_batch_nbytes = 10 * MB
        num_data_partitions = 5
        num_sort_partitions = 1 << 3
        for shuffle_engine in ("duckdb", "arrow"):
            for sort_engine in ("duckdb", "arrow", "polars"):
                with self.subTest(shuffle_engine=shuffle_engine, sort_engine=sort_engine):
                    ctx = Context()
                    random_records = generate_random_records(
                        ctx,
                        record_nbytes,
                        key_nbytes,
                        total_data_nbytes,
                        gensort_batch_nbytes,
                        num_data_partitions,
                        num_sort_partitions,
                    )
                    plan = LogicalPlan(ctx, random_records)
                    exec_plan = self.execute_plan(plan, enable_profiling=True)

                    plan = gray_sort_benchmark(
                        record_nbytes,
                        key_nbytes,
                        total_data_nbytes,
                        gensort_batch_nbytes,
                        num_data_partitions,
                        num_sort_partitions,
                        input_paths=exec_plan.final_output.resolved_paths,
                        shuffle_engine=shuffle_engine,
                        sort_engine=sort_engine,
                        hive_partitioning=True,
                        validate_results=True,
                    )
                    self.execute_plan(plan, enable_profiling=True)

    def test_hash_partition_benchmark(self):
        for engine_type in ("duckdb", "arrow"):
            with self.subTest(engine_type=engine_type):
                plan = hash_partition_benchmark(
                    ["tests/data/mock_urls/*.parquet"],
                    npartitions=5,
                    hash_columns=["url"],
                    engine_type=engine_type,
                    hive_partitioning=True,
                    partition_stats=True,
                )
                self.execute_plan(plan, enable_profiling=True)
