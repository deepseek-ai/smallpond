import os
import tempfile
import unittest

from examples.fstest import fstest
from examples.shuffle_data import shuffle_data
from examples.shuffle_mock_urls import shuffle_mock_urls
from examples.sort_mock_urls import sort_mock_urls
from examples.sort_mock_urls_v2 import sort_mock_urls_v2
from smallpond.dataframe import Session
from tests.test_fabric import TestFabric


class TestPlan(TestFabric, unittest.TestCase):
    def test_sort_mock_urls(self):
        for engine_type in ("duckdb", "arrow"):
            with self.subTest(engine_type=engine_type):
                plan = sort_mock_urls(
                    ["tests/data/mock_urls/*.tsv"],
                    npartitions=3,
                    engine_type=engine_type,
                )
                self.execute_plan(plan)

    def test_sort_mock_urls_external_output_path(self):
        with tempfile.TemporaryDirectory(dir=self.output_root_abspath) as output_dir:
            plan = sort_mock_urls(
                ["tests/data/mock_urls/*.tsv"],
                npartitions=3,
                external_output_path=output_dir,
            )
            self.execute_plan(plan)

    def test_shuffle_mock_urls(self):
        for engine_type in ("duckdb", "arrow"):
            with self.subTest(engine_type=engine_type):
                plan = shuffle_mock_urls(
                    ["tests/data/mock_urls/*.parquet"],
                    npartitions=3,
                    sort_rand_keys=True,
                )
                self.execute_plan(plan)

    def test_shuffle_data(self):
        for engine_type in ("duckdb", "arrow"):
            with self.subTest(engine_type=engine_type):
                plan = shuffle_data(
                    ["tests/data/mock_urls/*.parquet"],
                    num_data_partitions=3,
                    num_out_data_partitions=3,
                    engine_type=engine_type,
                )
                self.execute_plan(plan)


def test_fstest(sp: Session):
    path = sp._runtime_ctx.output_root
    fstest(
        sp,
        input_path=os.path.join(path, "*"),
        output_path=path,
        size="10M",
        npartitions=3,
    )


def test_sort_mock_urls_v2(sp: Session):
    sort_mock_urls_v2(sp, ["tests/data/mock_urls/*.tsv"], sp._runtime_ctx.output_root, npartitions=3)
