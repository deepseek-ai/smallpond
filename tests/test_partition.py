import os.path
import tempfile
import unittest
from typing import List

import pyarrow.compute as pc

from smallpond.common import DATA_PARTITION_COLUMN_NAME, GB
from smallpond.execution.task import RuntimeContext
from smallpond.logical.dataset import DataSet, ParquetDataSet
from smallpond.logical.node import (
    ArrowComputeNode,
    ConsolidateNode,
    Context,
    DataSetPartitionNode,
    DataSinkNode,
    DataSourceNode,
    EvenlyDistributedPartitionNode,
    HashPartitionNode,
    LoadPartitionedDataSetNode,
    LogicalPlan,
    ProjectionNode,
    SqlEngineNode,
    UnionNode,
    UserDefinedPartitionNode,
    UserPartitionedDataSourceNode,
)
from tests.test_execution import parse_url
from tests.test_fabric import TestFabric


class CalculatePartitionFromFilename(UserDefinedPartitionNode):
    def partition(self, runtime_ctx: RuntimeContext, dataset: DataSet) -> List[DataSet]:
        partitioned_datasets: List[ParquetDataSet] = [ParquetDataSet([]) for _ in range(self.npartitions)]
        for path in dataset.resolved_paths:
            partition_idx = hash(path) % self.npartitions
            partitioned_datasets[partition_idx].paths.append(path)
        return partitioned_datasets


class TestPartition(TestFabric, unittest.TestCase):
    def test_many_file_partitions(self):
        ctx = Context()
        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"] * 10)
        data_files = DataSourceNode(ctx, dataset)
        data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=dataset.num_files)
        count_rows = SqlEngineNode(
            ctx,
            (data_partitions,),
            "select count(*) from {0}",
            cpu_limit=1,
            memory_limit=1 * GB,
        )
        plan = LogicalPlan(ctx, count_rows)
        self.execute_plan(plan)

    def test_many_row_partitions(self):
        ctx = Context()
        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_files = DataSourceNode(ctx, dataset)
        data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=dataset.num_rows, partition_by_rows=True)
        count_rows = SqlEngineNode(
            ctx,
            (data_partitions,),
            "select count(*) from {0}",
            cpu_limit=1,
            memory_limit=1 * GB,
        )
        plan = LogicalPlan(ctx, count_rows)
        exec_plan = self.execute_plan(plan, num_executors=5)
        self.assertEqual(exec_plan.final_output.to_arrow_table().num_rows, dataset.num_rows)

    def test_empty_dataset_partition(self):
        ctx = Context()
        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_files = DataSourceNode(ctx, dataset)
        # create more partitions than files
        data_partitions = EvenlyDistributedPartitionNode(ctx, (data_files,), npartitions=dataset.num_files * 2)
        data_partitions.max_num_producer_tasks = 3
        unique_urls = SqlEngineNode(
            ctx,
            (data_partitions,),
            r"select distinct url from {0}",
            cpu_limit=1,
            memory_limit=1 * GB,
        )
        # nested partition
        nested_partitioned_urls = EvenlyDistributedPartitionNode(ctx, (unique_urls,), npartitions=3, dimension="nested", nested=True)
        parsed_urls = ArrowComputeNode(
            ctx,
            (nested_partitioned_urls,),
            process_func=parse_url,
            cpu_limit=1,
            memory_limit=1 * GB,
        )
        plan = LogicalPlan(ctx, parsed_urls)
        final_output = self.execute_plan(plan, remove_empty_parquet=True, skip_task_with_empty_input=True).final_output
        self.assertTrue(isinstance(final_output, ParquetDataSet))
        self.assertEqual(dataset.num_rows, final_output.num_rows)

    def test_hash_partition(self):
        for engine_type in ("duckdb", "arrow"):
            for partition_by_rows in (False, True):
                for hive_partitioning in (False, True) if engine_type == "duckdb" else (False,):
                    with self.subTest(
                        engine_type=engine_type,
                        partition_by_rows=partition_by_rows,
                        hive_partitioning=hive_partitioning,
                    ):
                        ctx = Context()
                        dataset = ParquetDataSet(["tests/data/arrow/*.parquet"])
                        data_files = DataSourceNode(ctx, dataset)
                        npartitions = 3
                        data_partitions = DataSetPartitionNode(
                            ctx,
                            (data_files,),
                            npartitions=npartitions,
                            partition_by_rows=partition_by_rows,
                        )
                        hash_partitions = HashPartitionNode(
                            ctx,
                            (ProjectionNode(ctx, data_partitions, ["url"]),),
                            npartitions=npartitions,
                            hash_columns=["url"],
                            engine_type=engine_type,
                            hive_partitioning=hive_partitioning,
                            cpu_limit=2,
                            memory_limit=2 * GB,
                            output_name="hash_partitions",
                        )
                        row_count = SqlEngineNode(
                            ctx,
                            (hash_partitions,),
                            r"select count(*) as row_count from {0}",
                            cpu_limit=1,
                            memory_limit=1 * GB,
                        )
                        plan = LogicalPlan(ctx, row_count)
                        exec_plan = self.execute_plan(plan)
                        self.assertEqual(
                            dataset.num_rows,
                            pc.sum(exec_plan.final_output.to_arrow_table().column("row_count")).as_py(),
                        )
                        self.assertEqual(
                            npartitions,
                            len(exec_plan.final_output.load_partitioned_datasets(npartitions, DATA_PARTITION_COLUMN_NAME)),
                        )
                        self.assertEqual(
                            npartitions,
                            len(
                                exec_plan.get_output("hash_partitions").load_partitioned_datasets(
                                    npartitions,
                                    DATA_PARTITION_COLUMN_NAME,
                                    hive_partitioning,
                                )
                            ),
                        )

    def test_empty_hash_partition(self):
        for engine_type in ("duckdb", "arrow"):
            for partition_by_rows in (False, True):
                for hive_partitioning in (False, True) if engine_type == "duckdb" else (False,):
                    with self.subTest(
                        engine_type=engine_type,
                        partition_by_rows=partition_by_rows,
                        hive_partitioning=hive_partitioning,
                    ):
                        ctx = Context()
                        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
                        data_files = DataSourceNode(ctx, dataset)
                        npartitions = 3
                        npartitions_nested = 4
                        num_rows = 1
                        head_rows = SqlEngineNode(ctx, (data_files,), f"select * from {{0}} limit {num_rows}")
                        data_partitions = DataSetPartitionNode(
                            ctx,
                            (head_rows,),
                            npartitions=npartitions,
                            partition_by_rows=partition_by_rows,
                        )
                        hash_partitions = HashPartitionNode(
                            ctx,
                            (data_partitions,),
                            npartitions=npartitions,
                            hash_columns=["url"],
                            data_partition_column="hash_partitions",
                            engine_type=engine_type,
                            hive_partitioning=hive_partitioning,
                            output_name="hash_partitions",
                            cpu_limit=2,
                            memory_limit=1 * GB,
                        )
                        nested_hash_partitions = HashPartitionNode(
                            ctx,
                            (hash_partitions,),
                            npartitions=npartitions_nested,
                            hash_columns=["url"],
                            data_partition_column="nested_hash_partitions",
                            nested=True,
                            engine_type=engine_type,
                            hive_partitioning=hive_partitioning,
                            output_name="nested_hash_partitions",
                            cpu_limit=2,
                            memory_limit=1 * GB,
                        )
                        select_every_row = SqlEngineNode(
                            ctx,
                            (nested_hash_partitions,),
                            r"select * from {0}",
                            cpu_limit=1,
                            memory_limit=1 * GB,
                        )
                        plan = LogicalPlan(ctx, select_every_row)
                        exec_plan = self.execute_plan(plan, skip_task_with_empty_input=True)
                        self.assertEqual(num_rows, exec_plan.final_output.num_rows)
                        self.assertEqual(
                            npartitions,
                            len(exec_plan.final_output.load_partitioned_datasets(npartitions, "hash_partitions")),
                        )
                        self.assertEqual(
                            npartitions_nested,
                            len(exec_plan.final_output.load_partitioned_datasets(npartitions_nested, "nested_hash_partitions")),
                        )
                        self.assertEqual(
                            npartitions,
                            len(exec_plan.get_output("hash_partitions").load_partitioned_datasets(npartitions, "hash_partitions")),
                        )
                        self.assertEqual(
                            npartitions_nested,
                            len(
                                exec_plan.get_output("nested_hash_partitions").load_partitioned_datasets(npartitions_nested, "nested_hash_partitions")
                            ),
                        )
                        if hive_partitioning:
                            self.assertEqual(
                                npartitions,
                                len(
                                    exec_plan.get_output("hash_partitions").load_partitioned_datasets(
                                        npartitions,
                                        "hash_partitions",
                                        hive_partitioning=True,
                                    )
                                ),
                            )
                            self.assertEqual(
                                npartitions_nested,
                                len(
                                    exec_plan.get_output("nested_hash_partitions").load_partitioned_datasets(
                                        npartitions_nested,
                                        "nested_hash_partitions",
                                        hive_partitioning=True,
                                    )
                                ),
                            )

    def test_load_partitioned_datasets(self):
        def run_test_plan(
            npartitions: int,
            data_partition_column: str,
            engine_type: str,
            hive_partitioning: bool,
        ):
            ctx = Context()
            input_dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
            input_data_files = DataSourceNode(ctx, input_dataset)
            # create hash partitions
            input_partitions = HashPartitionNode(
                ctx,
                (input_data_files,),
                npartitions=npartitions,
                hash_columns=["url"],
                data_partition_column=data_partition_column,
                engine_type=engine_type,
                hive_partitioning=hive_partitioning,
                output_name="input_partitions",
                cpu_limit=1,
                memory_limit=1 * GB,
            )
            split_urls = SqlEngineNode(
                ctx,
                (input_partitions,),
                f"select url, string_split(url, '/')[0] as host from {{0}}",
                cpu_limit=1,
                memory_limit=1 * GB,
            )
            plan = LogicalPlan(ctx, split_urls)
            exec_plan = self.execute_plan(plan)
            self.assertEqual(
                npartitions,
                len(exec_plan.final_output.load_partitioned_datasets(npartitions, data_partition_column)),
            )
            self.assertEqual(
                npartitions,
                len(exec_plan.get_output("input_partitions").load_partitioned_datasets(npartitions, data_partition_column, hive_partitioning)),
            )
            return exec_plan

        npartitions = 5
        data_partition_column = "_human_readable_column_name_"

        for engine_type in ("duckdb", "arrow"):
            with self.subTest(engine_type=engine_type):
                exec_plan1 = run_test_plan(
                    npartitions,
                    data_partition_column,
                    engine_type,
                    hive_partitioning=engine_type == "duckdb",
                )
                exec_plan2 = run_test_plan(
                    npartitions,
                    data_partition_column,
                    engine_type,
                    hive_partitioning=False,
                )

                ctx = Context()
                output1 = DataSourceNode(ctx, dataset=exec_plan1.get_output("input_partitions"))
                output2 = DataSourceNode(ctx, dataset=exec_plan2.get_output("input_partitions"))
                split_urls1 = LoadPartitionedDataSetNode(
                    ctx,
                    (output1,),
                    npartitions=npartitions,
                    data_partition_column=data_partition_column,
                    hive_partitioning=engine_type == "duckdb",
                )
                split_urls2 = LoadPartitionedDataSetNode(
                    ctx,
                    (output2,),
                    npartitions=npartitions,
                    data_partition_column=data_partition_column,
                    hive_partitioning=False,
                )
                split_urls3 = SqlEngineNode(
                    ctx,
                    (split_urls1, split_urls2),
                    f"""
            select split_urls1.url, string_split(split_urls2.url, '/')[0] as host
            from {{0}} as split_urls1
            join {{1}} as split_urls2
            on split_urls1.url = split_urls2.url
          """,
                    cpu_limit=1,
                    memory_limit=1 * GB,
                )
                plan = LogicalPlan(ctx, split_urls3)
                exec_plan3 = self.execute_plan(plan)
                # load each partition as arrow table and compare
                final_output_partitions1 = exec_plan1.final_output.load_partitioned_datasets(npartitions, data_partition_column)
                final_output_partitions3 = exec_plan3.final_output.load_partitioned_datasets(npartitions, data_partition_column)
                self.assertEqual(npartitions, len(final_output_partitions3))
                for x, y in zip(final_output_partitions1, final_output_partitions3):
                    self._compare_arrow_tables(x.to_arrow_table(), y.to_arrow_table())

    def test_nested_partition(self):
        ctx = Context()
        parquet_files = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_source = DataSourceNode(ctx, parquet_files)

        SqlEngineNode.default_cpu_limit = 1
        SqlEngineNode.default_memory_limit = 1 * GB
        initial_reduce = r"select host, count(*) as cnt from {0} group by host"
        combine_reduce_results = r"select host, cast(sum(cnt) as bigint) as cnt from {0} group by host"
        join_query = r"select host, cnt from {0} where (exists (select * from {1} where {1}.host = {0}.host)) and (exists (select * from {2} where {2}.host = {0}.host))"

        partition_by_hosts = HashPartitionNode(
            ctx,
            (data_source,),
            npartitions=3,
            hash_columns=["host"],
            data_partition_column="host_partition",
        )
        partition_by_hosts_x_urls = HashPartitionNode(
            ctx,
            (partition_by_hosts,),
            npartitions=5,
            hash_columns=["url"],
            data_partition_column="url_partition",
            nested=True,
        )
        url_count_by_hosts_x_urls1 = SqlEngineNode(
            ctx,
            (partition_by_hosts_x_urls,),
            initial_reduce,
            output_name="url_count_by_hosts_x_urls1",
        )
        url_count_by_hosts1 = SqlEngineNode(
            ctx,
            (ConsolidateNode(ctx, url_count_by_hosts_x_urls1, ["host_partition"]),),
            combine_reduce_results,
            output_name="url_count_by_hosts1",
        )
        join_count_by_hosts_x_urls1 = SqlEngineNode(
            ctx,
            (url_count_by_hosts_x_urls1, url_count_by_hosts1, data_source),
            join_query,
            output_name="join_count_by_hosts_x_urls1",
        )

        partitioned_urls = LoadPartitionedDataSetNode(
            ctx,
            (partition_by_hosts_x_urls,),
            data_partition_column="url_partition",
            npartitions=5,
        )
        partitioned_hosts_x_urls = LoadPartitionedDataSetNode(
            ctx,
            (partitioned_urls,),
            data_partition_column="host_partition",
            npartitions=3,
            nested=True,
        )
        partitioned_3dims = EvenlyDistributedPartitionNode(
            ctx,
            (partitioned_hosts_x_urls,),
            npartitions=2,
            dimension="inner_partition",
            partition_by_rows=True,
            nested=True,
        )
        url_count_by_3dims = SqlEngineNode(ctx, (partitioned_3dims,), initial_reduce)
        url_count_by_hosts_x_urls2 = SqlEngineNode(
            ctx,
            (ConsolidateNode(ctx, url_count_by_3dims, ["host_partition", "url_partition"]),),
            combine_reduce_results,
            output_name="url_count_by_hosts_x_urls2",
        )
        url_count_by_hosts2 = SqlEngineNode(
            ctx,
            (ConsolidateNode(ctx, url_count_by_hosts_x_urls2, ["host_partition"]),),
            combine_reduce_results,
            output_name="url_count_by_hosts2",
        )
        url_count_by_hosts_expected = SqlEngineNode(
            ctx,
            (data_source,),
            initial_reduce,
            per_thread_output=False,
            output_name="url_count_by_hosts_expected",
        )
        join_count_by_hosts_x_urls2 = SqlEngineNode(
            ctx,
            (url_count_by_hosts_x_urls2, url_count_by_hosts2, data_source),
            join_query,
            output_name="join_count_by_hosts_x_urls2",
        )

        union_url_count_by_hosts = UnionNode(ctx, (url_count_by_hosts1, url_count_by_hosts2))
        union_url_count_by_hosts_x_urls = UnionNode(
            ctx,
            (
                url_count_by_hosts_x_urls1,
                url_count_by_hosts_x_urls2,
                join_count_by_hosts_x_urls1,
                join_count_by_hosts_x_urls2,
            ),
        )

        with tempfile.TemporaryDirectory(dir=self.output_root_abspath) as output_dir:
            data_sink = DataSinkNode(
                ctx,
                (
                    url_count_by_hosts_expected,
                    union_url_count_by_hosts,
                    union_url_count_by_hosts_x_urls,
                ),
                output_path=output_dir,
                manifest_only=True,
            )
            plan = LogicalPlan(ctx, data_sink)
            exec_plan = self.execute_plan(plan, remove_empty_parquet=True)
            # verify results
            self._compare_arrow_tables(
                exec_plan.get_output("url_count_by_hosts_x_urls1").to_arrow_table(),
                exec_plan.get_output("url_count_by_hosts_x_urls2").to_arrow_table(),
            )
            self._compare_arrow_tables(
                exec_plan.get_output("join_count_by_hosts_x_urls1").to_arrow_table(),
                exec_plan.get_output("join_count_by_hosts_x_urls2").to_arrow_table(),
            )
            self._compare_arrow_tables(
                exec_plan.get_output("url_count_by_hosts_x_urls1").to_arrow_table(),
                exec_plan.get_output("join_count_by_hosts_x_urls1").to_arrow_table(),
            )
            self._compare_arrow_tables(
                exec_plan.get_output("url_count_by_hosts1").to_arrow_table(),
                exec_plan.get_output("url_count_by_hosts2").to_arrow_table(),
            )
            self._compare_arrow_tables(
                exec_plan.get_output("url_count_by_hosts_expected").to_arrow_table(),
                exec_plan.get_output("url_count_by_hosts1").to_arrow_table(),
            )

    def test_user_defined_partition(self):
        ctx = Context()
        parquet_files = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_source = DataSourceNode(ctx, parquet_files)
        file_partitions1 = CalculatePartitionFromFilename(ctx, (data_source,), npartitions=3, dimension="by_filename_hash1")
        url_count1 = SqlEngineNode(
            ctx,
            (file_partitions1,),
            r"select host, count(*) as cnt from {0} group by host",
            output_name="url_count1",
        )
        file_partitions2 = CalculatePartitionFromFilename(ctx, (url_count1,), npartitions=3, dimension="by_filename_hash2")
        url_count2 = SqlEngineNode(
            ctx,
            (file_partitions2,),
            r"select host, cnt from {0}",
            output_name="url_count2",
        )
        plan = LogicalPlan(ctx, url_count2)

        exec_plan = self.execute_plan(plan, enable_diagnostic_metrics=True)
        self._compare_arrow_tables(
            exec_plan.get_output("url_count1").to_arrow_table(),
            exec_plan.get_output("url_count2").to_arrow_table(),
        )

    def test_user_partitioned_data_source(self):
        ctx = Context()
        parquet_dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_source = DataSourceNode(ctx, parquet_dataset)
        evenly_dist_data_source = EvenlyDistributedPartitionNode(ctx, (data_source,), npartitions=parquet_dataset.num_files)

        parquet_datasets = [ParquetDataSet([p]) for p in parquet_dataset.resolved_paths]
        partitioned_data_source = UserPartitionedDataSourceNode(ctx, parquet_datasets)

        url_count_by_host1 = SqlEngineNode(
            ctx,
            (evenly_dist_data_source,),
            r"select host, count(*) as cnt from {0} group by host",
            output_name="url_count_by_host1",
            cpu_limit=1,
            memory_limit=1 * GB,
        )

        url_count_by_host2 = SqlEngineNode(
            ctx,
            (evenly_dist_data_source, partitioned_data_source),
            r"select {1}.host, count(*) as cnt from {0} join {1} on {0}.host = {1}.host group by {1}.host",
            output_name="url_count_by_host2",
            cpu_limit=1,
            memory_limit=1 * GB,
        )

        plan = LogicalPlan(ctx, UnionNode(ctx, [url_count_by_host1, url_count_by_host2]))
        exec_plan = self.execute_plan(plan, enable_diagnostic_metrics=True)
        self._compare_arrow_tables(
            exec_plan.get_output("url_count_by_host1").to_arrow_table(),
            exec_plan.get_output("url_count_by_host2").to_arrow_table(),
        )

    def test_partition_info_in_sql_query(self):
        """
        User can refer to the partition info in the SQL query.
        """
        ctx = Context()
        parquet_dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_source = DataSourceNode(ctx, parquet_dataset)
        evenly_dist_data_source = EvenlyDistributedPartitionNode(ctx, (data_source,), npartitions=parquet_dataset.num_files)
        sql_query = SqlEngineNode(
            ctx,
            (evenly_dist_data_source,),
            r"select host, {__data_partition__} as partition_info from {0}",
        )
        plan = LogicalPlan(ctx, sql_query)
        exec_plan = self.execute_plan(plan)
