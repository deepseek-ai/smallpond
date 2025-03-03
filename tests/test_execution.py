import functools
import os.path
import socket
import tempfile
import time
import unittest
from datetime import datetime
from typing import Iterable, List, Tuple

import pandas
import pyarrow as arrow
from loguru import logger
from pandas.core.api import DataFrame as DataFrame

from smallpond.common import GB, MB, split_into_rows
from smallpond.execution.task import (
    DataSinkTask,
    DataSourceTask,
    JobId,
    PartitionInfo,
    PythonScriptTask,
    RuntimeContext,
    StreamOutput,
)
from smallpond.execution.workqueue import WorkStatus
from smallpond.logical.dataset import (
    ArrowTableDataSet,
    DataSet,
    ParquetDataSet,
    SqlQueryDataSet,
)
from smallpond.logical.node import (
    ArrowBatchNode,
    ArrowComputeNode,
    ArrowStreamNode,
    Context,
    DataSetPartitionNode,
    DataSinkNode,
    DataSourceNode,
    EvenlyDistributedPartitionNode,
    HashPartitionNode,
    LogicalPlan,
    Node,
    PandasBatchNode,
    PandasComputeNode,
    ProjectionNode,
    PythonScriptNode,
    RootNode,
    SqlEngineNode,
)
from smallpond.logical.udf import UDFListType, UDFType
from tests.test_fabric import TestFabric


class OutputMsgPythonTask(PythonScriptTask):
    def __init__(self, msg: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.msg = msg

    def initialize(self):
        pass

    def finalize(self):
        pass

    def process(
        self,
        runtime_ctx: RuntimeContext,
        input_datasets: List[DataSet],
        output_path: str,
    ) -> bool:
        logger.info(f"msg: {self.msg}, num files: {input_datasets[0].num_files}, local gpu ranks: {self.local_gpu_ranks}")
        self.inject_fault()
        return True


# method1: inherit Task class and override spawn method
class OutputMsgPythonNode(PythonScriptNode):
    def spawn(self, *args, **kwargs) -> OutputMsgPythonTask:
        return OutputMsgPythonTask("python script", *args, **kwargs)


# method2: override process method
# this usage is not recommended and only for testing. use `process_func` instead.
class OutputMsgPythonNode2(PythonScriptNode):
    def __init__(self, ctx: Context, input_deps: Tuple[Node, ...], msg: str) -> None:
        super().__init__(ctx, input_deps)
        self.msg = msg

    def process(
        self,
        runtime_ctx: RuntimeContext,
        input_datasets: List[DataSet],
        output_path: str,
    ) -> bool:
        logger.info(f"msg: {self.msg}, num files: {input_datasets[0].num_files}")
        return True


# this usage is not recommended and only for testing. use `process_func` instead.
class CopyInputArrowNode(ArrowComputeNode):
    def __init__(self, ctx: Context, input_deps: Tuple[Node, ...], msg: str) -> None:
        super().__init__(ctx, input_deps)
        self.msg = msg

    def process(self, runtime_ctx: RuntimeContext, input_tables: List[arrow.Table]) -> arrow.Table:
        return copy_input_arrow(runtime_ctx, input_tables, self.msg)


# this usage is not recommended and only for testing. use `process_func` instead.
class CopyInputStreamNode(ArrowStreamNode):
    def __init__(self, ctx: Context, input_deps: Tuple[Node, ...], msg: str) -> None:
        super().__init__(ctx, input_deps)
        self.msg = msg

    def process(self, runtime_ctx: RuntimeContext, input_readers: List[arrow.RecordBatchReader]) -> Iterable[arrow.Table]:
        return copy_input_stream(runtime_ctx, input_readers, self.msg)


def copy_input_arrow(runtime_ctx: RuntimeContext, input_tables: List[arrow.Table], msg: str) -> arrow.Table:
    logger.info(f"msg: {msg}, num rows: {input_tables[0].num_rows}")
    time.sleep(runtime_ctx.secs_executor_probe_interval)
    runtime_ctx.task.inject_fault()
    return input_tables[0]


def copy_input_stream(runtime_ctx: RuntimeContext, input_readers: List[arrow.RecordBatchReader], msg: str) -> Iterable[arrow.Table]:
    for index, batch in enumerate(input_readers[0]):
        logger.info(f"msg: {msg}, batch index: {index}, num rows: {batch.num_rows}")
        time.sleep(runtime_ctx.secs_executor_probe_interval)
        yield StreamOutput(
            arrow.Table.from_batches([batch]),
            batch_indices=[index],
            force_checkpoint=True,
        )
        runtime_ctx.task.inject_fault()


def copy_input_batch(runtime_ctx: RuntimeContext, input_batches: List[arrow.Table], msg: str) -> arrow.Table:
    logger.info(f"msg: {msg}, num rows: {input_batches[0].num_rows}")
    time.sleep(runtime_ctx.secs_executor_probe_interval)
    runtime_ctx.task.inject_fault()
    return input_batches[0]


def copy_input_data_frame(runtime_ctx: RuntimeContext, input_dfs: List[DataFrame]) -> DataFrame:
    runtime_ctx.task.inject_fault()
    return input_dfs[0]


def copy_input_data_frame_batch(runtime_ctx: RuntimeContext, input_dfs: List[DataFrame]) -> DataFrame:
    runtime_ctx.task.inject_fault()
    return input_dfs[0]


def merge_input_tables(runtime_ctx: RuntimeContext, input_batches: List[arrow.Table]) -> arrow.Table:
    runtime_ctx.task.inject_fault()
    output = arrow.concat_tables(input_batches)
    logger.info(f"input rows: {[len(batch) for batch in input_batches]}, output rows: {len(output)}")
    return output


def merge_input_data_frames(runtime_ctx: RuntimeContext, input_dfs: List[DataFrame]) -> DataFrame:
    runtime_ctx.task.inject_fault()
    output = pandas.concat(input_dfs)
    logger.info(f"input rows: {[len(df) for df in input_dfs]}, output rows: {len(output)}")
    return output


def parse_url(runtime_ctx: RuntimeContext, input_tables: List[arrow.Table]) -> arrow.Table:
    urls = input_tables[0].columns[0]
    hosts = [url.as_py().split("/", maxsplit=2)[0] for url in urls]
    return input_tables[0].append_column("host", arrow.array(hosts))


def nonzero_exit_code(runtime_ctx: RuntimeContext, input_datasets: List[DataSet], output_path: str) -> bool:
    import sys

    if runtime_ctx.task._memory_boost == 1:
        sys.exit(1)
    return True


# create an empty file with a fixed name
def empty_file(runtime_ctx: RuntimeContext, input_datasets: List[DataSet], output_path: str) -> bool:
    import os

    with open(os.path.join(output_path, "file"), "w") as fout:
        pass
    return True


def return_fake_gpus(count: int = 8):
    import GPUtil

    return [GPUtil.GPU(i, *list(range(11))) for i in range(count)]


def split_url(urls: arrow.array) -> arrow.array:
    url_parts = [url.as_py().split("/") for url in urls]
    return arrow.array(url_parts, type=arrow.list_(arrow.string()))


def choose_random_urls(runtime_ctx: RuntimeContext, input_tables: List[arrow.Table], k: int = 5) -> arrow.Table:
    # get the current running task
    runtime_task = runtime_ctx.task
    # access task-specific attributes
    cpu_limit = runtime_task.cpu_limit
    random_gen = runtime_task.python_random_gen
    # input data
    (url_table,) = input_tables
    hosts, urls = url_table.columns
    logger.info(f"{cpu_limit=} {len(urls)=}")
    # generate ramdom samples
    random_urls = random_gen.choices(urls.to_pylist(), k=k)
    return arrow.Table.from_arrays([arrow.array(random_urls)], names=["random_urls"])


class TestExecution(TestFabric, unittest.TestCase):

    fault_inject_prob = 0.05

    def test_arrow_task(self):
        for use_duckdb_reader in (False, True):
            with self.subTest(use_duckdb_reader=use_duckdb_reader):
                with tempfile.TemporaryDirectory(dir=self.output_root_abspath) as output_dir:
                    ctx = Context()
                    dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
                    data_table = dataset.to_arrow_table()
                    data_files = DataSourceNode(ctx, dataset)
                    data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=7)
                    if use_duckdb_reader:
                        data_partitions = ProjectionNode(
                            ctx,
                            data_partitions,
                            columns=["*", "string_split(url, '/')[0] as parsed_host"],
                        )
                    arrow_compute = ArrowComputeNode(
                        ctx,
                        (data_partitions,),
                        process_func=functools.partial(copy_input_arrow, msg="arrow compute"),
                        use_duckdb_reader=use_duckdb_reader,
                        output_name="arrow_compute",
                        output_path=output_dir,
                        cpu_limit=2,
                    )
                    arrow_stream = ArrowStreamNode(
                        ctx,
                        (data_partitions,),
                        process_func=functools.partial(copy_input_stream, msg="arrow stream"),
                        streaming_batch_size=10,
                        secs_checkpoint_interval=0.5,
                        use_duckdb_reader=use_duckdb_reader,
                        output_name="arrow_stream",
                        output_path=output_dir,
                        cpu_limit=2,
                    )
                    arrow_batch = ArrowBatchNode(
                        ctx,
                        (data_partitions,),
                        process_func=functools.partial(copy_input_batch, msg="arrow batch"),
                        streaming_batch_size=10,
                        secs_checkpoint_interval=0.5,
                        use_duckdb_reader=use_duckdb_reader,
                        output_name="arrow_batch",
                        output_path=output_dir,
                        cpu_limit=2,
                    )
                    data_sink = DataSinkNode(
                        ctx,
                        (arrow_compute, arrow_stream, arrow_batch),
                        output_path=output_dir,
                    )
                    plan = LogicalPlan(ctx, data_sink)
                    exec_plan = self.execute_plan(plan, fault_inject_prob=0.1, secs_executor_probe_interval=0.5)
                    self.assertTrue(all(map(os.path.exists, exec_plan.final_output.resolved_paths)))
                    arrow_compute_output = ParquetDataSet(
                        [os.path.join(output_dir, "arrow_compute", "**/*.parquet")],
                        recursive=True,
                    )
                    arrow_stream_output = ParquetDataSet(
                        [os.path.join(output_dir, "arrow_stream", "**/*.parquet")],
                        recursive=True,
                    )
                    arrow_batch_output = ParquetDataSet(
                        [os.path.join(output_dir, "arrow_batch", "**/*.parquet")],
                        recursive=True,
                    )
                    self._compare_arrow_tables(
                        data_table,
                        arrow_compute_output.to_arrow_table().select(data_table.column_names),
                    )
                    self._compare_arrow_tables(
                        data_table,
                        arrow_stream_output.to_arrow_table().select(data_table.column_names),
                    )
                    self._compare_arrow_tables(
                        data_table,
                        arrow_batch_output.to_arrow_table().select(data_table.column_names),
                    )

    def test_pandas_task(self):
        with tempfile.TemporaryDirectory(dir=self.output_root_abspath) as output_dir:
            ctx = Context()
            dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
            data_table = dataset.to_arrow_table()
            data_files = DataSourceNode(ctx, dataset)
            data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=7)
            pandas_compute = PandasComputeNode(
                ctx,
                (data_partitions,),
                process_func=copy_input_data_frame,
                output_name="pandas_compute",
                output_path=output_dir,
                cpu_limit=2,
            )
            pandas_batch = PandasBatchNode(
                ctx,
                (data_partitions,),
                process_func=copy_input_data_frame_batch,
                streaming_batch_size=10,
                secs_checkpoint_interval=0.5,
                output_name="pandas_batch",
                output_path=output_dir,
                cpu_limit=2,
            )
            data_sink = DataSinkNode(ctx, (pandas_compute, pandas_batch), output_path=output_dir)
            plan = LogicalPlan(ctx, data_sink)
            exec_plan = self.execute_plan(plan, fault_inject_prob=0.1, secs_executor_probe_interval=0.5)
            self.assertTrue(all(map(os.path.exists, exec_plan.final_output.resolved_paths)))
            pandas_compute_output = ParquetDataSet(
                [os.path.join(output_dir, "pandas_compute", "**/*.parquet")],
                recursive=True,
            )
            pandas_batch_output = ParquetDataSet(
                [os.path.join(output_dir, "pandas_batch", "**/*.parquet")],
                recursive=True,
            )
            self._compare_arrow_tables(data_table, pandas_compute_output.to_arrow_table())
            self._compare_arrow_tables(data_table, pandas_batch_output.to_arrow_table())

    def test_variable_length_input_datasets(self):
        ctx = Context()
        small_dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        large_dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"] * 10)
        small_partitions = DataSetPartitionNode(ctx, (DataSourceNode(ctx, small_dataset),), npartitions=7)
        large_partitions = DataSetPartitionNode(ctx, (DataSourceNode(ctx, large_dataset),), npartitions=7)
        arrow_batch = ArrowBatchNode(
            ctx,
            (small_partitions, large_partitions),
            process_func=merge_input_tables,
            streaming_batch_size=100,
            secs_checkpoint_interval=0.5,
            output_name="arrow_batch",
            cpu_limit=2,
        )
        pandas_batch = PandasBatchNode(
            ctx,
            (small_partitions, large_partitions),
            process_func=merge_input_data_frames,
            streaming_batch_size=100,
            secs_checkpoint_interval=0.5,
            output_name="pandas_batch",
            cpu_limit=2,
        )
        plan = LogicalPlan(ctx, RootNode(ctx, (arrow_batch, pandas_batch)))
        exec_plan = self.execute_plan(plan, fault_inject_prob=0.1, secs_executor_probe_interval=0.5)
        self.assertTrue(all(map(os.path.exists, exec_plan.final_output.resolved_paths)))
        arrow_batch_output = ParquetDataSet(
            [os.path.join(exec_plan.ctx.output_root, "arrow_batch", "**/*.parquet")],
            recursive=True,
        )
        pandas_batch_output = ParquetDataSet(
            [os.path.join(exec_plan.ctx.output_root, "pandas_batch", "**/*.parquet")],
            recursive=True,
        )
        self.assertEqual(small_dataset.num_rows + large_dataset.num_rows, arrow_batch_output.num_rows)
        self.assertEqual(
            small_dataset.num_rows + large_dataset.num_rows,
            pandas_batch_output.num_rows,
        )

    def test_projection_task(self):
        ctx = Context()
        # select columns when defining dataset
        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"], columns=["url"])
        data_files = DataSourceNode(ctx, dataset)
        data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=3, partition_by_rows=True)
        # projection as input of arrow node
        generated_columns = ["filename", "file_row_number"]
        urls_with_host = ArrowComputeNode(
            ctx,
            (ProjectionNode(ctx, data_partitions, ["url"], generated_columns),),
            process_func=parse_url,
            use_duckdb_reader=True,
        )
        # projection as input of sql node
        distinct_urls_with_host = SqlEngineNode(
            ctx,
            (
                ProjectionNode(
                    ctx,
                    data_partitions,
                    ["url", "string_split(url, '/')[0] as host"],
                    generated_columns,
                ),
            ),
            r"select distinct host, url, filename from {0}",
        )
        # unify different schemas
        merged_diff_schemas = ProjectionNode(
            ctx,
            DataSetPartitionNode(ctx, (distinct_urls_with_host, urls_with_host), npartitions=1),
            union_by_name=True,
        )
        host_partitions = HashPartitionNode(
            ctx,
            (merged_diff_schemas,),
            npartitions=3,
            hash_columns=["host"],
            engine_type="duckdb",
            output_name="host_partitions",
        )
        host_partitions.max_num_producer_tasks = 1
        plan = LogicalPlan(ctx, host_partitions)
        final_output = self.execute_plan(plan, fault_inject_prob=0.1).final_output
        final_table = final_output.to_arrow_table()
        self.assertEqual(
            sorted(
                [
                    "url",
                    "host",
                    *generated_columns,
                    HashPartitionNode.default_data_partition_column,
                ]
            ),
            sorted(final_table.column_names),
        )

    def test_arrow_type_in_udfs(self):
        ctx = Context()
        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_files = DataSourceNode(ctx, dataset)
        data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=dataset.num_files)
        ctx.create_function(
            "split_url",
            split_url,
            [UDFType.VARCHAR],
            UDFListType(UDFType.VARCHAR),
            use_arrow_type=True,
        )
        uniq_hosts = SqlEngineNode(
            ctx,
            (data_partitions,),
            r"select split_url(url) as url_parts from {0}",
            udfs=["split_url"],
        )
        plan = LogicalPlan(ctx, uniq_hosts)
        self.execute_plan(plan)

    def test_many_simple_tasks(self):
        ctx = Context()
        npartitions = 1000
        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"] * npartitions)
        data_files = DataSourceNode(ctx, dataset)
        data_partitions = EvenlyDistributedPartitionNode(ctx, (data_files,), npartitions=npartitions)
        output_msg = OutputMsgPythonNode(ctx, (data_partitions,))
        plan = LogicalPlan(ctx, output_msg)
        self.execute_plan(
            plan,
            num_executors=10,
            secs_executor_probe_interval=5,
            enable_profiling=True,
        )

    def test_many_producers_and_partitions(self):
        ctx = Context()
        npartitions = 10000
        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"] * (npartitions * 10))
        data_files = DataSourceNode(ctx, dataset)
        data_partitions = EvenlyDistributedPartitionNode(ctx, (data_files,), npartitions=npartitions, cpu_limit=1)
        data_partitions.max_num_producer_tasks = 20
        output_msg = OutputMsgPythonNode(ctx, (data_partitions,))
        plan = LogicalPlan(ctx, output_msg)
        self.execute_plan(
            plan,
            num_executors=10,
            secs_executor_probe_interval=5,
            enable_profiling=True,
        )

    def test_local_gpu_rank(self):
        ctx = Context()
        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_files = DataSourceNode(ctx, dataset)
        data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=dataset.num_files)
        output_msg = OutputMsgPythonNode(ctx, (data_partitions,), cpu_limit=1, gpu_limit=0.5)
        plan = LogicalPlan(ctx, output_msg)
        runtime_ctx = RuntimeContext(
            JobId.new(),
            datetime.now(),
            self.output_root_abspath,
            console_log_level="WARNING",
        )
        runtime_ctx.get_local_gpus = return_fake_gpus
        runtime_ctx.initialize(socket.gethostname(), cleanup_root=True)
        self.execute_plan(plan, runtime_ctx=runtime_ctx)

    def test_python_node_with_process_method(self):
        ctx = Context()
        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_files = DataSourceNode(ctx, dataset)
        copy_input_arrow_node = CopyInputArrowNode(ctx, (data_files,), "hello")
        copy_input_stream_node = CopyInputStreamNode(ctx, (data_files,), "hello")
        output_msg = OutputMsgPythonNode2(ctx, (copy_input_arrow_node, copy_input_stream_node), "hello")
        plan = LogicalPlan(ctx, output_msg)
        self.execute_plan(plan)

    def test_sql_engine_oom(self):
        ctx = Context()
        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_files = DataSourceNode(ctx, dataset)
        uniq_urls = SqlEngineNode(ctx, (data_files,), r"select distinct * from {0}", memory_limit=2 * MB)
        uniq_url_partitions = DataSetPartitionNode(ctx, (uniq_urls,), 2)
        uniq_url_count = SqlEngineNode(
            ctx,
            (uniq_url_partitions,),
            sql_query=r"select count(distinct columns(*)) from {0}",
            memory_limit=2 * MB,
        )
        plan = LogicalPlan(ctx, uniq_url_count)
        self.execute_plan(plan, max_fail_count=10)

    @unittest.skip("flaky on CI")
    def test_enforce_memory_limit(self):
        ctx = Context()
        dataset = ParquetDataSet(["tests/data/arrow/*.parquet"])
        data_files = DataSourceNode(ctx, dataset)
        arrow_compute = ArrowComputeNode(
            ctx,
            (data_files,),
            process_func=functools.partial(copy_input_arrow, msg="arrow compute"),
            memory_limit=1 * GB,
        )
        arrow_stream = ArrowStreamNode(
            ctx,
            (data_files,),
            process_func=functools.partial(copy_input_stream, msg="arrow stream"),
            memory_limit=1 * GB,
        )
        with tempfile.TemporaryDirectory(dir=self.output_root_abspath) as output_dir:
            data_sink = DataSinkNode(ctx, (arrow_compute, arrow_stream), output_path=output_dir)
            plan = LogicalPlan(ctx, data_sink)
            self.execute_plan(
                plan,
                max_fail_count=10,
                enforce_memory_limit=True,
                nonzero_exitcode_as_oom=True,
            )

    def test_task_crash_as_oom(self):
        ctx = Context()
        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_files = DataSourceNode(ctx, dataset)
        nonzero_exitcode = PythonScriptNode(ctx, (data_files,), process_func=nonzero_exit_code)
        plan = LogicalPlan(ctx, nonzero_exitcode)
        exec_plan = self.execute_plan(plan, num_executors=1, check_result=False, nonzero_exitcode_as_oom=False)
        self.assertFalse(exec_plan.successful)
        exec_plan = self.execute_plan(plan, num_executors=1, check_result=False, nonzero_exitcode_as_oom=True)
        self.assertTrue(exec_plan.successful)

    def test_manifest_only_data_sink(self):
        with open("tests/data/long_path_list.txt", buffering=16 * MB) as fin:
            filenames = list(map(str.strip, fin.readlines()))
            logger.info(f"loaded {len(filenames)} filenames")

        with tempfile.TemporaryDirectory(dir=self.output_root_abspath) as output_dir:
            ctx = Context()
            dataset = ParquetDataSet(filenames)
            data_files = DataSourceNode(ctx, dataset)
            data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=512)
            data_sink = DataSinkNode(ctx, (data_partitions,), output_path=output_dir, manifest_only=True)
            plan = LogicalPlan(ctx, data_sink)
            self.execute_plan(plan)

            with open(
                os.path.join(output_dir, DataSinkTask.manifest_filename),
                buffering=16 * MB,
            ) as fin:
                num_lines = len(fin.readlines())
            self.assertEqual(len(filenames), num_lines)

    def test_sql_batched_processing(self):
        for materialize_in_memory in (False, True):
            with self.subTest(materialize_in_memory=materialize_in_memory):
                ctx = Context()
                dataset = ParquetDataSet(["tests/data/large_array/*.parquet"] * 2)
                data_files = DataSourceNode(ctx, dataset)
                content_length = SqlEngineNode(
                    ctx,
                    (data_files,),
                    r"select url, octet_length(content) as content_len from {0}",
                    materialize_in_memory=materialize_in_memory,
                    batched_processing=True,
                    cpu_limit=2,
                    memory_limit=2 * GB,
                )
                plan = LogicalPlan(ctx, content_length)
                final_output: ParquetDataSet = self.execute_plan(plan).final_output
                self.assertEqual(dataset.num_rows, final_output.num_rows)

    def test_multiple_sql_queries(self):
        for materialize_in_memory in (False, True):
            with self.subTest(materialize_in_memory=materialize_in_memory):
                ctx = Context()
                dataset = ParquetDataSet(["tests/data/large_array/*.parquet"] * 2)
                data_files = DataSourceNode(ctx, dataset)
                content_length = SqlEngineNode(
                    ctx,
                    (data_files,),
                    [
                        r"create or replace temp table content_len_data as select url, octet_length(content) as content_len from {0}",
                        r"select * from content_len_data",
                    ],
                    materialize_in_memory=materialize_in_memory,
                    batched_processing=True,
                    cpu_limit=2,
                    memory_limit=2 * GB,
                )
                plan = LogicalPlan(ctx, content_length)
                final_output: ParquetDataSet = self.execute_plan(plan).final_output
                self.assertEqual(dataset.num_rows, final_output.num_rows)

    def test_temp_outputs_in_final_results(self):
        ctx = Context()
        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_files = DataSourceNode(ctx, dataset)
        data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=10)
        url_counts = SqlEngineNode(ctx, (data_partitions,), r"select count(url) as cnt from {0}")
        distinct_url_counts = SqlEngineNode(ctx, (data_partitions,), r"select count(distinct url) as cnt from {0}")
        merged_counts = DataSetPartitionNode(
            ctx,
            (
                ProjectionNode(ctx, url_counts, ["cnt"]),
                ProjectionNode(ctx, distinct_url_counts, ["cnt"]),
            ),
            npartitions=1,
        )
        split_counts = DataSetPartitionNode(ctx, (merged_counts,), npartitions=10)
        plan = LogicalPlan(ctx, split_counts)
        final_output: ParquetDataSet = self.execute_plan(plan).final_output
        self.assertEqual(data_partitions.npartitions * 2, final_output.num_rows)

    def test_override_output_path(self):
        ctx = Context()
        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_files = DataSourceNode(ctx, dataset)
        data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=10)
        url_counts = SqlEngineNode(
            ctx,
            (data_partitions,),
            r"select count(url) as cnt from {0}",
            output_name="url_counts",
        )
        distinct_url_counts = SqlEngineNode(ctx, (data_partitions,), r"select count(distinct url) as cnt from {0}")
        merged_counts = DataSetPartitionNode(
            ctx,
            (
                ProjectionNode(ctx, url_counts, ["cnt"]),
                ProjectionNode(ctx, distinct_url_counts, ["cnt"]),
            ),
            npartitions=1,
        )
        plan = LogicalPlan(ctx, merged_counts)

        output_path = os.path.join(self.runtime_ctx.output_root, "final_output")
        final_output = self.execute_plan(plan, output_path=output_path).final_output
        self.assertTrue(os.path.exists(os.path.join(output_path, "url_counts")))
        self.assertTrue(os.path.exists(os.path.join(output_path, "FinalResults")))

    def test_data_sink_avoid_filename_conflicts(self):
        ctx = Context()
        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_files = DataSourceNode(ctx, dataset)
        data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=10)
        empty_files1 = PythonScriptNode(ctx, (data_partitions,), process_func=empty_file)
        empty_files2 = PythonScriptNode(ctx, (data_partitions,), process_func=empty_file)
        link_path = os.path.join(self.runtime_ctx.output_root, "link")
        copy_path = os.path.join(self.runtime_ctx.output_root, "copy")
        copy_input_path = os.path.join(self.runtime_ctx.output_root, "copy_input")
        data_link = DataSinkNode(ctx, (empty_files1, empty_files2), type="link", output_path=link_path)
        data_copy = DataSinkNode(ctx, (empty_files1, empty_files2), type="copy", output_path=copy_path)
        data_copy_input = DataSinkNode(ctx, (data_partitions,), type="copy", output_path=copy_input_path)
        plan = LogicalPlan(ctx, RootNode(ctx, (data_link, data_copy, data_copy_input)))

        self.execute_plan(plan)
        # there should be 21 files (20 input files + 1 manifest file) in the sink dir
        self.assertEqual(21, len(os.listdir(link_path)))
        self.assertEqual(21, len(os.listdir(copy_path)))
        # file name should not be modified if no conflict
        self.assertEqual(
            set(filename for filename in os.listdir("tests/data/mock_urls") if filename.endswith(".parquet")),
            set(filename for filename in os.listdir(copy_input_path) if filename.endswith(".parquet")),
        )

    def test_literal_datasets_as_data_sources(self):
        ctx = Context()
        num_rows = 10
        query_dataset = SqlQueryDataSet(f"select i from range({num_rows}) as x(i)")
        table_dataset = ArrowTableDataSet(arrow.Table.from_arrays([list(range(num_rows))], names=["i"]))
        query_source = DataSourceNode(ctx, query_dataset)
        table_source = DataSourceNode(ctx, table_dataset)
        query_partitions = DataSetPartitionNode(ctx, (query_source,), npartitions=num_rows, partition_by_rows=True)
        table_partitions = DataSetPartitionNode(ctx, (table_source,), npartitions=num_rows, partition_by_rows=True)
        joined_rows = SqlEngineNode(
            ctx,
            (query_partitions, table_partitions),
            r"select a.i as i, b.i as j from {0} as a join {1} as b on a.i = b.i",
        )
        plan = LogicalPlan(ctx, joined_rows)
        final_output: ParquetDataSet = self.execute_plan(plan).final_output
        self.assertEqual(num_rows, final_output.num_rows)

    def test_partial_process_func(self):
        ctx = Context()
        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_files = DataSourceNode(ctx, dataset)
        data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=3)
        # use default value of k
        random_urls_k5 = ArrowComputeNode(
            ctx,
            (data_partitions,),
            process_func=choose_random_urls,
            output_name="random_urls_k5",
        )
        # set value of k using functools.partial
        random_urls_k10 = ArrowComputeNode(
            ctx,
            (data_partitions,),
            process_func=functools.partial(choose_random_urls, k=10),
            output_name="random_urls_k10",
        )
        random_urls_all = SqlEngineNode(
            ctx,
            (random_urls_k5, random_urls_k10),
            r"select * from {0} union select * from {1}",
            output_name="random_urls_all",
        )
        plan = LogicalPlan(ctx, random_urls_all)
        exec_plan = self.execute_plan(plan)
        self.assertEqual(
            data_partitions.npartitions * 5,
            exec_plan.get_output("random_urls_k5").to_arrow_table().num_rows,
        )
        self.assertEqual(
            data_partitions.npartitions * 10,
            exec_plan.get_output("random_urls_k10").to_arrow_table().num_rows,
        )
