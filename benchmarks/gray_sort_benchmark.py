import functools
import logging
import os.path
import shutil
import subprocess
import tempfile
from pathlib import PurePath
from typing import Iterable, List

import duckdb
import polars
import psutil
import pyarrow as arrow
import pyarrow.compute as pc

from smallpond.common import GB, MB, next_power_of_two, pytest_running
from smallpond.execution.driver import Driver
from smallpond.execution.task import (
    ArrowStreamTask,
    PythonScriptTask,
    RuntimeContext,
    StreamOutput,
)
from smallpond.logical.dataset import ArrowTableDataSet, DataSet, ParquetDataSet
from smallpond.logical.node import (
    ArrowStreamNode,
    Context,
    DataSetPartitionNode,
    DataSourceNode,
    LogicalPlan,
    ProjectionNode,
    PythonScriptNode,
    ShuffleNode,
)


class SortBenchTool(object):

    gensort_path = shutil.which("gensort")
    valsort_path = shutil.which("valsort")

    @staticmethod
    def ensure_installed():
        if not SortBenchTool.gensort_path or not SortBenchTool.valsort_path:
            raise Exception("gensort or valsort not found")


def generate_records(
    runtime_ctx: RuntimeContext,
    input_readers: List[arrow.RecordBatchReader],
    record_nbytes=100,
    key_nbytes=10,
    bucket_nbits=12,
    gensort_batch_nbytes=500 * MB,
) -> Iterable[arrow.Table]:
    runtime_task: ArrowStreamTask = runtime_ctx.task
    batch_size = gensort_batch_nbytes // record_nbytes
    schema = arrow.schema(
        [
            arrow.field("buckets", arrow.uint16()),
            arrow.field("keys", arrow.binary()),
            arrow.field("records", arrow.binary()),
        ]
    )

    with tempfile.NamedTemporaryFile(dir="/dev/shm", buffering=0) as shm_file:
        for batch_idx, batch in enumerate(input_readers[0]):
            for begin_at, num_records in zip(*batch.columns):
                begin_at, num_records = begin_at.as_py(), num_records.as_py()
                for offset in range(begin_at, begin_at + num_records, batch_size):
                    record_count = min(batch_size, begin_at + num_records - offset)
                    gensort_cmd = f"{SortBenchTool.gensort_path} -t2 -b{offset} {record_count} {shm_file.name},buf,trans=100m"
                    subprocess.run(gensort_cmd.split()).check_returncode()
                    runtime_task.add_elapsed_time("generate records (secs)")
                    shm_file.seek(0)
                    buffer = arrow.py_buffer(shm_file.read(record_count * record_nbytes))
                    runtime_task.add_elapsed_time("read records (secs)")
                    # https://arrow.apache.org/docs/format/Columnar.html#fixed-size-primitive-layout
                    records = arrow.Array.from_buffers(arrow.binary(record_nbytes), record_count, [None, buffer])
                    keys = pc.binary_slice(records, 0, key_nbytes)
                    # get first 2 bytes and convert to big-endian uint16
                    binary_prefix = pc.binary_slice(records, 0, 2).cast(arrow.binary())
                    reversed_prefix = pc.binary_reverse(binary_prefix).cast(arrow.binary(2))
                    uint16_prefix = reversed_prefix.view(arrow.uint16())
                    buckets = pc.shift_right(uint16_prefix, 16 - bucket_nbits)
                    runtime_task.add_elapsed_time("build arrow table (secs)")
                    yield arrow.Table.from_arrays([buckets, keys, records], schema=schema)
            yield StreamOutput(
                schema.empty_table(),
                batch_indices=[batch_idx],
                force_checkpoint=pytest_running(),
            )


def sort_records(
    runtime_ctx: RuntimeContext,
    input_datasets: List[DataSet],
    output_path: str,
    sort_engine="polars",
    write_io_nbytes=500 * MB,
) -> bool:
    runtime_task: PythonScriptTask = runtime_ctx.task
    data_file_path = os.path.join(runtime_task.runtime_output_abspath, f"{runtime_task.output_filename}.dat")

    if sort_engine == "polars":
        input_data = polars.read_parquet(
            input_datasets[0].resolved_paths,
            rechunk=False,
            hive_partitioning=False,
            columns=input_datasets[0].columns,
        )
        runtime_task.perf_metrics["num input rows"] += len(input_data)
        runtime_task.add_elapsed_time("input load time (secs)")
        sorted_records = input_data.sort("keys").get_column("records")
        runtime_task.add_elapsed_time("sort by keys (secs)")
        record_arrays = [chunk.to_arrow() for chunk in sorted_records.get_chunks()]
        runtime_task.add_elapsed_time("convert to chunks (secs)")
    elif sort_engine == "arrow":
        input_table = input_datasets[0].to_arrow_table(runtime_task.cpu_limit)
        runtime_task.perf_metrics["num input rows"] += input_table.num_rows
        runtime_task.add_elapsed_time("input load time (secs)")
        sorted_table = input_table.sort_by("keys")
        runtime_task.add_elapsed_time("sort by keys (secs)")
        record_arrays = sorted_table.column("records").chunks
        runtime_task.add_elapsed_time("convert to chunks (secs)")
    elif sort_engine == "duckdb":
        with duckdb.connect(database=":memory:", config={"allow_unsigned_extensions": "true"}) as conn:
            runtime_task.prepare_connection(conn)
            input_views = runtime_task.create_input_views(conn, input_datasets)
            sql_query = "select records from {0} order by keys".format(*input_views)
            sorted_table = conn.sql(sql_query).to_arrow_table()
            runtime_task.add_elapsed_time("sort by keys (secs)")
            record_arrays = sorted_table.column("records").chunks
            runtime_task.add_elapsed_time("convert to chunks (secs)")
    else:
        raise Exception(f"unknown sort engine: {sort_engine}")

    with open(data_file_path, "wb") as fout:
        for record_array in record_arrays:
            # https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-layout
            validity_bitmap, offsets, values = record_array.buffers()
            buffer_mem = memoryview(values)

            total_write_nbytes = sum(
                fout.write(buffer_mem[offset : offset + write_io_nbytes]) for offset in range(0, len(buffer_mem), write_io_nbytes)
            )
            assert total_write_nbytes == len(buffer_mem)

    runtime_task.perf_metrics["num output rows"] += len(record_array)
    runtime_task.add_elapsed_time("output dump time (secs)")
    return True


def validate_records(runtime_ctx: RuntimeContext, input_datasets: List[DataSet], output_path: str) -> bool:
    for data_path in input_datasets[0].resolved_paths:
        summary_path = os.path.join(output_path, PurePath(data_path).with_suffix(".sum").name)
        cmdstr = f"{SortBenchTool.valsort_path} -o {summary_path} {data_path},buf,trans=10m"
        logging.debug(f"running command: {cmdstr}")
        result = subprocess.run(cmdstr.split(), capture_output=True, encoding="utf8")
        if result.stderr:
            logging.info(f"valsort stderr: {result.stderr}")
        if result.stdout:
            logging.info(f"valsort stdout: {result.stdout}")
        if result.returncode != 0:
            return False
    return True


def validate_summary(runtime_ctx: RuntimeContext, input_datasets: List[DataSet], output_path: str) -> bool:
    concated_summary_path = os.path.join(output_path, "merged.sum")
    with open(concated_summary_path, "wb") as fout:
        for path in input_datasets[0].resolved_paths:
            with open(path, "rb") as fin:
                fout.write(fin.read())
    cmdstr = f"{SortBenchTool.valsort_path} -s {concated_summary_path}"
    logging.debug(f"running command: {cmdstr}")
    result = subprocess.run(cmdstr.split(), capture_output=True, encoding="utf8")
    if result.stderr:
        logging.info(f"valsort stderr: {result.stderr}")
    if result.stdout:
        logging.info(f"valsort stdout: {result.stdout}")
    return result.returncode == 0


def generate_random_records(
    ctx,
    record_nbytes,
    key_nbytes,
    total_data_nbytes,
    gensort_batch_nbytes,
    num_data_partitions,
    num_sort_partitions,
    parquet_compression=None,
    parquet_compression_level=None,
):
    num_record_ranges = num_data_partitions * 10
    total_num_records = total_data_nbytes // record_nbytes
    record_range_size = (total_num_records + num_record_ranges - 1) // num_record_ranges
    logging.warning(
        f"{record_nbytes} bytes/record x total {total_num_records:,d} records = "
        f"{total_data_nbytes/GB:.3f}GB / {num_record_ranges} record ranges = "
        f"{record_range_size * record_nbytes/GB:.3f}GB ({record_range_size:,d} records) per record range"
    )

    range_begin_at = [pos for pos in range(0, total_num_records, record_range_size)]
    range_num_records = [min(total_num_records, record_range_size * (range_idx + 1)) - begin_at for range_idx, begin_at in enumerate(range_begin_at)]
    assert sum(range_num_records) == total_num_records
    record_range = DataSourceNode(
        ctx,
        ArrowTableDataSet(arrow.Table.from_arrays([range_begin_at, range_num_records], names=["begin_at", "num_records"])),
    )
    record_range_partitions = DataSetPartitionNode(ctx, (record_range,), npartitions=num_data_partitions, partition_by_rows=True)

    random_records = ArrowStreamNode(
        ctx,
        (record_range_partitions,),
        process_func=functools.partial(
            generate_records,
            record_nbytes=record_nbytes,
            key_nbytes=key_nbytes,
            bucket_nbits=num_sort_partitions.bit_length() - 1,
            gensort_batch_nbytes=gensort_batch_nbytes,
        ),
        background_io_thread=True,
        streaming_batch_size=10,
        parquet_row_group_size=1024 * 1024,
        parquet_compression=parquet_compression,
        parquet_compression_level=parquet_compression_level,
        output_name="random_records",
        cpu_limit=2,
    )
    return random_records


def gray_sort_benchmark(
    record_nbytes,
    key_nbytes,
    total_data_nbytes,
    gensort_batch_nbytes,
    num_data_partitions,
    num_sort_partitions,
    input_paths=None,
    shuffle_engine="duckdb",
    sort_engine="polars",
    hive_partitioning=False,
    validate_results=False,
    shuffle_cpu_limit=32,
    shuffle_memory_limit=None,
    sort_cpu_limit=8,
    sort_memory_limit=None,
    parquet_compression=None,
    parquet_compression_level=None,
    **kwargs,
) -> LogicalPlan:
    ctx = Context()
    num_sort_partitions = next_power_of_two(num_sort_partitions)

    if input_paths:
        input_dataset = ParquetDataSet(input_paths)
        input_nbytes = sum(os.path.getsize(p) for p in input_dataset.resolved_paths)
        logging.warning(f"input data size: {input_nbytes/GB:.3f}GB, {input_dataset.num_files} files")
        random_records = DataSourceNode(ctx, input_dataset)
    else:
        random_records = generate_random_records(
            ctx,
            record_nbytes,
            key_nbytes,
            total_data_nbytes,
            gensort_batch_nbytes,
            num_data_partitions,
            num_sort_partitions,
            parquet_compression,
            parquet_compression_level,
        )

    partitioned_records = ShuffleNode(
        ctx,
        (random_records,),
        npartitions=num_sort_partitions,
        data_partition_column="buckets",
        engine_type=shuffle_engine,
        hive_partitioning=hive_partitioning,
        parquet_row_group_size=10 * 1024 * 1024,
        parquet_compression=parquet_compression,
        parquet_compression_level=parquet_compression_level,
        cpu_limit=shuffle_cpu_limit,
        memory_limit=shuffle_memory_limit,
    )

    sorted_records = PythonScriptNode(
        ctx,
        (ProjectionNode(ctx, partitioned_records, ["keys", "records"]),),
        process_func=functools.partial(sort_records, sort_engine=sort_engine),
        output_name="sorted_records",
        cpu_limit=sort_cpu_limit,
        memory_limit=sort_memory_limit,
    )

    if validate_results:
        partitioned_summaries = PythonScriptNode(
            ctx,
            (sorted_records,),
            process_func=validate_records,
            output_name="partitioned_summaries",
        )
        merged_summaries = DataSetPartitionNode(ctx, (partitioned_summaries,), npartitions=1)
        final_check = PythonScriptNode(ctx, (merged_summaries,), process_func=validate_summary)
        root = final_check
    else:
        root = sorted_records

    return LogicalPlan(ctx, root)


def main():
    SortBenchTool.ensure_installed()

    driver = Driver()
    driver.add_argument("-R", "--record_nbytes", type=int, default=100)
    driver.add_argument("-K", "--key_nbytes", type=int, default=10)
    driver.add_argument("-T", "--total_data_nbytes", type=int, default=None)
    driver.add_argument("-B", "--gensort_batch_nbytes", type=int, default=512 * MB)
    driver.add_argument("-n", "--num_data_partitions", type=int, default=None)
    driver.add_argument("-t", "--num_sort_partitions", type=int, default=None)
    driver.add_argument("-i", "--input_paths", nargs="+", default=[])
    driver.add_argument("-e", "--shuffle_engine", default="duckdb", choices=("duckdb", "arrow"))
    driver.add_argument("-s", "--sort_engine", default="duckdb", choices=("duckdb", "arrow", "polars"))
    driver.add_argument("-H", "--hive_partitioning", action="store_true")
    driver.add_argument("-V", "--validate_results", action="store_true")
    driver.add_argument("-C", "--shuffle_cpu_limit", type=int, default=ShuffleNode.default_cpu_limit)
    driver.add_argument(
        "-M",
        "--shuffle_memory_limit",
        type=int,
        default=ShuffleNode.default_memory_limit,
    )
    driver.add_argument("-TC", "--sort_cpu_limit", type=int, default=8)
    driver.add_argument("-TM", "--sort_memory_limit", type=int, default=None)
    driver.add_argument("-NC", "--cpus_per_node", type=int, default=psutil.cpu_count(logical=False))
    driver.add_argument("-NM", "--memory_per_node", type=int, default=psutil.virtual_memory().total)
    driver.add_argument("-CP", "--parquet_compression", default=None)
    driver.add_argument("-LV", "--parquet_compression_level", type=int, default=None)

    user_args, driver_args = driver.parse_arguments()
    assert len(user_args.input_paths) == 0 or user_args.num_sort_partitions is not None

    total_num_cpus = max(1, driver_args.num_executors) * user_args.cpus_per_node
    memory_per_cpu = user_args.memory_per_node // user_args.cpus_per_node

    user_args.sort_cpu_limit = 1 if user_args.sort_engine == "arrow" else user_args.sort_cpu_limit
    sort_memory_limit = user_args.sort_memory_limit or user_args.sort_cpu_limit * memory_per_cpu
    user_args.total_data_nbytes = user_args.total_data_nbytes or max(1, driver_args.num_executors) * user_args.memory_per_node
    user_args.num_data_partitions = user_args.num_data_partitions or total_num_cpus // 2
    user_args.num_sort_partitions = user_args.num_sort_partitions or max(
        total_num_cpus // user_args.sort_cpu_limit,
        user_args.total_data_nbytes // (sort_memory_limit // 4),
    )

    plan = gray_sort_benchmark(**vars(user_args))
    driver.run(plan)


if __name__ == "__main__":
    main()
