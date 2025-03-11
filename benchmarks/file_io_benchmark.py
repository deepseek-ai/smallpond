from smallpond.common import DEFAULT_BATCH_SIZE, DEFAULT_ROW_GROUP_SIZE, GB
from smallpond.contrib.copy_table import CopyArrowTable, StreamCopy
from smallpond.execution.driver import Driver
from smallpond.logical.dataset import ParquetDataSet
from smallpond.logical.node import (
    Context,
    DataSetPartitionNode,
    DataSourceNode,
    LogicalPlan,
    SqlEngineNode,
)


def file_io_benchmark(
    input_paths,
    npartitions,
    io_engine="duckdb",
    batch_size=DEFAULT_BATCH_SIZE,
    row_group_size=DEFAULT_ROW_GROUP_SIZE,
    output_name="data",
    **kwargs,
) -> LogicalPlan:
    ctx = Context()
    dataset = ParquetDataSet(input_paths)
    data_files = DataSourceNode(ctx, dataset)
    data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=npartitions)

    if io_engine == "duckdb":
        data_copy = SqlEngineNode(
            ctx,
            (data_partitions,),
            r"select * from {0}",
            parquet_row_group_size=row_group_size,
            per_thread_output=False,
            output_name=output_name,
            cpu_limit=1,
            memory_limit=10 * GB,
        )
    elif io_engine == "arrow":
        data_copy = CopyArrowTable(
            ctx,
            (data_partitions,),
            parquet_row_group_size=row_group_size,
            output_name=output_name,
            cpu_limit=1,
            memory_limit=10 * GB,
        )
    elif io_engine == "stream":
        data_copy = StreamCopy(
            ctx,
            (data_partitions,),
            streaming_batch_size=batch_size,
            parquet_row_group_size=row_group_size,
            output_name=output_name,
            cpu_limit=1,
            memory_limit=10 * GB,
        )

    plan = LogicalPlan(ctx, data_copy)
    return plan


def main():
    driver = Driver()
    driver.add_argument("-i", "--input_paths", nargs="+")
    driver.add_argument("-n", "--npartitions", type=int, default=None)
    driver.add_argument("-e", "--io_engine", default="duckdb", choices=("duckdb", "arrow", "stream"))
    driver.add_argument("-b", "--batch_size", type=int, default=1024 * 1024)
    driver.add_argument("-s", "--row_group_size", type=int, default=1024 * 1024)
    driver.add_argument("-o", "--output_name", default="data")
    driver.add_argument("-NC", "--cpus_per_node", type=int, default=128)

    user_args, driver_args = driver.parse_arguments()
    total_num_cpus = driver_args.num_executors * user_args.cpus_per_node
    user_args.npartitions = user_args.npartitions or total_num_cpus

    plan = file_io_benchmark(**driver.get_arguments())
    driver.run(plan)


if __name__ == "__main__":
    main()
