"""
Large-Scale Data Shuffling Example
================================

This script demonstrates advanced data processing capabilities of Smallpond using
the Driver class for complex pipeline execution. It shows how to:
- Create multi-stage data processing pipelines
- Implement different partitioning strategies (row and hash-based)
- Apply SQL transformations on partitioned data
- Use memory-efficient streaming writes

Note: Uses Driver class for advanced execution control instead of smallpond.init()
"""

from smallpond.contrib.copy_table import StreamCopy
from smallpond.execution.driver import Driver
from smallpond.logical.dataset import ParquetDataSet
from smallpond.logical.node import (
    Context,
    DataSetPartitionNode,
    DataSourceNode,
    HashPartitionNode,
    LogicalPlan,
    SqlEngineNode,
)


def shuffle_data(
    input_paths,
    num_out_data_partitions: int = 0,
    num_data_partitions: int = 10,
    num_hash_partitions: int = 10,
    engine_type="duckdb",
    skip_hash_partition=False,
) -> LogicalPlan:
    ctx = Context()
    dataset = ParquetDataSet(input_paths, union_by_name=True)
    data_files = DataSourceNode(ctx, dataset)
    # Node 1: Partitions data by splitting based on file boundaries or row counts
    data_partitions = DataSetPartitionNode(
        ctx,
        (data_files,),
        npartitions=num_data_partitions,
        partition_by_rows=True, # distributes rows across partitions
        random_shuffle=skip_hash_partition,
    )
    if skip_hash_partition:
        urls_partitions = data_partitions
    else:
        # Node 2: HashPartiion to ensure even distribution of rows across partitions
        urls_partitions = HashPartitionNode(
            ctx,
            (data_partitions,),
            npartitions=num_hash_partitions,
            hash_columns=None,
            random_shuffle=True,
            engine_type=engine_type,
        )
    # Node 3: Example adding SQL Tranformations, here we are adding a sort_key column and sorting one it
    shuffled_urls = SqlEngineNode(
        ctx,
        (urls_partitions,),
        r"select *, cast(random() * 2147483647 as integer) as sort_key from {0} order by sort_key",
        cpu_limit=16,
    )
    # Node 4: Repartition again to fit output partition count
    repartitioned = DataSetPartitionNode(
        ctx,
        (shuffled_urls,),
        npartitions=num_out_data_partitions,
        partition_by_rows=True,
    )
    # Node 5: Write file in partitioned parquet format using Streaming for memory efficiency
    shuffled_urls = StreamCopy(ctx, (repartitioned,), output_name="data_copy", cpu_limit=1)

    # Logical Plan DAG with all the Nodes dependencies and creates lazy execution plan 
    plan = LogicalPlan(ctx, shuffled_urls)
    return plan


def main():
    driver = Driver()
    driver.add_argument("-i", "--input_paths", nargs="+")
    driver.add_argument("-nd", "--num_data_partitions", type=int, default=1024)
    driver.add_argument("-nh", "--num_hash_partitions", type=int, default=3840)
    driver.add_argument("-no", "--num_out_data_partitions", type=int, default=1920)
    driver.add_argument("-e", "--engine_type", default="duckdb", choices=("duckdb", "arrow"))
    driver.add_argument("-x", "--skip_hash_partition", action="store_true")
    plan = shuffle_data(**driver.get_arguments())
    # Executes logical plan parallelly, handling task scheduling dependencies & resource management
    driver.run(plan)


if __name__ == "__main__":
    main()
