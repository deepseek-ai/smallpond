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
    data_partitions = DataSetPartitionNode(
        ctx,
        (data_files,),
        npartitions=num_data_partitions,
        partition_by_rows=True,
        random_shuffle=skip_hash_partition,
    )
    if skip_hash_partition:
        urls_partitions = data_partitions
    else:
        urls_partitions = HashPartitionNode(
            ctx,
            (data_partitions,),
            npartitions=num_hash_partitions,
            hash_columns=None,
            random_shuffle=True,
            engine_type=engine_type,
        )
    shuffled_urls = SqlEngineNode(
        ctx,
        (urls_partitions,),
        r"select *, cast(random() * 2147483647 as integer) as sort_key from {0} order by sort_key",
        cpu_limit=16,
    )
    repartitioned = DataSetPartitionNode(
        ctx,
        (shuffled_urls,),
        npartitions=num_out_data_partitions,
        partition_by_rows=True,
    )
    shuffled_urls = StreamCopy(ctx, (repartitioned,), output_name="data_copy", cpu_limit=1)

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
    driver.run(plan)


if __name__ == "__main__":
    main()
