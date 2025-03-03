from smallpond.common import GB
from smallpond.contrib.log_dataset import LogDataSet
from smallpond.execution.driver import Driver
from smallpond.logical.dataset import ParquetDataSet
from smallpond.logical.node import (
    ConsolidateNode,
    Context,
    DataSourceNode,
    HashPartitionNode,
    LogicalPlan,
    SqlEngineNode,
)


def hash_partition_benchmark(
    input_paths,
    npartitions,
    hash_columns,
    engine_type="duckdb",
    use_parquet_writer=False,
    hive_partitioning=False,
    cpu_limit=None,
    memory_limit=None,
    partition_stats=True,
    **kwargs,
) -> LogicalPlan:
    ctx = Context()
    dataset = ParquetDataSet(input_paths)
    data_files = DataSourceNode(ctx, dataset)

    partitioned_datasets = HashPartitionNode(
        ctx,
        (data_files,),
        npartitions=npartitions,
        hash_columns=hash_columns,
        data_partition_column="partition_keys",
        engine_type=engine_type,
        use_parquet_writer=use_parquet_writer,
        hive_partitioning=hive_partitioning,
        output_name="partitioned_datasets",
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
    )

    if partition_stats:
        partition_stats = SqlEngineNode(
            ctx,
            (partitioned_datasets,),
            f"""
      select partition_keys, count(*) as row_cnt, count( distinct ( {', '.join(hash_columns)} ) ) as uniq_key_cnt from {{0}}
      group by partition_keys""",
            output_name="partition_stats",
            cpu_limit=1,
            memory_limit=10 * GB,
        )
        sorted_stats = SqlEngineNode(
            ctx,
            (ConsolidateNode(ctx, partition_stats, []),),
            r"select * from {0} order by row_cnt desc",
        )
        plan = LogicalPlan(ctx, LogDataSet(ctx, (sorted_stats,), num_rows=npartitions))
    else:
        plan = LogicalPlan(ctx, partitioned_datasets)

    return plan


def main():
    driver = Driver()
    driver.add_argument("-i", "--input_paths", nargs="+", required=True)
    driver.add_argument("-n", "--npartitions", type=int, default=None)
    driver.add_argument("-c", "--hash_columns", nargs="+", required=True)
    driver.add_argument("-e", "--engine_type", default="duckdb", choices=("duckdb", "arrow"))
    driver.add_argument("-S", "--partition_stats", action="store_true")
    driver.add_argument("-W", "--use_parquet_writer", action="store_true")
    driver.add_argument("-H", "--hive_partitioning", action="store_true")
    driver.add_argument("-C", "--cpu_limit", type=int, default=HashPartitionNode.default_cpu_limit)
    driver.add_argument("-M", "--memory_limit", type=int, default=HashPartitionNode.default_memory_limit)
    driver.add_argument("-NC", "--cpus_per_node", type=int, default=192)
    driver.add_argument("-NM", "--memory_per_node", type=int, default=2000 * GB)

    user_args, driver_args = driver.parse_arguments()
    total_num_cpus = driver_args.num_executors * user_args.cpus_per_node
    user_args.npartitions = user_args.npartitions or total_num_cpus

    plan = hash_partition_benchmark(**vars(user_args))
    driver.run(plan)


if __name__ == "__main__":
    main()
