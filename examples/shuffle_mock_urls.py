from smallpond.common import GB
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


def shuffle_mock_urls(input_paths, npartitions: int = 10, sort_rand_keys=True, engine_type="duckdb") -> LogicalPlan:
    ctx = Context()
    dataset = ParquetDataSet(input_paths)
    data_files = DataSourceNode(ctx, dataset)
    data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=npartitions)

    urls_partitions = HashPartitionNode(
        ctx,
        (data_partitions,),
        npartitions=npartitions,
        hash_columns=None,
        random_shuffle=True,
        engine_type=engine_type,
        output_name="urls_partitions",
        cpu_limit=1,
        memory_limit=20 * GB,
    )

    if sort_rand_keys:
        # shuffle as sorting partition keys
        shuffled_urls = SqlEngineNode(
            ctx,
            (urls_partitions,),
            r"select *, random() as partition_key from {0} order by partition_key",
            output_name="shuffled_urls",
            cpu_limit=1,
            memory_limit=40 * GB,
        )
    else:
        # shuffle as reservoir sampling
        shuffled_urls = SqlEngineNode(
            ctx,
            (urls_partitions,),
            r"select * from {0} using sample 100% (reservoir, {rand_seed})",
            output_name="shuffled_urls",
            cpu_limit=1,
            memory_limit=40 * GB,
        )

    plan = LogicalPlan(ctx, shuffled_urls)
    return plan


def main():
    driver = Driver()
    driver.add_argument("-i", "--input_paths", nargs="+")
    driver.add_argument("-n", "--npartitions", type=int, default=500)
    driver.add_argument("-s", "--sort_rand_keys", action="store_true")
    driver.add_argument("-e", "--engine_type", default="duckdb", choices=("duckdb", "arrow"))

    plan = shuffle_mock_urls(**driver.get_arguments())
    driver.run(plan)


if __name__ == "__main__":
    main()
