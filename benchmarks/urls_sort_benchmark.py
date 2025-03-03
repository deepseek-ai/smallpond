from typing import List, OrderedDict

from smallpond.common import GB
from smallpond.dataframe import Session
from smallpond.execution.driver import Driver
from smallpond.logical.dataset import CsvDataSet
from smallpond.logical.node import (
    Context,
    DataSetPartitionNode,
    DataSourceNode,
    HashPartitionNode,
    LogicalPlan,
    SqlEngineNode,
)


def urls_sort_benchmark(
    input_paths: List[str],
    num_data_partitions: int,
    num_hash_partitions: int,
    engine_type="duckdb",
    sort_cpu_limit=8,
    sort_memory_limit=None,
) -> LogicalPlan:
    ctx = Context()
    dataset = CsvDataSet(
        input_paths,
        schema=OrderedDict([("urlstr", "varchar"), ("valstr", "varchar")]),
        delim=r"\t",
    )
    data_files = DataSourceNode(ctx, dataset)
    data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=num_data_partitions)

    imported_urls = SqlEngineNode(
        ctx,
        (data_partitions,),
        r"""
    select urlstr, valstr from {0}
    """,
        output_name="imported_urls",
        parquet_row_group_size=1024 * 1024,
        cpu_limit=1,
        memory_limit=16 * GB,
    )

    urls_partitions = HashPartitionNode(
        ctx,
        (imported_urls,),
        npartitions=num_hash_partitions,
        hash_columns=["urlstr"],
        engine_type=engine_type,
        parquet_row_group_size=1024 * 1024,
        cpu_limit=1,
        memory_limit=16 * GB,
    )

    sorted_urls = SqlEngineNode(
        ctx,
        (urls_partitions,),
        r"select * from {0} order by urlstr",
        output_name="sorted_urls",
        parquet_row_group_size=1024 * 1024,
        cpu_limit=sort_cpu_limit,
        memory_limit=sort_memory_limit,
    )

    plan = LogicalPlan(ctx, sorted_urls)
    return plan


def urls_sort_benchmark_v2(
    sp: Session,
    input_paths: List[str],
    output_path: str,
    num_data_partitions: int,
    num_hash_partitions: int,
    engine_type="duckdb",
    sort_cpu_limit=8,
    sort_memory_limit=None,
):
    dataset = sp.read_csv(input_paths, schema={"urlstr": "varchar", "valstr": "varchar"}, delim=r"\t")
    data_partitions = dataset.repartition(num_data_partitions)
    urls_partitions = data_partitions.repartition(num_hash_partitions, hash_by="urlstr", engine_type=engine_type)
    sorted_urls = urls_partitions.partial_sort(by="urlstr", cpu_limit=sort_cpu_limit, memory_limit=sort_memory_limit)
    sorted_urls.write_parquet(output_path)


def main():
    driver = Driver()
    driver.add_argument("-i", "--input_paths", nargs="+")
    driver.add_argument("-n", "--num_data_partitions", type=int, default=None)
    driver.add_argument("-m", "--num_hash_partitions", type=int, default=None)
    driver.add_argument("-e", "--engine_type", default="duckdb")
    driver.add_argument("-TC", "--sort_cpu_limit", type=int, default=8)
    driver.add_argument("-TM", "--sort_memory_limit", type=int, default=None)
    user_args, driver_args = driver.parse_arguments()

    num_nodes = driver_args.num_executors
    cpus_per_node = 120
    partition_rounds = 2
    user_args.num_data_partitions = user_args.num_data_partitions or num_nodes * cpus_per_node * partition_rounds
    user_args.num_hash_partitions = user_args.num_hash_partitions or num_nodes * cpus_per_node

    # v1
    plan = urls_sort_benchmark(**vars(user_args))
    driver.run(plan)

    # v2
    # sp = smallpond.init()
    # urls_sort_benchmark_v2(sp, **vars(user_args))


if __name__ == "__main__":
    main()
