import logging
import os.path
from typing import List, Optional, OrderedDict

import pyarrow as arrow

from smallpond.execution.driver import Driver
from smallpond.execution.task import RuntimeContext
from smallpond.logical.dataset import CsvDataSet
from smallpond.logical.node import (
    ArrowComputeNode,
    Context,
    DataSetPartitionNode,
    DataSinkNode,
    DataSourceNode,
    HashPartitionNode,
    LogicalPlan,
    SqlEngineNode,
)


class SortUrlsNode(ArrowComputeNode):
    def process(self, runtime_ctx: RuntimeContext, input_tables: List[arrow.Table]) -> arrow.Table:
        logging.info(f"sorting urls by 'host', table shape: {input_tables[0].shape}")
        return input_tables[0].sort_by("host")


def sort_mock_urls(
    input_paths,
    npartitions: int,
    engine_type="duckdb",
    external_output_path: Optional[str] = None,
) -> LogicalPlan:
    ctx = Context()
    dataset = CsvDataSet(
        input_paths,
        schema=OrderedDict([("urlstr", "varchar"), ("valstr", "varchar")]),
        delim=r"\t",
    )
    data_files = DataSourceNode(ctx, dataset)
    data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=npartitions)
    imported_urls = SqlEngineNode(
        ctx,
        (data_partitions,),
        r"""
    select split_part(urlstr, '/', 1) as host, split_part(urlstr, ' ', 1) as url, from_base64(valstr) AS payload from {0}
    """,
        output_name="imported_urls",
        output_path=external_output_path,
    )
    urls_partitions = HashPartitionNode(
        ctx,
        (imported_urls,),
        npartitions=npartitions,
        hash_columns=["host"],
        engine_type=engine_type,
        output_name="urls_partitions",
        output_path=external_output_path,
    )

    if engine_type == "duckdb":
        sorted_urls = SqlEngineNode(
            ctx,
            (urls_partitions,),
            r"select * from {0} order by host",
            output_name="sorted_urls",
        )
    else:
        sorted_urls = SortUrlsNode(
            ctx,
            (urls_partitions,),
            output_name="sorted_urls",
            output_path=external_output_path,
        )

    final_result = DataSetPartitionNode(ctx, (sorted_urls,), npartitions=1)

    if external_output_path:
        final_result = DataSinkNode(
            ctx,
            (final_result,),
            output_path=os.path.join(external_output_path, "data_sink"),
        )

    plan = LogicalPlan(ctx, final_result)
    return plan


def main():
    driver = Driver()
    driver.add_argument("-i", "--input_paths", nargs="+", default=["tests/data/mock_urls/*.tsv"])
    driver.add_argument("-n", "--npartitions", type=int, default=10)
    driver.add_argument("-e", "--engine_type", default="duckdb")

    plan = sort_mock_urls(**driver.get_arguments())
    driver.run(plan)


if __name__ == "__main__":
    main()
