from typing import Iterable, List

import pyarrow as arrow
from loguru import logger

from smallpond.execution.task import RuntimeContext
from smallpond.logical.node import ArrowComputeNode, ArrowStreamNode


class CopyArrowTable(ArrowComputeNode):
    def process(self, runtime_ctx: RuntimeContext, input_tables: List[arrow.Table]) -> arrow.Table:
        logger.info(f"copying table: {input_tables[0].num_rows} rows ...")
        return input_tables[0]


class StreamCopy(ArrowStreamNode):
    def process(self, runtime_ctx: RuntimeContext, input_readers: List[arrow.RecordBatchReader]) -> Iterable[arrow.Table]:
        for batch in input_readers[0]:
            logger.info(f"copying batch: {batch.num_rows} rows ...")
            yield arrow.Table.from_batches([batch])
