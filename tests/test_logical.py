import unittest

from loguru import logger

from smallpond.logical.dataset import ParquetDataSet
from smallpond.logical.node import (
    Context,
    DataSetPartitionNode,
    DataSourceNode,
    EvenlyDistributedPartitionNode,
    HashPartitionNode,
    LogicalPlan,
    SqlEngineNode,
)
from smallpond.logical.planner import Planner
from tests.test_fabric import TestFabric


class TestLogicalPlan(TestFabric, unittest.TestCase):

    def test_join_chunkmeta_inodes(self):
        ctx = Context()

        chunkmeta_dump = DataSourceNode(ctx, dataset=ParquetDataSet(["tests/data/chunkmeta*.parquet"]))
        chunkmeta_partitions = HashPartitionNode(ctx, (chunkmeta_dump,), npartitions=2, hash_columns=["inodeId"])

        inodes_dump = DataSourceNode(ctx, dataset=ParquetDataSet(["tests/data/inodes*.parquet"]))
        inodes_partitions = HashPartitionNode(ctx, (inodes_dump,), npartitions=2, hash_columns=["inode_id"])

        num_gc_chunks = SqlEngineNode(
            ctx,
            (chunkmeta_partitions, inodes_partitions),
            r"""
                                  select count(chunkmeta_chunkId) from {0}
                                    where chunkmeta.chunkmeta_chunkId NOT LIKE "F%" AND
                                    chunkmeta.inodeId not in ( select distinct inode_id from {1} )""",
        )

        plan = LogicalPlan(ctx, num_gc_chunks)
        logger.info(str(plan))
        exec_plan = Planner(self.runtime_ctx).create_exec_plan(plan)
        logger.info(str(exec_plan))

    def test_partition_dims_not_compatible(self):
        ctx = Context()
        parquet_dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_source = DataSourceNode(ctx, parquet_dataset)
        partition_dim_a = EvenlyDistributedPartitionNode(ctx, (data_source,), npartitions=parquet_dataset.num_files, dimension="A")
        partition_dim_b = EvenlyDistributedPartitionNode(ctx, (data_source,), npartitions=parquet_dataset.num_files, dimension="B")
        join_two_inputs = SqlEngineNode(
            ctx,
            (partition_dim_a, partition_dim_b),
            r"select a.* from {0} as a join {1} as b on a.host = b.host",
        )
        plan = LogicalPlan(ctx, join_two_inputs)
        logger.info(str(plan))
        with self.assertRaises(AssertionError) as context:
            Planner(self.runtime_ctx).create_exec_plan(plan)

    def test_npartitions_not_compatible(self):
        ctx = Context()
        parquet_dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_source = DataSourceNode(ctx, parquet_dataset)
        partition_dim_a = EvenlyDistributedPartitionNode(ctx, (data_source,), npartitions=parquet_dataset.num_files, dimension="A")
        partition_dim_a2 = EvenlyDistributedPartitionNode(
            ctx,
            (data_source,),
            npartitions=parquet_dataset.num_files * 2,
            dimension="A",
        )
        join_two_inputs1 = SqlEngineNode(
            ctx,
            (partition_dim_a, partition_dim_a2),
            r"select a.* from {0} as a join {1} as b on a.host = b.host",
        )
        join_two_inputs2 = SqlEngineNode(
            ctx,
            (partition_dim_a2, partition_dim_a),
            r"select a.* from {0} as a join {1} as b on a.host = b.host",
        )
        plan = LogicalPlan(
            ctx,
            DataSetPartitionNode(ctx, (join_two_inputs1, join_two_inputs2), npartitions=1),
        )
        logger.info(str(plan))
        with self.assertRaises(AssertionError) as context:
            Planner(self.runtime_ctx).create_exec_plan(plan)
