import multiprocessing
import multiprocessing.dummy
import multiprocessing.queues
import queue
import tempfile
import time
import unittest

from loguru import logger

from smallpond.execution.workqueue import (
    WorkItem,
    WorkQueue,
    WorkQueueInMemory,
    WorkQueueOnFilesystem,
)
from tests.test_fabric import TestFabric


class PrintWork(WorkItem):
    def __init__(self, name: str, message: str) -> None:
        super().__init__(name, cpu_limit=1, gpu_limit=0, memory_limit=0)
        self.message = message

    def run(self) -> bool:
        logger.debug(f"{self.key}: {self.message}")
        return True


def producer(wq: WorkQueue, id: int, numItems: int, numConsumers: int) -> None:
    print(f"wq.outbound_works: {wq.outbound_works}")
    for i in range(numItems):
        wq.push(PrintWork(f"item-{i}", message="hello"), buffering=(i % 3 == 1))
        # wq.push(PrintWork(f"item-{i}", message="hello"))
        if i % 5 == 0:
            wq.flush()
    for i in range(numConsumers):
        wq.push(PrintWork(f"stop-{i}", message="stop"))
    logger.success(f"producer {id} generated {numItems} items")


def consumer(wq: WorkQueue, id: int) -> int:
    numItems = 0
    numWaits = 0
    running = True
    while running:
        items = wq.pop(count=1)
        if not items:
            numWaits += 1
            time.sleep(0.01)
            continue
        for item in items:
            assert isinstance(item, PrintWork)
            if item.message == "stop":
                running = False
                break
            item.exec()
            numItems += 1
    logger.success(f"consumer {id} collected {numItems} items, {numWaits} waits")
    logger.complete()
    return numItems


class WorkQueueTestBase(object):

    wq: WorkQueue = None
    pool: multiprocessing.Pool = None

    def setUp(self) -> None:
        logger.disable("smallpond.execution.workqueue")
        return super().setUp()

    def test_basics(self):
        numItems = 200
        for i in range(numItems):
            self.wq.push(PrintWork(f"item-{i}", message="hello"))
        numCollected = 0
        for _ in range(numItems):
            items = self.wq.pop()
            logger.info(f"{len(items)} items")
            numCollected += len(items)
            if numItems == numCollected:
                break

    def test_multi_consumers(self):
        numConsumers = 10
        numItems = 200
        result = self.pool.starmap_async(consumer, [(self.wq, id) for id in range(numConsumers)])
        producer(self.wq, 0, numItems, numConsumers)

        logger.info("waiting for result")
        numCollected = sum(result.get(timeout=20))
        logger.info(f"expected vs collected: {numItems} vs {numCollected}")
        self.assertEqual(numItems, numCollected)
        logger.success("all done")

        self.pool.terminate()
        self.pool.join()
        logger.success("workers stopped")


class TestWorkQueueInMemory(WorkQueueTestBase, TestFabric, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.wq = WorkQueueInMemory(queue_type=queue.Queue)
        self.pool = multiprocessing.dummy.Pool(10)


class TestWorkQueueOnFilesystem(WorkQueueTestBase, TestFabric, unittest.TestCase):

    workq_root: str

    def setUp(self) -> None:
        super().setUp()
        self.workq_root = tempfile.mkdtemp(dir=self.runtime_ctx.queue_root)
        self.wq = WorkQueueOnFilesystem(self.workq_root, sort=True)
        self.pool = multiprocessing.Pool(10)
