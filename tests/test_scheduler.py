import os.path
import random
import time
import unittest
from typing import List, Tuple

from loguru import logger

from smallpond.execution.scheduler import ExecutorState
from smallpond.execution.task import PythonScriptTask, RuntimeContext
from smallpond.logical.dataset import DataSet, ParquetDataSet
from smallpond.logical.node import (
    Context,
    DataSetPartitionNode,
    DataSourceNode,
    LogicalPlan,
    Node,
    PythonScriptNode,
)
from tests.test_fabric import TestFabric


class RandomSleepTask(PythonScriptTask):
    def __init__(self, *args, sleep_secs: float, fail_first_try: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sleep_secs = sleep_secs
        self.fail_first_try = fail_first_try

    def process(
        self,
        runtime_ctx: RuntimeContext,
        input_datasets: List[DataSet],
        output_path: str,
    ) -> bool:
        logger.info(f"sleeping {self.sleep_secs} secs")
        time.sleep(self.sleep_secs)
        with open(os.path.join(output_path, self.output_filename), "w") as fout:
            fout.write(f"{repr(self)}")
        if self.fail_first_try and self.retry_count == 0:
            return False
        return True


class RandomSleepNode(PythonScriptNode):
    def __init__(
        self,
        ctx: Context,
        input_deps: Tuple[Node, ...],
        *,
        max_sleep_secs=5,
        fail_first_try=False,
        **kwargs,
    ):
        super().__init__(ctx, input_deps, **kwargs)
        self.max_sleep_secs = max_sleep_secs
        self.fail_first_try = fail_first_try

    def spawn(self, *args, **kwargs) -> RandomSleepTask:
        sleep_secs = random.random() if len(self.generated_tasks) % 20 else self.max_sleep_secs
        return RandomSleepTask(*args, **kwargs, sleep_secs=sleep_secs, fail_first_try=self.fail_first_try)


class TestScheduler(TestFabric, unittest.TestCase):
    def create_random_sleep_plan(self, npartitions, max_sleep_secs, fail_first_try=False):
        ctx = Context()
        dataset = ParquetDataSet(["tests/data/mock_urls/*.parquet"])
        data_files = DataSourceNode(ctx, dataset)
        data_partitions = DataSetPartitionNode(ctx, (data_files,), npartitions=npartitions, partition_by_rows=True)
        random_sleep = RandomSleepNode(
            ctx,
            (data_partitions,),
            max_sleep_secs=max_sleep_secs,
            fail_first_try=fail_first_try,
        )
        return LogicalPlan(ctx, random_sleep)

    def check_executor_state(self, target_state: ExecutorState, nloops=200):
        for _ in range(nloops):
            latest_sched_state = self.get_latest_sched_state()
            if any(executor.state == target_state for executor in latest_sched_state.remote_executors):
                logger.info(f"found {target_state} executor in: {latest_sched_state.remote_executors}")
                break
            time.sleep(0.1)
        else:
            self.assertTrue(
                False,
                f"cannot find any executor in state {target_state}: {latest_sched_state.remote_executors}",
            )

    def test_standalone_mode(self):
        plan = self.create_random_sleep_plan(npartitions=10, max_sleep_secs=1)
        self.execute_plan(plan, num_executors=0)

    def test_failed_executors(self):
        num_exec = 6
        num_fail = 4
        plan = self.create_random_sleep_plan(npartitions=300, max_sleep_secs=10)

        _, executors, processes = self.start_execution(
            plan,
            num_executors=num_exec,
            secs_wq_poll_interval=0.1,
            secs_executor_probe_interval=0.5,
            console_log_level="WARNING",
        )
        latest_sched_state = self.get_latest_sched_state()
        self.check_executor_state(ExecutorState.GOOD)

        for i, (executor, process) in enumerate(random.sample(list(zip(executors, processes[1:])), k=num_fail)):
            if i % 2 == 0:
                logger.warning(f"kill executor: {executor}")
                process.kill()
            else:
                logger.warning(f"skip probes: {executor}")
                executor.skip_probes(latest_sched_state.ctx.max_num_missed_probes * 2)

        self.join_running_procs()
        latest_sched_state = self.get_latest_sched_state()
        self.assertTrue(latest_sched_state.success)
        self.assertGreater(len(latest_sched_state.abandoned_tasks), 0)
        self.assertLessEqual(
            1,
            len(latest_sched_state.stopped_executors),
            f"remote_executors: {latest_sched_state.remote_executors}",
        )
        self.assertLessEqual(
            num_fail / 2,
            len(latest_sched_state.failed_executors),
            f"remote_executors: {latest_sched_state.remote_executors}",
        )

    def test_speculative_scheduling(self):
        for speculative_exec in ("disable", "enable", "aggressive"):
            with self.subTest(speculative_exec=speculative_exec):
                plan = self.create_random_sleep_plan(npartitions=100, max_sleep_secs=10)
                self.execute_plan(
                    plan,
                    num_executors=3,
                    secs_wq_poll_interval=0.1,
                    secs_executor_probe_interval=0.5,
                    prioritize_retry=(speculative_exec == "aggressive"),
                    speculative_exec=speculative_exec,
                )
                latest_sched_state = self.get_latest_sched_state()
                if speculative_exec == "disable":
                    self.assertEqual(len(latest_sched_state.abandoned_tasks), 0)
                else:
                    self.assertGreater(len(latest_sched_state.abandoned_tasks), 0)

    def test_stop_executor_on_failure(self):
        plan = self.create_random_sleep_plan(npartitions=3, max_sleep_secs=5, fail_first_try=True)
        exec_plan = self.execute_plan(
            plan,
            num_executors=5,
            secs_wq_poll_interval=0.1,
            secs_executor_probe_interval=0.5,
            check_result=False,
            stop_executor_on_failure=True,
        )
        latest_sched_state = self.get_latest_sched_state()
        self.assertGreater(len(latest_sched_state.abandoned_tasks), 0)
