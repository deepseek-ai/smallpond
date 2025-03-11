import os.path
import queue
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import Manager, Process
from typing import List, Optional

import fsspec
import numpy as np
import psutil
import pyarrow as arrow
import pyarrow.compute as pc
import pyarrow.parquet as parquet
from loguru import logger

from smallpond.common import DEFAULT_MAX_FAIL_COUNT, DEFAULT_MAX_RETRY_COUNT, GB, MB
from smallpond.execution.executor import Executor
from smallpond.execution.scheduler import Scheduler
from smallpond.execution.task import ExecutionPlan, JobId, RuntimeContext
from smallpond.io.arrow import cast_columns_to_large_string
from smallpond.logical.node import LogicalPlan
from smallpond.logical.planner import Planner
from tests.datagen import generate_data

generate_data()


def run_scheduler(runtime_ctx: RuntimeContext, scheduler: Scheduler, queue: queue.Queue):
    runtime_ctx.initialize("scheduler")
    scheduler.add_state_observer(Scheduler.StateObserver(SaveSchedState(queue)))
    retval = scheduler.run()
    print(f"scheduler exited with value {retval}", file=sys.stderr)


def run_executor(runtime_ctx: RuntimeContext, executor: Executor):
    runtime_ctx.initialize(executor.id)
    retval = executor.run()
    print(f"{executor.id} exited with value {retval}", file=sys.stderr)


class SaveSchedState:
    """
    A state observer that push the scheduler state into a queue when finished.
    """

    def __init__(self, queue: queue.Queue):
        self.queue = queue

    def __call__(self, sched_state: Scheduler) -> bool:
        if sched_state.num_local_running_works == 0:
            self.queue.put(sched_state)
        return True


class TestFabric(unittest.TestCase):
    """
    A helper class that includes boilerplate code to test a logical plan.
    """

    runtime_root = os.getenv("TEST_RUNTIME_ROOT") or f"tests/runtime"
    runtime_ctx = None
    fault_inject_prob = 0.00

    queue_manager = None
    sched_states: queue.Queue = None
    latest_state: Scheduler = None
    executors: List[Executor] = None
    processes: List[Process] = None

    @property
    def output_dir(self):
        return os.path.join(self.__class__.__name__, self._testMethodName)

    @property
    def output_root_abspath(self):
        output_root = os.path.abspath(os.path.join(self.runtime_root, self.output_dir))
        os.makedirs(output_root, exist_ok=True)
        return output_root

    def setUp(self) -> None:
        try:
            from pytest_cov.embed import cleanup_on_sigterm
        except ImportError:
            pass
        else:
            cleanup_on_sigterm()
        self.runtime_ctx = RuntimeContext(
            JobId.new(),
            datetime.now(),
            self.output_root_abspath,
            console_log_level="WARNING",
        )
        self.runtime_ctx.initialize("setup")
        return super().setUp()

    def tearDown(self) -> None:
        if self.sched_states is not None:
            self.get_latest_sched_state()
            assert self.sched_states.qsize() == 0
            self.sched_states = None
        if self.queue_manager is not None:
            self.queue_manager.shutdown()
            self.queue_manager = None
        return super().tearDown()

    def get_latest_sched_state(self) -> Scheduler:
        while True:
            try:
                self.latest_state = self.sched_states.get(block=False)
            except queue.Empty:
                return self.latest_state

    def join_running_procs(self, timeout=30):
        for i, process in enumerate(self.processes):
            if process.is_alive():
                logger.info(f"join #{i} process: {process.name}")
                process.join(timeout=None if i == 0 else timeout)

            if process.exitcode is None:
                logger.info(f"terminate #{i} process: {process.name}")
                process.terminate()
                process.join(timeout=timeout)

            if process.exitcode is None:
                logger.info(f"kill #{i} process: {process.name}")
                process.kill()
                process.join()

            logger.info(f"#{i} process {process.name} exited with code {process.exitcode}")

    def start_execution(
        self,
        plan: LogicalPlan,
        num_executors: int = 2,
        secs_wq_poll_interval: float = 0.1,
        secs_executor_probe_interval: float = 1,
        max_num_missed_probes: int = 10,
        max_retry_count: int = DEFAULT_MAX_RETRY_COUNT,
        max_fail_count: int = DEFAULT_MAX_FAIL_COUNT,
        prioritize_retry=False,
        speculative_exec="enable",
        stop_executor_on_failure=False,
        enforce_memory_limit=False,
        nonzero_exitcode_as_oom=False,
        fault_inject_prob=None,
        enable_profiling=False,
        enable_diagnostic_metrics=False,
        remove_empty_parquet=False,
        skip_task_with_empty_input=False,
        console_log_level="WARNING",
        file_log_level="DEBUG",
        output_path: Optional[str] = None,
        runtime_ctx: Optional[RuntimeContext] = None,
    ):
        """
        Start a scheduler and `num_executors` executors to execute `plan`.
        When this function returns, the execution is mostly still running.

        Parameters
        ----------
        plan
            A logical plan.
        num_executors, optional
            The number of executors
        console_log_level, optional
            Set to logger.INFO if more verbose loguru is needed for debug, by default "WARNING".

        Returns
        -------
            A 3-tuple of type (Scheduler, List[Executor], List[Process]).
        """
        if runtime_ctx is None:
            runtime_ctx = RuntimeContext(
                JobId.new(),
                datetime.now(),
                self.output_root_abspath,
                num_executors=num_executors,
                random_seed=123456,
                enforce_memory_limit=enforce_memory_limit,
                max_usable_cpu_count=min(64, psutil.cpu_count(logical=False)),
                max_usable_gpu_count=0,
                max_usable_memory_size=min(64 * GB, psutil.virtual_memory().total),
                secs_wq_poll_interval=secs_wq_poll_interval,
                secs_executor_probe_interval=secs_executor_probe_interval,
                max_num_missed_probes=max_num_missed_probes,
                fault_inject_prob=(fault_inject_prob if fault_inject_prob is not None else self.fault_inject_prob),
                enable_profiling=enable_profiling,
                enable_diagnostic_metrics=enable_diagnostic_metrics,
                remove_empty_parquet=remove_empty_parquet,
                skip_task_with_empty_input=skip_task_with_empty_input,
                console_log_level=console_log_level,
                file_log_level=file_log_level,
                output_path=output_path,
            )

        self.queue_manager = Manager()
        self.sched_states = self.queue_manager.Queue()

        exec_plan = Planner(runtime_ctx).create_exec_plan(plan)
        scheduler = Scheduler(
            exec_plan,
            max_retry_count=max_retry_count,
            max_fail_count=max_fail_count,
            prioritize_retry=prioritize_retry,
            speculative_exec=speculative_exec,
            stop_executor_on_failure=stop_executor_on_failure,
            nonzero_exitcode_as_oom=nonzero_exitcode_as_oom,
        )
        self.latest_state = scheduler
        self.executors = [Executor.create(runtime_ctx, f"executor-{i}") for i in range(num_executors)]
        self.processes = [
            Process(
                target=run_scheduler,
                # XXX: on macOS, scheduler state observer will be cleared when cross-process
                #      so we pass the queue and add the observer in the new process
                args=(runtime_ctx, scheduler, self.sched_states),
                name="scheduler",
            )
        ]
        self.processes += [Process(target=run_executor, args=(runtime_ctx, executor), name=executor.id) for executor in self.executors]

        for process in reversed(self.processes):
            process.start()

        return self.sched_states, self.executors, self.processes

    def execute_plan(self, *args, check_result=True, **kvargs) -> ExecutionPlan:
        """
        Start a scheduler and `num_executors` executors to execute `plan`,
        and wait the execution completed, then assert if it succeeds.

        Parameters
        ----------
        plan
            A logical plan.
        num_executors, optional
            The number of executors
        console_log_level, optional
            Set to logger.INFO if more verbose loguru is needed for debug, by default "WARNING".

        Returns
        -------
            The completed ExecutionPlan instance.
        """
        self.start_execution(*args, **kvargs)
        self.join_running_procs()
        latest_state = self.get_latest_sched_state()
        if check_result:
            self.assertTrue(latest_state.success)
        return latest_state.exec_plan

    def _load_parquet_files(self, paths, filesystem: fsspec.AbstractFileSystem = None) -> arrow.Table:
        def read_parquet_file(path):
            return arrow.Table.from_batches(parquet.ParquetFile(path, buffer_size=16 * MB, filesystem=filesystem).iter_batches())

        with ThreadPoolExecutor(16) as pool:
            return arrow.concat_tables(pool.map(read_parquet_file, paths))

    def _compare_arrow_tables(self, expected: arrow.Table, actual: arrow.Table):
        def sorted_table(t: arrow.Table):
            return t.sort_by([(col, "ascending") for col in t.column_names])

        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.column_names, actual.column_names)
        expected = sorted_table(cast_columns_to_large_string(expected))
        actual = sorted_table(cast_columns_to_large_string(actual))
        for col, x, y in zip(expected.column_names, expected.columns, actual.columns):
            if not pc.equal(x, y):
                x = x.to_numpy(zero_copy_only=False)
                y = y.to_numpy(zero_copy_only=False)
                logger.error(f"  expect {col}: {x}")
                logger.error(f"  actual {col}: {y}")
                np.testing.assert_array_equal(x, y, verbose=True)
