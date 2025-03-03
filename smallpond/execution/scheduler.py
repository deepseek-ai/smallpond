import copy
import cProfile
import itertools
import multiprocessing as mp
import os
import queue
import shutil
import socket
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import cached_property
from typing import Any, Callable, Dict, Iterable, List, Literal, Set, Tuple, Union

import numpy as np
from loguru import logger

from smallpond.common import (
    DEFAULT_MAX_FAIL_COUNT,
    DEFAULT_MAX_RETRY_COUNT,
    GB,
    MB,
    pytest_running,
)
from smallpond.execution.task import (
    ExecutionPlan,
    Probe,
    RuntimeContext,
    Task,
    TaskRuntimeId,
    WorkStatus,
)
from smallpond.execution.workqueue import (
    StopExecutor,
    StopWorkItem,
    WorkItem,
    WorkQueue,
    WorkQueueInMemory,
    WorkQueueOnFilesystem,
)
from smallpond.io.filesystem import dump, remove_path
from smallpond.logical.node import LogicalPlan, Node
from smallpond.utility import cprofile_to_string


class ExecutorState(Enum):
    GOOD = 1
    FAIL = 2
    RESOURCE_LOW = 3
    STOPPING = 4
    STOPPED = 5


class RemoteExecutor(object):
    def __init__(self, ctx: RuntimeContext, id: str, wq: WorkQueue, cq: WorkQueue, init_epoch=0) -> None:
        self.ctx = ctx
        self.id = id
        self.wq = wq
        self.cq = cq
        self.running_works: Dict[str, WorkItem] = {}
        self.state = ExecutorState.RESOURCE_LOW
        self.last_acked_probe = Probe(self.ctx, f"Probe-{self.id}#{0}", init_epoch)
        self.stop_request_sent = False
        self.stop_request_acked = False
        self._allocated_cpus = 0
        self._allocated_gpus = 0
        self._allocated_memory = 0

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.id}), running_works[{len(self.running_works)}]={list(self.running_works.keys())[:3]}..., \
allocated={self.allocated_cpus}CPUs/{self.allocated_gpus}GPUs/{self.allocated_memory//GB}GB, \
state={self.state}, probe={self.last_acked_probe}"

    def __repr__(self) -> str:
        return f"RemoteExecutor({self.id}):{self.state}"

    @staticmethod
    def create(ctx: RuntimeContext, id: str, queue_dir: str, init_epoch=0) -> "RemoteExecutor":
        wq = WorkQueueOnFilesystem(os.path.join(queue_dir, "wq"))
        cq = WorkQueueOnFilesystem(os.path.join(queue_dir, "cq"))
        return RemoteExecutor(ctx, id, wq, cq, init_epoch)

    @property
    def idle(self) -> bool:
        return len(self.running_works) == 0

    @property
    def busy(self) -> bool:
        return (
            len(self.running_works) >= self.max_running_works
            or (self.cpu_count > 0 and self.allocated_cpus >= self.cpu_count)
            or (self.gpu_count > 0 and self.allocated_gpus >= self.gpu_count)
            or (self.memory_size > 0 and self.allocated_memory >= self.memory_size)
        )

    @property
    def local(self) -> bool:
        return False

    @property
    def good(self) -> bool:
        return self.state == ExecutorState.GOOD

    @property
    def fail(self) -> bool:
        return self.state == ExecutorState.FAIL

    @property
    def stopping(self) -> bool:
        return self.state == ExecutorState.STOPPING

    @property
    def stopped(self) -> bool:
        return self.state == ExecutorState.STOPPED

    @property
    def resource_low(self) -> bool:
        return self.state == ExecutorState.RESOURCE_LOW

    @property
    def working(self) -> bool:
        return self.state in (ExecutorState.GOOD, ExecutorState.RESOURCE_LOW)

    @property
    def alive(self) -> bool:
        return self.state in (
            ExecutorState.GOOD,
            ExecutorState.RESOURCE_LOW,
            ExecutorState.STOPPING,
        )

    @property
    def cpu_count(self) -> int:
        return self.last_acked_probe.cpu_count

    @property
    def gpu_count(self) -> int:
        return self.last_acked_probe.gpu_count

    @property
    def memory_size(self) -> int:
        return self.last_acked_probe.total_memory

    @property
    def allocated_cpus(self) -> int:
        return self._allocated_cpus

    @property
    def allocated_gpus(self) -> int:
        return self._allocated_gpus

    @property
    def allocated_memory(self) -> int:
        return self._allocated_memory

    @property
    def available_cpus(self) -> int:
        return self.cpu_count - self.allocated_cpus

    @property
    def available_memory(self) -> int:
        return self.memory_size - self.allocated_memory

    @property
    def max_running_works(self) -> int:
        # limit max number of running works on an executor: reserve 1/16 cpu cores for filesystem and others
        return self.cpu_count - self.cpu_count // 16

    def add_running_work(self, item: WorkItem):
        assert item.key not in self.running_works, f"duplicate work item assigned to {repr(self)}: {item.key}"
        self.running_works[item.key] = item
        self._allocated_cpus += item.cpu_limit
        self._allocated_gpus += item.gpu_limit
        self._allocated_memory += item.memory_limit

    def pop_running_work(self, key: str):
        if (item := self.running_works.pop(key, None)) is None:
            logger.debug(f"work item {key} not found in running works of {self}")
        else:
            self._allocated_cpus -= item.cpu_limit
            self._allocated_gpus -= item.gpu_limit
            self._allocated_memory -= item.memory_limit
        return item

    def pop(self) -> List[Task]:
        finished_items = self.cq.pop(count=max(1, len(self.running_works)))
        finished_tasks = []

        for item in finished_items:
            if isinstance(item, Probe):
                self.last_acked_probe = max(self.last_acked_probe, item)
                continue

            if isinstance(item, StopWorkItem):
                self.pop_running_work(item.work_to_stop)
                logger.debug(f"work item stopped: {item}")
                continue

            if isinstance(item, StopExecutor):
                self.stop_request_acked |= self.state != ExecutorState.STOPPED
                logger.info(f"executor stopped: {item}")
            else:
                assert isinstance(item, Task), f"unexpected work item type: {item}"
                item.finish_time = item.finish_time or time.time()
                finished_tasks.append(item)

            if item.status != WorkStatus.INCOMPLETE:
                self.pop_running_work(item.key)

        return finished_tasks

    def push(self, item: WorkItem, buffering=False) -> bool:
        if item.key in self.running_works:
            logger.warning(f"work item {item.key} already exists in running works of {self}")
            return False
        item.start_time = time.time()
        item.exec_id = self.id
        self.add_running_work(item)
        return self.wq.push(item, buffering)

    def flush(self) -> bool:
        return self.wq.flush()

    def probe(self, epoch: int):
        self.wq.push(Probe(self.ctx, f".Probe-{self.id}#{epoch:06d}", epoch))

    def stop(self):
        if self.working and not self.stop_request_sent:
            logger.info(f"stopping remote executor: {self}")
            self.push(StopExecutor(f".StopExecutor-{self.id}"))
            self.stop_request_sent = True

    def reset_state(self, current_epoch: int):
        self.__init__(self.ctx, self.id, self.wq, self.cq, current_epoch)

    def update_state(self, current_epoch: int) -> bool:
        num_missed_probes = current_epoch - self.last_acked_probe.epoch
        if self.state == ExecutorState.STOPPED:
            return False
        elif num_missed_probes > self.ctx.max_num_missed_probes:
            if self.state != ExecutorState.FAIL:
                self.state = ExecutorState.FAIL
                logger.error(f"find failed executor: {self}, missed probes: {num_missed_probes}, current epoch: {current_epoch}")
                return True
        elif self.state == ExecutorState.STOPPING:
            if self.stop_request_acked:
                self.state = ExecutorState.STOPPED
                logger.info(f"find stopped executor: {self}")
                return True
        elif self.stop_request_sent:
            if self.state != ExecutorState.STOPPING:
                self.state = ExecutorState.STOPPING
                return True
        elif self.last_acked_probe.resource_low:
            if self.state != ExecutorState.RESOURCE_LOW:
                self.state = ExecutorState.RESOURCE_LOW
                logger.warning(f"find low-resource executor: {self}")
                return True
        elif self.last_acked_probe.status == WorkStatus.SUCCEED:
            if self.state != ExecutorState.GOOD:
                self.state = ExecutorState.GOOD
                logger.info(f"find working executor: {self}")
                return True
        return False


class LocalExecutor(RemoteExecutor):
    def __init__(self, ctx: RuntimeContext, id: str, wq: WorkQueue, cq: WorkQueue) -> None:
        super().__init__(ctx, id, wq, cq)
        self.work = None
        self.running = False

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["wq"]
        del state["cq"]
        del state["work"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.wq = WorkQueueInMemory(queue_type=queue.Queue)
        self.cq = WorkQueueInMemory(queue_type=queue.Queue)
        self.work = None

    @staticmethod
    def create(ctx: RuntimeContext, id: str) -> "LocalExecutor":
        wq = WorkQueueInMemory(queue_type=queue.Queue)
        cq = WorkQueueInMemory(queue_type=queue.Queue)
        return LocalExecutor(ctx, id, wq, cq)

    @logger.catch(reraise=True, message="local executor terminated unexpectedly")
    def run(self):
        logger.info(f"local executor started: {self.id}")
        local_gpus = self.ctx.get_local_gpus()

        while self.running:
            items = self.wq.pop()

            if len(items) == 0:
                time.sleep(self.ctx.secs_wq_poll_interval)
                continue

            for item in items:
                if not self.running:
                    break
                if item.gpu_limit > 0:
                    assert len(local_gpus) > 0
                    item._local_gpu = local_gpus[0]
                    logger.info(f"{repr(item)} is assigned to run on GPU #{item.local_rank}: {item.local_gpu}")

                item = copy.copy(item)
                item.exec()
                self.cq.push(item, buffering=True)

            self.cq.flush()

        logger.info(f"local executor exits: {self.id}")
        logger.complete()

    @property
    def local(self) -> bool:
        return True

    def start(self, pool: ThreadPoolExecutor):
        self.running = True
        self.running_works.clear()
        self.work = pool.submit(self.run)
        self.state = ExecutorState.GOOD

    def stop(self):
        if self.working:
            logger.info(f"stopping local executor: {self}")
            self.running = False
            self.state = ExecutorState.STOPPING
            self.work.result(timeout=self.ctx.secs_executor_probe_interval)
            self.state = ExecutorState.STOPPED


class Scheduler(object):
    """
    The task scheduler.
    """

    large_num_nontrivial_tasks = 200 if pytest_running() else 20000
    StateCallback = Callable[["Scheduler"], Any]

    class StateObserver(object):
        def __init__(self, callback: "Scheduler.StateCallback" = None) -> None:
            assert callback is None or isinstance(callback, Callable)
            self.enabled = True
            self.callback = callback

        def __repr__(self) -> str:
            return repr(self.callback) if self.callback is not None else super().__repr__()

        __str__ = __repr__

        def update(self, sched_state: "Scheduler"):
            assert self.callback is not None
            self.callback(sched_state)

    def __init__(
        self,
        exec_plan: ExecutionPlan,
        max_retry_count: int = DEFAULT_MAX_RETRY_COUNT,
        max_fail_count: int = DEFAULT_MAX_FAIL_COUNT,
        prioritize_retry=False,
        speculative_exec: Literal["disable", "enable", "aggressive"] = "enable",
        stop_executor_on_failure=False,
        nonzero_exitcode_as_oom=False,
        remove_output_root=False,
        sched_state_observers=None,
    ) -> None:
        self.ctx = exec_plan.ctx
        self.exec_plan = exec_plan
        self.logical_plan: LogicalPlan = self.exec_plan.logical_plan
        self.logical_nodes = self.logical_plan.nodes
        self.max_retry_count = max_retry_count
        self.max_fail_count = max_fail_count
        self.standalone_mode = self.ctx.num_executors == 0
        self.prioritize_retry = prioritize_retry
        self.disable_speculative_exec = speculative_exec == "disable"
        self.aggressive_speculative_exec = speculative_exec == "aggressive"
        self.stop_executor_on_failure = stop_executor_on_failure
        self.nonzero_exitcode_as_oom = nonzero_exitcode_as_oom
        self.remove_output_root = remove_output_root
        self.sched_state_observers: List[Scheduler.StateObserver] = sched_state_observers or []
        self.secs_state_notify_interval = self.ctx.secs_executor_probe_interval * 2
        # task states
        self.local_queue: List[Task] = []
        self.sched_queue: List[Task] = []
        self.tasks: Dict[str, Task] = self.exec_plan.tasks
        self.scheduled_tasks: Dict[TaskRuntimeId, Task] = OrderedDict()
        self.finished_tasks: Dict[TaskRuntimeId, Task] = OrderedDict()
        self.succeeded_tasks: Dict[str, Task] = OrderedDict()
        self.nontrivial_tasks = dict((key, task) for (key, task) in self.tasks.items() if not task.exec_on_scheduler)
        self.succeeded_nontrivial_tasks: Dict[str, Task] = OrderedDict()
        # executor pool
        self.local_executor = LocalExecutor.create(self.ctx, "localhost")
        self.available_executors = {self.local_executor.id: self.local_executor}
        # other runtime states
        self.sched_running = False
        self.sched_start_time = 0
        self.last_executor_probe_time = 0
        self.last_state_notify_time = 0
        self.probe_epoch = 0
        self.sched_epoch = 0

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["sched_state_observers"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.sched_state_observers = []

    @property
    def elapsed_time(self):
        return time.time() - self.sched_start_time

    @property
    def success(self) -> bool:
        return self.exec_plan.root_task.key in self.succeeded_tasks

    @property
    def progress(self) -> Tuple[int, int, float]:
        num_succeeded = len(self.succeeded_nontrivial_tasks)
        num_tasks = len(self.nontrivial_tasks)
        return num_succeeded, num_tasks, num_succeeded / num_tasks * 100.0

    @property
    def large_runtime_state(self) -> bool:
        return len(self.nontrivial_tasks) > self.large_num_nontrivial_tasks

    @property
    def running_works(self) -> Iterable[WorkItem]:
        return (work for executor in (self.alive_executors + self.local_executors) for work in executor.running_works.values())

    @property
    def num_running_works(self) -> int:
        return sum(len(executor.running_works) for executor in (self.alive_executors + self.local_executors))

    @property
    def num_local_running_works(self) -> int:
        return sum(len(executor.running_works) for executor in self.local_executors)

    @property
    def num_pending_tasks(self) -> int:
        assert len(self.tasks) >= len(
            self.succeeded_tasks
        ), f"number of tasks {len(self.tasks)} < number of succeeded tasks {len(self.succeeded_tasks)}"
        return len(self.tasks) - len(self.succeeded_tasks)

    @property
    def pending_nontrivial_tasks(self) -> Dict[str, Task]:
        return dict((key, task) for key, task in self.nontrivial_tasks.items() if key not in self.succeeded_nontrivial_tasks)

    @property
    def num_pending_nontrivial_tasks(self) -> int:
        assert len(self.nontrivial_tasks) >= len(
            self.succeeded_nontrivial_tasks
        ), f"number of nontrivial tasks {len(self.nontrivial_tasks)} < number of succeeded nontrivial tasks {len(self.succeeded_nontrivial_tasks)}"
        return len(self.nontrivial_tasks) - len(self.succeeded_nontrivial_tasks)

    @property
    def succeeded_task_ids(self) -> Set[TaskRuntimeId]:
        return set(TaskRuntimeId(task.id, task.sched_epoch, task.retry_count) for task in self.succeeded_tasks.values())

    @property
    def abandoned_tasks(self) -> List[Task]:
        succeeded_task_ids = self.succeeded_task_ids
        return [task for task in {**self.scheduled_tasks, **self.finished_tasks}.values() if task.runtime_id not in succeeded_task_ids]

    @cached_property
    def remote_executors(self) -> List[RemoteExecutor]:
        return [executor for executor in self.available_executors.values() if not executor.local]

    @cached_property
    def local_executors(self) -> List[RemoteExecutor]:
        return [executor for executor in self.available_executors.values() if executor.local]

    @cached_property
    def working_executors(self) -> List[RemoteExecutor]:
        return [executor for executor in self.remote_executors if executor.working]

    @cached_property
    def alive_executors(self) -> List[RemoteExecutor]:
        return [executor for executor in self.remote_executors if executor.alive]

    @cached_property
    def good_executors(self) -> List[RemoteExecutor]:
        return [executor for executor in self.remote_executors if executor.good]

    @cached_property
    def failed_executors(self) -> List[RemoteExecutor]:
        return [executor for executor in self.remote_executors if executor.fail]

    @cached_property
    def stopped_executors(self) -> List[RemoteExecutor]:
        return [executor for executor in self.remote_executors if executor.stopped]

    @cached_property
    def stopping_executors(self) -> List[RemoteExecutor]:
        return [executor for executor in self.remote_executors if executor.stopping]

    @cached_property
    def low_resource_executors(self) -> List[RemoteExecutor]:
        return [executor for executor in self.remote_executors if executor.resource_low]

    def suspend_good_executors(self):
        for executor in self.good_executors:
            executor.reset_state(self.probe_epoch)
        self.clear_cached_executor_lists()

    def clear_cached_executor_lists(self):
        if hasattr(self, "remote_executors"):
            del self.remote_executors
        if hasattr(self, "local_executors"):
            del self.local_executors
        if hasattr(self, "working_executors"):
            del self.working_executors
        if hasattr(self, "alive_executors"):
            del self.alive_executors
        if hasattr(self, "good_executors"):
            del self.good_executors
        if hasattr(self, "failed_executors"):
            del self.failed_executors
        if hasattr(self, "stopped_executors"):
            del self.stopped_executors
        if hasattr(self, "stopping_executors"):
            del self.stopping_executors
        if hasattr(self, "low_resource_executors"):
            del self.low_resource_executors

    def stop_executors(self):
        for exec in self.available_executors.values():
            exec.stop()

    def start_speculative_execution(self):
        for executor in self.working_executors:
            for idx, item in enumerate(executor.running_works.values()):
                aggressive_retry = self.aggressive_speculative_exec and len(self.good_executors) >= self.ctx.num_executors
                short_sched_queue = len(self.sched_queue) < len(self.good_executors)
                if (
                    isinstance(item, Task)
                    and item.key not in self.succeeded_tasks
                    and item.allow_speculative_exec
                    and item.retry_count < self.max_retry_count
                    and item.retry_count == self.tasks[item.key].retry_count
                    and (logical_node := self.logical_nodes.get(item.node_id, None)) is not None
                ):
                    perf_stats = logical_node.get_perf_stats("elapsed wall time (secs)")
                    if perf_stats is not None and perf_stats.cnt >= 20:
                        if short_sched_queue:
                            retry_threshold = max(
                                self.ctx.secs_executor_probe_timeout,
                                perf_stats.p95 - perf_stats.p50,
                            )
                        elif aggressive_retry:
                            retry_threshold = max(
                                self.ctx.secs_executor_probe_timeout,
                                perf_stats.p99 - perf_stats.p50,
                            )
                        else:
                            retry_threshold = max(
                                self.ctx.secs_executor_probe_timeout,
                                perf_stats.p99 - perf_stats.p50,
                            ) * (2 + item.retry_count)
                        excess_time = item.elapsed_time - perf_stats.p50
                        if excess_time >= retry_threshold:
                            logger.warning(
                                f"retry long-running task: {repr(item)} on {repr(executor)}, elapsed time: {item.elapsed_time:.1f} secs, elapsed time stats: {perf_stats}"
                            )
                            self.try_enqueue(self.get_retry_task(item.key))

    def probe_executors(self):
        secs_since_last_executor_probe = time.time() - self.last_executor_probe_time
        if secs_since_last_executor_probe >= self.ctx.secs_executor_probe_interval:
            # discover new executors
            with os.scandir(self.ctx.queue_root) as dir_iter:
                for entry in dir_iter:
                    if entry.is_dir():
                        _, exec_id = os.path.split(entry.path)
                        if exec_id not in self.available_executors:
                            self.available_executors[exec_id] = RemoteExecutor.create(self.ctx, exec_id, entry.path, self.probe_epoch)
                            logger.info(f"find a new executor #{len(self.available_executors)}: {self.available_executors[exec_id]}")
                            self.clear_cached_executor_lists()
            # start a new probe epoch
            self.last_executor_probe_time = time.time()
            self.probe_epoch += 1
            logger.info(
                f"send a new round of probes #{self.probe_epoch} to {len(self.working_executors)} working executors: {self.working_executors}"
            )
            for executor in self.working_executors:
                executor.probe(self.probe_epoch)
            # start speculative execution of tasks
            if not self.disable_speculative_exec:
                self.start_speculative_execution()

    def update_executor_states(self):
        executor_state_changed = []
        for executor in self.alive_executors:
            old_state = executor.state
            executor_state_changed.append(executor.update_state(self.probe_epoch))
            if executor.state == ExecutorState.FAIL and executor.state != old_state:
                for item in executor.running_works.values():
                    item.status = WorkStatus.EXEC_FAILED
                    item.finish_time = time.time()
                    if isinstance(item, Task) and item.key not in self.succeeded_tasks:
                        logger.warning(f"reschedule {repr(item)} on failed executor: {repr(executor)}")
                        self.try_enqueue(self.get_retry_task(item.key))

        if any(executor_state_changed):
            self.clear_cached_executor_lists()
            logger.info(
                f"in total {len(self.available_executors)} executors: "
                f"{len(self.local_executors)} local, "
                f"{len(self.good_executors)} good, "
                f"{len(self.failed_executors)} failed, "
                f"{len(self.stopped_executors)} stopped, "
                f"{len(self.stopping_executors)} stopping, "
                f"{len(self.low_resource_executors)} low-resource"
            )

    def copy_task_for_execution(self, task: Task) -> Task:
        task = copy.copy(task)
        # remove the reference to input deps
        task.input_deps = {dep_key: None for dep_key in task.input_deps}
        # feed input datasets
        task.input_datasets = [self.succeeded_tasks[dep_key].output for dep_key in task.input_deps]
        task.sched_epoch = self.sched_epoch
        return task

    def save_task_final_state(self, finished_task: Task):
        # update perf metrics of logical node
        logical_node: Node = self.logical_nodes.get(finished_task.node_id, None)
        if logical_node is not None:
            for name, value in finished_task.perf_metrics.items():
                logical_node.add_perf_metrics(name, value)

        # update task instance in execution plan
        task = self.tasks[finished_task.key]
        task.status = finished_task.status
        task.start_time = finished_task.start_time
        task.finish_time = finished_task.finish_time
        task.retry_count = finished_task.retry_count
        task.sched_epoch = finished_task.sched_epoch
        task.dataset = finished_task.dataset

    def get_runnable_tasks(self, finished_task: Task) -> Iterable[Task]:
        assert finished_task.status == WorkStatus.SUCCEED, f"task not succeeded: {finished_task}"
        for output_key in finished_task.output_deps:
            output_dep = self.tasks[output_key]
            if all(key in self.succeeded_tasks for key in output_dep.input_deps):
                logger.trace(
                    "{} initiates a new runnable task: {}",
                    repr(finished_task),
                    repr(output_dep),
                )
                yield output_dep

    def stop_running_tasks(self, task_key: str):
        for executor in self.remote_executors:
            running_task = executor.running_works.get(task_key, None)
            if running_task is not None:
                logger.info(f"try to stop {repr(running_task)} running on {repr(executor)}")
                executor.wq.push(StopWorkItem(f".StopWorkItem-{repr(running_task)}", running_task.key))

    def try_relax_memory_limit(self, task: Task, executor: RemoteExecutor) -> bool:
        if task.memory_limit >= executor.memory_size:
            logger.warning(f"failed to relax memory limit of {task}")
            return False
        relaxed_memory_limit = min(executor.memory_size, task.memory_limit * 2)
        task._memory_boost = relaxed_memory_limit / task._memory_limit
        logger.warning(f"relax memory limit of {task.key} to {task.memory_limit/GB:.3f}GB and retry ...")
        return True

    def try_boost_resource(self, item: WorkItem, executor: RemoteExecutor):
        if (
            item._cpu_boost == 1
            and item._memory_boost == 1
            and isinstance(item, Task)
            and item.node_id in self.logical_nodes
            and self.logical_nodes[item.node_id].enable_resource_boost
        ):
            boost_cpu = max(
                item._cpu_limit,
                min(
                    item._cpu_limit * 2,
                    executor.available_cpus,
                    executor.cpu_count // 2,
                ),
            )
            boost_mem = max(
                item._memory_limit,
                min(
                    item._memory_limit * 2,
                    executor.available_memory,
                    executor.memory_size // 2,
                ),
            )
            if item._cpu_limit < boost_cpu or item._memory_limit < boost_mem:
                item._cpu_boost = boost_cpu / item._cpu_limit
                item._memory_boost = boost_mem / item._memory_limit
                logger.info(f"boost resource usage of {repr(item)}: {item.cpu_limit} CPUs, {item.memory_limit/GB:.3f}GB")

    def get_retry_task(self, key: str) -> Task:
        task = self.tasks[key]
        task.retry_count += 1
        assert task.status != WorkStatus.SUCCEED or task.sched_epoch != self.sched_epoch
        return task

    @logger.catch(reraise=pytest_running(), message="failed to clean temp files")
    def clean_temp_files(self, pool: ThreadPoolExecutor):
        remove_path(self.ctx.queue_root)
        remove_path(self.ctx.temp_root)
        remove_path(self.ctx.staging_root)

        if abandoned_tasks := self.abandoned_tasks:
            logger.info(f"removing outputs of {len(abandoned_tasks)} abandoned tasks: {abandoned_tasks[:3]} ...")
            assert list(pool.map(lambda t: t.clean_output(force=True), abandoned_tasks))

    @logger.catch(reraise=pytest_running(), message="failed to export task metrics")
    def export_task_metrics(self):
        import pyarrow as arrow
        import pyarrow.csv as csv

        def pristine_attrs_dict(task: Task):
            return {
                key: str(val) if isinstance(val, Enum) else val
                for key in task._pristine_attrs
                if isinstance(
                    val := getattr(task, key),
                    (bool, str, int, float, Enum, np.integer, np.floating),
                )
            }

        dump(
            self.finished_tasks,
            os.path.join(self.ctx.config_root, "finished_tasks.pickle"),
            buffering=32 * MB,
        )
        dump(
            self.scheduled_tasks,
            os.path.join(self.ctx.config_root, "scheduled_tasks.pickle"),
            buffering=32 * MB,
        )

        task_props = arrow.array(pristine_attrs_dict(task) for task in self.finished_tasks.values())
        partition_infos = arrow.array(task.partition_infos_as_dict for task in self.finished_tasks.values())
        perf_metrics = arrow.array(dict(task.perf_metrics) for task in self.finished_tasks.values())
        task_metrics = arrow.Table.from_arrays(
            [task_props, partition_infos, perf_metrics],
            names=["task_props", "partition_infos", "perf_metrics"],
        )

        task_metrics_csv = os.path.join(self.ctx.log_root, "task_metrics.csv")
        csv.write_csv(task_metrics.flatten(), task_metrics_csv)

        if self.ctx.shared_log_root:
            shutil.copy(task_metrics_csv, self.ctx.shared_log_root)
        logger.debug(f"exported task metrics to {task_metrics_csv}")

    @logger.catch(reraise=pytest_running(), message="failed to export timeline figures")
    def export_timeline_figs(self):
        from datetime import datetime

        import pandas as pd
        import plotly.express as px

        if self.large_runtime_state:
            logger.debug(f"pause exporting timeline figure")
            return

        now = datetime.now()
        task_data = pd.DataFrame(
            [
                dict(
                    task=repr(task),
                    node=(repr(node) if (node := self.logical_nodes.get(task.node_id, None)) is not None else "StandaloneTasks"),
                    status=str(task.status),
                    executor=task.exec_id,
                    start_time=datetime.fromtimestamp(task.start_time),
                    finish_time=datetime.fromtimestamp(
                        max(
                            task.finish_time or now.timestamp(),
                            task.start_time + 0.0001,
                        )
                    ),
                    elapsed_time=task.elapsed_time,
                    partition=str(task.partition_infos),
                    cpu_limit=task.cpu_limit,
                    gpu_limit=task.gpu_limit,
                    mem_limit=task.memory_limit,
                )
                for task in {**self.scheduled_tasks, **self.finished_tasks}.values()
                if task.start_time is not None
            ]
        )

        if task_data.empty:
            return

        timeline_figs = [
            px.timeline(
                task_data,
                x_start="start_time",
                x_end="finish_time",
                y="node",
                color="executor",
                hover_name="task",
                hover_data=task_data.columns,
                title="plan_timeline - progress: {}/{} ({:.1f}%), elapsed: {:.1f} secs, job: {}".format(
                    *self.progress, self.elapsed_time, self.ctx.job_id
                ),
                opacity=0.3,
            ),
            px.timeline(
                task_data,
                x_start="start_time",
                x_end="finish_time",
                y="executor",
                color="node",
                hover_name="task",
                hover_data=task_data.columns,
                title="exec_timeline - progress: {}/{} ({:.1f}%), elapsed: {:.1f} secs, job: {}".format(
                    *self.progress, self.elapsed_time, self.ctx.job_id
                ),
                opacity=0.3,
            ),
        ]

        for fig in timeline_figs:
            fig_title = str(fig.layout["title_text"])
            fig_filename, _ = fig_title.split(" - ", maxsplit=1)
            fig_filename += ".html"
            fig_path = os.path.join(self.ctx.log_root, fig_filename)
            fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
            fig.update_traces(marker_line_color="black", marker_line_width=1, opacity=1)
            fig.write_html(fig_path, include_plotlyjs="cdn" if pytest_running() else True)
            if self.ctx.shared_log_root:
                shutil.copy(fig_path, self.ctx.shared_log_root)
            logger.debug(f"exported timeline figure to {fig_path}")

    def notify_state_observers(self, force_notify=False) -> bool:
        secs_since_last_state_notify = time.time() - self.last_state_notify_time
        if force_notify or secs_since_last_state_notify >= self.secs_state_notify_interval:
            self.last_state_notify_time = time.time()
            for observer in self.sched_state_observers:
                if force_notify or observer.enabled:
                    start_time = time.time()
                    observer.update(self)
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= self.ctx.secs_executor_probe_interval / 2:
                        self.secs_state_notify_interval = self.ctx.secs_executor_probe_timeout
                    if elapsed_time >= self.ctx.secs_executor_probe_interval:
                        observer.enabled = False
                        logger.warning(f"disabled slow scheduler state observer (elapsed time: {elapsed_time:.1f} secs): {observer}")
            return True
        else:
            return False

    def add_state_observer(self, observer: StateObserver):
        self.sched_state_observers.append(observer)
        logger.info(f"added a scheduler state observer: {observer}")

    def log_overall_progress(self):
        logger.info(
            "overall progress: {}/{} ({:.1f}%), ".format(*self.progress)
            + f"{len(self.local_queue) + len(self.sched_queue)} queued works: {self.local_queue[:3] + self.sched_queue[:3]}, "
            + f"{self.num_running_works} running works: {list(itertools.islice(self.running_works, 3))} ..."
        )

    def log_current_status(self):
        with open(self.ctx.job_status_path, "w") as fout:
            if self.sched_running:
                status = "running"
            elif self.success:
                status = "success"
            else:
                status = "failure"
            fout.write(f"{status}@{int(time.time())}")

    def run(self) -> bool:
        mp.current_process().name = f"SchedulerMainProcess#{self.sched_epoch}"
        logger.info(f"start to run scheduler #{self.sched_epoch} on {socket.gethostname()}")

        perf_profile = None
        if self.ctx.enable_profiling:
            perf_profile = cProfile.Profile()
            perf_profile.enable()

        with ThreadPoolExecutor(32) as pool:
            self.sched_running = True
            self.sched_start_time = time.time()
            self.last_executor_probe_time = 0
            self.last_state_notify_time = 0
            self.prioritize_retry |= self.sched_epoch > 0

            if self.local_queue or self.sched_queue:
                pending_tasks = [item for item in self.local_queue + self.sched_queue if isinstance(item, Task)]
                self.local_queue.clear()
                self.sched_queue.clear()
                logger.info(f"requeue {len(pending_tasks)} pending tasks with latest epoch #{self.sched_epoch}: {pending_tasks[:3]} ...")
                self.try_enqueue(pending_tasks)

            if self.sched_epoch == 0:
                leaf_tasks = self.exec_plan.leaves
                logger.info(f"enqueue {len(leaf_tasks)} leaf tasks: {leaf_tasks[:3]} ...")
                self.try_enqueue(leaf_tasks)

            self.log_overall_progress()
            while (num_finished_tasks := self.process_finished_tasks(pool)) > 0:
                logger.info(f"processed {num_finished_tasks} finished tasks during startup")
                self.log_overall_progress()

            earlier_running_tasks = [item for item in self.running_works if isinstance(item, Task)]
            if earlier_running_tasks:
                logger.info(f"enqueue {len(earlier_running_tasks)} earlier running tasks: {earlier_running_tasks[:3]} ...")
                self.try_enqueue(earlier_running_tasks)

            self.suspend_good_executors()
            self.add_state_observer(Scheduler.StateObserver(Scheduler.log_current_status))
            self.add_state_observer(Scheduler.StateObserver(Scheduler.export_timeline_figs))
            self.notify_state_observers(force_notify=True)

            try:
                self.local_executor.start(pool)
                self.sched_loop(pool)
            finally:
                logger.info(f"schedule loop stopped")
                self.sched_running = False
                self.notify_state_observers(force_notify=True)
                self.export_task_metrics()
                self.stop_executors()

            # if --output_path is specified, remove the output root as well
            if self.remove_output_root or self.ctx.final_output_path:
                remove_path(self.ctx.staging_root)
                remove_path(self.ctx.output_root)

            if self.success:
                self.clean_temp_files(pool)
                logger.success(f"final output path: {self.exec_plan.final_output_path}")
                logger.info(f"analyzed plan:{os.linesep}{self.exec_plan.analyzed_logical_plan.explain_str()}")

        if perf_profile is not None:
            logger.debug(f"scheduler perf profile:{os.linesep}{cprofile_to_string(perf_profile)}")

        logger.info(f"scheduler of job {self.ctx.job_id} exits")
        logger.complete()
        return self.success

    def try_enqueue(self, tasks: Union[Iterable[Task], Task]):
        tasks = tasks if isinstance(tasks, Iterable) else [tasks]
        for task in tasks:
            task = self.copy_task_for_execution(task)
            if task.key in self.succeeded_tasks:
                logger.debug(f"task {repr(task)} already succeeded, skipping")
                self.try_enqueue(self.get_runnable_tasks(self.succeeded_tasks[task.key]))
                continue
            if task.runtime_id in self.scheduled_tasks:
                logger.debug(f"task {repr(task)} already scheduled, skipping")
                continue
            # save enqueued task
            self.scheduled_tasks[task.runtime_id] = task
            if self.standalone_mode or task.exec_on_scheduler or task.skip_when_any_input_empty:
                self.local_queue.append(task)
            else:
                self.sched_queue.append(task)

    def sched_loop(self, pool: ThreadPoolExecutor) -> bool:
        has_progress = True
        do_notify = False

        if self.success:
            logger.success(f"job already succeeded, stopping scheduler ...")
            return True

        while self.sched_running:
            self.probe_executors()
            self.update_executor_states()

            if self.local_queue:
                assert self.local_executor.alive
                logger.info(f"running {len(self.local_queue)} works on local executor: {self.local_queue[:3]} ...")
                self.local_queue = [item for item in self.local_queue if not self.local_executor.push(item, buffering=True)]
                self.local_executor.flush()

            has_progress |= self.dispatch_tasks(pool) > 0

            if len(self.sched_queue) == 0 and self.num_pending_nontrivial_tasks + 1 < len(self.good_executors):
                for executor in self.good_executors:
                    if executor.idle:
                        logger.info(f"{len(self.good_executors)} remote executors running, stopping {executor}")
                        executor.stop()
                        break

            if len(self.sched_queue) == 0 and len(self.local_queue) == 0 and self.num_running_works == 0:
                self.log_overall_progress()
                assert (
                    self.num_pending_tasks == 0
                ), f"scheduler stuck in idle state, but still have {self.num_pending_tasks} pending tasks: {self.tasks.keys() - self.succeeded_tasks.keys()}"
                logger.info(f"no queued or running works, stopping scheduler ...")
                break

            if has_progress:
                has_progress = False
                do_notify = True
                self.log_overall_progress()
            else:
                time.sleep(self.ctx.secs_wq_poll_interval)

            if do_notify:
                do_notify = not self.notify_state_observers()

            has_progress |= self.process_finished_tasks(pool) > 0

        # out of loop
        return self.success

    def dispatch_tasks(self, pool: ThreadPoolExecutor):
        # sort pending tasks
        item_sort_key = (lambda item: (-item.retry_count, item.id)) if self.prioritize_retry else (lambda item: (item.retry_count, item.id))
        items_sorted_by_node_id = sorted(self.sched_queue, key=lambda t: t.node_id)
        items_group_by_node_id = itertools.groupby(items_sorted_by_node_id, key=lambda t: t.node_id)
        sorted_item_groups = [sorted(items, key=item_sort_key) for _, items in items_group_by_node_id]
        self.sched_queue = [item for batch in itertools.zip_longest(*sorted_item_groups, fillvalue=None) for item in batch if item is not None]

        final_phase = self.num_pending_nontrivial_tasks - self.num_running_works <= len(self.good_executors) * 2
        num_dispatched_tasks = 0
        unassigned_tasks = []

        while self.sched_queue and self.good_executors:
            first_item = self.sched_queue[0]

            # assign tasks to executors in round-robin fashion
            usable_executors = [executor for executor in self.good_executors if not executor.busy]
            for executor in sorted(usable_executors, key=lambda exec: len(exec.running_works)):
                if not self.sched_queue:
                    break
                item = self.sched_queue[0]

                if item._memory_limit is None:
                    item._memory_limit = np.int64(executor.memory_size * item._cpu_limit // executor.cpu_count)

                if item.key in self.succeeded_tasks:
                    logger.debug(f"task {repr(item)} already succeeded, skipping")
                    self.sched_queue.pop(0)
                    self.try_enqueue(self.get_runnable_tasks(self.succeeded_tasks[item.key]))
                elif (
                    len(executor.running_works) < executor.max_running_works
                    and executor.allocated_cpus + item.cpu_limit <= executor.cpu_count
                    and executor.allocated_gpus + item.gpu_limit <= executor.gpu_count
                    and executor.allocated_memory + item.memory_limit <= executor.memory_size
                    and item.key not in executor.running_works
                ):
                    if final_phase:
                        self.try_boost_resource(item, executor)
                    # push to wq of executor but not flushed yet
                    executor.push(item, buffering=True)
                    logger.info(f"appended {repr(item)} ({item.cpu_limit} CPUs, {item.memory_limit/GB:.3f}GB) to the queue of {executor}")
                    self.sched_queue.pop(0)
                    num_dispatched_tasks += 1

            if self.sched_queue and self.sched_queue[0] is first_item:
                unassigned_tasks.append(self.sched_queue.pop(0))

        # append unassigned tasks to the queue
        self.sched_queue.extend(unassigned_tasks)

        # flush the buffered work items into wq
        assert all(pool.map(RemoteExecutor.flush, self.good_executors)), f"failed to flush work queues"
        return num_dispatched_tasks

    def process_finished_tasks(self, pool: ThreadPoolExecutor) -> int:
        pop_results = pool.map(RemoteExecutor.pop, self.available_executors.values())
        num_finished_tasks = 0

        for executor, finished_tasks in zip(self.available_executors.values(), pop_results):

            for finished_task in finished_tasks:
                assert isinstance(finished_task, Task)

                scheduled_task = self.scheduled_tasks.get(finished_task.runtime_id, None)
                if scheduled_task is None:
                    logger.info(f"task not initiated by current scheduler: {finished_task}")
                    if finished_task.status != WorkStatus.SUCCEED and (
                        missing_inputs := [key for key in finished_task.input_deps if key not in self.succeeded_tasks]
                    ):
                        logger.info(f"ignore {repr(finished_task)} since some of the input deps are missing: {missing_inputs}")
                        continue

                if finished_task.status == WorkStatus.INCOMPLETE:
                    logger.trace(f"{repr(finished_task)} checkpoint created on {executor.id}: {finished_task.runtime_state}")
                    self.tasks[finished_task.key].runtime_state = finished_task.runtime_state
                    continue

                prior_task = self.finished_tasks.get(finished_task.runtime_id, None)
                if prior_task is not None:
                    logger.info(f"found duplicate tasks, current: {repr(finished_task)}, prior: {repr(prior_task)}")
                    continue
                else:
                    self.finished_tasks[finished_task.runtime_id] = finished_task
                    num_finished_tasks += 1

                succeeded_task = self.succeeded_tasks.get(finished_task.key, None)
                if succeeded_task is not None:
                    logger.info(f"task already succeeded, current: {repr(finished_task)}, succeeded: {repr(succeeded_task)}")
                    continue

                if finished_task.status in (WorkStatus.FAILED, WorkStatus.CRASHED):
                    logger.warning(f"task failed on {executor.id}: {finished_task}, error: {finished_task.exception}")
                    finished_task.dump()

                    task = self.tasks[finished_task.key]
                    task.fail_count += 1

                    if task.fail_count > self.max_fail_count:
                        logger.critical(f"task failed too many times: {finished_task}, stopping ...")
                        self.stop_executors()
                        self.sched_running = False

                    if not executor.local and finished_task.oom(self.nonzero_exitcode_as_oom):
                        if task._memory_limit is None:
                            task._memory_limit = finished_task._memory_limit
                        self.try_relax_memory_limit(task, executor)

                    if not executor.local and self.stop_executor_on_failure:
                        logger.warning(f"stop executor: {executor}")
                        executor.stop()

                    self.try_enqueue(self.get_retry_task(finished_task.key))
                else:
                    assert finished_task.status == WorkStatus.SUCCEED, f"unexpected task status: {finished_task}"
                    logger.log(
                        "TRACE" if finished_task.exec_on_scheduler else "INFO",
                        "task succeeded on {}: {}",
                        finished_task.exec_id,
                        finished_task,
                    )

                    self.succeeded_tasks[finished_task.key] = finished_task
                    if not finished_task.exec_on_scheduler:
                        self.succeeded_nontrivial_tasks[finished_task.key] = finished_task

                    # stop the redundant retries of finished task
                    self.stop_running_tasks(finished_task.key)
                    self.save_task_final_state(finished_task)
                    self.try_enqueue(self.get_runnable_tasks(finished_task))

                    if finished_task.id == self.exec_plan.root_task.id:
                        self.sched_queue = []
                        self.stop_executors()
                        logger.success(f"all tasks completed, root task: {finished_task}")
                        logger.success(
                            f"{len(self.succeeded_tasks)} succeeded tasks, success: {self.success}, elapsed time: {self.elapsed_time:.3f} secs"
                        )

                # clear inputs since they are not needed anymore
                finished_task.input_datasets = []

        return num_finished_tasks
