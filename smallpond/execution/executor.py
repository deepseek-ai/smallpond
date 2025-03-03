import multiprocessing as mp
import os.path
import socket
import time
from collections import OrderedDict
from typing import Dict, List, Tuple

from GPUtil import GPU
from loguru import logger

from smallpond.common import NonzeroExitCode, pytest_running
from smallpond.execution.task import Probe, RuntimeContext
from smallpond.execution.workqueue import (
    StopExecutor,
    StopWorkItem,
    WorkItem,
    WorkQueue,
    WorkQueueOnFilesystem,
    WorkStatus,
)


class SimplePoolTask(object):
    def __init__(self, func, args, name: str):
        self.proc: mp.Process = mp.Process(target=func, args=args, name=name)
        self.stopping = False

    def start(self):
        self.proc.start()

    def terminate(self):
        self.proc.terminate()
        self.stopping = True

    def join(self, timeout=None):
        self.proc.join(timeout)
        if not self.ready() and timeout is not None:
            logger.warning(f"worker process {self.proc.name}({self.proc.pid}) does not exit after {timeout} secs, stopping it")
            self.terminate()
            self.proc.join()

    def ready(self):
        return self.proc.pid and not self.proc.is_alive()

    def exitcode(self):
        assert self.ready(), f"worker process {self.proc.name}({self.proc.pid}) has not exited yet"
        if self.stopping:
            logger.info(f"worker process stopped: {self.proc.name}({self.proc.pid}), exitcode: {self.proc.exitcode}")
        elif self.proc.exitcode != 0:
            logger.error(f"worker process crashed: {self.proc.name}({self.proc.pid}), exitcode: {self.proc.exitcode}")
        return self.proc.exitcode


class SimplePool(object):
    def __init__(self, pool_size: int):
        self.pool_size = pool_size
        self.queued_tasks: List[SimplePoolTask] = []
        self.running_tasks: List[SimplePoolTask] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.join(1)

    def apply_async(self, func, args, name: str = None):
        task = SimplePoolTask(func, args, name)
        self.queued_tasks.append(task)
        return task

    def update_queue(self):
        self.running_tasks = [t for t in self.running_tasks if not t.ready()]
        tasks_to_run = self.queued_tasks[: self.pool_size - len(self.running_tasks)]
        self.queued_tasks = self.queued_tasks[self.pool_size - len(self.running_tasks) :]
        for task in tasks_to_run:
            task.start()
        self.running_tasks += tasks_to_run

    def join(self, timeout=None):
        for task in self.running_tasks:
            logger.info(f"joining process: {task.proc.name}({task.proc.pid})")
            task.join(timeout)


class Executor(object):
    """
    The task executor.
    """

    def __init__(self, ctx: RuntimeContext, id: str, wq: WorkQueue, cq: WorkQueue) -> None:
        self.ctx = ctx
        self.id = id
        self.wq = wq
        self.cq = cq
        self.running_works: Dict[str, Tuple[SimplePoolTask, WorkItem]] = OrderedDict()
        self.running = True
        self.epochs_to_skip = 0
        self.numa_node = self.ctx.numa_node_id
        self.local_gpus = {gpu: 1.0 for gpu in self.ctx.get_local_gpus()}
        """ { GPU: available_quota } """

    def __str__(self) -> str:
        return f"Executor({self.id}), running_works[{len(self.running_works)}]={list(self.running_works.keys())[:3]}..."

    @property
    def busy(self) -> bool:
        return len(self.running_works) > 0

    @property
    def available_gpu_quota(self) -> float:
        return sum(self.local_gpus.values())

    @staticmethod
    def create(ctx: RuntimeContext, id: str) -> "Executor":
        queue_dir = os.path.join(ctx.queue_root, id)
        wq = WorkQueueOnFilesystem(os.path.join(queue_dir, "wq"))
        cq = WorkQueueOnFilesystem(os.path.join(queue_dir, "cq"))
        executor = Executor(ctx, id, wq, cq)
        return executor

    @staticmethod
    @logger.catch(reraise=True, message="work item failed unexpectedly")
    def process_work(item: WorkItem, cq: WorkQueue):
        item.exec(cq)
        cq.push(item)
        logger.info(f"finished work: {repr(item)}, status: {item.status}, elapsed time: {item.elapsed_time:.3f} secs")
        logger.complete()

    # for test
    def stop(self):
        self.wq.push(StopExecutor(f".FailStop-{self.id}", ack=False))

    # for test
    def skip_probes(self, epochs: int):
        self.wq.push(Probe(self.ctx, f".FalseFail-{self.id}", epoch=0, epochs_to_skip=epochs))

    @logger.catch(reraise=True, message="executor terminated unexpectedly")
    def run(self) -> bool:
        mp.current_process().name = "ExecutorMainProcess"
        logger.info(f"start to run executor {self.id} on numa node #{self.ctx.numa_node_id} of {socket.gethostname()}")

        with SimplePool(self.ctx.usable_cpu_count + 1) as pool:
            retval = self.exec_loop(pool)

        logger.info(f"executor exits: {self}")
        logger.complete()
        return retval

    def exec_loop(self, pool: SimplePool) -> bool:
        stop_request = None
        latest_probe_time = time.time()

        while self.running:
            # get new work items
            try:
                items = self.wq.pop(count=self.ctx.usable_cpu_count)
            except Exception as ex:
                logger.opt(exception=ex).critical(f"failed to pop from work queue: {self.wq}")
                self.running = False
                items = []

            if not items:
                secs_quiet_period = time.time() - latest_probe_time
                if secs_quiet_period > self.ctx.secs_executor_probe_interval * 2 and os.path.exists(self.ctx.job_status_path):
                    with open(self.ctx.job_status_path) as status_file:
                        if (status := status_file.read().strip()) and not status.startswith("running"):
                            logger.critical(f"job scheduler already stopped: {status}, stopping executor")
                            self.running = False
                            break
                if secs_quiet_period > self.ctx.secs_executor_probe_timeout * 2 and not pytest_running():
                    logger.critical(f"no probe received for {secs_quiet_period:.1f} secs, stopping executor")
                    self.running = False
                    break
                # no pending works, so wait a few seconds before checking results
                time.sleep(self.ctx.secs_wq_poll_interval)

            for item in items:
                if isinstance(item, StopExecutor):
                    logger.info(f"stop request received from scheduler: {item}")
                    stop_request = item
                    self.running = False
                    break

                if isinstance(item, StopWorkItem):
                    running_work = self.running_works.get(item.work_to_stop, None)
                    if running_work is None:
                        logger.debug(f"cannot find {item.work_to_stop} in running works of {self.id}")
                        self.cq.push(item)
                    else:
                        logger.info(f"stopping work: {item.work_to_stop}")
                        task, _ = running_work
                        task.terminate()
                    continue

                if isinstance(item, Probe):
                    latest_probe_time = time.time()
                    if item.epochs_to_skip > 0:
                        self.epochs_to_skip += item.epochs_to_skip
                    if self.epochs_to_skip > 0:
                        self.epochs_to_skip -= 1
                        continue

                if self.numa_node is not None:
                    item._numa_node = self.numa_node

                # wait and allocate GPU to work item
                if item.gpu_limit > 0:
                    if item.gpu_limit > len(self.local_gpus):
                        logger.warning(
                            f"task {item.key} requires more GPUs than physical GPUs, downgrading from {item.gpu_limit} to {len(self.local_gpus)}"
                        )
                        item.gpu_limit = len(self.local_gpus)
                    # FIXME: this will block the executor if there is no available GPU
                    while not (granted_gpus := self.acquire_gpu(item.gpu_limit)):
                        logger.info(f"collecting finished works to find available GPUs")
                        self.collect_finished_works()
                        time.sleep(self.ctx.secs_wq_poll_interval)
                    item._local_gpu = granted_gpus
                    logger.info(f"{repr(item)} is assigned to run on GPU: { {gpu.id: quota for gpu, quota in item._local_gpu.items()} }")

                # enqueue work item to the pool
                self.running_works[item.key] = (
                    pool.apply_async(func=Executor.process_work, args=(item, self.cq), name=item.key),
                    item,
                )
                logger.info(f"started work: {repr(item)}, {len(self.running_works)} running works: {list(self.running_works.keys())[:10]}...")

            # start to run works
            pool.update_queue()
            self.collect_finished_works()

        pool.join(self.ctx.secs_executor_probe_interval)

        if stop_request and stop_request.ack:
            self.collect_finished_works()
            stop_request.exec()
            self.cq.push(stop_request)

        return True

    def collect_finished_works(self):
        finished_works: List[WorkItem] = []
        for work, item in self.running_works.values():
            if not work.ready():
                continue
            else:
                work.join()
            if (exitcode := work.exitcode()) != 0:
                item.status = WorkStatus.CRASHED
                item.exception = NonzeroExitCode(f"worker process {work.proc.name}({work.proc.pid}) exited with non-zero code {exitcode}")
                try:
                    self.cq.push(item)
                except Exception as ex:
                    logger.opt(exception=ex).critical(f"failed to push into completion queue: {self.cq}")
                    self.running = False
            finished_works.append(item)

        # remove finished works
        for item in finished_works:
            self.running_works.pop(item.key)
            if item._local_gpu:
                self.release_gpu(item._local_gpu)
                logger.info(f"{repr(item)} released GPU: { {gpu.id: quota for gpu, quota in item._local_gpu.items()} }")

    def acquire_gpu(self, quota: float) -> Dict[GPU, float]:
        """
        Acquire GPU resources with the given quota.
        Return a dict of granted GPUs with their quotas.
        `release_gpu` should be called later to release GPUs.
        """
        if self.available_gpu_quota < quota:
            # no enough GPU resources, return empty dict
            return {}
        granted_gpus: Dict[GPU, float] = {}
        for gpu in self.local_gpus:
            gpu_available_quota = self.local_gpus[gpu]
            if gpu_available_quota <= 0:
                continue
            granted_quota = min(gpu_available_quota, quota)
            granted_gpus[gpu] = granted_quota
            self.local_gpus[gpu] -= granted_quota
            quota -= granted_quota
            if quota <= 0:
                break
        return granted_gpus

    def release_gpu(self, gpus: Dict[GPU, float]):
        """
        Release GPU resources to the pool.
        """
        for gpu, quota in gpus.items():
            self.local_gpus[gpu] += quota
            assert self.local_gpus[gpu] <= 1.0, f"GPU {gpu} quota is greater than 1.0: {self.local_gpus[gpu]}"
