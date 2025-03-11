import os
import os.path
import queue
import random
import sys
import time
import uuid
from enum import Enum
from typing import Dict, Iterable, List, Optional

import numpy as np
from GPUtil import GPU
from loguru import logger

from smallpond.common import MB, NonzeroExitCode, OutOfMemory
from smallpond.io.filesystem import dump, load


class WorkStatus(Enum):
    INCOMPLETE = 1
    SUCCEED = 2
    FAILED = 3
    CRASHED = 4
    EXEC_FAILED = 5


class WorkItem(object):

    __slots__ = (
        "_cpu_limit",
        "_gpu_limit",
        "_memory_limit",
        "_cpu_boost",
        "_memory_boost",
        "_numa_node",
        "_local_gpu",
        "key",
        "status",
        "start_time",
        "finish_time",
        "retry_count",
        "fail_count",
        "exception",
        "exec_id",
        "exec_cq",
        "location",
    )

    def __init__(
        self,
        key: str,
        cpu_limit: int = None,
        gpu_limit: float = None,
        memory_limit: int = None,
    ) -> None:
        self._cpu_limit = cpu_limit
        self._gpu_limit = gpu_limit
        self._memory_limit = np.int64(memory_limit) if memory_limit is not None else None
        self._cpu_boost = 1
        self._memory_boost = 1
        self._numa_node = None
        self._local_gpu: Dict[GPU, float] = {}
        self.key = key
        self.status = WorkStatus.INCOMPLETE
        self.start_time = None
        self.finish_time = None
        self.retry_count = 0
        self.fail_count = 0
        self.exception = None
        self.exec_id = "unknown"
        self.exec_cq = None
        self.location: Optional[str] = None

    def __repr__(self) -> str:
        return self.key

    __str__ = __repr__

    @property
    def cpu_limit(self) -> int:
        return int(self._cpu_boost * self._cpu_limit)

    @property
    def gpu_limit(self) -> int:
        return self._gpu_limit

    @property
    def memory_limit(self) -> np.int64:
        return np.int64(self._memory_boost * self._memory_limit) if self._memory_limit else 0

    @property
    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0
        if self.finish_time is None:
            return time.time() - self.start_time
        return self.finish_time - self.start_time

    @property
    def exec_on_scheduler(self) -> bool:
        return False

    @property
    def local_gpu(self) -> Optional[GPU]:
        """
        Return the first GPU granted to this task.
        If gpu_limit is 0, return None.
        If gpu_limit is greater than 1, only the first GPU is returned.
        """
        return next(iter(self._local_gpu.keys()), None)

    @property
    # @deprecated("use `local_gpu_ranks` instead")
    def local_rank(self) -> Optional[int]:
        """
        Return the first GPU rank granted to this task.
        If gpu_limit is 0, return None.
        If gpu_limit is greater than 1, only the first GPU rank is returned. Caller should use `local_gpu_ranks` instead.
        """
        if self.gpu_limit > 1:
            logger.warning(
                f"task {self.key} requires more than 1 GPU, but only the first GPU rank is returned. please use `local_gpu_ranks` instead."
            )
        return next(iter(self._local_gpu.keys())).id if self._local_gpu else None

    @property
    def local_gpu_ranks(self) -> List[int]:
        """Return all GPU ranks granted to this task."""
        return [gpu.id for gpu in self._local_gpu.keys()]

    @property
    def numa_node(self) -> int:
        return self._numa_node

    def oom(self, nonzero_exitcode_as_oom=False):
        return (
            self._memory_limit is not None
            and self.status == WorkStatus.CRASHED
            and (isinstance(self.exception, (OutOfMemory, MemoryError)) or (isinstance(self.exception, NonzeroExitCode) and nonzero_exitcode_as_oom))
        )

    def run(self) -> bool:
        return True

    def initialize(self):
        """Called before run() to prepare for running the task."""

    def finalize(self):
        """Called after run() to finalize the processing."""

    def cleanup(self):
        """Called after run() even if there is an exception."""

    def exec(self, cq: Optional["WorkQueue"] = None) -> WorkStatus:
        if self.status == WorkStatus.INCOMPLETE:
            try:
                self.start_time = time.time()
                self.exec_cq = cq
                self.initialize()
                if self.run():
                    self.status = WorkStatus.SUCCEED
                    self.finalize()
                else:
                    self.status = WorkStatus.FAILED
            except Exception as ex:
                logger.opt(exception=ex).error(f"{repr(self)} crashed with error. node location at {self.location}")
                self.status = WorkStatus.CRASHED
                self.exception = ex
            finally:
                self.cleanup()
                self.exec_cq = None
                self.finish_time = time.time()
        return self.status


class StopExecutor(WorkItem):
    def __init__(self, key: str, ack=True) -> None:
        super().__init__(key, cpu_limit=0, gpu_limit=0, memory_limit=0)
        self.ack = ack


class StopWorkItem(WorkItem):
    def __init__(self, key: str, work_to_stop: str) -> None:
        super().__init__(key, cpu_limit=0, gpu_limit=0, memory_limit=0)
        self.work_to_stop = work_to_stop


class WorkBatch(WorkItem):
    def __init__(self, key: str, works: List[WorkItem]) -> None:
        cpu_limit = max(w.cpu_limit for w in works)
        gpu_limit = max(w.gpu_limit for w in works)
        memory_limit = max(w.memory_limit for w in works)
        super().__init__(f"{self.__class__.__name__}-{key}", cpu_limit, gpu_limit, memory_limit)
        self.works = works

    def __str__(self) -> str:
        return super().__str__() + f", works[{len(self.works)}]={self.works[:1]}...{self.works[-1:]}"

    def run(self) -> bool:
        logger.info(f"processing {len(self.works)} works in the batch")
        for index, work in enumerate(self.works):
            work.exec_id = self.exec_id
            if work.exec(self.exec_cq) != WorkStatus.SUCCEED:
                logger.error(f"work item #{index+1}/{len(self.works)} in {self.key} failed: {work}")
                return False
        logger.info(f"done {len(self.works)} works in the batch")
        return True


class WorkQueue(object):
    def __init__(self) -> None:
        self.outbound_works: List[WorkItem] = []

    def _pop_unbuffered(self, count: int) -> List[WorkItem]:
        raise NotImplementedError

    def _push_unbuffered(self, item: WorkItem) -> bool:
        raise NotImplementedError

    def pop(self, count=1) -> List[WorkItem]:
        inbound_works: List[WorkItem] = []
        for item in self._pop_unbuffered(count):
            if isinstance(item, WorkBatch):
                inbound_works.extend(item.works)
            else:
                inbound_works.append(item)
        return inbound_works

    def push(self, item: WorkItem, buffering=False) -> bool:
        if buffering:
            self.outbound_works.append(item)
            return True
        elif len(self.outbound_works) > 0:
            self.outbound_works.append(item)
            return self.flush()
        else:
            return self._push_unbuffered(item)

    def flush(self) -> bool:
        if len(self.outbound_works) == 0:
            return True
        batch = WorkBatch(self.outbound_works[0].key, self.outbound_works)
        self.outbound_works = []
        return self._push_unbuffered(batch)


class WorkQueueInMemory(WorkQueue):
    def __init__(self, queue_type=queue.Queue) -> None:
        super().__init__()
        self.queue = queue_type()

    def _pop_unbuffered(self, count: int) -> List[WorkItem]:
        try:
            return [self.queue.get(block=False)]
        except queue.Empty:
            return []

    def _push_unbuffered(self, item: WorkItem) -> bool:
        self.queue.put(item)
        return True


class WorkQueueOnFilesystem(WorkQueue):

    buffer_size = 16 * MB

    def __init__(self, workq_root: str, sort=True, random=False) -> None:
        super().__init__()
        self.workq_root = workq_root
        self.sort = sort
        self.random = random
        self.buffered_works: List[WorkItem] = []
        self.temp_dir = os.path.join(self.workq_root, "temp")
        self.enqueue_dir = os.path.join(self.workq_root, "enqueued")
        self.dequeue_dir = os.path.join(self.workq_root, "dequeued")
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.enqueue_dir, exist_ok=True)
        os.makedirs(self.dequeue_dir, exist_ok=True)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}@{self.workq_root}"

    def _get_entries(self, path=None) -> List[os.DirEntry]:
        dentries: List[os.DirEntry] = []
        with os.scandir(path or self.enqueue_dir) as dir_iter:
            for entry in dir_iter:
                if entry.is_file():
                    dentries.append(entry)
        return dentries

    def size(self) -> int:
        return len(self._get_entries())

    def _list_works(self, path: str, expand_batch=True) -> Iterable[WorkItem]:
        dentries = self._get_entries(path)
        logger.info("listing {} entries in {}", len(dentries), path)
        for entry in dentries:
            item: WorkItem = load(entry.path, self.buffer_size)
            if expand_batch and isinstance(item, WorkBatch):
                for work in item.works:
                    yield work
            else:
                yield item

    def list_enqueued(self, expand_batch=True) -> Iterable[WorkItem]:
        yield from self._list_works(self.enqueue_dir, expand_batch)

    def list_dequeued(self, expand_batch=True) -> Iterable[WorkItem]:
        yield from self._list_works(self.dequeue_dir, expand_batch)

    def list_works(self, expand_batch=True) -> Iterable[WorkItem]:
        yield from self.list_enqueued(expand_batch)
        yield from self.list_dequeued(expand_batch)

    def _pop_unbuffered(self, count: int) -> List[WorkItem]:
        items = []
        dentries = self._get_entries()

        if self.sort:
            dentries = sorted(dentries, key=lambda entry: entry.name)
        elif self.random:
            random.shuffle(dentries)

        for entry in dentries:
            uuid_hex = uuid.uuid4().hex
            filename = f"{entry.name}-DEQ{uuid_hex}"
            dequeued_path = os.path.join(self.dequeue_dir, filename)

            try:
                os.rename(entry.path, dequeued_path)
            except OSError as err:
                logger.debug(f"cannot rename {entry.path} to {dequeued_path}: {err}")
                if items:
                    break
                else:
                    continue

            items.append(load(dequeued_path, self.buffer_size))
            if len(items) >= count:
                break

        return items

    def _push_unbuffered(self, item: WorkItem) -> bool:
        timestamp = time.time_ns()
        uuid_hex = uuid.uuid4().hex
        filename = f"{item.key}-{timestamp:x}-ENQ{uuid_hex}"
        tempfile_path = os.path.join(self.temp_dir, filename)
        enqueued_path = os.path.join(self.enqueue_dir, filename)

        dump(item, tempfile_path, self.buffer_size)

        try:
            os.rename(tempfile_path, enqueued_path)
            return True
        except OSError as err:
            logger.critical(f"failed to rename {tempfile_path} to {enqueued_path}: {err}")
            return False


def count_objects(obj, object_cnt=None, visited_objs=None, depth=0):
    object_cnt = {} if object_cnt is None else object_cnt
    visited_objs = set() if visited_objs is None else visited_objs

    if id(obj) in visited_objs:
        return
    else:
        visited_objs.add(id(obj))

    if isinstance(obj, dict):
        for key, value in obj.items():
            count_objects(key, object_cnt, visited_objs, depth + 1)
            count_objects(value, object_cnt, visited_objs, depth + 1)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for item in obj:
            count_objects(item, object_cnt, visited_objs, depth + 1)
    else:
        class_name = obj.__class__.__name__
        if class_name not in object_cnt:
            object_cnt[class_name] = (0, 0)
        cnt, size = object_cnt[class_name]
        object_cnt[class_name] = (cnt + 1, size + sys.getsizeof(obj))

        key_attributes = ("__self__", "__dict__", "__slots__")
        if not isinstance(obj, (bool, str, int, float, type(None))) and any(attr_name in key_attributes for attr_name in dir(obj)):
            logger.debug(f"{' ' * depth}{class_name}@{id(obj):x}")
            for attr_name in dir(obj):
                try:
                    if (not attr_name.startswith("__") or attr_name in key_attributes) and not isinstance(
                        getattr(obj.__class__, attr_name, None), property
                    ):
                        logger.debug(f"{' ' * depth}{class_name}.{attr_name}@{id(obj):x}")
                        count_objects(getattr(obj, attr_name), object_cnt, visited_objs, depth + 1)
                except Exception as ex:
                    logger.warning(f"failed to get '{attr_name}' from {repr(obj)}: {ex}")


def main():
    import argparse

    from smallpond.execution.task import Probe

    parser = argparse.ArgumentParser(prog="workqueue.py", description="Work Queue Reader")
    parser.add_argument("wq_root", help="Work queue root path")
    parser.add_argument("-f", "--work_filter", default="", help="Work item filter")
    parser.add_argument("-x", "--expand_batch", action="store_true", help="Expand batched works")
    parser.add_argument("-c", "--count_object", action="store_true", help="Count number of objects")
    parser.add_argument("-n", "--top_n_class", default=20, type=int, help="Show the top n classes")
    parser.add_argument("-l", "--log_level", default="INFO", help="Logging message level")
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stdout,
        format=r"[{time:%Y-%m-%d %H:%M:%S.%f}] {level} {message}",
        level=args.log_level,
    )

    wq = WorkQueueOnFilesystem(args.wq_root)
    for work in wq.list_works(args.expand_batch):
        if isinstance(work, Probe):
            continue
        if args.work_filter in work.key:
            logger.info(work)
            if args.count_object:
                object_cnt = {}
                count_objects(work, object_cnt)
                sorted_counts = sorted([(v, k) for k, v in object_cnt.items()], reverse=True)
                for count, class_name in sorted_counts[: args.top_n_class]:
                    logger.info(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
