import contextlib
import copy
import cProfile
import io
import itertools
import json
import logging
import math
import os
import pprint
import random
import resource
import shutil
import sys
import time
import uuid
from collections import OrderedDict, defaultdict, namedtuple
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path, PurePath
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import duckdb
import fsspec
import GPUtil
import numpy as np
import pandas as pd
import psutil
import pyarrow as arrow
import pyarrow.parquet as parquet
import ray
from loguru import logger

from smallpond.common import (
    DATA_PARTITION_COLUMN_NAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_RETRY_COUNT,
    DEFAULT_ROW_GROUP_SIZE,
    GB,
    GENERATED_COLUMNS,
    INPUT_VIEW_PREFIX,
    KB,
    MAX_NUM_ROW_GROUPS,
    MAX_ROW_GROUP_BYTES,
    MAX_ROW_GROUP_SIZE,
    MB,
    PARQUET_METADATA_KEY_PREFIX,
    RAND_SEED_BYTE_LEN,
    TB,
    InjectedFault,
    OutOfMemory,
    clamp_row_group_bytes,
    clamp_row_group_size,
    pytest_running,
    round_up,
    split_into_rows,
)
from smallpond.execution.workqueue import WorkItem, WorkStatus
from smallpond.io.arrow import (
    cast_columns_to_large_string,
    dump_to_parquet_files,
    filter_schema,
)
from smallpond.io.filesystem import dump, find_mount_point, load, remove_path
from smallpond.logical.dataset import (
    ArrowTableDataSet,
    CsvDataSet,
    DataSet,
    FileSet,
    JsonDataSet,
    ParquetDataSet,
    PartitionedDataSet,
    SqlQueryDataSet,
)
from smallpond.logical.udf import UDFContext
from smallpond.utility import ConcurrentIter, InterceptHandler, cprofile_to_string


class JobId(uuid.UUID):
    """
    A unique identifier for a job.
    """

    @staticmethod
    def new():
        return JobId(int=uuid.uuid4().int)


class TaskId(int):
    """
    A unique identifier for a task.
    """

    def __str__(self) -> str:
        return f"{self:06d}"


@dataclass(frozen=True)
class TaskRuntimeId:
    """
    A unique identifier for a task at runtime.

    Besides the task id, it also includes the epoch and retry count.
    """

    id: TaskId
    epoch: int
    retry: int
    """How many times the task has been retried."""

    def __str__(self) -> str:
        return f"{self.id}.{self.epoch}.{self.retry}"


class PerfStats(namedtuple("PerfStats", ("cnt", "sum", "min", "max", "avg", "p50", "p75", "p95", "p99"))):
    """
    Performance statistics for a task.
    """

    def __str__(self) -> str:
        return ", ".join([f"{k}={v:,.1f}" for k, v in self._asdict().items()])

    __repr__ = __str__


class RuntimeContext(object):
    """
    The configuration and state for a running job.
    """

    def __init__(
        self,
        job_id: JobId,
        job_time: datetime,
        data_root: str,
        *,
        num_executors: int = 1,
        random_seed: int = None,
        env_overrides: Dict[str, str] = None,
        bind_numa_node=False,
        enforce_memory_limit=False,
        max_usable_cpu_count: int = 1024,
        max_usable_gpu_count: int = 1024,
        max_usable_memory_size: int = 16 * TB,
        secs_wq_poll_interval: float = 0.5,
        secs_executor_probe_interval: float = 30,
        max_num_missed_probes: int = 6,
        fault_inject_prob=0.0,
        enable_profiling=False,
        enable_diagnostic_metrics=False,
        remove_empty_parquet=False,
        skip_task_with_empty_input=False,
        shared_log_root: Optional[str] = None,
        console_log_level="INFO",
        file_log_level="DEBUG",
        disable_log_rotation=False,
        output_path: Optional[str] = None,
    ) -> None:
        self.job_id = job_id
        self.job_time = job_time
        self.data_root = data_root
        self.next_task_id = 0
        self.num_executors = num_executors
        self.random_seed: int = random_seed or int.from_bytes(os.urandom(RAND_SEED_BYTE_LEN), byteorder=sys.byteorder)
        self.env_overrides = env_overrides or {}
        self.bind_numa_node = bind_numa_node
        self.numa_node_id: Optional[int] = None
        self.enforce_memory_limit = enforce_memory_limit
        self.max_usable_cpu_count = max_usable_cpu_count
        self.max_usable_gpu_count = max_usable_gpu_count
        self.max_usable_memory_size = max_usable_memory_size
        self.secs_wq_poll_interval = secs_wq_poll_interval
        self.secs_executor_probe_interval = secs_executor_probe_interval
        self.max_num_missed_probes = max_num_missed_probes
        self.fault_inject_prob = fault_inject_prob
        self.enable_profiling = enable_profiling
        self.enable_diagnostic_metrics = enable_diagnostic_metrics
        self.remove_empty_parquet = remove_empty_parquet
        self.skip_task_with_empty_input = skip_task_with_empty_input

        self.shared_log_root = os.path.join(shared_log_root, self.job_root_dirname) if shared_log_root else None
        self.console_log_level = console_log_level
        self.file_log_level = file_log_level
        self.disable_log_rotation = disable_log_rotation

        self.job_root = os.path.abspath(os.path.join(data_root, self.job_root_dirname))
        self.config_root = os.path.join(self.job_root, "config")
        self.queue_root = os.path.join(self.job_root, "queue")
        self.output_root = os.path.join(self.job_root, "output")
        self.staging_root = os.path.join(self.job_root, "staging")
        self.temp_root = os.path.join(self.job_root, "temp")
        self.log_root = os.path.join(self.job_root, "log")
        self.final_output_path = os.path.abspath(output_path) if output_path else None
        self.current_task: Task = None

        # used by ray executors to checkpoint task states
        self.started_task_dir = os.path.join(self.staging_root, "started_tasks")
        self.completed_task_dir = os.path.join(self.staging_root, "completed_tasks")

    @property
    def job_root_dirname(self):
        return f"{self.job_time:%Y-%m-%d-%H-%M-%S}.{self.job_id}"

    @property
    def job_status_path(self):
        return os.path.join(self.log_root, ".STATUS")

    @property
    def runtime_ctx_path(self):
        return os.path.join(self.config_root, f"runtime_ctx.pickle")

    @property
    def logcial_plan_path(self):
        return os.path.join(self.config_root, f"logical_plan.pickle")

    @property
    def logcial_plan_graph_path(self):
        return os.path.join(self.log_root, "graph")

    @property
    def ray_log_path(self):
        return os.path.join(self.log_root, "ray.log")

    @property
    def exec_plan_path(self):
        return os.path.join(self.config_root, f"exec_plan.pickle")

    @property
    def sched_state_path(self):
        return os.path.join(self.config_root, f"sched_state.pickle")

    @property
    def numa_node_count(self):
        if sys.platform == "darwin":
            # numa is not supported on macos
            return 1
        import numa

        return numa.info.get_num_configured_nodes()

    @property
    def physical_cpu_count(self):
        cpu_count = psutil.cpu_count(logical=False)
        return cpu_count // self.numa_node_count if self.bind_numa_node else cpu_count

    @property
    def available_memory(self):
        available_memory = psutil.virtual_memory().available
        return available_memory // self.numa_node_count if self.bind_numa_node else available_memory

    @property
    def total_memory(self):
        total_memory = psutil.virtual_memory().total
        return total_memory // self.numa_node_count if self.bind_numa_node else total_memory

    @property
    def usable_cpu_count(self):
        return min(self.max_usable_cpu_count, self.physical_cpu_count)

    @property
    def usable_memory_size(self):
        return min(self.max_usable_memory_size, self.total_memory)

    @property
    def secs_executor_probe_timeout(self):
        return self.secs_executor_probe_interval * self.max_num_missed_probes

    def get_local_gpus(self) -> List[GPUtil.GPU]:
        gpus = GPUtil.getGPUs()
        gpus_on_node = split_into_rows(gpus, self.numa_node_count)
        return gpus_on_node[self.numa_node_id] if self.bind_numa_node and self.numa_node_id is not None else gpus

    @property
    def usable_gpu_count(self):
        return min(self.max_usable_gpu_count, len(self.get_local_gpus()))

    @property
    def task(self):
        return self.current_task

    def set_current_task(self, task: "Task" = None) -> "RuntimeContext":
        self.current_task = None
        ctx = copy.copy(self)
        ctx.current_task = copy.copy(task)
        return ctx

    def new_task_id(self) -> TaskId:
        self.next_task_id += 1
        return TaskId(self.next_task_id)

    def initialize(self, exec_id: str, root_exist_ok=True, cleanup_root=False) -> None:
        import smallpond

        self._make_dirs(root_exist_ok, cleanup_root)
        self._init_logs(exec_id)
        self._init_envs()
        logger.info(f"smallpond version: {smallpond.__version__}")
        logger.info(f"runtime context:{os.linesep}{pprint.pformat(vars(self))}")
        logger.info(f"local GPUs: {[gpu.id for gpu in self.get_local_gpus()]}")

    def cleanup_root(self):
        if os.path.exists(self.job_root):
            remove_path(self.job_root)

    def _make_dirs(self, root_exist_ok, cleanup_root):
        if os.path.exists(self.job_root):
            if cleanup_root:
                remove_path(self.job_root)
            elif not root_exist_ok:
                raise FileExistsError(f"Folder already exists: {self.job_root}")
        os.makedirs(self.config_root, exist_ok=root_exist_ok)
        os.makedirs(self.queue_root, exist_ok=root_exist_ok)
        os.makedirs(self.output_root, exist_ok=root_exist_ok)
        os.makedirs(self.staging_root, exist_ok=root_exist_ok)
        os.makedirs(self.temp_root, exist_ok=root_exist_ok)
        os.makedirs(self.log_root, exist_ok=root_exist_ok)
        os.makedirs(self.started_task_dir, exist_ok=root_exist_ok)
        os.makedirs(self.completed_task_dir, exist_ok=root_exist_ok)

    def _init_envs(self):
        sys.setrecursionlimit(100000)
        env_overrides = copy.copy(self.env_overrides)
        ld_library_path = os.getenv("LD_LIBRARY_PATH", "")
        py_library_path = os.path.join(sys.exec_prefix, "lib")
        if py_library_path not in ld_library_path:
            env_overrides["LD_LIBRARY_PATH"] = ":".join([py_library_path, ld_library_path])
        for key, val in env_overrides.items():
            if (old := os.getenv(key, None)) is not None:
                logger.info(f"overwrite environment variable '{key}': '{old}' -> '{val}'")
            else:
                logger.info(f"set environment variable '{key}': '{val}'")
            os.environ[key] = val
        logger.info(
            f"RANK='{os.getenv('RANK', '')}' LD_LIBRARY_PATH='{os.getenv('LD_LIBRARY_PATH', '')}' LD_PRELOAD='{os.getenv('LD_PRELOAD', '')}' MALLOC_CONF='{os.getenv('MALLOC_CONF', '')}'"
        )

    def _init_logs(self, exec_id: str, capture_stdout_stderr: bool = False) -> None:
        log_rotation = {"rotation": "100 MB", "retention": 5} if not self.disable_log_rotation else {}
        log_file_paths = [os.path.join(self.log_root, f"{exec_id}.log")]
        user_log_only = {"": self.file_log_level, "smallpond": False}
        user_log_path = os.path.join(self.log_root, f"{exec_id}-user.log")
        # create shared log dir
        if self.shared_log_root is not None:
            os.makedirs(self.shared_log_root, exist_ok=True)
            shared_log_path = os.path.join(self.shared_log_root, f"{exec_id}.log")
            log_file_paths.append(shared_log_path)
        # remove existing handlers
        logger.remove()
        # register stdout log handler
        format_str = f"[{{time:%Y-%m-%d %H:%M:%S.%f}}] [{exec_id}] [{{process.name}}({{process.id}})] [{{file}}:{{line}}] {{level}} {{message}}"
        logger.add(
            sys.stdout,
            format=format_str,
            colorize=False,
            enqueue=True,
            backtrace=False,
            level=self.console_log_level,
        )
        # register file log handlers
        for log_path in log_file_paths:
            logger.add(
                log_path,
                format=format_str,
                colorize=False,
                enqueue=True,
                backtrace=False,
                level=self.file_log_level,
                **log_rotation,
            )
            logger.info(f"initialized logging to file: {log_path}")
        # register user log handler
        logger.add(
            user_log_path,
            format=format_str,
            colorize=False,
            enqueue=True,
            backtrace=False,
            level=self.file_log_level,
            filter=user_log_only,
            **log_rotation,
        )
        logger.info(f"initialized user logging to file: {user_log_path}")
        # intercept messages from logging
        logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
        # capture stdout as INFO level
        # https://loguru.readthedocs.io/en/stable/resources/recipes.html#capturing-standard-stdout-stderr-and-warnings
        if capture_stdout_stderr:

            class StreamToLogger(io.TextIOBase):
                def __init__(self, level="INFO"):
                    super().__init__()
                    self._level = level

                def write(self, buffer):
                    for line in buffer.rstrip().splitlines():
                        logger.opt(depth=1).log(self._level, line.rstrip())

                def flush(self):
                    pass

            sys.stdout = StreamToLogger()
            sys.stderr = StreamToLogger()

    def cleanup(self, remove_output_root: bool = True):
        """
        Clean up the runtime directory. This will be called when the job is finished.
        """
        remove_path(self.queue_root)
        remove_path(self.temp_root)
        remove_path(self.staging_root)
        if remove_output_root:
            remove_path(self.output_root)


class Probe(WorkItem):
    def __init__(self, ctx: RuntimeContext, key: str, epoch: int, epochs_to_skip=0) -> None:
        super().__init__(key, cpu_limit=0, gpu_limit=0, memory_limit=0)
        self.ctx = ctx
        self.epoch = epoch
        self.epochs_to_skip = epochs_to_skip
        self.resource_low = True
        self.cpu_count = 0
        self.gpu_count = 0
        self.cpu_percent = 0
        self.total_memory = 0
        self.available_memory = 0

    def __str__(self) -> str:
        return (
            super().__str__()
            + f", epoch={self.epoch}, resource_low={self.resource_low}, cpu_count={self.cpu_count}, gpu_count={self.gpu_count}, cpu_usage={self.cpu_percent}%, available_memory={self.available_memory//GB}GB/{self.total_memory//GB}GB"
        )

    def __lt__(self, other: "Probe") -> bool:
        return self.epoch < other.epoch

    def run(self) -> bool:
        self.cpu_percent = psutil.cpu_percent(interval=min(self.ctx.secs_executor_probe_interval / 2, 3))
        self.total_memory = self.ctx.usable_memory_size
        self.available_memory = self.ctx.available_memory
        self.resource_low = self.cpu_percent >= 80.0 or self.available_memory < self.total_memory // 16
        self.cpu_count = self.ctx.usable_cpu_count
        self.gpu_count = self.ctx.usable_gpu_count
        logger.info("resource usage: {}", self)
        return True


class PartitionInfo(object):
    """
    Information about a partition of a dataset.
    """

    toplevel_dimension = "@toplevel@"
    default_dimension = DATA_PARTITION_COLUMN_NAME

    __slots__ = (
        "index",
        "npartitions",
        "dimension",
    )

    def __init__(self, index: int = 0, npartitions: int = 1, dimension: str = toplevel_dimension) -> None:
        self.index = index
        self.npartitions = npartitions
        self.dimension = dimension

    def __str__(self):
        return f"{self.dimension}[{self.index}/{self.npartitions}]"

    __repr__ = __str__

    def __lt__(self, other: "PartitionInfo"):
        return (self.dimension, self.index) < (other.dimension, other.index)

    def __eq__(self, other: "PartitionInfo"):
        return (self.dimension, self.index) == (other.dimension, other.index)

    def __hash__(self):
        return hash((self.dimension, self.index))


class Task(WorkItem):
    """
    The base class for all tasks.

    Task is the basic unit of work in smallpond.
    Each task represents a specific operation that takes a series of input datasets and produces an output dataset.
    Tasks can depend on other tasks, forming a directed acyclic graph (DAG).
    Tasks can specify resource requirements such as CPU, GPU, and memory limits.
    Tasks should be idempotent. They can be retried if they fail.

    Lifetime of a task object:

    - instantiated at planning time on the scheduler node
    - pickled and sent to a worker node
    - `initialize()` is called to prepare for execution
    - `run()` is called to execute the task
    - `finalize()` or `cleanup()` is called to release resources
    - pickled and sent back to the scheduler node
    """

    __slots__ = (
        "ctx",
        "id",
        "node_id",
        "sched_epoch",
        "output_name",
        "output_root",
        "_temp_output",
        "dataset",
        "input_deps",
        "output_deps",
        "_np_randgen",
        "_py_randgen",
        "_timer_start",
        "perf_metrics",
        "perf_profile",
        "_partition_infos",
        "runtime_state",
        "input_datasets",
        "_dataset_ref",
    )

    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: "List[Task]",
        partition_infos: List[PartitionInfo],
        output_name: Optional[str] = None,
        output_path: Optional[str] = None,
        cpu_limit: int = 1,
        gpu_limit: float = 0,
        memory_limit: Optional[int] = None,
    ) -> None:
        assert isinstance(input_deps, Iterable), f"{input_deps} is not iterable"
        assert all(isinstance(dep, Task) for dep in input_deps), f"not every element of {input_deps} is a task"
        assert isinstance(partition_infos, Iterable), f"{partition_infos} is not iterable"
        assert all(isinstance(info, PartitionInfo) for info in partition_infos), f"not every element of {partition_infos} is a partition info"
        assert any(
            info.dimension == PartitionInfo.toplevel_dimension for info in partition_infos
        ), f"cannot find toplevel partition dimension: {partition_infos}"
        assert cpu_limit > 0, f"cpu_limit should be greater than zero"
        self.ctx = ctx
        self.id = ctx.new_task_id()
        self.node_id = 0
        self.sched_epoch = 0
        self._np_randgen = None
        self._py_randgen = None
        self._timer_start = None
        self.perf_metrics: Dict[str, np.int64] = defaultdict(np.int64)
        self.perf_profile = None
        self._partition_infos = sorted(partition_infos) or []
        assert len(self.partition_dims) == len(set(self.partition_dims)), f"found duplicate partition dimensions: {self.partition_dims}"
        super().__init__(f"{self.__class__.__name__}-{self.id}", cpu_limit, gpu_limit, memory_limit)
        self.output_name = output_name
        self.output_root = output_path
        self._temp_output = output_name is None and output_path is None

        # dependency references
        # NOTICE: `input_deps` is only used to maintain the task graph at planning time.
        #         before execution, references to dependencies are cleared so that the
        #         task can be sent to executors independently.
        #         DO NOT use `input_deps.values()` in execution time.
        self.input_deps = dict((dep.key, dep) for dep in input_deps)
        self.output_deps: Set[str] = set()
        for dep in input_deps:
            dep.output_deps.add(self.key)

        # input datasets for each dependency
        # implementor should read input from here
        self.input_datasets: List[DataSet] = None
        # the output dataset
        # implementor should set this variable as the output
        # if not set, the output dataset will be all parquet files in the output directory
        self.dataset: Optional[DataSet] = None
        # runtime state
        # implementor can use this variable as a checkpoint and restore from it after interrupted
        self.runtime_state = None

        # if the task is executed by ray, this is the reference to the output dataset
        # do not use this variable directly, use `self.run_on_ray()` instead
        self._dataset_ref: Optional[ray.ObjectRef] = None

    def __repr__(self) -> str:
        return f"{self.key}.{self.sched_epoch}.{self.retry_count},{self.node_id}"

    def __str__(self) -> str:
        return (
            f"{repr(self)}: status={self.status}, elapsed_time={self.elapsed_time:.3f}s, epoch={self.sched_epoch}, #retries={self.retry_count}, "
            f"input_deps[{len(self.input_deps)}]={list(self.input_deps.keys())[:3]}..., output_dir={self.output_dirname}, "
            f"resource_limit={self.cpu_limit}CPUs/{self.gpu_limit}GPUs/{(self.memory_limit or 0)//GB}GB, "
            f"partition_infos={self.partition_infos}"
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.exec()

    @property
    def _pristine_attrs(self) -> Set[str]:
        """All attributes in __slots__."""
        return set(itertools.chain.from_iterable(getattr(cls, "__slots__", []) for cls in type(self).__mro__))

    @property
    def partition_infos(self) -> Tuple[PartitionInfo]:
        return tuple(self._partition_infos)

    @property
    def partition_infos_as_dict(self):
        return dict((info.dimension, info.index) for info in self.partition_infos)

    def parquet_kv_metadata_str(self, extra_partitions: List[PartitionInfo] = None):
        task_infos = [
            ("__task_key__", self.key),
            ("__task_id__", str(self.id)),
            ("__node_id__", str(self.node_id)),
            ("__job_id__", str(self.ctx.job_id)),
            ("__job_root__", self.ctx.job_root),
        ]
        partition_infos = [(info.dimension, str(info.index)) for info in self._partition_infos + (extra_partitions or [])]
        return dict((PARQUET_METADATA_KEY_PREFIX + k, v) for k, v in task_infos + partition_infos)

    def parquet_kv_metadata_bytes(self, extra_partitions: List[PartitionInfo] = None):
        return dict((k.encode("utf-8"), v.encode("utf-8")) for k, v in self.parquet_kv_metadata_str(extra_partitions).items())

    @property
    def partition_dims(self):
        return tuple(info.dimension for info in self._partition_infos)

    def get_partition_info(self, dimension: str):
        for info in self._partition_infos:
            if info.dimension == dimension:
                return info
        raise KeyError(f"cannot find dimension {dimension} in {self._partition_infos}")

    @property
    def any_input_empty(self) -> bool:
        return any(dataset.empty for dataset in self.input_datasets)

    @property
    def skip_when_any_input_empty(self) -> bool:
        return self.ctx.skip_task_with_empty_input and self.any_input_empty

    @property
    def runtime_id(self):
        return TaskRuntimeId(self.id, self.sched_epoch, self.retry_count)

    @property
    def default_output_name(self):
        return ".".join(map(str, filter(None, [self.__class__.__name__, self.node_id])))

    @property
    def output_filename(self):
        output_name = self.output_name or self.default_output_name
        return f"{output_name}-{self.ctx.job_id}.{self.runtime_id}"

    @property
    def output_dirname(self):
        output_name = self.output_name or self.default_output_name
        return os.path.join(output_name, f"{self.ctx.job_id}.{self.runtime_id}")

    @property
    def staging_root(self) -> Optional[str]:
        """
        If the task has a special output directory, its runtime output directory will be under it.
        """
        return None if self.output_root is None else os.path.join(self.output_root, ".staging")

    @property
    def _final_output_root(self):
        return self.ctx.staging_root if self.temp_output else (self.output_root or self.ctx.output_root)

    @property
    def _runtime_output_root(self):
        return self.staging_root or self.ctx.staging_root

    @property
    def final_output_abspath(self) -> str:
        return os.path.join(self._final_output_root, self.output_dirname)

    @property
    def runtime_output_abspath(self) -> str:
        """
        Output data will be produced in this directory at runtime.

        When the task is finished, this directory will be atomically moved to `final_output_abspath`.
        """
        return os.path.join(self._runtime_output_root, self.output_dirname)

    @property
    def temp_abspath(self) -> str:
        return os.path.join(self.ctx.temp_root, self.output_dirname)

    @property
    def output(self) -> DataSet:
        return self.dataset or ParquetDataSet(["*"], root_dir=self.final_output_abspath)

    @property
    def self_contained_output(self) -> bool:
        """
        Whether the output of this node is not dependent on any input nodes.
        """
        return True

    @property
    def temp_output(self) -> bool:
        """
        Whether the output of this node is stored in a temporary directory.
        """
        return self._temp_output

    @temp_output.setter
    def temp_output(self, temp_output: bool):
        assert temp_output == False, f"cannot change temp_output to True in {self}"
        self._temp_output = False
        if not self.self_contained_output:
            for task in self.input_deps.values():
                if task.temp_output:
                    task.temp_output = False

    @property
    def allow_speculative_exec(self) -> bool:
        """
        Whether the task is allowed to be executed by speculative execution.
        """
        return True

    @property
    def ray_marker_path(self) -> str:
        """
        The path of an empty file that is used to determine if the task has been started.
        Only used by the ray executor.
        """
        return os.path.join(self.ctx.started_task_dir, f"{self.node_id}.{self.key}.{self.retry_count}")

    @property
    def ray_dataset_path(self) -> str:
        """
        The path of a pickle file that contains the output dataset of the task.
        If this file exists, the task is considered finished.
        Only used by the ray executor.
        """
        return os.path.join(self.ctx.completed_task_dir, str(self.node_id), f"{self.key}.pickle")

    @property
    def random_seed_bytes(self) -> bytes:
        return self.id.to_bytes(4, sys.byteorder) + self.ctx.random_seed.to_bytes(RAND_SEED_BYTE_LEN, sys.byteorder)

    @property
    def numpy_random_gen(self):
        if self._np_randgen is None:
            self._np_randgen = np.random.default_rng(int.from_bytes(self.random_seed_bytes, sys.byteorder))
        return self._np_randgen

    @property
    def python_random_gen(self):
        if self._py_randgen is None:
            self._py_randgen = random.Random(int.from_bytes(self.random_seed_bytes, sys.byteorder))
        return self._py_randgen

    def random_uint32(self) -> int:
        return self.python_random_gen.randint(0, 0x7FFFFFFF)

    def random_float(self) -> float:
        return self.python_random_gen.random()

    @property
    def uniform_failure_prob(self):
        return 1.0 / (self.ctx.next_task_id - self.id + 1)

    def inject_fault(self):
        if self.ctx.fault_inject_prob > 0 and self.fail_count <= 1:
            random_value = self.random_float()
            if random_value < self.uniform_failure_prob and random_value < self.ctx.fault_inject_prob:
                raise InjectedFault(
                    f"inject fault to {repr(self)}, uniform_failure_prob={self.uniform_failure_prob:.6f}, fault_inject_prob={self.ctx.fault_inject_prob:.6f}"
                )

    def compute_avg_row_size(self, nbytes, num_rows):
        return max(1, nbytes // num_rows) if num_rows > 0 else 1

    def adjust_row_group_size(
        self,
        nbytes,
        num_rows,
        max_row_group_size=MAX_ROW_GROUP_SIZE,
        max_row_group_bytes=MAX_ROW_GROUP_BYTES,
        max_num_row_groups=MAX_NUM_ROW_GROUPS,
    ):
        parquet_row_group_size = self.parquet_row_group_size
        num_row_groups = num_rows // parquet_row_group_size

        if num_row_groups > max_num_row_groups:
            parquet_row_group_size = round_up(
                clamp_row_group_size(num_rows // max_num_row_groups, maxval=max_row_group_size),
                KB,
            )
        avg_row_size = self.compute_avg_row_size(nbytes, num_rows)
        parquet_row_group_size = round_up(min(parquet_row_group_size, max_row_group_bytes // avg_row_size), KB)

        if self.parquet_row_group_size != parquet_row_group_size:
            parquet_row_group_bytes = round_up(
                clamp_row_group_bytes(parquet_row_group_size * avg_row_size, maxval=max_row_group_bytes),
                MB,
            )
            logger.info(
                f"adjust row group size for dataset ({num_rows} rows, {nbytes/MB:.3f}MB): {self.parquet_row_group_size} -> {parquet_row_group_size} rows, {parquet_row_group_bytes/MB:.1f}MB"
            )
            self.parquet_row_group_size = parquet_row_group_size
            self.parquet_row_group_bytes = parquet_row_group_bytes

    def run(self) -> bool:
        return True

    def set_memory_limit(self, soft_limit: int, hard_limit: int):
        soft_old, hard_old = resource.getrlimit(resource.RLIMIT_DATA)
        resource.setrlimit(resource.RLIMIT_DATA, (soft_limit, hard_limit))
        logger.debug(f"updated memory limit from ({soft_old/GB:.3f}GB, {hard_old/GB:.3f}GB) to ({soft_limit/GB:.3f}GB, {hard_limit/GB:.3f}GB)")

    def initialize(self):
        self.inject_fault()

        if self._memory_limit is None:
            self._memory_limit = np.int64(self.ctx.usable_memory_size * self._cpu_limit // self.ctx.usable_cpu_count)
        assert self.partition_infos, f"empty partition infos: {self}"
        os.makedirs(self.runtime_output_abspath, exist_ok=self.output_root is not None)
        os.makedirs(self.temp_abspath, exist_ok=False)

        if not self.exec_on_scheduler:
            if self.ctx.enable_profiling:
                self.perf_profile = cProfile.Profile()
                self.perf_profile.enable()
            if self.ctx.enforce_memory_limit:
                self.set_memory_limit(round_up(self.memory_limit * 1.2), round_up(self.memory_limit * 1.5))
            if self.ctx.remove_empty_parquet:
                for dataset in self.input_datasets:
                    if isinstance(dataset, ParquetDataSet):
                        dataset.remove_empty_files()
            logger.info("running task: {}", self)
            logger.debug("input datasets: {}", self.input_datasets)
            logger.trace(f"final output directory: {self.final_output_abspath}")
            logger.trace(f"runtime output directory: {self.runtime_output_abspath}")
            logger.trace(f"resource limit: {self.cpu_limit} cpus, {self.gpu_limit} gpus, {self.memory_limit/GB:.3f}GB memory")
            random.seed(self.random_seed_bytes)
            arrow.set_cpu_count(self.cpu_limit)
            arrow.set_io_thread_count(self.cpu_limit)
            os.environ["OMP_NUM_THREADS"] = str(self.cpu_limit)
            os.environ["POLARS_MAX_THREADS"] = str(self.cpu_limit)

    def finalize(self):
        self.inject_fault()
        assert self.status == WorkStatus.SUCCEED
        logger.info("finished task: {}", self)

        # move the task output from staging dir to output dir
        if self.runtime_output_abspath != self.final_output_abspath and os.path.exists(self.runtime_output_abspath):
            os.makedirs(os.path.dirname(self.final_output_abspath), exist_ok=True)
            os.rename(self.runtime_output_abspath, self.final_output_abspath)

        def collect_file_sizes(file_paths):
            if not file_paths:
                return []
            try:
                with ThreadPoolExecutor(min(32, len(file_paths))) as pool:
                    file_sizes = list(pool.map(os.path.getsize, file_paths))
            except FileNotFoundError:
                logger.warning(f"some of the output files not found: {file_paths[:3]}...")
                file_sizes = []
            return file_sizes

        if self.ctx.enable_diagnostic_metrics:
            input_file_paths = [path for dataset in self.input_datasets for path in dataset.resolved_paths]
            output_file_paths = self.output.resolved_paths
            for metric_name, file_paths in [
                ("input", input_file_paths),
                ("output", output_file_paths),
            ]:
                file_sizes = collect_file_sizes(file_paths)
                if file_paths and file_sizes:
                    self.perf_metrics[f"num {metric_name} files"] += len(file_paths)
                    self.perf_metrics[f"total {metric_name} size (MB)"] += sum(file_sizes) / MB

        self.perf_metrics["elapsed wall time (secs)"] += self.elapsed_time
        if not self.exec_on_scheduler:
            resource_usage = resource.getrusage(resource.RUSAGE_SELF)
            self.perf_metrics["max resident set size (MB)"] += resource_usage.ru_maxrss / 1024
            self.perf_metrics["user mode cpu time (secs)"] += resource_usage.ru_utime
            self.perf_metrics["system mode cpu time (secs)"] += resource_usage.ru_stime
            logger.debug(f"{self.key} perf metrics:{os.linesep}{os.linesep.join(f'{name}: {value}' for name, value in self.perf_metrics.items())}")

        if self.perf_profile is not None and self.elapsed_time > 3:
            logger.debug(f"{self.key} perf profile:{os.linesep}{cprofile_to_string(self.perf_profile)}")

    def cleanup(self):
        if self.perf_profile is not None:
            self.perf_profile.disable()
            self.perf_profile = None
        self.clean_complex_attrs()

    def clean_complex_attrs(self):

        self._np_randgen = None
        self._py_randgen = None
        self.perf_profile = None

        def is_primitive(obj: Any):
            return isinstance(obj, (bool, str, int, float))

        def is_primitive_iterable(obj: Any):
            if isinstance(obj, dict):
                return all(is_primitive(key) and is_primitive(value) for key, value in obj.items())
            elif isinstance(obj, Iterable):
                return all(is_primitive(elem) for elem in obj)
            return False

        if hasattr(self, "__dict__"):
            complex_attrs = [
                attr for attr, obj in vars(self).items() if not (attr in self._pristine_attrs or is_primitive(obj) or is_primitive_iterable(obj))
            ]
            if complex_attrs:
                logger.debug(f"removing complex attributes not explicitly declared in __slots__: {complex_attrs}")
                for attr in complex_attrs:
                    delattr(self, attr)

    def clean_output(self, force=False) -> None:
        if force or self.temp_output:
            remove_path(self.runtime_output_abspath)
            remove_path(self.final_output_abspath)

    @logger.catch(reraise=pytest_running(), message="failed to dump task")
    def dump(self):
        os.makedirs(self.temp_abspath, exist_ok=True)
        dump_path = os.path.join(self.temp_abspath, f"{self.key}.pickle")
        dump(self, dump_path)
        logger.info(f"{self.key} saved to {dump_path}")

    def add_elapsed_time(self, metric_name: str = None) -> float:
        """
        Start or stop the timer. If `metric_name` is provided, add the elapsed time to the task's performance metrics.

        Example:
        ```
        task.add_elapsed_time()                               # @t0 start timer
        e1 = task.add_elapsed_time("input load time (secs)")  # @t1 stop timer and add elapsed time e1=t1-t0 to metric
        e2 = task.add_elapsed_time("compute time (secs)")     # @t2 stop timer and add elapsed time e2=t2-t1 to metric
        ```
        """
        self.inject_fault()
        assert self._timer_start is not None or metric_name is None, f"timer not started, cannot save '{metric_name}'"
        if self._timer_start is None or metric_name is None:
            self._timer_start = time.time()
            return 0.0
        else:
            current_time = time.time()
            elapsed_time = current_time - self._timer_start
            self.perf_metrics[metric_name] += elapsed_time
            self._timer_start = current_time
            return elapsed_time

    def merge_metrics(self, metrics: Dict[str, int]):
        for name, value in metrics.items():
            self.perf_metrics[name] += value

    def run_on_ray(self) -> ray.ObjectRef:
        """
        Run the task on Ray.
        Return an `ObjectRef`, which can be used with `ray.get` to wait for the output dataset.
        """
        if self._dataset_ref is not None:
            # already started
            return self._dataset_ref

        # read the output dataset if the task has already finished
        if os.path.exists(self.ray_dataset_path):
            logger.info(f"task {self.key} already finished, skipping")
            output = load(self.ray_dataset_path)
            self._dataset_ref = ray.put(output)
            return self._dataset_ref

        task = copy.copy(self)
        task.input_deps = {dep_key: None for dep_key in task.input_deps}

        @ray.remote
        def exec_task(task: Task, *inputs: DataSet) -> DataSet:
            import multiprocessing as mp
            import os
            from pathlib import Path

            from loguru import logger

            # ray use a process pool to execute tasks
            # we set the current process name to the task name
            # so that we can see task name in the logs
            mp.current_process().name = task.key

            # probe the retry count
            task.retry_count = 0
            while os.path.exists(task.ray_marker_path):
                task.retry_count += 1
                if task.retry_count > DEFAULT_MAX_RETRY_COUNT:
                    raise RuntimeError(f"task {task.key} failed after {task.retry_count} retries")
            if task.retry_count > 0:
                logger.warning(f"task {task.key} is being retried for the {task.retry_count}th time")
            # create the marker file
            Path(task.ray_marker_path).touch()

            # put the inputs into the task
            assert len(inputs) == len(task.input_deps)
            task.input_datasets = list(inputs)
            # execute the task
            status = task.exec()
            if status != WorkStatus.SUCCEED:
                raise task.exception or RuntimeError(f"task {task.key} failed with status {status}")

            # dump the output dataset atomically
            os.makedirs(os.path.dirname(task.ray_dataset_path), exist_ok=True)
            dump(task.output, task.ray_dataset_path, atomic_write=True)
            return task.output

        # this shows as {"name": ...} in timeline
        exec_task._function_name = repr(task)

        remote_function = exec_task.options(
            # ray task name
            # do not include task id so that they can be grouped by node in ray dashboard
            name=f"{task.node_id}.{self.__class__.__name__}",
            num_cpus=self.cpu_limit,
            num_gpus=self.gpu_limit,
            memory=int(self.memory_limit),
            # note: `exec_on_scheduler` is ignored here,
            #       because dataset is distributed on ray
        )
        try:
            self._dataset_ref = remote_function.remote(task, *[dep.run_on_ray() for dep in self.input_deps.values()])
        except RuntimeError as e:
            if "SimpleQueue objects should only be shared between processes through inheritance" in str(e):
                raise RuntimeError(
                    f"Can't pickle task '{task.key}'. Please check if your function has captured unpicklable objects. {task.location}\n"
                    f"HINT: DO NOT use externally imported loguru logger in your task. Please import it within the task."
                ) from e
            raise e
        return self._dataset_ref


class ExecSqlQueryMixin(Task):

    enable_temp_directory = False
    cpu_overcommit_ratio = 1.0
    memory_overcommit_ratio = 1.0
    input_view_index = 0
    query_udfs: List[UDFContext] = []
    parquet_compression = None
    parquet_compression_level = None

    @cached_property
    def rand_seed_float(self) -> int:
        return self.random_float()

    @cached_property
    def rand_seed_uint32(self) -> int:
        return self.random_uint32()

    @property
    def input_udfs(self) -> List[UDFContext]:
        if self.input_datasets is None:
            return []
        return [udf for dataset in self.input_datasets for udf in dataset.udfs]

    @property
    def udfs(self):
        return set(self.query_udfs + self.input_udfs)

    @property
    def compression_type_str(self):
        return f"COMPRESSION '{self.parquet_compression}'" if self.parquet_compression is not None else "COMPRESSION 'uncompressed'"

    @property
    def compression_level_str(self):
        return (
            f"COMPRESSION_LEVEL {self.parquet_compression_level}"
            if self.parquet_compression == "ZSTD" and self.parquet_compression_level is not None
            else ""
        )

    @property
    def compression_options(self):
        return ", ".join(filter(None, (self.compression_type_str, self.compression_level_str)))

    def prepare_connection(self, conn: duckdb.DuckDBPyConnection):
        logger.debug(f"duckdb version: {duckdb.__version__}")
        # set random seed
        self.exec_query(conn, f"select setseed({self.rand_seed_float})")
        # prepare connection
        effective_cpu_count = math.ceil(self.cpu_limit * self.cpu_overcommit_ratio)
        effective_memory_size = round_up(self.memory_limit * self.memory_overcommit_ratio, MB)
        self.exec_query(
            conn,
            f"""
  SET threads TO {effective_cpu_count};
  SET memory_limit='{effective_memory_size // MB}MB';
  SET temp_directory='{self.temp_abspath if self.enable_temp_directory else ""}';
  SET enable_object_cache=true;
  SET arrow_large_buffer_size=true;
  SET preserve_insertion_order=false;
  SET max_expression_depth=10000;
""",
        )
        for udf in self.udfs:
            logger.debug("bind udf: {}", udf)
            udf.bind(conn)

    def create_input_views(
        self,
        conn: duckdb.DuckDBPyConnection,
        input_datasets: List[DataSet],
        filesystem: fsspec.AbstractFileSystem = None,
    ) -> List[str]:
        input_views = OrderedDict()
        for input_dataset in input_datasets:
            self.input_view_index += 1
            view_name = f"{INPUT_VIEW_PREFIX}_{self.id}_{self.input_view_index:06d}"
            input_views[view_name] = f"CREATE VIEW {view_name} AS {input_dataset.sql_query_fragment(filesystem, conn)};"
            logger.debug(f"create input view '{view_name}': {input_views[view_name]}")
            conn.sql(input_views[view_name])
        return list(input_views.keys())

    def exec_query(
        self,
        conn: duckdb.DuckDBPyConnection,
        query_statement: str,
        enable_profiling=False,
        log_query=True,
        log_output=False,
    ) -> Dict[str, int]:
        perf_metrics: Dict[str, np.int64] = defaultdict(np.int64)

        try:
            if log_query:
                logger.debug(f"running sql query: {query_statement}")
            start_time = time.time()
            query_output = conn.sql("SET enable_profiling='json';" if enable_profiling else "RESET enable_profiling;")
            query_output = conn.sql(query_statement)
            elapsed_time = time.time() - start_time
            if log_query:
                logger.debug(f"query elapsed time: {elapsed_time:.3f} secs")
        except duckdb.OutOfMemoryException as ex:
            raise OutOfMemory(f"{self.key} failed with OOM error") from ex
        except Exception as ex:
            # attach the query statement to the exception
            raise RuntimeError(f"failed to run query: {query_statement}") from ex

        def sum_children_metrics(obj: Dict, metric: str):
            value = obj.get(metric, None)
            if value is not None:
                return value
            if "children" not in obj:
                return 0
            return sum(sum_children_metrics(child, metric) for child in obj["children"])

        def extract_perf_metrics(obj: Dict):
            name = obj.get("operator_type", "")
            if name.startswith("TABLE_SCAN"):
                perf_metrics["num input rows"] += obj["operator_cardinality"]
                perf_metrics["input load time (secs)"] += obj["operator_timing"]
            elif name.startswith("COPY_TO_FILE"):
                perf_metrics["num output rows"] += sum(sum_children_metrics(child, "operator_cardinality") for child in obj["children"])
                perf_metrics["output dump time (secs)"] += obj["operator_timing"]
            return obj

        if query_output is not None:
            output_rows = query_output.fetchall()
            if log_output or (enable_profiling and self.ctx.enable_profiling):
                for row in output_rows:
                    logger.debug(f"query output:{os.linesep}{''.join(filter(None, row))}")
            if enable_profiling:
                _, json_str = output_rows[0]
                json.loads(json_str, object_hook=extract_perf_metrics)

        return perf_metrics


class DataSourceTask(Task):
    def __init__(
        self,
        ctx: RuntimeContext,
        dataset: DataSet,
        partition_infos: List[PartitionInfo],
    ) -> None:
        super().__init__(ctx, [], partition_infos)
        self.dataset = dataset

    def __str__(self) -> str:
        return super().__str__() + f", dataset=<{self.dataset}>"

    @property
    def exec_on_scheduler(self) -> bool:
        return True

    def run(self) -> bool:
        logger.info(f"added data source: {self.dataset}")
        if isinstance(self.dataset, (SqlQueryDataSet, ArrowTableDataSet)):
            self.dataset = ParquetDataSet.create_from(self.dataset.to_arrow_table(), self.runtime_output_abspath)
        return True


class MergeDataSetsTask(Task):
    @property
    def exec_on_scheduler(self) -> bool:
        return True

    @property
    def self_contained_output(self):
        return False

    def initialize(self):
        pass

    def finalize(self):
        pass

    def run(self) -> bool:
        datasets = self.input_datasets
        assert datasets, f"empty list of input datasets: {self}"
        assert all(isinstance(dataset, (DataSet, type(datasets[0]))) for dataset in datasets)
        self.dataset = datasets[0].merge(datasets)
        logger.info(f"created merged dataset: {self.dataset}")
        return True


class SplitDataSetTask(Task):

    __slots__ = (
        "partition",
        "npartitions",
    )

    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> None:
        assert len(input_deps) == 1, f"wrong number of input deps for data set partition: {input_deps}"
        super().__init__(ctx, input_deps, partition_infos)
        self.partition = partition_infos[-1].index
        self.npartitions = partition_infos[-1].npartitions

    @property
    def exec_on_scheduler(self) -> bool:
        return True

    @property
    def self_contained_output(self):
        return False

    def initialize(self):
        pass

    def finalize(self):
        pass

    def run(self) -> bool:
        self.dataset = self.input_datasets[0].partition_by_files(self.npartitions)[self.partition]
        return True


class PartitionProducerTask(Task):

    __slots__ = (
        "npartitions",
        "dimension",
        "partitioned_datasets",
    )

    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
        npartitions: int,
        dimension: str,
        output_name: str = None,
        output_path: str = None,
        cpu_limit: int = None,
        memory_limit: int = None,
    ) -> None:
        assert len(input_deps) == 1, f"wrong number of inputs: {input_deps}"
        assert isinstance(npartitions, int), f"npartitions is not an integer: {npartitions}"
        super().__init__(
            ctx,
            input_deps,
            partition_infos,
            output_name,
            output_path,
            cpu_limit,
            memory_limit=memory_limit,
        )
        self.npartitions = npartitions
        self.dimension = dimension
        # implementor should set this rather than `dataset`
        self.partitioned_datasets: List[DataSet] = None

    def __str__(self) -> str:
        return super().__str__() + f", npartitions={self.npartitions}, dimension={self.dimension}"

    def _create_empty_file(self, partition_idx: int, dataset: DataSet) -> str:
        """
        Create an empty file for a partition according to the schema of the dataset.
        Return the path relative to the output directory.
        """
        if isinstance(self, HashPartitionDuckDbTask) and self.hive_partitioning:
            empty_file_prefix = os.path.join(
                self.runtime_output_abspath,
                f"{self.data_partition_column}={partition_idx}",
                f"{self.output_filename}-{partition_idx}-empty",
            )
            Path(empty_file_prefix).parent.mkdir(exist_ok=True)
        else:
            empty_file_prefix = os.path.join(
                self.runtime_output_abspath,
                f"{self.output_filename}-{partition_idx}-empty",
            )

        if isinstance(dataset, CsvDataSet):
            empty_file_path = Path(empty_file_prefix + ".csv")
            empty_file_path.touch()
        elif isinstance(dataset, JsonDataSet):
            empty_file_path = Path(empty_file_prefix + ".json")
            empty_file_path.touch()
        elif isinstance(dataset, ParquetDataSet):
            with duckdb.connect(database=":memory:") as conn:
                conn.sql(f"SET threads TO 1")
                dataset_schema = dataset.to_batch_reader(batch_size=1, conn=conn).schema
            extra_partitions = (
                [PartitionInfo(partition_idx, self.npartitions, self.dimension)]
                if not isinstance(self, HashPartitionTask)
                else [
                    PartitionInfo(partition_idx, self.npartitions, self.dimension),
                    PartitionInfo(partition_idx, self.npartitions, self.data_partition_column),
                ]
            )
            schema_with_metadata = filter_schema(dataset_schema, excluded_cols=GENERATED_COLUMNS).with_metadata(
                self.parquet_kv_metadata_bytes(extra_partitions)
            )
            empty_file_path = Path(empty_file_prefix + ".parquet")
            parquet.ParquetWriter(empty_file_path, schema_with_metadata).close()
        else:
            raise ValueError(f"unsupported dataset type: {type(dataset)}")

        return str(empty_file_path.relative_to(self.runtime_output_abspath))

    def finalize(self):
        assert (
            len(self.partitioned_datasets) == self.npartitions
        ), f"number of partitions {len(self.partitioned_datasets)} not equal to {self.npartitions}"
        is_empty_partition = [dataset.empty for dataset in self.partitioned_datasets]

        if all(is_empty_partition):
            for dataset in self.partitioned_datasets:
                dataset.paths.clear()
        else:
            # Create an empty file for each empty partition.
            # This is to ensure that partition consumers have at least one file to read.
            empty_partitions = [idx for idx, empty in enumerate(is_empty_partition) if empty]
            nonempty_partitions = [idx for idx, empty in enumerate(is_empty_partition) if not empty]
            first_nonempty_dataset = self.partitioned_datasets[nonempty_partitions[0]]
            if empty_partitions:
                with ThreadPoolExecutor(self.cpu_limit) as pool:
                    empty_file_paths = list(
                        pool.map(
                            lambda idx: self._create_empty_file(idx, first_nonempty_dataset),
                            empty_partitions,
                        )
                    )
                    for partition_idx, empty_file_path in zip(empty_partitions, empty_file_paths):
                        self.partitioned_datasets[partition_idx].reset([empty_file_path], self.runtime_output_abspath)
                    logger.debug(f"created empty output files in partitions {empty_partitions} of {repr(self)}: {empty_file_paths[:3]}...")

        # reset root_dir from runtime_output_abspath to final_output_abspath
        for dataset in self.partitioned_datasets:
            # XXX: if the task has output in `runtime_output_abspath`,
            #      `root_dir` must be set and all row ranges must be full ranges.
            if dataset.root_dir == self.runtime_output_abspath:
                dataset.reset(dataset.paths, self.final_output_abspath, dataset.recursive)
            # XXX: otherwise, we assume there is no output in `runtime_output_abspath`.
            #      do nothing to the dataset.
        self.dataset = PartitionedDataSet(self.partitioned_datasets)

        super().finalize()

    def run(self) -> bool:
        raise NotImplementedError


class RepeatPartitionProducerTask(PartitionProducerTask):
    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
        npartitions: int,
        dimension: str,
        cpu_limit: int = None,
        memory_limit: int = None,
    ) -> None:
        super().__init__(
            ctx,
            input_deps,
            partition_infos,
            npartitions,
            dimension,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
        )

    @property
    def exec_on_scheduler(self) -> bool:
        return True

    @property
    def self_contained_output(self):
        return False

    def initialize(self):
        pass

    def run(self) -> bool:
        self.partitioned_datasets = [self.input_datasets[0] for _ in range(self.npartitions)]
        return True


class UserDefinedPartitionProducerTask(PartitionProducerTask):

    __slots__ = ("partition_func",)

    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
        npartitions: int,
        dimension: str,
        partition_func: Callable[[RuntimeContext, DataSet], List[DataSet]],
        cpu_limit: int = None,
        memory_limit: int = None,
    ) -> None:
        super().__init__(
            ctx,
            input_deps,
            partition_infos,
            npartitions,
            dimension,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
        )
        self.partition_func = partition_func

    def run(self) -> bool:
        try:
            self.partitioned_datasets = self.partition_func(self.ctx, self.input_datasets[0])
            return True
        finally:
            self.partition_func = None


class EvenlyDistributedPartitionProducerTask(PartitionProducerTask):

    __slots__ = (
        "partition_by_rows",
        "random_shuffle",
    )

    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
        npartitions: int,
        dimension: str,
        partition_by_rows=False,
        random_shuffle=False,
        cpu_limit: int = None,
        memory_limit: int = None,
    ) -> None:
        super().__init__(
            ctx,
            input_deps,
            partition_infos,
            npartitions,
            dimension,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
        )
        self.partition_by_rows = partition_by_rows
        self.random_shuffle = random_shuffle

    @property
    def exec_on_scheduler(self) -> bool:
        return True

    @property
    def self_contained_output(self):
        return False

    def run(self) -> bool:
        input_dataset = self.input_datasets[0]
        assert not (
            self.partition_by_rows and not isinstance(input_dataset, ParquetDataSet)
        ), f"Only parquet dataset supports partition by rows, found: {input_dataset}"
        if isinstance(input_dataset, ParquetDataSet) and self.partition_by_rows:
            self.partitioned_datasets = input_dataset.partition_by_rows(self.npartitions, self.random_shuffle)
        else:
            self.partitioned_datasets = input_dataset.partition_by_files(self.npartitions, self.random_shuffle)
        return True


class LoadPartitionedDataSetProducerTask(PartitionProducerTask):

    __slots__ = (
        "data_partition_column",
        "hive_partitioning",
    )

    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
        npartitions: int,
        dimension: str,
        data_partition_column: str,
        hive_partitioning: bool,
        cpu_limit: int = None,
        memory_limit: int = None,
    ) -> None:
        super().__init__(
            ctx,
            input_deps,
            partition_infos,
            npartitions,
            dimension,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
        )
        self.data_partition_column = data_partition_column
        self.hive_partitioning = hive_partitioning

    def run(self) -> bool:
        input_dataset = self.input_datasets[0]
        assert isinstance(input_dataset, ParquetDataSet), f"Not parquet dataset: {input_dataset}"
        self.partitioned_datasets = input_dataset.load_partitioned_datasets(self.npartitions, self.data_partition_column, self.hive_partitioning)
        return True


class PartitionConsumerTask(Task):

    __slots__ = ("last_partition",)

    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: List[PartitionProducerTask],
        partition_infos: List[PartitionInfo],
    ) -> None:
        assert all(isinstance(task, PartitionProducerTask) for task in input_deps)
        super().__init__(ctx, input_deps, partition_infos)
        self.last_partition = partition_infos[-1]

    def __str__(self) -> str:
        return super().__str__() + f", dataset=<{self.dataset}>"

    @property
    def exec_on_scheduler(self) -> bool:
        return not self.ctx.remove_empty_parquet

    @property
    def self_contained_output(self):
        return False

    def initialize(self):
        pass

    def finalize(self):
        pass

    def run(self) -> bool:
        # Build the dataset only after all `input_deps` finished, since `input_deps` could be tried multiple times.
        # Consumers always follow producers, so the input is a list of partitioned datasets.
        assert all(isinstance(dataset, PartitionedDataSet) for dataset in self.input_datasets)
        datasets = [dataset[self.last_partition.index] for dataset in self.input_datasets]
        self.dataset = datasets[0].merge(datasets)

        if self.ctx.remove_empty_parquet and isinstance(self.dataset, ParquetDataSet):
            self.dataset.remove_empty_files()

        assert (
            self.ctx.skip_task_with_empty_input or not self.dataset.empty
        ), f"found unexpected empty partition {self.last_partition} generated by {self.input_deps.keys()}"
        return True


class RangePartitionTask(Task):
    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> None:
        super().__init__(ctx, input_deps, partition_infos)


class PythonScriptTask(ExecSqlQueryMixin, Task):

    __slots__ = ("process_func",)

    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
        process_func: Callable[[RuntimeContext, List[DataSet], str], bool] = None,
        output_name: str = None,
        output_path: str = None,
        cpu_limit: int = None,
        gpu_limit: float = None,
        memory_limit: int = None,
    ) -> None:
        super().__init__(
            ctx,
            input_deps,
            partition_infos,
            output_name,
            output_path,
            cpu_limit,
            gpu_limit,
            memory_limit,
        )
        self.process_func = process_func

    def process(
        self,
        runtime_ctx: RuntimeContext,
        input_datasets: List[DataSet],
        output_path: str,
    ) -> bool:
        """
        This method can be overridden in subclass of `PythonScriptTask`.

        Parameters
        ----------
        runtime_ctx
            The runtime context, which defines a few global configuration info.
        input_datasets
            A list of input datasets. The number of datasets equal to the number of input_deps.
        output_path
            The absolute path of output directory created for each task generated from this node.
            The outputs generated by this node would be consumed by tasks of downstream nodes.

        Returns
        -------
            Return true if success. Return false or throw an exception if there is any error.
        """
        return self.process_func(runtime_ctx, input_datasets, output_path)

    def run(self) -> bool:
        try:
            self.add_elapsed_time()
            if self.skip_when_any_input_empty:
                return True
            return self.process(
                self.ctx.set_current_task(self),
                self.input_datasets,
                self.runtime_output_abspath,
            )
        finally:
            self.process_func = None
            self.dataset = FileSet(["*"], root_dir=self.final_output_abspath)


class ArrowComputeTask(ExecSqlQueryMixin, Task):

    cpu_overcommit_ratio = 0.5
    memory_overcommit_ratio = 0.5

    __slots__ = (
        "process_func",
        "parquet_row_group_size",
        "parquet_row_group_bytes",
        "parquet_dictionary_encoding",
        "use_duckdb_reader",
    )

    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
        process_func: Callable[[RuntimeContext, List[arrow.Table]], arrow.Table] = None,
        parquet_row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
        parquet_dictionary_encoding=False,
        parquet_compression="ZSTD",
        parquet_compression_level=3,
        use_duckdb_reader=False,
        output_name: str = None,
        output_path: str = None,
        cpu_limit: int = None,
        gpu_limit: float = None,
        memory_limit: int = None,
    ) -> None:
        super().__init__(
            ctx,
            input_deps,
            partition_infos,
            output_name,
            output_path,
            cpu_limit,
            gpu_limit,
            memory_limit,
        )
        self.process_func = process_func
        self.parquet_row_group_size = parquet_row_group_size
        self.parquet_row_group_bytes = clamp_row_group_bytes(parquet_row_group_size * 4 * KB)
        self.parquet_dictionary_encoding = parquet_dictionary_encoding
        self.parquet_compression = parquet_compression
        self.parquet_compression_level = parquet_compression_level
        self.use_duckdb_reader = use_duckdb_reader

    def clean_complex_attrs(self):
        self.exec_cq = None
        self.process_func = None
        super().clean_complex_attrs()

    def _call_process(self, runtime_ctx: RuntimeContext, input_tables: List[arrow.Table]) -> arrow.Table:
        return self.process(runtime_ctx, input_tables)

    def process(self, runtime_ctx: RuntimeContext, input_tables: List[arrow.Table]) -> arrow.Table:
        """
        This method can be overridden in subclass of `ArrowComputeTask`.

        Parameters
        ----------
        runtime_ctx
            The runtime context, which defines a few global configuration info.
        input_tables
            A list of arrow tables. The number of arrow tables equal to the number of input_deps.

        Returns
        -------
            Return the output as a arrow table. Throw an exception if there is any error.
        """
        return self.process_func(runtime_ctx, input_tables)

    def run(self) -> bool:
        try:
            conn = None
            self.add_elapsed_time()
            if self.skip_when_any_input_empty:
                return True

            if self.use_duckdb_reader:
                conn = duckdb.connect(database=":memory:", config={"allow_unsigned_extensions": "true"})
                self.prepare_connection(conn)

            input_tables = [dataset.to_arrow_table(max_workers=self.cpu_limit, conn=conn) for dataset in self.input_datasets]
            self.perf_metrics["num input rows"] += sum(table.num_rows for table in input_tables)
            self.add_elapsed_time("input load time (secs)")
            if conn is not None:
                conn.close()

            output_table = self._call_process(self.ctx.set_current_task(self), input_tables)
            self.add_elapsed_time("compute time (secs)")

            return self.dump_output(output_table)
        except arrow.lib.ArrowMemoryError as ex:
            raise OutOfMemory(f"{self.key} failed with OOM error") from ex
        finally:
            if conn is not None:
                conn.close()

    def dump_output(self, output_table: arrow.Table):
        if output_table is None:
            logger.warning(f"user's process method returns none")
            return True

        if self.parquet_row_group_size == DEFAULT_ROW_GROUP_SIZE:
            # adjust row group size if it is not set by user
            self.adjust_row_group_size(output_table.nbytes, output_table.num_rows)

        # write arrow table to parquet files
        dump_to_parquet_files(
            output_table.replace_schema_metadata(self.parquet_kv_metadata_bytes()),
            self.runtime_output_abspath,
            self.output_filename,
            compression=(self.parquet_compression if self.parquet_compression is not None else "NONE"),
            compression_level=self.parquet_compression_level,
            row_group_size=self.parquet_row_group_size,
            row_group_bytes=self.parquet_row_group_bytes,
            use_dictionary=self.parquet_dictionary_encoding,
            max_workers=self.cpu_limit,
        )
        self.perf_metrics["num output rows"] += output_table.num_rows
        self.add_elapsed_time("output dump time (secs)")

        return True


class StreamOutput(object):

    __slots__ = (
        "output_table",
        "batch_indices",
        "force_checkpoint",
    )

    def __init__(
        self,
        output_table: arrow.Table,
        batch_indices: List[int] = None,
        force_checkpoint=False,
    ) -> None:
        self.output_table = cast_columns_to_large_string(output_table)
        self.batch_indices = batch_indices or []
        self.force_checkpoint = force_checkpoint and bool(batch_indices)


class ArrowStreamTask(ExecSqlQueryMixin, Task):

    cpu_overcommit_ratio = 0.5
    memory_overcommit_ratio = 0.5

    __slots__ = (
        "process_func",
        "background_io_thread",
        "streaming_batch_size",
        "streaming_batch_count",
        "parquet_row_group_size",
        "parquet_row_group_bytes",
        "parquet_dictionary_encoding",
        "parquet_compression",
        "parquet_compression_level",
        "secs_checkpoint_interval",
    )

    class RuntimeState(object):

        __slots__ = (
            "last_batch_indices",
            "input_batch_offsets",
            "streaming_output_paths",
            "streaming_batch_size",
            "streaming_batch_count",
        )

        def __init__(self, streaming_batch_size: int, streaming_batch_count: int) -> None:
            self.last_batch_indices: List[int] = None
            self.input_batch_offsets: List[int] = None
            self.streaming_output_paths: List[str] = []
            self.streaming_batch_size: int = streaming_batch_size
            self.streaming_batch_count: int = streaming_batch_count

        def __str__(self) -> str:
            return f"streaming_batch_size={self.streaming_batch_size}, input_batch_offsets={self.input_batch_offsets}, streaming_output_paths[{len(self.streaming_output_paths)}]={self.streaming_output_paths[:3]}..."

        @property
        def max_batch_offsets(self):
            return max(self.input_batch_offsets)

        def update_batch_offsets(self, batch_indices: Optional[List[int]]):
            if batch_indices is None:
                return
            if self.last_batch_indices is None:
                self.last_batch_indices = [-1] * len(batch_indices)
            if self.input_batch_offsets is None:
                self.input_batch_offsets = [0] * len(batch_indices)
            self.input_batch_offsets = [i + j - k for i, j, k in zip(self.input_batch_offsets, batch_indices, self.last_batch_indices)]
            self.last_batch_indices = batch_indices

        def reset(self):
            self.input_batch_offsets.clear()
            self.streaming_output_paths.clear()

    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
        process_func: Callable[[RuntimeContext, List[arrow.RecordBatchReader]], Iterable[arrow.Table]] = None,
        background_io_thread=True,
        streaming_batch_size: int = DEFAULT_BATCH_SIZE,
        secs_checkpoint_interval: int = None,
        parquet_row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
        parquet_dictionary_encoding=False,
        parquet_compression="ZSTD",
        parquet_compression_level=3,
        use_duckdb_reader=False,
        output_name: str = None,
        output_path: str = None,
        cpu_limit: int = None,
        gpu_limit: float = None,
        memory_limit: int = None,
    ) -> None:
        super().__init__(
            ctx,
            input_deps,
            partition_infos,
            output_name,
            output_path,
            cpu_limit,
            gpu_limit,
            memory_limit,
        )
        self.process_func = process_func
        self.background_io_thread = background_io_thread
        self.streaming_batch_size = streaming_batch_size
        self.streaming_batch_count = 1
        self.parquet_row_group_size = parquet_row_group_size
        self.parquet_row_group_bytes = clamp_row_group_bytes(parquet_row_group_size * 4 * KB)
        self.parquet_dictionary_encoding = parquet_dictionary_encoding
        self.parquet_compression = parquet_compression
        self.parquet_compression_level = parquet_compression_level
        self.use_duckdb_reader = use_duckdb_reader
        self.secs_checkpoint_interval = secs_checkpoint_interval or self.ctx.secs_executor_probe_timeout
        self.runtime_state: Optional[ArrowStreamTask.RuntimeState] = None

    def __str__(self) -> str:
        return (
            super().__str__()
            + f", background_io_thread={self.background_io_thread}, streaming_batch_size={self.streaming_batch_size}, checkpoint_interval={self.secs_checkpoint_interval}s"
        )

    @property
    def max_batch_size(self) -> int:
        return self._memory_limit // 2

    def finalize(self):
        if self.runtime_state is not None:
            for path in self.runtime_state.streaming_output_paths:
                if not path.startswith(self.runtime_output_abspath):
                    os.link(
                        path,
                        os.path.join(self.runtime_output_abspath, os.path.basename(path)),
                    )
            self.runtime_state = None
        super().finalize()

    def clean_complex_attrs(self):
        self.exec_cq = None
        self.process_func = None
        super().clean_complex_attrs()

    def _wrap_output(self, output: Union[arrow.Table, StreamOutput], batch_indices: List[int] = None) -> StreamOutput:
        if isinstance(output, StreamOutput):
            assert len(output.batch_indices) == 0 or len(output.batch_indices) == len(
                self.input_deps
            ), f"num of batch indices {len(output.batch_indices)} not equal to num of inputs {len(self.input_deps)}"
            return output
        else:
            assert isinstance(output, arrow.Table)
            return StreamOutput(output, batch_indices)

    def _call_process(self, runtime_ctx: RuntimeContext, input_readers: List[arrow.RecordBatchReader]) -> Iterable[StreamOutput]:
        for output in self.process(runtime_ctx, input_readers):
            yield self._wrap_output(output)

    def process(self, runtime_ctx: RuntimeContext, input_readers: List[arrow.RecordBatchReader]) -> Iterable[arrow.Table]:
        """
        This method can be overridden in subclass of `ArrowStreamTask`.

        Parameters
        ----------
        runtime_ctx
            The runtime context, which defines a few global configuration info.
        input_readers
            A list of RecordBatchReader. The number of readers equal to the number of input_deps.

        Returns
        -------
            Return the output as a arrow table. Throw an exception if there is any error.
        """
        return self.process_func(runtime_ctx, input_readers)

    def restore_input_state(self, runtime_state: RuntimeState, input_readers: List[arrow.RecordBatchReader]):
        logger.info(f"restore input state to: {runtime_state}")
        assert len(runtime_state.input_batch_offsets) == len(
            input_readers
        ), f"num of batch offsets {len(runtime_state.input_batch_offsets)} not equal to num of input readers {len(input_readers)}"

        for batch_offset, input_reader in zip(runtime_state.input_batch_offsets, input_readers):
            if batch_offset <= 0:
                continue
            for (
                batch_index,
                input_batch,
            ) in enumerate(input_reader):
                logger.debug(f"skipped input batch #{batch_index}: {input_batch.num_rows} rows")
                if batch_index + 1 == batch_offset:
                    break
            assert batch_index + 1 <= batch_offset

    def run(self) -> bool:
        self.add_elapsed_time()
        if self.skip_when_any_input_empty:
            return True

        input_row_ranges = [dataset.resolved_row_ranges for dataset in self.input_datasets if isinstance(dataset, ParquetDataSet)]
        input_byte_size = [sum(row_range.estimated_data_size for row_range in row_ranges) for row_ranges in input_row_ranges]
        input_num_rows = [sum(row_range.num_rows for row_range in row_ranges) for row_ranges in input_row_ranges]
        input_files = [set(row_range.path for row_range in row_ranges) for row_ranges in input_row_ranges]
        self.perf_metrics["num input rows"] += sum(input_num_rows)
        self.perf_metrics["input data size (MB)"] += sum(input_byte_size) / MB

        # calculate the max streaming batch size based on memory limit
        avg_input_row_size = sum(self.compute_avg_row_size(nbytes, num_rows) for nbytes, num_rows in zip(input_byte_size, input_num_rows))
        max_batch_rows = self.max_batch_size // avg_input_row_size

        if self.runtime_state is None:
            if self.streaming_batch_size > max_batch_rows:
                logger.warning(
                    f"reduce streaming batch size from {self.streaming_batch_size} to {max_batch_rows} (approx. {self.max_batch_size/GB:.3f}GB)"
                )
                self.streaming_batch_size = max_batch_rows
            self.streaming_batch_count = max(
                1,
                max(map(len, input_files)),
                math.ceil(max(input_num_rows) / self.streaming_batch_size),
            )
        else:
            self.streaming_batch_size = self.runtime_state.streaming_batch_size
            self.streaming_batch_count = self.runtime_state.streaming_batch_count

        try:
            conn = None
            if self.use_duckdb_reader:
                conn = duckdb.connect(database=":memory:", config={"allow_unsigned_extensions": "true"})
                self.prepare_connection(conn)

            input_readers = [
                dataset.to_batch_reader(
                    batch_size=self.streaming_batch_size,
                    conn=conn,
                )
                for dataset in self.input_datasets
            ]

            if self.runtime_state is None:
                self.runtime_state = ArrowStreamTask.RuntimeState(self.streaming_batch_size, self.streaming_batch_count)
            else:
                self.restore_input_state(self.runtime_state, input_readers)
                self.runtime_state.last_batch_indices = None

            output_iter = self._call_process(self.ctx.set_current_task(self), input_readers)
            self.add_elapsed_time("compute time (secs)")

            if self.background_io_thread:
                with ConcurrentIter(output_iter) as concurrent_iter:
                    return self.dump_output(concurrent_iter)
            else:
                return self.dump_output(output_iter)
        except arrow.lib.ArrowMemoryError as ex:
            raise OutOfMemory(f"{self.key} failed with OOM error") from ex
        finally:
            if conn is not None:
                conn.close()

    def dump_output(self, output_iter: Iterable[StreamOutput]):
        def write_table(writer: parquet.ParquetWriter, table: arrow.Table):
            if table.num_rows == 0:
                return
            writer.write_table(table, self.parquet_row_group_size)
            self.perf_metrics["num output rows"] += table.num_rows
            self.add_elapsed_time("output dump time (secs)")

        create_checkpoint = False
        last_checkpoint_time = time.time() - self.random_float() * self.secs_checkpoint_interval / 2

        output: StreamOutput = next(output_iter, None)
        self.add_elapsed_time("compute time (secs)")

        if output is None:
            logger.warning(f"user's process method returns none")
            return True

        if self.parquet_row_group_size == DEFAULT_ROW_GROUP_SIZE:
            # adjust row group size if it is not set by user
            self.adjust_row_group_size(
                self.streaming_batch_count * output.output_table.nbytes,
                self.streaming_batch_count * output.output_table.num_rows,
            )

        output_iter = itertools.chain([output], output_iter)
        buffered_output = output.output_table.slice(length=0)

        for output_file_idx in itertools.count():
            output_path = os.path.join(
                self.runtime_output_abspath,
                f"{self.output_filename}-{output_file_idx}.parquet",
            )
            output_file = open(output_path, "wb", buffering=32 * MB)

            try:
                with parquet.ParquetWriter(
                    where=output_file,
                    schema=buffered_output.schema.with_metadata(self.parquet_kv_metadata_bytes()),
                    use_dictionary=self.parquet_dictionary_encoding,
                    compression=(self.parquet_compression if self.parquet_compression is not None else "NONE"),
                    compression_level=self.parquet_compression_level,
                    write_batch_size=max(16 * 1024, self.parquet_row_group_size // 8),
                    data_page_size=max(64 * MB, self.parquet_row_group_bytes // 8),
                ) as writer:

                    while (output := next(output_iter, None)) is not None:
                        self.add_elapsed_time("compute time (secs)")

                        if buffered_output.num_rows + output.output_table.num_rows < self.parquet_row_group_size:
                            buffered_output = arrow.concat_tables((buffered_output, output.output_table))
                        else:
                            write_table(writer, buffered_output)
                            buffered_output = output.output_table

                        periodic_checkpoint = bool(output.batch_indices) and (time.time() - last_checkpoint_time) >= self.secs_checkpoint_interval
                        create_checkpoint = output.force_checkpoint or periodic_checkpoint

                        if create_checkpoint:
                            self.runtime_state.update_batch_offsets(output.batch_indices)
                            last_checkpoint_time = time.time()
                            break

                    if buffered_output is not None:
                        write_table(writer, buffered_output)
                        buffered_output = buffered_output.slice(length=0)

            finally:
                if isinstance(output_file, io.IOBase):
                    output_file.close()

            assert buffered_output is None or buffered_output.num_rows == 0
            self.runtime_state.streaming_output_paths.append(output_path)

            if output is None:
                break

            if create_checkpoint and self.exec_cq is not None:
                checkpoint = copy.copy(self)
                checkpoint.clean_complex_attrs()
                self.exec_cq.push(checkpoint, buffering=False)
                logger.debug(
                    f"created and sent checkpoint #{self.runtime_state.max_batch_offsets}/{self.streaming_batch_count}: {self.runtime_state}"
                )

        return True


class ArrowBatchTask(ArrowStreamTask):
    @property
    def max_batch_size(self) -> int:
        return self._memory_limit // 3

    def _call_process(self, runtime_ctx: RuntimeContext, input_readers: List[arrow.RecordBatchReader]) -> Iterable[arrow.Table]:
        with contextlib.ExitStack() as stack:
            opened_readers = [stack.enter_context(ConcurrentIter(reader) if self.background_io_thread else reader) for reader in input_readers]
            for batch_index, input_batches in enumerate(itertools.zip_longest(*opened_readers, fillvalue=None)):
                input_tables = [
                    (reader.schema.empty_table() if batch is None else arrow.Table.from_batches([batch], reader.schema))
                    for reader, batch in zip(input_readers, input_batches)
                ]
                output_table = self._process_batches(runtime_ctx, input_tables)
                yield self._wrap_output(output_table, [batch_index] * len(input_batches))

    def _process_batches(self, runtime_ctx: RuntimeContext, input_tables: List[arrow.Table]) -> arrow.Table:
        return self.process(runtime_ctx, input_tables)

    def process(self, runtime_ctx: RuntimeContext, input_tables: List[arrow.Table]) -> arrow.Table:
        return self.process_func(runtime_ctx, input_tables)


class PandasComputeTask(ArrowComputeTask):
    def _call_process(self, runtime_ctx: RuntimeContext, input_tables: List[arrow.Table]) -> arrow.Table:
        input_dfs = [table.to_pandas() for table in input_tables]
        output_df = self.process(runtime_ctx, input_dfs)
        return arrow.Table.from_pandas(output_df, preserve_index=False) if output_df is not None else None

    def process(self, runtime_ctx: RuntimeContext, input_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        return self.process_func(runtime_ctx, input_dfs)


class PandasBatchTask(ArrowBatchTask):
    def _process_batches(self, runtime_ctx: RuntimeContext, input_tables: List[arrow.Table]) -> arrow.Table:
        input_dfs = [table.to_pandas() for table in input_tables]
        output_df = self.process(runtime_ctx, input_dfs)
        return arrow.Table.from_pandas(output_df, preserve_index=False)

    def process(self, runtime_ctx: RuntimeContext, input_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        return self.process_func(runtime_ctx, input_dfs)


class SqlEngineTask(ExecSqlQueryMixin, Task):

    __slots__ = (
        "sql_queries",
        "per_thread_output",
        "materialize_output",
        "materialize_in_memory",
        "batched_processing",
        "parquet_row_group_size",
        "parquet_row_group_bytes",
        "parquet_dictionary_encoding",
        "parquet_compression",
        "parquet_compression_level",
    )

    memory_overcommit_ratio = 0.9

    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
        sql_queries: List[str],
        udfs: List[UDFContext] = None,
        per_thread_output=True,
        materialize_output=True,
        materialize_in_memory=False,
        batched_processing=False,
        enable_temp_directory=False,
        parquet_row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
        parquet_dictionary_encoding=False,
        parquet_compression="ZSTD",
        parquet_compression_level=3,
        output_name: str = None,
        output_path: str = None,
        cpu_limit: int = None,
        gpu_limit: float = None,
        memory_limit: int = None,
        cpu_overcommit_ratio: float = 1.0,
        memory_overcommit_ratio: float = 0.9,
    ) -> None:
        super().__init__(
            ctx,
            input_deps,
            partition_infos,
            output_name,
            output_path,
            cpu_limit,
            gpu_limit,
            memory_limit,
        )
        self.cpu_overcommit_ratio = cpu_overcommit_ratio
        self.memory_overcommit_ratio = memory_overcommit_ratio
        self.sql_queries = sql_queries
        self.query_udfs: List[UDFContext] = udfs or []
        self.per_thread_output = per_thread_output
        self.materialize_output = materialize_output
        self.materialize_in_memory = materialize_in_memory
        self.batched_processing = batched_processing and len(self.input_deps) == 1
        self.enable_temp_directory = enable_temp_directory
        self.parquet_row_group_size = parquet_row_group_size
        self.parquet_row_group_bytes = clamp_row_group_bytes(parquet_row_group_size * 4 * KB)
        self.parquet_dictionary_encoding = parquet_dictionary_encoding
        self.parquet_compression = parquet_compression
        self.parquet_compression_level = parquet_compression_level

    def __str__(self) -> str:
        return super().__str__() + f", sql_query=<{self.oneline_query[:100]}...>, udfs={self.udfs}, batched_processing={self.batched_processing}"

    @property
    def oneline_query(self) -> str:
        return "; ".join(" ".join(filter(None, map(str.strip, query.splitlines()))) for query in self.sql_queries)

    @property
    def max_batch_size(self) -> int:
        return self._memory_limit // 2

    def cleanup(self):
        self.udfs.clear()
        super().cleanup()

    def run(self) -> bool:
        if self.skip_when_any_input_empty:
            return True

        if self.batched_processing and isinstance(self.input_datasets[0], ParquetDataSet):
            input_batches = [[batch] for batch in self.input_datasets[0].partition_by_size(self.max_batch_size)]
        else:
            input_batches = [self.input_datasets]

        for batch_index, input_batch in enumerate(input_batches):
            with duckdb.connect(database=":memory:", config={"allow_unsigned_extensions": "true"}) as conn:
                self.prepare_connection(conn)
                self.process_batch(batch_index, input_batch, conn)

        return True

    def process_batch(
        self,
        batch_index: int,
        input_datasets: List[DataSet],
        conn: duckdb.DuckDBPyConnection,
    ):
        # define inputs as views
        input_views = self.create_input_views(conn, input_datasets)

        if isinstance(self.parquet_dictionary_encoding, bool):
            dictionary_encoding_cfg = "DICTIONARY_ENCODING TRUE," if self.parquet_dictionary_encoding else ""
        else:
            dictionary_encoding_cfg = "DICTIONARY_ENCODING ({}),".format(", ".join(self.parquet_dictionary_encoding))

        for query_index, sql_query in enumerate(self.sql_queries):
            last_query = query_index + 1 == len(self.sql_queries)
            output_filename = f"{self.output_filename}-{batch_index}.{query_index}"
            output_path = self.runtime_output_abspath

            if not self.per_thread_output:
                output_path = os.path.join(output_path, f"{output_filename}.parquet")

            sql_query = sql_query.format(
                *input_views,
                batch_index=batch_index,
                query_index=query_index,
                cpu_limit=self.cpu_limit,
                memory_limit=self.memory_limit,
                rand_seed=self.rand_seed_uint32,
                output_filename=output_filename,
                **self.partition_infos_as_dict,
            )

            if last_query and self.materialize_in_memory:
                self.merge_metrics(
                    self.exec_query(
                        conn,
                        f"EXPLAIN ANALYZE CREATE OR REPLACE TEMP TABLE temp_query_result AS {sql_query}",
                        enable_profiling=True,
                    )
                )
                sql_query = f"select * from temp_query_result"

            if last_query and self.materialize_output:
                sql_query = f"""
  COPY (
    {sql_query}
  ) TO '{output_path}' (
      FORMAT PARQUET,
      KV_METADATA {self.parquet_kv_metadata_str()},
      {self.compression_options},
      ROW_GROUP_SIZE {self.parquet_row_group_size},
      ROW_GROUP_SIZE_BYTES {self.parquet_row_group_bytes},
      {dictionary_encoding_cfg}
      PER_THREAD_OUTPUT {self.per_thread_output},
      FILENAME_PATTERN '{output_filename}.{{i}}',
      OVERWRITE_OR_IGNORE true)
      """

            self.merge_metrics(self.exec_query(conn, f"EXPLAIN ANALYZE {sql_query}", enable_profiling=True))


class HashPartitionTask(PartitionProducerTask):

    __slots__ = (
        "hash_columns",
        "data_partition_column",
        "random_shuffle",
        "shuffle_only",
        "drop_partition_column",
        "use_parquet_writer",
        "hive_partitioning",
        "parquet_row_group_size",
        "parquet_row_group_bytes",
        "parquet_dictionary_encoding",
        "parquet_compression",
        "parquet_compression_level",
        "partitioned_datasets",
        "_io_workers",
        "_partition_files",
        "_partition_writers",
        "_pending_write_works",
        "_file_writer_closed",
    )

    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
        npartitions: int,
        dimension: str,
        hash_columns: List[str],
        data_partition_column: str,
        random_shuffle: bool = False,
        shuffle_only: bool = False,
        drop_partition_column=False,
        use_parquet_writer=False,
        hive_partitioning=False,
        parquet_row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
        parquet_dictionary_encoding=False,
        parquet_compression="ZSTD",
        parquet_compression_level=3,
        output_name: str = None,
        output_path: str = None,
        cpu_limit: int = None,
        memory_limit: int = None,
    ) -> None:
        self.hash_columns = ["random()"] if random_shuffle else hash_columns
        self.data_partition_column = data_partition_column
        self.random_shuffle = random_shuffle
        self.shuffle_only = shuffle_only
        self.drop_partition_column = drop_partition_column
        self.use_parquet_writer = use_parquet_writer
        self.hive_partitioning = hive_partitioning
        self.parquet_row_group_size = parquet_row_group_size
        self.parquet_row_group_bytes = clamp_row_group_bytes(parquet_row_group_size * 4 * KB)
        self.parquet_dictionary_encoding = parquet_dictionary_encoding
        self.parquet_compression = parquet_compression
        self.parquet_compression_level = parquet_compression_level
        super().__init__(
            ctx,
            input_deps,
            partition_infos,
            npartitions,
            dimension,
            output_name,
            output_path,
            cpu_limit,
            memory_limit,
        )
        self.partitioned_datasets = None
        self._io_workers: ThreadPoolExecutor = None
        self._partition_files: List[BinaryIO] = None
        self._partition_writers: List[parquet.ParquetWriter] = None
        self._pending_write_works: List[Future] = None
        self._file_writer_closed = True

    def __str__(self) -> str:
        return super().__str__() + f", hash_columns={self.hash_columns}, data_partition_column={self.data_partition_column}"

    @staticmethod
    def create(engine_type: Literal["duckdb", "arrow"], *args, **kwargs) -> "HashPartitionTask":
        if engine_type == "duckdb":
            return HashPartitionDuckDbTask(*args, *kwargs)
        if engine_type == "arrow":
            return HashPartitionArrowTask(*args, *kwargs)
        raise ValueError(f"Unknown hash partition engine: '{engine_type}'")

    @property
    def max_batch_size(self) -> int:
        return self._memory_limit // 6

    @property
    def write_buffer_size(self) -> int:
        write_buffer_size = min(
            4 * MB,
            round_up(min(16 * GB, self.max_batch_size) // self.npartitions, 16 * KB),
        )
        return write_buffer_size if write_buffer_size >= 128 * KB else -1  # disable write buffer if too small

    @property
    def num_workers(self) -> int:
        return min(self.npartitions, self.cpu_limit)

    @property
    def io_workers(self):
        if self._io_workers is None:
            self._io_workers = ThreadPoolExecutor(self.num_workers)
        return self._io_workers

    def _wait_pending_writes(self):
        for i in range(len(self._pending_write_works)):
            if self._pending_write_works[i] is not None:
                self._pending_write_works[i].result()
                self._pending_write_works[i] = None

    def _close_file_writers(self):
        if self._file_writer_closed:
            return
        self._file_writer_closed = True
        self.add_elapsed_time()
        self._wait_pending_writes()
        if self._io_workers is not None:
            list(self._io_workers.map(lambda w: w.close(), filter(None, self._partition_writers)))
            list(self._io_workers.map(lambda f: f.close(), filter(None, self._partition_files)))
            self._io_workers.shutdown(wait=True)
        self.add_elapsed_time("output dump time (secs)")

    def _create_file_writer(self, partition_idx: int, schema: arrow.Schema):
        partition_filename = f"{self.output_filename}-{partition_idx}.parquet"
        partition_path = os.path.join(self.runtime_output_abspath, partition_filename)

        self._partition_files[partition_idx] = open(partition_path, "wb", buffering=self.write_buffer_size)
        output_file = self._partition_files[partition_idx]

        self.partitioned_datasets[partition_idx].paths.append(partition_filename)
        self._partition_writers[partition_idx] = parquet.ParquetWriter(
            where=output_file,
            schema=schema.with_metadata(
                self.parquet_kv_metadata_bytes(
                    [
                        PartitionInfo(partition_idx, self.npartitions, self.dimension),
                        PartitionInfo(partition_idx, self.npartitions, self.data_partition_column),
                    ]
                )
            ),
            use_dictionary=self.parquet_dictionary_encoding,
            compression=(self.parquet_compression if self.parquet_compression is not None else "NONE"),
            compression_level=self.parquet_compression_level,
            write_batch_size=max(16 * 1024, self.parquet_row_group_size // 8),
            data_page_size=max(64 * MB, self.parquet_row_group_bytes // 8),
        )
        return self._partition_writers[partition_idx]

    def _write_to_partition(self, partition_idx, partition, pending_write: Future = None):
        if pending_write is not None:
            pending_write.result()
        if partition is not None:
            writer = self._partition_writers[partition_idx] or self._create_file_writer(partition_idx, partition.schema)
            writer.write_table(partition, self.parquet_row_group_size)

    def _write_partitioned_tables(self, partitioned_tables):
        assert len(partitioned_tables) == self.npartitions
        assert len(self._pending_write_works) == self.npartitions

        self._pending_write_works = [
            self.io_workers.submit(self._write_to_partition, partition_idx, partition, pending_write)
            for partition_idx, (partition, pending_write) in enumerate(zip(partitioned_tables, self._pending_write_works))
        ]
        self.perf_metrics["num output rows"] += sum(partition.num_rows for partition in partitioned_tables if partition is not None)
        self._wait_pending_writes()

    def initialize(self):
        super().initialize()
        if isinstance(self, HashPartitionDuckDbTask) and self.hive_partitioning:
            self.partitioned_datasets = [
                ParquetDataSet(
                    [os.path.join(f"{self.data_partition_column}={partition_idx}", "*.parquet")],
                    root_dir=self.runtime_output_abspath,
                )
                for partition_idx in range(self.npartitions)
            ]
        else:
            self.partitioned_datasets = [ParquetDataSet([], root_dir=self.runtime_output_abspath) for _ in range(self.npartitions)]
        self._partition_files = [None] * self.npartitions
        self._partition_writers = [None] * self.npartitions
        self._pending_write_works = [None] * self.npartitions
        self._file_writer_closed = False

    def finalize(self):
        # first close all writers
        self._close_file_writers()
        assert (
            self.perf_metrics["num input rows"] == self.perf_metrics["num output rows"]
        ), f'num input rows {self.perf_metrics["num input rows"]} != num output rows {self.perf_metrics["num output rows"]}'
        super().finalize()

    def cleanup(self):
        self._close_file_writers()
        self._io_workers = None
        self._partition_files = None
        self._partition_writers = None
        super().cleanup()

    def partition(self, input_dataset: ParquetDataSet):
        raise NotImplementedError

    def run(self) -> bool:
        self.add_elapsed_time()
        if self.skip_when_any_input_empty:
            return True

        input_dataset = self.input_datasets[0]
        assert isinstance(input_dataset, ParquetDataSet), f"only parquet dataset supported, found {input_dataset}"
        input_paths = input_dataset.resolved_paths
        input_byte_size = input_dataset.estimated_data_size
        input_num_rows = input_dataset.num_rows

        logger.info(f"partitioning dataset: {len(input_paths)} files, {input_byte_size/GB:.3f}GB, {input_num_rows} rows")
        input_batches = input_dataset.partition_by_size(self.max_batch_size)

        for batch_index, input_batch in enumerate(input_batches):
            batch_start_time = time.time()
            batch_byte_size = input_batch.estimated_data_size
            batch_num_rows = input_batch.num_rows
            logger.info(
                f"start to partition batch #{batch_index+1}/{len(input_batches)}: {len(input_batch.resolved_paths)} files, {batch_byte_size/GB:.3f}GB, {batch_num_rows} rows"
            )
            self.partition(batch_index, input_batch)
            logger.info(f"finished to partition batch #{batch_index+1}/{len(input_batches)}: {time.time() - batch_start_time:.3f} secs")

        return True


class HashPartitionDuckDbTask(ExecSqlQueryMixin, HashPartitionTask):

    memory_overcommit_ratio = 1.0

    def __str__(self) -> str:
        return super().__str__() + f", hive_partitioning={self.hive_partitioning}"

    @property
    def partition_query(self):
        if self.shuffle_only:
            partition_query = r"SELECT * FROM {0}"
        else:
            if self.random_shuffle:
                hash_values = f"random() * {2147483647 // self.npartitions * self.npartitions}"
            else:
                hash_values = f"hash( concat_ws( '##', {', '.join(self.hash_columns)} ) )"
            partition_keys = f"CAST({hash_values} AS UINT64) % {self.npartitions}::UINT64 AS {self.data_partition_column}"
            partition_query = f"""
      SELECT *,
        {partition_keys}
      FROM (
        SELECT COLUMNS(c -> c != '{self.data_partition_column}') FROM {{0}}
      )"""
        return partition_query

    def partition(self, batch_index: int, input_dataset: ParquetDataSet):
        with duckdb.connect(database=":memory:", config={"allow_unsigned_extensions": "true"}) as conn:
            self.prepare_connection(conn)
            if self.hive_partitioning:
                self.load_input_batch(conn, batch_index, input_dataset, sort_by_partition_key=True)
                self.write_hive_partitions(conn, batch_index, input_dataset)
            else:
                self.load_input_batch(conn, batch_index, input_dataset, sort_by_partition_key=True)
                self.write_flat_partitions(conn, batch_index, input_dataset)

    def load_input_batch(
        self,
        conn: duckdb.DuckDBPyConnection,
        batch_index: int,
        input_dataset: ParquetDataSet,
        sort_by_partition_key=False,
    ):
        input_views = self.create_input_views(conn, [input_dataset])
        partition_query = self.partition_query.format(*input_views, **self.partition_infos_as_dict)
        if sort_by_partition_key:
            partition_query += f" ORDER BY {self.data_partition_column}"

        perf_metrics = self.exec_query(
            conn,
            f"EXPLAIN ANALYZE CREATE OR REPLACE TABLE temp_query_result AS {partition_query}",
            enable_profiling=True,
        )
        self.perf_metrics["num input rows"] += perf_metrics["num input rows"]
        elapsed_time = self.add_elapsed_time("input load time (secs)")

        # make sure partition keys are in the range of [0, npartitions)
        min_partition_key, max_partition_key = conn.sql(
            f"SELECT MIN({self.data_partition_column}), MAX({self.data_partition_column}) FROM temp_query_result"
        ).fetchall()[0]
        assert min_partition_key >= 0, f"partition key {min_partition_key} is out of range 0-{self.npartitions-1}"
        assert max_partition_key < self.npartitions, f"partition key {max_partition_key} is out of range 0-{self.npartitions-1}"

        logger.debug(f"load input dataset #{batch_index+1}: {elapsed_time:.3f} secs")

    def write_hive_partitions(
        self,
        conn: duckdb.DuckDBPyConnection,
        batch_index: int,
        input_dataset: ParquetDataSet,
    ):
        batch_num_rows = input_dataset.num_rows
        self.exec_query(
            conn,
            f"SET partitioned_write_flush_threshold={round_up(batch_num_rows / self.cpu_limit / 4, KB)}",
        )
        copy_query_result = f"""
  COPY (
    SELECT * FROM temp_query_result
  ) TO '{self.runtime_output_abspath}' (
      FORMAT PARQUET,
      OVERWRITE_OR_IGNORE,
      WRITE_PARTITION_COLUMNS,
      PARTITION_BY {self.data_partition_column},
      KV_METADATA {self.parquet_kv_metadata_str()},
      {self.compression_options},
      ROW_GROUP_SIZE {self.parquet_row_group_size},
      ROW_GROUP_SIZE_BYTES {self.parquet_row_group_bytes},
      {"DICTIONARY_ENCODING TRUE," if self.parquet_dictionary_encoding else ""}
      FILENAME_PATTERN '{self.output_filename}-{batch_index}.{{i}}')
    """
        perf_metrics = self.exec_query(conn, f"EXPLAIN ANALYZE {copy_query_result}", enable_profiling=True)
        self.perf_metrics["num output rows"] += perf_metrics["num output rows"]
        elapsed_time = self.add_elapsed_time("output dump time (secs)")
        logger.debug(f"write partition data #{batch_index+1}: {elapsed_time:.3f} secs")

    def write_flat_partitions(
        self,
        conn: duckdb.DuckDBPyConnection,
        batch_index: int,
        input_dataset: ParquetDataSet,
    ):
        def write_partition_data(conn: duckdb.DuckDBPyConnection, partition_batch: List[Tuple[int, str]]) -> int:
            total_num_rows = 0
            for partition_idx, partition_filter in partition_batch:
                if self.use_parquet_writer:
                    partition_data = conn.sql(partition_filter).fetch_arrow_table()
                    self._write_to_partition(partition_idx, partition_data)
                    total_num_rows += partition_data.num_rows
                else:
                    partition_filename = f"{self.output_filename}-{partition_idx}.{batch_index}.parquet"
                    partition_path = os.path.join(self.runtime_output_abspath, partition_filename)
                    self.partitioned_datasets[partition_idx].paths.append(partition_filename)
                    perf_metrics = self.exec_query(
                        conn,
                        f"""
            EXPLAIN ANALYZE
            COPY (
              {partition_filter}
            ) TO '{partition_path}' (
                FORMAT PARQUET,
                KV_METADATA {self.parquet_kv_metadata_str(
                  [PartitionInfo(partition_idx, self.npartitions, self.dimension), PartitionInfo(partition_idx, self.npartitions, self.data_partition_column)])},
                {self.compression_options},
                ROW_GROUP_SIZE {self.parquet_row_group_size},
                ROW_GROUP_SIZE_BYTES {self.parquet_row_group_bytes},
                {"DICTIONARY_ENCODING TRUE," if self.parquet_dictionary_encoding else ""}
                OVERWRITE_OR_IGNORE true)
          """,
                        enable_profiling=True,
                        log_query=partition_idx == 0,
                        log_output=False,
                    )  # avoid duplicate logs
                    total_num_rows += perf_metrics["num output rows"]
            return total_num_rows

        column_projection = f"* EXCLUDE ({self.data_partition_column})" if self.drop_partition_column else "*"
        partition_filters = [
            (
                partition_idx,
                f"SELECT {column_projection} FROM temp_query_result WHERE {self.data_partition_column} = {partition_idx}",
            )
            for partition_idx in range(self.npartitions)
        ]
        partition_batches = split_into_rows(partition_filters, self.num_workers)

        with contextlib.ExitStack() as stack:
            db_conns = [stack.enter_context(conn.cursor()) for _ in range(self.num_workers)]
            self.perf_metrics["num output rows"] += sum(self.io_workers.map(write_partition_data, db_conns, partition_batches))
        elapsed_time = self.add_elapsed_time("output dump time (secs)")
        logger.debug(f"write partition data #{batch_index+1}: {elapsed_time:.3f} secs")


class HashPartitionArrowTask(HashPartitionTask):

    # WARNING: totally different hash partitions are generated if the random seeds changed.
    fixed_rand_seeds = (
        14592751030717519312,
        9336845975743342460,
        1211974630270170534,
        6392954943940246686,
    )

    def partition(self, batch_index: int, input_dataset: ParquetDataSet):
        import polars

        self.add_elapsed_time()
        table = input_dataset.to_arrow_table(max_workers=self.cpu_limit)
        self.perf_metrics["num input rows"] += table.num_rows
        elapsed_time = self.add_elapsed_time("input load time (secs)")
        logger.debug(f"load input dataset: {table.nbytes/MB:.3f}MB, {table.num_rows} rows, {elapsed_time:.3f} secs")

        if self.shuffle_only:
            partition_keys = table.column(self.data_partition_column)
        elif self.random_shuffle:
            partition_keys = arrow.array(self.numpy_random_gen.integers(self.npartitions, size=table.num_rows))
        else:
            hash_columns = polars.from_arrow(table.select(self.hash_columns))
            hash_values = hash_columns.hash_rows(*self.fixed_rand_seeds)
            partition_keys = (hash_values % self.npartitions).to_arrow()

        if self.data_partition_column in table.column_names:
            table = table.drop_columns(self.data_partition_column)
        table = table.append_column(self.data_partition_column, partition_keys)
        elapsed_time = self.add_elapsed_time("compute time (secs)")
        logger.debug(f"generate partition keys: {elapsed_time:.3f} secs")

        table_slice_size = max(DEFAULT_BATCH_SIZE, min(table.num_rows // 2, 100 * 1024 * 1024))
        num_iterations = math.ceil(table.num_rows / table_slice_size)

        def write_partition_data(
            partition_batch: List[Tuple[int, polars.DataFrame]],
        ) -> int:
            total_num_rows = 0
            for partition_idx, partition_data in partition_batch:
                total_num_rows += len(partition_data)
                self._write_to_partition(partition_idx, partition_data.to_arrow())
            return total_num_rows

        for table_slice_idx, table_slice_offset in enumerate(range(0, table.num_rows, table_slice_size)):
            table_slice = table.slice(table_slice_offset, table_slice_size)
            logger.debug(f"table slice #{table_slice_idx+1}/{num_iterations}: {table_slice.nbytes/MB:.3f}MB, {table_slice.num_rows} rows")

            df = polars.from_arrow(table_slice)
            del table_slice
            elapsed_time = self.add_elapsed_time("compute time (secs)")
            logger.debug(f"convert from arrow table #{table_slice_idx+1}/{num_iterations}: {elapsed_time:.3f} secs")

            partitioned_dfs = df.partition_by(
                [self.data_partition_column],
                maintain_order=False,
                include_key=not self.drop_partition_column,
                as_dict=True,
            )
            partitioned_dfs = [(partition_idx, df) for (partition_idx,), df in partitioned_dfs.items()]
            del df
            elapsed_time = self.add_elapsed_time("compute time (secs)")
            logger.debug(f"build partition data #{table_slice_idx+1}/{num_iterations}: {elapsed_time:.3f} secs")

            partition_batches = split_into_rows(partitioned_dfs, self.num_workers)
            self.perf_metrics["num output rows"] += sum(self.io_workers.map(write_partition_data, partition_batches))
            elapsed_time = self.add_elapsed_time("output dump time (secs)")
            logger.debug(f"write partition data #{table_slice_idx+1}/{num_iterations}: {elapsed_time:.3f} secs")


class ProjectionTask(Task):

    __slots__ = (
        "columns",
        "generated_columns",
        "union_by_name",
    )

    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
        columns: List[str],
        generated_columns: List[str],
        union_by_name: Optional[bool],
    ) -> None:
        super().__init__(ctx, input_deps, partition_infos)
        self.columns = list(columns)
        self.generated_columns = list(generated_columns)
        self.union_by_name = union_by_name

    @property
    def exec_on_scheduler(self) -> bool:
        return True

    @property
    def self_contained_output(self):
        return False

    def initialize(self):
        pass

    def finalize(self):
        pass

    def run(self) -> bool:
        self.dataset = copy.copy(self.input_datasets[0])
        assert not self.generated_columns or isinstance(
            self.dataset, ParquetDataSet
        ), f"generated columns can be only applied to parquet dataset, but found: {self.dataset}"
        self.dataset.columns = self.columns
        self.dataset.generated_columns = self.generated_columns
        if self.union_by_name is not None:
            self.dataset._union_by_name = self.union_by_name
        return True


DataSinkType = Literal["link_manifest", "copy", "link_or_copy", "manifest"]


class DataSinkTask(Task):

    __slots__ = (
        "type",
        "is_final_node",
    )

    manifest_filename = ".MANIFEST.txt"

    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
        output_path: str,
        type: DataSinkType = "link_manifest",
        manifest_only=False,
        is_final_node=False,
    ) -> None:
        assert type in (
            "link_manifest",
            "copy",
            "link_or_copy",
            "manifest",
        ), f"invalid sink type: {type}"
        assert output_path is not None, f"output path is required for {repr(self)}"
        super().__init__(ctx, input_deps, partition_infos, output_path=output_path)
        self.type = "manifest" if manifest_only else type
        # just a flag to indicate if this is the final results
        # there should be only one final results in the execution plan
        self.is_final_node = is_final_node
        self.temp_output = False

    @property
    def allow_speculative_exec(self) -> bool:
        return False

    @property
    def self_contained_output(self) -> bool:
        return self.type != "manifest"

    @property
    def final_output_abspath(self) -> str:
        if self.type in ("copy", "link_or_copy"):
            # in the first phase, we copy or link files to the staging directory
            return os.path.join(self.staging_root, self.output_dirname)
        else:
            # in the second phase, these files will be linked to the output directory
            return self.output_root

    @property
    def runtime_output_abspath(self) -> str:
        return self.final_output_abspath

    def clean_output(self, force=False) -> None:
        pass

    def run(self) -> bool:
        with ThreadPoolExecutor(min(32, len(self.input_datasets))) as pool:
            return self.collect_output_files(pool)

    def collect_output_files(self, pool: ThreadPoolExecutor) -> bool:
        final_output_dir = PurePath(self.final_output_abspath)
        runtime_output_dir = Path(self.runtime_output_abspath)
        dst_mount_point = find_mount_point(self.runtime_output_abspath)
        sink_type = self.type

        src_paths = [
            p
            for paths in pool.map(
                lambda dataset: [Path(path) for path in dataset.resolved_paths],
                self.input_datasets,
            )
            for p in paths
        ]
        logger.info(f"collected {len(src_paths)} files from {len(self.input_datasets)} input datasets")

        if len(set(p.name for p in src_paths)) == len(src_paths):
            dst_paths = [runtime_output_dir / p.name for p in src_paths]
        else:
            logger.warning(f"found duplicate filenames, appending index to filename...")
            dst_paths = [runtime_output_dir / f"{p.stem}.{idx}{p.suffix}" for idx, p in enumerate(src_paths)]

        output_paths = src_paths if sink_type == "manifest" else [final_output_dir / p.name for p in dst_paths]
        self.dataset = ParquetDataSet([str(p) for p in output_paths])  # FIXME: what if the dataset is not parquet?

        def copy_file(src_path: Path, dst_path: Path):
            # XXX: DO NOT use shutil.{copy, copy2, copyfileobj}
            #   they use sendfile on Linux. although they set blocksize=8M, the actual io size is fixed to 64k, resulting in low throughput.
            with open(src_path, "rb") as src_file, open(dst_path, "wb") as dst_file:
                shutil.copyfileobj(src_file, dst_file, length=16 * MB)

        def create_link_or_copy(src_path: Path, dst_path: Path):
            if dst_path.exists():
                logger.warning(f"destination path already exists, replacing {dst_path} with {src_path}")
                dst_path.unlink(missing_ok=True)
            same_mount_point = str(src_path).startswith(dst_mount_point)
            if sink_type == "copy":
                copy_file(src_path, dst_path)
            elif sink_type == "link_manifest":
                if same_mount_point:
                    os.link(src_path, dst_path)
                else:
                    dst_path.symlink_to(src_path)
            elif sink_type == "link_or_copy":
                if same_mount_point:
                    os.link(src_path, dst_path)
                else:
                    copy_file(src_path, dst_path)
            else:
                raise RuntimeError(f"invalid sink type: {sink_type}")
            return True

        if sink_type in ("copy", "link_or_copy", "link_manifest"):
            if src_paths:
                assert all(pool.map(create_link_or_copy, src_paths, dst_paths))
            else:
                logger.warning(f"input of data sink is empty: {self}")

        if sink_type == "manifest" or sink_type == "link_manifest":
            # write to a temporary file and rename it atomically
            manifest_path = final_output_dir / self.manifest_filename
            manifest_tmp_path = runtime_output_dir / f"{self.manifest_filename}.tmp"
            with open(manifest_tmp_path, "w", buffering=2 * MB) as manifest_file:
                for path in output_paths:
                    print(str(path), file=manifest_file)
            manifest_tmp_path.rename(manifest_path)
            logger.info(f"created a manifest file at {manifest_path}")

        if sink_type == "link_manifest":
            # remove the staging directory
            remove_path(self.staging_root)

            # check the output parquet files
            # if any file is broken, an exception will be raised
            if len(dst_paths) > 0 and dst_paths[0].suffix == ".parquet":
                logger.info(f"checked dataset files and found {self.dataset.num_rows} rows")

        return True


class RootTask(Task):
    @property
    def exec_on_scheduler(self) -> bool:
        return True

    def __init__(
        self,
        ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> None:
        super().__init__(ctx, input_deps, partition_infos)

    def initialize(self):
        pass

    def finalize(self):
        pass

    def run(self) -> bool:
        return True


class ExecutionPlan(object):
    """
    A directed acyclic graph (DAG) of tasks.
    """

    def __init__(self, ctx: RuntimeContext, root_task: RootTask, logical_plan: "LogicalPlan") -> None:
        from smallpond.logical.node import LogicalPlan

        self.ctx = ctx
        self.root_task = root_task
        self.logical_plan: LogicalPlan = logical_plan

    def __str__(self) -> str:
        visited = set()

        def to_str(task: Task, depth: int = 0) -> List[str]:
            lines = ["  " * depth + str(task)]
            if task.id in visited:
                return lines + ["  " * depth + " (omitted ...)"]
            visited.add(task.id)
            for dep in task.input_deps.values():
                lines.extend(to_str(dep, depth + 1))
            return lines

        return os.linesep.join(to_str(self.root_task))

    @cached_property
    def _final_results(self) -> DataSinkTask:
        for task in self.root_task.input_deps.values():
            if isinstance(task, DataSinkTask) and task.is_final_node:
                return task
        raise RuntimeError("no final results found")

    @property
    def final_output(self) -> DataSet:
        return self._final_results.output

    @property
    def final_output_path(self) -> str:
        return self._final_results.final_output_abspath

    @property
    def successful(self) -> str:
        return self.root_task.status == WorkStatus.SUCCEED

    @property
    def leaves(self) -> List[Task]:
        return [task for task in self.tasks.values() if not task.input_deps]

    @staticmethod
    def iter_tasks(task: Task, visited: Set[str] = None):
        visited = visited or set()
        assert task.key not in visited
        visited.add(task.key)
        yield task
        for dep in task.input_deps.values():
            if dep.key not in visited:
                yield from ExecutionPlan.iter_tasks(dep, visited)

    @property
    def tasks(self) -> Dict[str, Task]:
        return dict((task.key, task) for task in self.iter_tasks(self.root_task))

    @cached_property
    def named_outputs(self):
        assert self.successful
        named_outputs: Dict[str, DataSet] = {}
        task_outputs: Dict[str, List[DataSet]] = {}

        for task in self.tasks.values():
            if task.output_name:
                if task.output_name not in task_outputs:
                    task_outputs[task.output_name] = [task.output]
                else:
                    task_outputs[task.output_name].append(task.output)

        for name, datasets in task_outputs.items():
            named_outputs[name] = datasets[0].merge(datasets)
        return named_outputs

    def get_output(self, output_name: str) -> Optional[DataSet]:
        return self.named_outputs.get(output_name, None)

    @property
    def analyzed_logical_plan(self):
        assert self.successful
        for node in self.logical_plan.nodes.values():
            for name in node.perf_metrics:
                node.get_perf_stats(name)
        return self.logical_plan


def main():
    import argparse

    from smallpond.execution.task import Task
    from smallpond.io.filesystem import load

    parser = argparse.ArgumentParser(prog="task.py", description="Task Local Runner")
    parser.add_argument("pickle_path", help="Path of pickled task(s)")
    parser.add_argument("-t", "--task_id", default=None, help="Task id")
    parser.add_argument("-r", "--retry_count", default=0, help="Task retry count")
    parser.add_argument("-o", "--output_path", default=None, help="Task output path")
    parser.add_argument("-l", "--log_level", default="DEBUG", help="Logging message level")
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stdout,
        format=r"[{time:%Y-%m-%d %H:%M:%S.%f}] {level} {message}",
        level=args.log_level,
    )

    def exec_task(task: Task, output_path: Optional[str]):
        for retry_count in range(1000):
            task.retry_count = retry_count
            if output_path is None:
                task.output_root = task.ctx.staging_root
            else:
                task.output_root = os.path.join(output_path, "output")
                task.ctx.temp_root = os.path.join(output_path, "temp")
            if any(
                os.path.exists(path)
                for path in (
                    task.temp_abspath,
                    task.final_output_abspath,
                    task.runtime_output_abspath,
                )
            ):
                continue
            task.status = WorkStatus.INCOMPLETE
            task.start_time = time.time()
            task.finish_time = None
            logger.info(f"start to debug: {task}")
            task.exec()
            break

    obj = load(args.pickle_path)
    logger.info(f"loaded an object of {type(obj)} from pickle file {args.pickle_path}")

    if isinstance(obj, Task):
        task: Task = obj
        exec_task(task, args.output_path)
    elif isinstance(obj, Dict):
        assert args.task_id is not None, f"error: no task id specified"
        tasks: List[Task] = obj.values()
        for task in tasks:
            if task.id == TaskId(args.task_id) and task.retry_count == args.retry_count:
                exec_task(task, args.output_path)
                break
        else:
            logger.error(f"cannot find task with id {args.task_id}")
    else:
        logger.error(f"unsupported type of object: {type(obj)}")


if __name__ == "__main__":
    main()
