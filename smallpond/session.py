"""
This module defines the `Session` class, which is the entry point for smallpond interactive mode.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import ray
from graphviz import Digraph
import graphviz.backend.execute
from loguru import logger

import smallpond
from smallpond.execution.task import JobId, RuntimeContext
from smallpond.logical.node import Context
from smallpond.platform import Platform, get_platform


class SessionBase:
    def __init__(self, **kwargs):
        """
        Create a smallpond environment.
        """
        super().__init__()
        self._ctx = Context()
        self.config, self._platform = Config.from_args_and_env(**kwargs)

        # construct runtime context for Tasks
        runtime_ctx = RuntimeContext(
            job_id=JobId(hex=self.config.job_id),
            job_time=self.config.job_time,
            data_root=self.config.data_root,
            num_executors=self.config.num_executors,
            bind_numa_node=self.config.bind_numa_node,
            shared_log_root=self._platform.shared_log_root(),
        )
        self._runtime_ctx = runtime_ctx

        # if `spawn` is specified, spawn a job and exit
        if os.environ.get("SP_SPAWN") == "1":
            self._spawn_self()
            exit(0)

        self._runtime_ctx.initialize(exec_id=socket.gethostname())
        logger.info(f"using platform: {self._platform}")
        logger.info(f"command-line arguments: {' '.join(sys.argv)}")
        logger.info(f"session config: {self.config}")

        def setup_worker():
            runtime_ctx._init_logs(exec_id=socket.gethostname(), capture_stdout_stderr=True)

        if self.config.ray_address is None:
            # find the memory allocator
            if self.config.memory_allocator == "system":
                malloc_path = ""
            elif self.config.memory_allocator == "jemalloc":
                malloc_path = shutil.which("libjemalloc.so.2")
                assert malloc_path is not None, "jemalloc is not installed"
            elif self.config.memory_allocator == "mimalloc":
                malloc_path = shutil.which("libmimalloc.so.2.1")
                assert malloc_path is not None, "mimalloc is not installed"
            else:
                raise ValueError(f"unsupported memory allocator: {self.config.memory_allocator}")
            memory_purge_delay = 10000

            # start ray head node
            # for ray head node to access grafana
            os.environ["RAY_GRAFANA_HOST"] = "http://localhost:8122"
            self._ray_address = ray.init(
                # start a new local cluster
                address="local",
                # disable local CPU resource if not running on localhost
                num_cpus=(0 if self.config.num_executors > 0 else self._runtime_ctx.usable_cpu_count),
                # set the memory limit to the available memory size
                _memory=self._runtime_ctx.usable_memory_size,
                # setup logging for workers
                log_to_driver=False,
                runtime_env={
                    "worker_process_setup_hook": setup_worker,
                    "env_vars": {
                        "LD_PRELOAD": malloc_path,
                        "MALLOC_CONF": f"percpu_arena:percpu,background_thread:true,metadata_thp:auto,dirty_decay_ms:{memory_purge_delay},muzzy_decay_ms:{memory_purge_delay},oversize_threshold:0,lg_tcache_max:16",
                        "MIMALLOC_PURGE_DELAY": f"{memory_purge_delay}",
                        "ARROW_DEFAULT_MEMORY_POOL": self.config.memory_allocator,
                        "ARROW_IO_THREADS": "2",
                        "OMP_NUM_THREADS": "2",
                        "POLARS_MAX_THREADS": "2",
                        "NUMEXPR_MAX_THREADS": "2",
                        "RAY_PROFILING": "1",
                    },
                },
                dashboard_host="0.0.0.0",
                dashboard_port=8008,
                # for prometheus to scrape metrics
                _metrics_export_port=8080,
            ).address_info["gcs_address"]
            logger.info(f"started ray cluster at {self._ray_address}")

            self._prometheus_process = self._start_prometheus()
            self._grafana_process = self._start_grafana()
        else:
            self._ray_address = self.config.ray_address
            self._prometheus_process = None
            self._grafana_process = None
            logger.info(f"connected to ray cluster at {self._ray_address}")

        # start workers
        if self.config.num_executors > 0:
            # override configs
            kwargs["job_id"] = self.config.job_id

            self._job_names = self._platform.start_job(
                self.config.num_executors,
                entrypoint=os.path.join(os.path.dirname(__file__), "worker.py"),
                args=[
                    f"--ray_address={self._ray_address}",
                    f"--log_dir={self._runtime_ctx.log_root}",
                    *(["--bind_numa_node"] if self.config.bind_numa_node else []),
                ],
                extra_opts=kwargs,
            )
        else:
            self._job_names = []

        # spawn a thread to periodically dump metrics
        self._stop_event = threading.Event()
        self._dump_thread = threading.Thread(name="dump_thread", target=self._dump_periodically, daemon=True)
        self._dump_thread.start()

    def shutdown(self):
        """
        Shutdown the session.
        """
        logger.info("shutting down session")
        self._stop_event.set()

        # stop all jobs
        for job_name in self._job_names:
            self._platform.stop_job(job_name)
        self._job_names = []

        self._dump_thread.join()
        if self.config.ray_address is None:
            ray.shutdown()
        if self._prometheus_process is not None:
            self._prometheus_process.terminate()
            self._prometheus_process.wait()
            self._prometheus_process = None
            logger.info("stopped prometheus")
        if self._grafana_process is not None:
            self._grafana_process.terminate()
            self._grafana_process.wait()
            self._grafana_process = None
            logger.info("stopped grafana")

    def _spawn_self(self):
        """
        Spawn a new job to run the current script.
        """
        self._platform.start_job(
            num_nodes=1,
            entrypoint=sys.argv[0],
            args=sys.argv[1:],
            extra_opts=dict(
                tags=["smallpond", "scheduler", smallpond.__version__],
            ),
            envs={k: v for k, v in os.environ.items() if k.startswith("SP_") and k != "SP_SPAWN"},
        )

    def _start_prometheus(self) -> Optional[subprocess.Popen]:
        """
        Start prometheus server if it exists.
        """
        prometheus_path = shutil.which("prometheus")
        if prometheus_path is None:
            logger.warning("prometheus is not found")
            return None
        os.makedirs(f"{self._runtime_ctx.log_root}/prometheus", exist_ok=True)
        proc = subprocess.Popen(
            [
                prometheus_path,
                "--config.file=/tmp/ray/session_latest/metrics/prometheus/prometheus.yml",
                f"--storage.tsdb.path={self._runtime_ctx.log_root}/prometheus/data",
            ],
            stderr=open(f"{self._runtime_ctx.log_root}/prometheus/prometheus.log", "w"),
        )
        logger.info("started prometheus")
        return proc

    def _start_grafana(self) -> Optional[subprocess.Popen]:
        """
        Start grafana server if it exists.
        """
        homepath = self._platform.grafana_homepath()
        if homepath is None:
            logger.warning("grafana is not found")
            return None
        os.makedirs(f"{self._runtime_ctx.log_root}/grafana", exist_ok=True)
        proc = subprocess.Popen(
            [
                shutil.which("grafana"),
                "server",
                "--config",
                "/tmp/ray/session_latest/metrics/grafana/grafana.ini",
                "-homepath",
                homepath,
                "web",
            ],
            stdout=open(f"{self._runtime_ctx.log_root}/grafana/grafana.log", "w"),
            env={
                "GF_SERVER_HTTP_PORT": "8122",  # redirect to an available port
                "GF_SERVER_ROOT_URL": os.environ.get("RAY_GRAFANA_IFRAME_HOST") or "http://localhost:8122",
                "GF_PATHS_DATA": f"{self._runtime_ctx.log_root}/grafana/data",
            },
        )
        logger.info(f"started grafana at http://localhost:8122")
        return proc

    @property
    def runtime_ctx(self) -> RuntimeContext:
        return self._runtime_ctx

    def graph(self) -> Digraph:
        """
        Get the logical plan graph.
        """
        # implemented in Session class
        raise NotImplementedError("graph")

    def dump_graph(self, path: Optional[str] = None):
        """
        Dump the logical plan graph to a file.
        """
        path = path or os.path.join(self.runtime_ctx.log_root, "graph")
        try:
            self.graph().render(path, format="png")
            logger.debug(f"dumped graph to {path}")
        except graphviz.backend.execute.ExecutableNotFound as e:
            logger.warning(f"graphviz is not installed, skipping graph dump")

    def dump_timeline(self, path: Optional[str] = None):
        """
        Dump the task timeline to a file.
        """
        path = path or os.path.join(self.runtime_ctx.log_root, "timeline")
        # the default timeline is grouped by worker
        exec_path = f"{path}_exec"
        ray.timeline(exec_path)
        logger.debug(f"dumped timeline to {exec_path}")

        # generate another timeline grouped by node
        with open(exec_path) as f:
            records = json.load(f)
        new_records = []
        for record in records:
            # swap record name and pid-tid
            name = record["name"]
            try:
                node_id = name.split(",")[-1]
                task_id = name.split("-")[1].split(".")[0]
                task_name = name.split("-")[0]
                record["pid"] = f"{node_id}-{task_name}"
                record["tid"] = f"task {task_id}"
                new_records.append(record)
            except Exception:
                # filter out other records
                pass
        node_path = f"{path}_plan"
        with open(node_path, "w") as f:
            json.dump(new_records, f)
        logger.debug(f"dumped timeline to {node_path}")

    def _summarize_task(self) -> Tuple[int, int]:
        # implemented in Session class
        raise NotImplementedError("summarize_task")

    def _dump_periodically(self):
        """
        Dump the graph and timeline every minute.
        Set `self._stop_event` to have a final dump and stop this thread.
        """
        while not self._stop_event.is_set():
            self._stop_event.wait(60)
            self.dump_graph()
            self.dump_timeline()
            num_total_tasks, num_finished_tasks = self._summarize_task()
            percent = num_finished_tasks / num_total_tasks * 100 if num_total_tasks > 0 else 0
            logger.info(f"progress: {num_finished_tasks}/{num_total_tasks} tasks ({percent:.1f}%)")


@dataclass
class Config:
    """
    Configuration for a session.
    """

    job_id: str  # JOBID
    job_time: datetime  # JOB_TIME
    data_root: str  # DATA_ROOT
    num_executors: int  # NUM_NODES_TOTAL
    ray_address: Optional[str]  # RAY_ADDRESS
    bind_numa_node: bool  # BIND_NUMA_NODE
    memory_allocator: str  # MEMORY_ALLOCATOR
    remove_output_root: bool

    @staticmethod
    def from_args_and_env(
        platform: Optional[str] = None,
        job_id: Optional[str] = None,
        job_time: Optional[datetime] = None,
        data_root: Optional[str] = None,
        num_executors: Optional[int] = None,
        ray_address: Optional[str] = None,
        bind_numa_node: Optional[bool] = None,
        memory_allocator: Optional[str] = None,
        _remove_output_root: bool = True,
        **kwargs,
    ) -> Config:
        """
        Load config from arguments and environment variables.
        If not specified, use the default value.
        """

        def get_env(key: str, type: type = str):
            """
            Get an environment variable and convert it to the given type.
            If the variable is not set, return None.
            """
            value = os.environ.get(f"SP_{key}")
            return type(value) if value is not None else None

        platform = get_platform(get_env("PLATFORM") or platform)
        job_id = get_env("JOBID") or job_id or platform.default_job_id()
        job_time = get_env("JOB_TIME", datetime.fromisoformat) or job_time or platform.default_job_time()
        data_root = get_env("DATA_ROOT") or data_root or platform.default_data_root()
        num_executors = get_env("NUM_EXECUTORS", int) or num_executors or 0
        ray_address = get_env("RAY_ADDRESS") or ray_address
        bind_numa_node = get_env("BIND_NUMA_NODE") == "1" or bind_numa_node
        memory_allocator = get_env("MEMORY_ALLOCATOR") or memory_allocator or platform.default_memory_allocator()

        config = Config(
            job_id=job_id,
            job_time=job_time,
            data_root=data_root,
            num_executors=num_executors,
            ray_address=ray_address,
            bind_numa_node=bind_numa_node,
            memory_allocator=memory_allocator,
            remove_output_root=_remove_output_root,
        )
        return config, platform
