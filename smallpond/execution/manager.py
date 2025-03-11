import os
import shutil
import socket
import sys
from datetime import datetime
from typing import Dict, List, Literal, Optional

from loguru import logger

import smallpond
from smallpond.common import DEFAULT_MAX_FAIL_COUNT, DEFAULT_MAX_RETRY_COUNT, MB
from smallpond.execution.scheduler import Scheduler
from smallpond.execution.task import ExecutionPlan, JobId, RuntimeContext
from smallpond.io.filesystem import dump, load
from smallpond.logical.node import LogicalPlan
from smallpond.logical.planner import Planner
from smallpond.platform import get_platform


class SchedStateExporter(Scheduler.StateObserver):
    def __init__(self, sched_state_path: str) -> None:
        super().__init__()
        self.sched_state_path = sched_state_path

    def update(self, sched_state: Scheduler):
        if sched_state.large_runtime_state:
            logger.debug(f"pause exporting scheduler state")
        elif sched_state.num_local_running_works > 0:
            logger.debug(f"pause exporting scheduler state: {sched_state.num_local_running_works} local running works")
        else:
            dump(sched_state, self.sched_state_path, buffering=32 * MB, atomic_write=True)
            sched_state.log_overall_progress()
            logger.debug(f"exported scheduler state to {self.sched_state_path}")


class JobManager(object):

    jemalloc_filename = "libjemalloc.so.2"
    mimalloc_filename = "libmimalloc.so.2.1"

    env_template = r"""
    ARROW_DEFAULT_MEMORY_POOL={arrow_default_malloc}
    ARROW_IO_THREADS=2
    OMP_NUM_THREADS=2
    POLARS_MAX_THREADS=2
    NUMEXPR_MAX_THREADS=2
"""

    def __init__(
        self,
        data_root: Optional[str] = None,
        python_venv: Optional[str] = None,
        task_image: str = "default",
        platform: Optional[str] = None,
    ) -> None:
        self.platform = get_platform(platform)
        self.data_root = os.path.abspath(data_root or self.platform.default_data_root())
        self.python_venv = python_venv
        self.task_image = task_image

    @logger.catch(reraise=True, message="job manager terminated unexpectedly")
    def run(
        self,
        plan: LogicalPlan,
        job_id: Optional[str] = None,
        job_time: Optional[float] = None,
        job_name: str = "smallpond",
        job_priority: Optional[int] = None,
        num_executors: int = 1,
        num_executors_per_task: int = 5,
        resource_group: str = "localhost",
        env_variables: List[str] = None,
        sidecars: List[str] = None,
        user_tags: List[str] = None,
        random_seed: int = None,
        max_retry_count: int = DEFAULT_MAX_RETRY_COUNT,
        max_fail_count: int = DEFAULT_MAX_FAIL_COUNT,
        prioritize_retry=False,
        speculative_exec: Literal["disable", "enable", "aggressive"] = "enable",
        stop_executor_on_failure=False,
        fail_fast_on_failure=False,
        nonzero_exitcode_as_oom=False,
        fault_inject_prob: float = 0.0,
        enable_profiling=False,
        enable_diagnostic_metrics=False,
        enable_sched_state_dump=False,
        remove_empty_parquet=False,
        remove_output_root=False,
        skip_task_with_empty_input=False,
        manifest_only_final_results=True,
        memory_allocator: Literal["system", "jemalloc", "mimalloc"] = "mimalloc",
        memory_purge_delay: int = 10000,
        bind_numa_node=False,
        enforce_memory_limit=False,
        share_log_analytics: Optional[bool] = None,
        console_log_level: Literal["CRITICAL", "ERROR", "WARNING", "SUCCESS", "INFO", "DEBUG", "TRACE"] = "INFO",
        file_log_level: Literal["CRITICAL", "ERROR", "WARNING", "SUCCESS", "INFO", "DEBUG", "TRACE"] = "DEBUG",
        disable_log_rotation=False,
        sched_state_observers: Optional[List[Scheduler.StateObserver]] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> ExecutionPlan:
        logger.info(f"using platform: {self.platform}")

        job_id = JobId(hex=job_id or self.platform.default_job_id())
        job_time = datetime.fromtimestamp(job_time) if job_time is not None else self.platform.default_job_time()

        malloc_path = ""
        if memory_allocator == "system":
            malloc_path = ""
        elif memory_allocator == "jemalloc":
            malloc_path = shutil.which(self.jemalloc_filename)
        elif memory_allocator == "mimalloc":
            malloc_path = shutil.which(self.mimalloc_filename)
        else:
            logger.critical(f"failed to find memory allocator: {memory_allocator}")

        env_overrides = self.env_template.format(
            arrow_default_malloc=memory_allocator,
        ).splitlines()
        env_overrides = env_overrides + (env_variables or [])
        env_overrides = dict(tuple(kv.strip().split("=", maxsplit=1)) for kv in filter(None, env_overrides))

        share_log_analytics = share_log_analytics if share_log_analytics is not None else self.platform.default_share_log_analytics()
        shared_log_root = self.platform.shared_log_root() if share_log_analytics else None

        runtime_ctx = RuntimeContext(
            job_id,
            job_time,
            self.data_root,
            num_executors=num_executors,
            random_seed=random_seed,
            env_overrides=env_overrides,
            bind_numa_node=bind_numa_node,
            enforce_memory_limit=enforce_memory_limit,
            fault_inject_prob=fault_inject_prob,
            enable_profiling=enable_profiling,
            enable_diagnostic_metrics=enable_diagnostic_metrics,
            remove_empty_parquet=remove_empty_parquet,
            skip_task_with_empty_input=skip_task_with_empty_input,
            shared_log_root=shared_log_root,
            console_log_level=console_log_level,
            file_log_level=file_log_level,
            disable_log_rotation=disable_log_rotation,
            output_path=output_path,
            **kwargs,
        )
        runtime_ctx.initialize(socket.gethostname(), root_exist_ok=True)
        logger.info(f"command-line arguments: {' '.join([os.path.basename(sys.argv[0]), *sys.argv[1:]])}")

        dump(runtime_ctx, runtime_ctx.runtime_ctx_path, atomic_write=True)
        logger.info(f"saved runtime context at {runtime_ctx.runtime_ctx_path}")

        dump(plan, runtime_ctx.logcial_plan_path, atomic_write=True)
        logger.info(f"saved logcial plan at {runtime_ctx.logcial_plan_path}")

        plan.graph().render(runtime_ctx.logcial_plan_graph_path, format="png")
        logger.info(f"saved logcial plan graph at {runtime_ctx.logcial_plan_graph_path}.png")

        exec_plan = Planner(runtime_ctx).create_exec_plan(plan, manifest_only_final_results)
        dump(exec_plan, runtime_ctx.exec_plan_path, atomic_write=True)
        logger.info(f"saved execution plan at {runtime_ctx.exec_plan_path}")

        sidecar_list = sidecars or []

        fs_name, cluster = self.data_root.split("/")[1:3]
        tag_list = [
            "smallpond",
            "executor",
            smallpond.__version__,
            job_name,
            fs_name,
            cluster,
            f"malloc:{memory_allocator}",
        ] + (user_tags or [])

        if self.python_venv:
            tag_list.append(self.python_venv)
        if fail_fast_on_failure:
            max_fail_count = 0
        if max_fail_count == 0:
            tag_list.append("fail_fast")
        tag_list.append(f"max_fail:{max_fail_count}")

        tag_list.append(f"speculative_exec:{speculative_exec}")
        tag_list.append(f"max_retry:{max_retry_count}")

        if prioritize_retry:
            tag_list.append("prioritize_retry")
        if stop_executor_on_failure:
            tag_list.append("stop_executor")
        if bind_numa_node:
            tag_list.append("bind_numa_node")
        if enforce_memory_limit:
            tag_list.append("enforce_memory_limit")
            nonzero_exitcode_as_oom = True

        sched_state_observers = sched_state_observers or []

        if enable_sched_state_dump:
            sched_state_exporter = SchedStateExporter(runtime_ctx.sched_state_path)
            sched_state_observers.insert(0, sched_state_exporter)

        if os.path.exists(runtime_ctx.sched_state_path):
            logger.warning(f"loading scheduler state from: {runtime_ctx.sched_state_path}")
            scheduler: Scheduler = load(runtime_ctx.sched_state_path)
            scheduler.sched_epoch += 1
            scheduler.sched_state_observers = sched_state_observers
        else:
            scheduler = Scheduler(
                exec_plan,
                max_retry_count,
                max_fail_count,
                prioritize_retry,
                speculative_exec,
                stop_executor_on_failure,
                nonzero_exitcode_as_oom,
                remove_output_root,
                sched_state_observers,
            )
            # start executors
            self.platform.start_job(
                num_nodes=num_executors,
                entrypoint=os.path.join(os.path.dirname(__file__), "driver.py"),
                args=[
                    "--job_id",
                    str(job_id),
                    "--data_root",
                    self.data_root,
                    "--runtime_ctx_path",
                    runtime_ctx.runtime_ctx_path,
                    "executor",
                ],
                envs={
                    "LD_PRELOAD": malloc_path,
                    "MALLOC_CONF": f"percpu_arena:percpu,background_thread:true,metadata_thp:auto,dirty_decay_ms:{memory_purge_delay},muzzy_decay_ms:{memory_purge_delay},oversize_threshold:0,lg_tcache_max:16",
                    "MIMALLOC_PURGE_DELAY": memory_purge_delay,
                },
                extra_opts=dict(
                    job_id=job_id,
                    job_name=job_name,
                    job_priority=job_priority,
                    num_executors_per_task=num_executors_per_task,
                    resource_group=resource_group,
                    image=self.task_image,
                    python_venv=self.python_venv,
                    tags=tag_list,
                    sidecars=sidecar_list,
                ),
            )

        # run scheduler
        scheduler.run()
        return scheduler.exec_plan
