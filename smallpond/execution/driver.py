import argparse
import os
import socket
import sys
from multiprocessing import Process
from typing import List, Optional

from loguru import logger

import smallpond
from smallpond.common import DEFAULT_MAX_FAIL_COUNT, DEFAULT_MAX_RETRY_COUNT
from smallpond.dataframe import DataFrame
from smallpond.execution.task import ExecutionPlan
from smallpond.io.filesystem import load
from smallpond.logical.node import LogicalPlan


class Driver(object):
    """
    A helper class that includes boilerplate code to execute a logical plan.
    """

    def __init__(self) -> None:
        self.driver_args_parser = self._create_driver_args_parser()
        self.user_args_parser = argparse.ArgumentParser(add_help=False)
        self.driver_args = None
        self.user_args = None
        self.all_args = None

    def _create_driver_args_parser(self):
        parser = argparse.ArgumentParser(prog="driver.py", description="Smallpond Driver", add_help=False)
        parser.add_argument("mode", choices=["executor", "scheduler", "ray"], default="executor")
        parser.add_argument("--exec_id", default=socket.gethostname(), help="Unique executor id")
        parser.add_argument("--job_id", type=str, help="Unique job id")
        parser.add_argument("--job_time", type=float, help="Job create time (seconds since epoch)")
        parser.add_argument("--job_name", default="smallpond", help="Display name of the job")
        parser.add_argument(
            "--job_priority",
            type=int,
            help="Job priority",
        )
        parser.add_argument("--resource_group", type=str, help="Resource group")
        parser.add_argument("--env_variables", nargs="*", default=[], help="Env variables for the job")
        parser.add_argument("--sidecars", nargs="*", default=[], help="Sidecars for the job")
        parser.add_argument("--tags", nargs="*", default=[], help="Tags for submitted platform task")
        parser.add_argument("--task_image", default="default", help="Container image of platform task")
        parser.add_argument("--python_venv", type=str, help="Python virtual env for the job")
        parser.add_argument(
            "--data_root",
            type=str,
            help="The root folder for all files generated at runtime",
        )
        parser.add_argument(
            "--runtime_ctx_path",
            default=None,
            help="The path of pickled runtime context passed from scheduler to executor",
        )
        parser.add_argument(
            "--num_executors",
            default=0,
            type=int,
            help="The number of nodes/executors (run all tasks on scheduler if set to zero)",
        )
        parser.add_argument(
            "--num_executors_per_task",
            default=5,
            type=int,
            help="The number of nodes/executors in each platform task.",
        )
        parser.add_argument(
            "--random_seed",
            type=int,
            default=None,
            help="Random seed for the job, default: int.from_bytes((os.urandom(128))",
        )
        parser.add_argument(
            "--max_retry",
            "--max_retry_count",
            dest="max_retry_count",
            default=DEFAULT_MAX_RETRY_COUNT,
            type=int,
            help="The max number of times a task is retried by speculative execution",
        )
        parser.add_argument(
            "--max_fail",
            "--max_fail_count",
            dest="max_fail_count",
            default=DEFAULT_MAX_FAIL_COUNT,
            type=int,
            help="The number of times a task is allowed to fail or crash before giving up",
        )
        parser.add_argument(
            "--prioritize_retry",
            action="store_true",
            help="Prioritize retry tasks in scheduling",
        )
        parser.add_argument(
            "--speculative_exec",
            dest="speculative_exec",
            choices=["disable", "enable", "aggressive"],
            help="Level of speculative execution",
        )
        parser.add_argument(
            "--enable_speculative_exec",
            dest="speculative_exec",
            action="store_const",
            const="enable",
            help="Enable speculative execution",
        )
        parser.add_argument(
            "--disable_speculative_exec",
            dest="speculative_exec",
            action="store_const",
            const="disable",
            help="Disable speculative execution",
        )
        parser.set_defaults(speculative_exec="enable")
        parser.add_argument(
            "--stop_executor_on_failure",
            action="store_true",
            help="Stop an executor if any task fails on it",
        )
        parser.add_argument(
            "--fail_fast",
            "--fail_fast_on_failure",
            dest="fail_fast_on_failure",
            action="store_true",
            help="Stop all executors if any task fails",
        )
        parser.add_argument(
            "--nonzero_exitcode_as_oom",
            action="store_true",
            help="Treat task crash as oom error",
        )
        parser.add_argument(
            "--fault_inject",
            "--fault_inject_prob",
            dest="fault_inject_prob",
            type=float,
            default=0.0,
            help="Inject random errors at runtime (for test)",
        )
        parser.add_argument(
            "--enable_profiling",
            action="store_true",
            help="Enable Python profiling for each task",
        )
        parser.add_argument(
            "--enable_diagnostic_metrics",
            action="store_true",
            help="Enable diagnostic metrcis which may have performance impact",
        )
        parser.add_argument(
            "--disable_sched_state_dump",
            dest="enable_sched_state_dump",
            action="store_false",
            help="Disable periodic dump of scheduler state",
        )
        parser.add_argument(
            "--enable_sched_state_dump",
            dest="enable_sched_state_dump",
            action="store_true",
            help="Enable periodic dump of scheduler state so that scheduler can resume execution after restart",
        )
        parser.set_defaults(enable_sched_state_dump=False)
        parser.add_argument(
            "--remove_empty_parquet",
            action="store_true",
            help="Remove empty parquet files from hash partition output",
        )
        parser.add_argument(
            "--remove_output_root",
            action="store_true",
            help="Remove all output files after job completed (for test)",
        )
        parser.add_argument(
            "--skip_task_with_empty_input",
            action="store_true",
            help="Skip running a task if any of its input datasets is empty",
        )
        parser.add_argument(
            "--self_contained_final_results",
            action="store_true",
            help="Build self-contained final results, i.e., create hard/symbolic links in output folder of final results",
        )
        parser.add_argument(
            "--malloc",
            "--memory_allocator",
            dest="memory_allocator",
            default="mimalloc",
            choices=["system", "jemalloc", "mimalloc"],
            help="Override memory allocator used by worker processes",
        )
        parser.add_argument(
            "--memory_purge_delay",
            default=10000,
            help="The delay in milliseconds after which jemalloc/mimalloc will purge memory pages that are not in use.",
        )
        parser.add_argument(
            "--bind_numa_node",
            action="store_true",
            help="Bind executor processes to numa nodes.",
        )
        parser.add_argument(
            "--enforce_memory_limit",
            action="store_true",
            help="Set soft/hard memory limit for each task process",
        )
        parser.add_argument(
            "--enable_log_analytics",
            dest="share_log_analytics",
            action="store_true",
            help="Share log analytics with smallpond team",
        )
        parser.add_argument(
            "--disable_log_analytics",
            dest="share_log_analytics",
            action="store_false",
            help="Do not share log analytics with smallpond team",
        )
        log_level_choices = [
            "CRITICAL",
            "ERROR",
            "WARNING",
            "SUCCESS",
            "INFO",
            "DEBUG",
            "TRACE",
        ]
        parser.add_argument(
            "--console_log_level",
            default="INFO",
            choices=log_level_choices,
        )
        parser.add_argument(
            "--file_log_level",
            default="DEBUG",
            choices=log_level_choices,
        )
        parser.add_argument("--disable_log_rotation", action="store_true", help="Disable log rotation")
        parser.add_argument(
            "--output_path",
            help="Set the output directory of final results and all nodes that have output_name but no output_path specified",
        )
        parser.add_argument(
            "--platform",
            type=str,
            help="The platform to use for the job. available: mpi",
        )
        return parser

    def add_argument(self, *args, **kwargs):
        """
        Add a command-line argument. This is a wrapper of `argparse.ArgumentParser.add_argument(...)`.
        """
        self.user_args_parser.add_argument(*args, **kwargs)

    def parse_arguments(self, args=None):
        if self.user_args is None or self.driver_args is None:
            args_parser = argparse.ArgumentParser(parents=[self.driver_args_parser, self.user_args_parser])
            self.all_args = args_parser.parse_args(args)
            self.user_args, other_args = self.user_args_parser.parse_known_args(args)
            self.driver_args = self.driver_args_parser.parse_args(other_args)
        return self.user_args, self.driver_args

    def get_user_arguments(self, to_dict=True):
        """
        Get user-defined arguments.
        """
        user_args, _ = self.parse_arguments()
        return vars(user_args) if to_dict else user_args

    get_arguments = get_user_arguments

    def get_driver_arguments(self, to_dict=True):
        _, driver_args = self.parse_arguments()
        return vars(driver_args) if to_dict else driver_args

    @property
    def mode(self) -> str:
        return self.get_driver_arguments(to_dict=False).mode

    @property
    def job_id(self) -> str:
        return self.get_driver_arguments(to_dict=False).job_id

    @property
    def data_root(self) -> str:
        return self.get_driver_arguments(to_dict=False).data_root

    @property
    def num_executors(self) -> str:
        return self.get_driver_arguments(to_dict=False).num_executors

    def run(
        self,
        plan: LogicalPlan,
        stop_process_on_done=True,
        tags: List[str] = None,
    ) -> Optional[ExecutionPlan]:
        """
        The entry point for executor and scheduler of `plan`.
        """
        from smallpond.execution.executor import Executor
        from smallpond.execution.manager import JobManager
        from smallpond.execution.task import RuntimeContext

        _, args = self.parse_arguments()
        retval = None

        def run_executor(runtime_ctx: RuntimeContext, exec_id: str, numa_node_id=None):
            if numa_node_id is not None:
                import numa

                exec_id += f".numa{numa_node_id}"
                numa.schedule.bind(numa_node_id)
                runtime_ctx.numa_node_id = numa_node_id
            runtime_ctx.initialize(exec_id)
            executor = Executor.create(runtime_ctx, exec_id)
            return executor.run()

        if args.mode == "ray":
            assert plan is not None
            sp = smallpond.init(_remove_output_root=args.remove_output_root)
            DataFrame(sp, plan.root_node).compute()
            retval = True
        elif args.mode == "executor":
            assert os.path.isfile(args.runtime_ctx_path), f"cannot find runtime context: {args.runtime_ctx_path}"
            runtime_ctx: RuntimeContext = load(args.runtime_ctx_path)

            if runtime_ctx.bind_numa_node:
                exec_procs = [
                    Process(
                        target=run_executor,
                        args=(runtime_ctx, args.exec_id, numa_node_id),
                    )
                    for numa_node_id in range(runtime_ctx.numa_node_count)
                ]
                for proc in exec_procs:
                    proc.start()
                for proc in exec_procs:
                    proc.join()
                retval = all(proc.exitcode == 0 for proc in exec_procs)
            else:
                retval = run_executor(runtime_ctx, args.exec_id)
        elif args.mode == "scheduler":
            assert plan is not None
            jobmgr = JobManager(args.data_root, args.python_venv, args.task_image, args.platform)
            exec_plan = jobmgr.run(
                plan,
                job_id=args.job_id,
                job_time=args.job_time,
                job_name=args.job_name,
                job_priority=args.job_priority,
                num_executors=args.num_executors,
                num_executors_per_task=args.num_executors_per_task,
                resource_group=args.resource_group,
                env_variables=args.env_variables,
                sidecars=args.sidecars,
                user_tags=args.tags + (tags or []),
                random_seed=args.random_seed,
                max_retry_count=args.max_retry_count,
                max_fail_count=args.max_fail_count,
                prioritize_retry=args.prioritize_retry,
                speculative_exec=args.speculative_exec,
                stop_executor_on_failure=args.stop_executor_on_failure,
                fail_fast_on_failure=args.fail_fast_on_failure,
                nonzero_exitcode_as_oom=args.nonzero_exitcode_as_oom,
                fault_inject_prob=args.fault_inject_prob,
                enable_profiling=args.enable_profiling,
                enable_diagnostic_metrics=args.enable_diagnostic_metrics,
                enable_sched_state_dump=args.enable_sched_state_dump,
                remove_empty_parquet=args.remove_empty_parquet,
                remove_output_root=args.remove_output_root,
                skip_task_with_empty_input=args.skip_task_with_empty_input,
                manifest_only_final_results=not args.self_contained_final_results,
                memory_allocator=args.memory_allocator,
                memory_purge_delay=args.memory_purge_delay,
                bind_numa_node=args.bind_numa_node,
                enforce_memory_limit=args.enforce_memory_limit,
                share_log_analytics=args.share_log_analytics,
                console_log_level=args.console_log_level,
                file_log_level=args.file_log_level,
                disable_log_rotation=args.disable_log_rotation,
                output_path=args.output_path,
            )
            retval = exec_plan if exec_plan.successful else None

        if stop_process_on_done:
            exit_code = os.EX_OK if retval else os.EX_SOFTWARE
            logger.info(f"exit code: {exit_code}")
            sys.exit(exit_code)
        else:
            logger.info(f"return value: {repr(retval)}")
            return retval


def main():
    # run in executor mode
    driver = Driver()
    driver.run(plan=None)


if __name__ == "__main__":
    main()
