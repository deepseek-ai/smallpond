import cProfile
import inspect
import io
import logging
import pstats
import queue
import subprocess
import sys
import threading
from typing import Any, Dict, Iterable

from loguru import logger


def overall_stats(
    ctx,
    inp,
    sql_per_part,
    sql_on_merged,
    output_name,
    output_dir=None,
    cpu_limit=2,
    memory_limit=30 << 30,
):
    from smallpond.logical.node import DataSetPartitionNode, DataSinkNode, SqlEngineNode

    n = SqlEngineNode(ctx, inp, sql_per_part, cpu_limit=cpu_limit, memory_limit=memory_limit)
    p = DataSetPartitionNode(ctx, (n,), npartitions=1)
    n2 = SqlEngineNode(
        ctx,
        (p,),
        sql_on_merged,
        output_name=output_name,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
    )
    if output_dir is not None:
        return DataSinkNode(ctx, (n2,), output_dir)
    else:
        return n2


def execute_command(cmd: str, env: Dict[str, str] = None, shell=False):
    with subprocess.Popen(
        cmd.split(),
        env=env,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding="utf8",
    ) as proc:
        for line in proc.stdout:
            yield line.rstrip()
        return_code = proc.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)


def cprofile_to_string(perf_profile: cProfile.Profile, order_by=pstats.SortKey.TIME, top_k=20):
    perf_profile.disable()
    pstats_output = io.StringIO()
    profile_stats = pstats.Stats(perf_profile, stream=pstats_output)
    profile_stats.strip_dirs().sort_stats(order_by).print_stats(top_k)
    return pstats_output.getvalue()


class Wrapper(object):
    def __init__(self, base_obj: Any):
        self._base_obj = base_obj

    def __str__(self) -> str:
        return str(self._base_obj)

    def __repr__(self) -> str:
        return repr(self._base_obj)

    def __getattr__(self, name):
        return getattr(self._base_obj, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            return setattr(self._base_obj, name, value)


class ConcurrentIterError(Exception):
    pass


class ConcurrentIter(object):
    """
    Use a background thread to iterate over an iterable.

    Examples
    --------
    The following code snippet is a common pattern to read record batches from parquet files asynchronously in arrow stream task.
    ```
    from smallpond.utility import ConcurrentIter

    with ConcurrentIter(input_readers[0], max_buffer_size=1) as async_reader:
      for batch_idx, batch in enumerate(async_reader):
        # your code here
        yield StreamOutput(output_table, batch_indices=[batch_idx])
    ```
    """

    def __init__(self, iterable: Iterable, max_buffer_size=1) -> None:
        assert isinstance(iterable, Iterable), f"expect an iterable but found: {repr(iterable)}"
        self.__iterable = iterable
        self.__queue = queue.Queue(max_buffer_size)
        self.__last = object()
        self.__stop = threading.Event()
        self.__thread = threading.Thread(target=self._producer)

    def __enter__(self):
        self.__thread.start()
        return iter(self)

    def __exit__(self, exc_type, exc_value, traceback):
        self.join()

    def __iter__(self):
        try:
            yield from self._consumer()
        finally:
            self.join()

    def join(self):
        self.__stop.set()
        self.clear()
        self.__thread.join(timeout=1)
        if self.__thread.is_alive():
            print(f"waiting {self.__thread.name} of {self}", file=sys.stderr)
            self.__thread.join()
            print(f"joined {self.__thread.name} of {self}", file=sys.stderr)

    def clear(self):
        try:
            while self.__queue.get_nowait() is not None:
                pass
        except queue.Empty:
            pass

    def _producer(self):
        try:
            for item in self.__iterable:
                self.__queue.put(item)
                if self.__stop.is_set():
                    self.clear()
                    break
        except Exception as ex:
            print(f"Error in {self}: {ex}", file=sys.stderr)
            self.clear()
            self.__queue.put(ConcurrentIterError(ex))
        else:
            self.__queue.put(self.__last)

    def _consumer(self):
        while True:
            item = self.__queue.get()
            if item is self.__last:
                break
            if isinstance(item, ConcurrentIterError):
                (ex,) = item.args
                raise item from ex
            yield item


class InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages toward loguru sinks.
    See https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
