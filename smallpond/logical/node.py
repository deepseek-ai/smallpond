import copy
import math
import os
import os.path
import re
import traceback
import warnings
from collections import defaultdict
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
import pyarrow as arrow
from graphviz import Digraph

from smallpond.common import (
    DATA_PARTITION_COLUMN_NAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_ROW_GROUP_SIZE,
    GB,
    GENERATED_COLUMNS,
)
from smallpond.execution.task import (
    ArrowBatchTask,
    ArrowComputeTask,
    ArrowStreamTask,
    DataSinkTask,
    DataSourceTask,
    EvenlyDistributedPartitionProducerTask,
    HashPartitionTask,
    LoadPartitionedDataSetProducerTask,
    MergeDataSetsTask,
    PandasBatchTask,
    PandasComputeTask,
    PartitionConsumerTask,
    PartitionInfo,
    PartitionProducerTask,
    PerfStats,
    ProjectionTask,
    PythonScriptTask,
    RepeatPartitionProducerTask,
    RuntimeContext,
    SplitDataSetTask,
    SqlEngineTask,
    Task,
    UserDefinedPartitionProducerTask,
)
from smallpond.logical.dataset import DataSet, ParquetDataSet
from smallpond.logical.udf import (
    DuckDbExtensionContext,
    ExternalModuleContext,
    PythonUDFContext,
    UDFContext,
    UDFType,
    UserDefinedFunction,
)


class NodeId(int):
    """
    A unique identifier for each node.
    """

    def __str__(self) -> str:
        return f"{self:06d}"


class Context(object):
    """
    Global context for each logical plan.
    Right now it's mainly used to keep a list of Python UDFs.
    """

    def __init__(self) -> None:
        self.next_node_id = 0
        self.udfs: Dict[str, UDFContext] = {}

    def _new_node_id(self) -> NodeId:
        """
        Generate a new node id.
        """
        self.next_node_id += 1
        return NodeId(self.next_node_id)

    def create_function(
        self,
        name: str,
        func: Callable,
        params: Optional[List[UDFType]],
        return_type: Optional[UDFType],
        use_arrow_type=False,
    ) -> str:
        """
        Define a Python UDF to be referenced in the logical plan.
        Currently only scalar functions (return one element per row) are supported.
        See https://duckdb.org/docs/archive/0.9.2/api/python/function.

        Parameters
        ----------
        name
            A unique function name, which can be referenced in SQL query.
        func
            The Python function you wish to register as a UDF.
        params
            A list of column types for function parameters, including basic types:
            `UDFType.INTEGER`, `UDFType.FLOAT`, `UDFType.VARCHAR`, `UDFType.BLOB` etc,
            and container types:
            ```
                UDFListType(UDFType.INTEGER),
                UDFMapType(UDFType.VARCHAR, UDFType.INTEGER),
                UDFListType(UDFStructType({'int': 'INTEGER', 'str': 'VARCHAR'})).
            ```
            These types are simple wrappers of duckdb types defined in https://duckdb.org/docs/api/python/types.html.
            Set params to `UDFAnyParameters()` allows the udf to accept parameters of any type.
        use_arrow_type, optional
            Specify true to use PyArrow Tables, by default use built-in Python types.
        return_type
            The return type of the function, see the above note for `params`.

        Returns
        -------
            The unique function name.
        """
        self.udfs[name] = PythonUDFContext(name, func, params, return_type, use_arrow_type)
        return name

    def create_external_module(self, module_path: str, name: str = None) -> str:
        """
        Load an external DuckDB module.
        """
        name = name or os.path.basename(module_path)
        self.udfs[name] = ExternalModuleContext(name, module_path)
        return name

    def create_duckdb_extension(self, extension_path: str, name: str = None) -> str:
        """
        Load a DuckDB extension.
        """
        name = name or os.path.basename(extension_path)
        self.udfs[name] = DuckDbExtensionContext(name, extension_path)
        return name


class Node(object):
    """
    The base class for all nodes.
    """

    enable_resource_boost = False

    def __init__(
        self,
        ctx: Context,
        input_deps: "Tuple[Node, ...]",
        output_name: Optional[str] = None,
        output_path: Optional[str] = None,
        cpu_limit: int = 1,
        gpu_limit: float = 0,
        memory_limit: Optional[int] = None,
    ) -> None:
        """
        The base class for all nodes in logical plan.

        Parameters
        ----------
        ctx
            The context of logical plan.
        input_deps
            Define the inputs of this node.
        output_name, optional
            The prefix of output directories and filenames for tasks generated from this node.
            The default `output_name` is the class name of the task created for this node, e.g.
            `HashPartitionTask, SqlEngineTask, PythonScriptTask`, etc.

            The `output_name` string should only include alphanumeric characters and underscore.
            In other words, it matches regular expression `[a-zA-Z0-9_]+`.

            If `output_name` is set and `output_path` is None, the path format of output files is:
            `{job_root_path}/output/{output_name}/{task_runtime_id}/{output_name}-{task_runtime_id}-{seqnum}.parquet`
            where `{task_runtime_id}` is defined as `{job_id}.{task_id}.{sched_epoch}.{task_retry_count}`.
        output_path, optional
            The absolute path of a customized output folder for tasks generated from this node.
            Any shared folder that can be accessed by executor and scheduler is allowed
            although IO performance varies across filesystems.

            If both `output_name` and `output_path` are specified, the path format of output files is:
            `{output_path}/{output_name}/{task_runtime_id}/{output_name}-{task_runtime_id}-{seqnum}.parquet`
            where `{task_runtime_id}` is defined as `{job_id}.{task_id}.{sched_epoch}.{task_retry_count}`.
        cpu_limit, optional
            The max number of CPUs would be used by tasks generated from this node.
            This is a resource requirement specified by the user and used to guide
            task scheduling. smallpond does NOT enforce this limit.
        gpu_limit, optional
            The max number of GPUs would be used by tasks generated from this node.
            This is a resource requirement specified by the user and used to guide
            task scheduling. smallpond does NOT enforce this limit.
        memory_limit, optional
            The max memory would be used by tasks generated from this node.
            The memory limit is automatically calculated based memory-to-cpu ratio of executor machine if not specified.
            This is a resource requirement specified by the user and used to guide
            task scheduling. smallpond does NOT enforce this limit.
        """
        assert isinstance(input_deps, Iterable), f"input_deps is not iterable: {input_deps}"
        assert all(isinstance(node, Node) for node in input_deps), f"some of input_deps are not instances of Node: {input_deps}"
        assert output_name is None or re.match("[a-zA-Z0-9_]+", output_name), f"output_name has invalid format: {output_name}"
        assert output_path is None or os.path.isabs(output_path), f"output_path is not an absolute path: {output_path}"
        self.ctx = ctx
        self.id = self.ctx._new_node_id()
        self.input_deps = input_deps
        self.output_name = output_name
        self.output_path = output_path
        self.cpu_limit = max(cpu_limit, gpu_limit * 8)
        self.gpu_limit = gpu_limit
        self.memory_limit = memory_limit
        self.generated_tasks: List[str] = []
        self.perf_stats: Dict[str, PerfStats] = {}
        self.perf_metrics: Dict[str, List[float]] = defaultdict(list)
        # record the location where the node is constructed in user code
        frame = next(
            frame for frame in reversed(traceback.extract_stack()) if frame.filename != __file__ and not frame.filename.endswith("/dataframe.py")
        )
        self.location = f"{frame.filename}:{frame.lineno}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}-{self.id}"

    def __str__(self) -> str:
        return (
            f"{repr(self)}: input_deps={self.input_deps}, output_name={self.output_name}, "
            f"tasks[{len(self.generated_tasks)}]={self.generated_tasks[:1]}...{self.generated_tasks[-1:]}, "
            f"resource_limit={self.cpu_limit}CPUs/{self.gpu_limit}GPUs/{(self.memory_limit or 0)//GB}GB"
        )

    @staticmethod
    def task_factory(task_builder):
        def wrapper(self: Node, *args, **kwargs):
            task: Task = task_builder(self, *args, **kwargs)
            task.node_id = self.id
            task.location = self.location
            self.generated_tasks.append(task.key)
            return task

        return wrapper

    def slim_copy(self):
        node = copy.copy(self)
        del node.input_deps, node.generated_tasks
        return node

    def create_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> Task:
        raise NotImplementedError

    def add_perf_metrics(self, name, value: Union[List[float], float]):
        self.perf_metrics[name].append(value)
        self.perf_stats.pop(name, None)

    def get_perf_stats(self, name):
        if name not in self.perf_stats:
            if name not in self.perf_metrics:
                return None
            values = self.perf_metrics[name]
            min, max, avg = np.min(values), np.max(values), np.average(values)
            p50, p75, p95, p99 = np.percentile(values, (50, 75, 95, 99))
            self.perf_stats[name] = PerfStats(len(values), sum(values), min, max, avg, p50, p75, p95, p99)
        return self.perf_stats[name]

    @property
    def num_partitions(self) -> int:
        raise NotImplementedError("num_partitions")


class DataSourceNode(Node):
    """
    All inputs of a logical plan are represented as `DataSourceNode`. It does not depend on any other node.
    """

    def __init__(self, ctx: Context, dataset: DataSet) -> None:
        """
        Construct a DataSourceNode. See :func:`Node.__init__` to find comments on other parameters.

        Parameters
        ----------
        dataset
            A DataSet instance serving as a input of the plan. Set to `None` to create a dummy data source.
        """
        assert dataset is None or isinstance(dataset, DataSet)
        super().__init__(ctx, [])
        self.dataset = dataset if dataset is not None else ParquetDataSet([])

    def __str__(self) -> str:
        return super().__str__() + f", dataset=<{self.dataset}>"

    @Node.task_factory
    def create_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> DataSourceTask:
        return DataSourceTask(runtime_ctx, self.dataset, partition_infos)

    @property
    def num_partitions(self) -> int:
        return 1


DataSinkType = Literal["link", "copy", "link_or_copy", "manifest"]


class DataSinkNode(Node):
    """
    Collect the output files of `input_deps` to `output_path`.
    Depending on the options, it may create hard links, symbolic links, manifest files, or copy files.
    """

    def __init__(
        self,
        ctx: Context,
        input_deps: Tuple[Node, ...],
        output_path: str,
        type: DataSinkType = "link",
        manifest_only=False,
        is_final_node=False,
    ) -> None:
        """
        Construct a DataSinkNode. See :func:`Node.__init__` to find comments on other parameters.

        Parameters
        ----------
        output_path
            The absolute path of a customized output folder. If set to None, an output
            folder would be created under the default output root.
            Any shared folder that can be accessed by executor and scheduler is allowed
            although IO performance varies across filesystems.
        type, optional
            The operation type of the sink node.
            "link" (default): If an output file is under the same mount point as `output_path`, a hard link is created; otherwise a symlink.
            "copy": Copies files to the output path.
            "link_or_copy": If an output file is under the same mount point as `output_path`, creates a hard link; otherwise copies the file.
            "manifest": Creates a manifest file under `output_path`. Every line of the manifest file is a path string.
        manifest_only, optional, deprecated
            Set type to "manifest".
        """
        assert type in (
            "link",
            "copy",
            "link_or_copy",
            "manifest",
        ), f"invalid sink type: {type}"
        super().__init__(ctx, input_deps, None, output_path, cpu_limit=1, gpu_limit=0, memory_limit=0)
        self.type: DataSinkType = "manifest" if manifest_only else type
        self.is_final_node = is_final_node

    def __str__(self) -> str:
        return super().__str__() + f", output_path={self.output_path}, type={self.type}"

    @Node.task_factory
    def create_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> DataSinkTask:
        # design considerations:
        # 1. data copy should start as soon as possible.
        # 2. file names may conflict across partitions of different tasks.
        #    we should rename files **if and only if** there are conflicts.
        # 3. resolving conflicts requires a single task.
        if self.type == "copy" or self.type == "link_or_copy":
            # so we create two phase tasks:
            # phase1: copy data to a temp directory, for each input partition in parallel
            input_deps = [self._create_phase1_task(runtime_ctx, task, [PartitionInfo(i, len(input_deps))]) for i, task in enumerate(input_deps)]
            # phase2: resolve file name conflicts, hard link files, create manifest file, and clean up temp directory
            return DataSinkTask(
                runtime_ctx,
                input_deps,
                [PartitionInfo()],
                self.output_path,
                type="link_manifest",
                is_final_node=self.is_final_node,
            )
        elif self.type == "link":
            return DataSinkTask(
                runtime_ctx,
                input_deps,
                [PartitionInfo()],
                self.output_path,
                type="link_manifest",
                is_final_node=self.is_final_node,
            )
        elif self.type == "manifest":
            return DataSinkTask(
                runtime_ctx,
                input_deps,
                [PartitionInfo()],
                self.output_path,
                type="manifest",
                is_final_node=self.is_final_node,
            )
        else:
            raise ValueError(f"invalid sink type: {self.type}")

    @Node.task_factory
    def _create_phase1_task(
        self,
        runtime_ctx: RuntimeContext,
        input_dep: Task,
        partition_infos: List[PartitionInfo],
    ) -> DataSinkTask:
        return DataSinkTask(runtime_ctx, [input_dep], partition_infos, self.output_path, type=self.type)


class PythonScriptNode(Node):
    """
    Run Python code to process the input datasets with `PythonScriptNode.process(...)`.

    If the code needs to access attributes of runtime task, e.g. `local_rank`, `random_seed_long`, `numpy_random_gen`,

    - create a subclass of `PythonScriptTask`, which implements `PythonScriptTask.process(...)`,
    - override `PythonScriptNode.spawn(...)` and return an instance of the subclass.

    Or use `runtime_ctx.task` in `process(runtime_ctx: RuntimeContext, ...)` function.
    """

    def __init__(
        self,
        ctx: Context,
        input_deps: Tuple[Node, ...],
        *,
        process_func: Optional[Callable[[RuntimeContext, List[DataSet], str], bool]] = None,
        output_name: Optional[str] = None,
        output_path: Optional[str] = None,
        cpu_limit: int = 1,
        gpu_limit: float = 0,
        memory_limit: Optional[int] = None,
    ):
        """
        Construct a PythonScriptNode. See :func:`Node.__init__` to find comments on other parameters.

        Parameters
        ----------
        process_func, optional
            User-defined process function, which should have the same signature as `self.process(...)`.
            If user-defined function has extra parameters, use `functools.partial(...)` to bind arguments.
            See `test_partial_process_func` in `test/test_execution.py` for examples of usage.
        """
        super().__init__(
            ctx,
            input_deps,
            output_name,
            output_path,
            cpu_limit,
            gpu_limit,
            memory_limit,
        )
        self.process_func = process_func

    def __str__(self) -> str:
        return super().__str__() + f", process_func={self.process_func}"

    @Node.task_factory
    def create_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> PythonScriptTask:
        return self.spawn(
            runtime_ctx,
            input_deps,
            partition_infos,
            self.process_func
            or self.slim_copy().process,  # warn: do not call self.slim_copy() in __init__ as attributes may not be fully initialized
            self.output_name,
            self.output_path,
            self.cpu_limit,
            self.gpu_limit,
            self.memory_limit,
        )

    def spawn(self, *args, **kwargs) -> PythonScriptTask:
        """
        Return an instance of subclass of `PythonScriptTask`. The subclass should override `PythonScriptTask.process(...)`.

        Examples
        --------
        ```
        class OutputMsgPythonTask(PythonScriptTask):

          def __init__(self, msg: str, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.msg = msg

          def process(self, runtime_ctx: RuntimeContext, input_datasets: List[DataSet], output_path: str) -> bool:
            fout = (Path(output_path) / f"{self.output_filename}-{self.local_rank}.SUCCESS")
            fout.write_text(f"msg: {self.msg}, seed: {self.random_seed_uint32}, rank: {self.local_rank}")
            return True


        class OutputMsgPythonNode(PythonScriptNode):

          def spawn(self, *args, **kwargs) -> OutputMsgPythonTask:
            return OutputMsgPythonTask("python script", *args, **kwargs)
        ```
        """
        return PythonScriptTask(*args, **kwargs)

    def process(
        self,
        runtime_ctx: RuntimeContext,
        input_datasets: List[DataSet],
        output_path: str,
    ) -> bool:
        """
        Put user-defined code here.

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
        raise NotImplementedError


class ArrowComputeNode(Node):
    """
    Run Python code to process the input datasets, which have been loaded as Apache Arrow tables.
    See https://arrow.apache.org/docs/python/generated/pyarrow.Table.html.

    If the code needs to access attributes of runtime task, e.g. `local_rank`, `random_seed_long`, `numpy_random_gen`,

    - create a subclass of `ArrowComputeTask`, which implements `ArrowComputeTask.process(...)`,
    - override `ArrowComputeNode.spawn(...)` and return an instance of the subclass.

    Or use `runtime_ctx.task` in `process(runtime_ctx: RuntimeContext, ...)` function.
    """

    default_row_group_size = DEFAULT_ROW_GROUP_SIZE

    def __init__(
        self,
        ctx: Context,
        input_deps: Tuple[Node, ...],
        *,
        process_func: Callable[[RuntimeContext, List[arrow.Table]], arrow.Table] = None,
        parquet_row_group_size: int = None,
        parquet_dictionary_encoding=False,
        parquet_compression="ZSTD",
        parquet_compression_level=3,
        use_duckdb_reader=False,
        output_name: str = None,
        output_path: str = None,
        cpu_limit: int = 1,
        gpu_limit: float = 0,
        memory_limit: Optional[int] = None,
    ) -> None:
        """
        Construct a ArrowComputeNode. See :func:`Node.__init__` to find comments on other parameters.

        Parameters
        ----------
        process_func, optional
            User-defined process function, which should have the same signature as `self.process(...)`.
            If user-defined function has extra parameters, use `functools.partial(...)` to bind arguments.
            See `test_partial_process_func` in `test/test_execution.py` for examples of usage.
        parquet_row_group_size, optional
            The number of rows stored in each row group of parquet file.
            Large row group size provides more opportunities to compress the data.
            Small row groups size could make filtering rows faster and achieve high concurrency.
            See https://duckdb.org/docs/data/parquet/tips.html#selecting-a-row_group_size.
        parquet_dictionary_encoding, optional
            Specify if we should use dictionary encoding in general or only for some columns.
            See `use_dictionary` in https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html.
        use_duckdb_reader, optional
            Use duckdb (instead of pyarrow parquet module) to load parquet files as arrow table.
        cpu_limit, optional
            The max number of CPUs would be used by tasks generated from this node.
            This is a resource requirement specified by the user and used to guide
            task scheduling. smallpond does NOT enforce this limit.
        gpu_limit, optional
            The max number of GPUs would be used by tasks generated from this node.
            This is a resource requirement specified by the user and used to guide
            task scheduling. smallpond does NOT enforce this limit.
        memory_limit, optional
            The max memory would be used by tasks generated from this node.
            This is a resource requirement specified by the user and used to guide
            task scheduling. smallpond does NOT enforce this limit.
        """
        super().__init__(
            ctx,
            input_deps,
            output_name,
            output_path,
            cpu_limit,
            gpu_limit,
            memory_limit,
        )
        self.parquet_row_group_size = parquet_row_group_size or self.default_row_group_size
        self.parquet_dictionary_encoding = parquet_dictionary_encoding
        self.parquet_compression = parquet_compression
        self.parquet_compression_level = parquet_compression_level
        self.use_duckdb_reader = use_duckdb_reader
        self.process_func = process_func

    def __str__(self) -> str:
        return super().__str__() + f", process_func={self.process_func}"

    @Node.task_factory
    def create_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> ArrowComputeTask:
        return self.spawn(
            runtime_ctx,
            input_deps,
            partition_infos,
            self.process_func
            or self.slim_copy().process,  # warn: do not call self.slim_copy() in __init__ as attributes may not be fully initialized
            self.parquet_row_group_size,
            self.parquet_dictionary_encoding,
            self.parquet_compression,
            self.parquet_compression_level,
            self.use_duckdb_reader,
            self.output_name,
            self.output_path,
            self.cpu_limit,
            self.gpu_limit,
            self.memory_limit,
        )

    def spawn(self, *args, **kwargs) -> ArrowComputeTask:
        """
        Return an instance of subclass of `ArrowComputeTask`. The subclass should override `ArrowComputeTask.process(...)`.

        Examples
        --------
        ```
        class CopyInputArrowTask(ArrowComputeTask):

          def __init__(self, msg: str, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.msg = msg

          def process(self, runtime_ctx: RuntimeContext, input_tables: List[arrow.Table]) -> arrow.Table:
            return input_tables[0]


        class CopyInputArrowNode(ArrowComputeNode):

          def spawn(self, *args, **kwargs) -> CopyInputArrowTask:
            return CopyInputArrowTask("arrow compute", *args, **kwargs)
        ```
        """
        return ArrowComputeTask(*args, **kwargs)

    def process(self, runtime_ctx: RuntimeContext, input_tables: List[arrow.Table]) -> arrow.Table:
        """
        Put user-defined code here.

        Parameters
        ----------
        runtime_ctx
            The runtime context, which defines a few global configuration info.
        input_datasets
            A list of arrow tables. The number of arrow tables equal to the number of input_deps.

        Returns
        -------
            Return the output as a arrow table. Throw an exception if there is any error.
        """
        raise NotImplementedError


class ArrowStreamNode(Node):
    """
    Run Python code to process the input datasets, which have been loaded as RecordBatchReader.
    See https://arrow.apache.org/docs/python/generated/pyarrow.RecordBatchReader.html.

    If the code needs to access attributes of runtime task, e.g. `local_rank`, `random_seed_long`, `numpy_random_gen`,
    - create a subclass of `ArrowStreamTask`, which implements `ArrowStreamTask.process(...)`,
    - override `ArrowStreamNode.spawn(...)` and return an instance of the subclass.

    Or use `runtime_ctx.task` in `process(runtime_ctx: RuntimeContext, ...)` function.
    """

    default_batch_size = DEFAULT_BATCH_SIZE
    default_row_group_size = DEFAULT_ROW_GROUP_SIZE
    default_secs_checkpoint_interval = 180

    def __init__(
        self,
        ctx: Context,
        input_deps: Tuple[Node, ...],
        *,
        process_func: Callable[[RuntimeContext, List[arrow.RecordBatchReader]], Iterable[arrow.Table]] = None,
        background_io_thread=True,
        streaming_batch_size: int = None,
        secs_checkpoint_interval: int = None,
        parquet_row_group_size: int = None,
        parquet_dictionary_encoding=False,
        parquet_compression="ZSTD",
        parquet_compression_level=3,
        use_duckdb_reader=False,
        output_name: str = None,
        output_path: str = None,
        cpu_limit: int = 1,
        gpu_limit: float = 0,
        memory_limit: Optional[int] = None,
    ) -> None:
        """
        Construct a ArrowStreamNode. See :func:`Node.__init__` to find comments on other parameters.

        Parameters
        ----------
        process_func, optional
            User-defined process function, which should have the same signature as `self.process(...)`.
            If user-defined function has extra parameters, use `functools.partial(...)` to bind arguments.
            See `test_partial_process_func` in `test/test_execution.py` for examples of usage.
        background_io_thread, optional
            Create a background IO thread for read/write.
        streaming_batch_size, optional
            Split the input datasets into batches, each of which has length less or equal to `streaming_batch_size`.
        secs_checkpoint_interval, optional
            Create a checkpoint of the stream task every `secs_checkpoint_interval` seconds.
        parquet_row_group_size, optional
            The number of rows stored in each row group of parquet file.
            Large row group size provides more opportunities to compress the data.
            Small row groups size could make filtering rows faster and achieve high concurrency.
            See https://duckdb.org/docs/data/parquet/tips.html#selecting-a-row_group_size.
        parquet_dictionary_encoding, optional
            Specify if we should use dictionary encoding in general or only for some columns.
            See `use_dictionary` in https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html.
        use_duckdb_reader, optional
            Use duckdb (instead of pyarrow parquet module) to load parquet files as arrow table.
        cpu_limit, optional
            The max number of CPUs would be used by tasks generated from this node.
            This is a resource requirement specified by the user and used to guide
            task scheduling. smallpond does NOT enforce this limit.
        gpu_limit, optional
            The max number of GPUs would be used by tasks generated from this node.
            This is a resource requirement specified by the user and used to guide
            task scheduling. smallpond does NOT enforce this limit.
        memory_limit, optional
            The max memory would be used by tasks generated from this node.
            This is a resource requirement specified by the user and used to guide
            task scheduling. smallpond does NOT enforce this limit.
        """
        super().__init__(
            ctx,
            input_deps,
            output_name,
            output_path,
            cpu_limit,
            gpu_limit,
            memory_limit,
        )
        self.background_io_thread = background_io_thread and self.cpu_limit > 1
        self.streaming_batch_size = streaming_batch_size or self.default_batch_size
        self.secs_checkpoint_interval = secs_checkpoint_interval or math.ceil(
            self.default_secs_checkpoint_interval / min(6, self.gpu_limit + 2, self.cpu_limit)
        )
        self.parquet_row_group_size = parquet_row_group_size or self.default_row_group_size
        self.parquet_dictionary_encoding = parquet_dictionary_encoding
        self.parquet_compression = parquet_compression
        self.parquet_compression_level = parquet_compression_level
        self.use_duckdb_reader = use_duckdb_reader
        self.process_func = process_func

    def __str__(self) -> str:
        return (
            super().__str__()
            + f", process_func={self.process_func}, background_io_thread={self.background_io_thread}, streaming_batch_size={self.streaming_batch_size}, checkpoint_interval={self.secs_checkpoint_interval}s"
        )

    @Node.task_factory
    def create_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> ArrowStreamTask:
        return self.spawn(
            runtime_ctx,
            input_deps,
            partition_infos,
            self.process_func
            or self.slim_copy().process,  # warn: do not call self.slim_copy() in __init__ as attributes may not be fully initialized
            self.background_io_thread,
            self.streaming_batch_size,
            self.secs_checkpoint_interval,
            self.parquet_row_group_size,
            self.parquet_dictionary_encoding,
            self.parquet_compression,
            self.parquet_compression_level,
            self.use_duckdb_reader,
            self.output_name,
            self.output_path,
            self.cpu_limit,
            self.gpu_limit,
            self.memory_limit,
        )

    def spawn(self, *args, **kwargs) -> ArrowStreamTask:
        """
        Return an instance of subclass of `ArrowStreamTask`. The subclass should override `ArrowStreamTask.process(...)`.

        Examples
        --------
        ```
        class CopyInputStreamTask(ArrowStreamTask):

          def __init__(self, msg: str, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.msg = msg

          def process(self, runtime_ctx: RuntimeContext, input_readers: List[arrow.RecordBatchReader]) -> Iterable[arrow.Table]:
            for batch in input_readers[0]:
              yield arrow.Table.from_batches([batch])


        class CopyInputStreamNode(ArrowStreamNode):

          default_batch_size = 10

          def spawn(self, *args, **kwargs) -> CopyInputStreamTask:
            return CopyInputStreamTask("arrow stream", *args, **kwargs)
        ```
        """
        return ArrowStreamTask(*args, **kwargs)

    def process(self, runtime_ctx: RuntimeContext, input_readers: List[arrow.RecordBatchReader]) -> Iterable[arrow.Table]:
        """
        Put user-defined code here.

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
        raise NotImplementedError


class ArrowBatchNode(ArrowStreamNode):
    """
    Run user-defined code to process the input datasets as a series of arrow tables.
    """

    def spawn(self, *args, **kwargs) -> ArrowBatchTask:
        return ArrowBatchTask(*args, **kwargs)

    def process(self, runtime_ctx: RuntimeContext, input_tables: List[arrow.Table]) -> arrow.Table:
        raise NotImplementedError


class PandasComputeNode(ArrowComputeNode):
    """
    Run Python code to process the input datasets as a single pandas DataFrame.
    """

    def spawn(self, *args, **kwargs) -> PandasComputeTask:
        return PandasComputeTask(*args, **kwargs)

    def process(self, runtime_ctx: RuntimeContext, input_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError


class PandasBatchNode(ArrowStreamNode):
    """
    Run Python code to process the input datasets as a series of pandas DataFrames.
    """

    def spawn(self, *args, **kwargs) -> PandasBatchTask:
        return PandasBatchTask(*args, **kwargs)

    def process(self, runtime_ctx: RuntimeContext, input_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError


class SqlEngineNode(Node):
    """
    Run SQL query against the outputs of input_deps.
    """

    max_udf_cpu_limit = 3
    default_cpu_limit = 1
    default_memory_limit = None
    default_row_group_size = DEFAULT_ROW_GROUP_SIZE
    enable_resource_boost = True

    def __init__(
        self,
        ctx: Context,
        input_deps: Tuple[Node, ...],
        sql_query: Union[str, Iterable[str]],
        *,
        udfs: List[Union[str, UserDefinedFunction]] = None,
        per_thread_output=True,
        materialize_output=True,
        materialize_in_memory=False,
        relax_memory_if_oom=None,
        batched_processing=False,
        extension_paths: List[str] = None,
        udf_module_paths: List[str] = None,
        enable_temp_directory=False,
        parquet_row_group_size: int = None,
        parquet_dictionary_encoding: bool = False,
        parquet_compression="ZSTD",
        parquet_compression_level=3,
        output_name: Optional[str] = None,
        output_path: Optional[str] = None,
        cpu_limit: Optional[int] = None,
        memory_limit: Optional[int] = None,
        cpu_overcommit_ratio: float = 1.0,
        memory_overcommit_ratio: float = 0.9,
    ) -> None:
        """
        Construct a SqlEngineNode. See :func:`Node.__init__` to find comments on other parameters.

        Parameters
        ----------
        sql_query
            SQL query string or a list of query strings, currently DuckDB query syntax is supported,
            see https://duckdb.org/docs/sql/query_syntax/select.
            All queries are executed. But only the results of the last query is persisted as the output.

            The output dataset of each `input_deps` can be referenced as `{0}`, `{1}`, `{2}`, etc.
            For example, the following query counts the total number of product items
            from `{0}` that have `category_id` included in `{1}`.

            .. code-block::

                select count(product_item.id) from {0}
                where product_item.id > 0 and
                    product_item.category_id in ( select category_id from {1} )

            The following placeholders are supported in the query:

            - `{batch_index}`: the index of the current batch.
            - `{query_index}`: the index of the current query.
            - `{rand_seed}`: the random seed of the current query.
            - `{__data_partition__}`: the index of the current data partition.
        udfs, optional
            A list of user-defined functions to be referenced in `sql_query`.
            Each element can be one of the following:

            - A `@udf` decorated function.
            - A path to a duckdb extension file, e.g. `path/to/udf.duckdb_extension`.
            - A string returned by `ctx.create_function()` or `ctx.create_duckdb_extension()`.

            If `udfs` is not empty, the resource requirement is downgraded to `min(cpu_limit, 3)` and `min(memory_limit, 50*GB)`
            since UDF execution in duckdb is not highly paralleled.
        per_thread_output, optional
            If the final number of Parquet files is not important, writing one file per thread can significantly improve performance.
            Also see https://duckdb.org/docs/data/parquet/tips.html#enabling-per_thread_output.
        materialize_output, optional
            Query result is materialized to the underlying filesystem as parquet files if enabled.
        materialize_in_memory, optional
            Materialize query result in memory before writing to the underlying filesystem, by default False.
        relax_memory_if_oom, optional
            Double the memory limit and retry if sql engine OOM, by default False.
        batched_processing, optional
            Split input dataset into multiple batches, each of which fits into memory limit, and then run sql query against each batch.
            Enabled only if `len(input_deps) == 1`.
        extension_paths, optional
            A list of duckdb extension paths to be loaded at runtime.
        enable_temp_directory, optional
            Write temp files when memory is low, by default False.
        parquet_row_group_size, optional
            The number of rows stored in each row group of parquet file.
            Large row group size provides more opportunities to compress the data.
            Small row groups size could make filtering rows faster and achieve high concurrency.
            See https://duckdb.org/docs/data/parquet/tips.html#selecting-a-row_group_size.
        parquet_dictionary_encoding, optional
            Specify if we should use dictionary encoding in general or only for some columns.
            When encoding the column, if the dictionary size is too large, the column will fallback to PLAIN encoding.
            By default, dictionary encoding is enabled for all columns. Set it to False to disable dictionary encoding,
            or pass in column names to enable it only for specific columns. eg: parquet_dictionary_encoding=['column_1']
        cpu_limit, optional
            The max number of CPUs used by the SQL engine.
        memory_limit, optional
            The max memory used by the SQL engine.
        cpu_overcommit_ratio, optional
            The effective number of threads used by the SQL engine is: `cpu_limit * cpu_overcommit_ratio`.
        memory_overcommit_ratio, optional
            The effective size of memory used by the SQL engine is: `memory_limit * memory_overcommit_ratio`.
        """

        cpu_limit = cpu_limit or self.default_cpu_limit
        memory_limit = memory_limit or self.default_memory_limit
        if udfs is not None:
            if self.max_udf_cpu_limit is not None and cpu_limit > self.max_udf_cpu_limit:
                warnings.warn(f"UDF execution is not highly paralleled, downgrade cpu_limit from {cpu_limit} to {self.max_udf_cpu_limit}")
                cpu_limit = self.max_udf_cpu_limit
                memory_limit = None
        if relax_memory_if_oom is not None:
            warnings.warn(
                "Argument 'relax_memory_if_oom' has been deprecated",
                DeprecationWarning,
                stacklevel=3,
            )

        assert isinstance(sql_query, str) or (isinstance(sql_query, Iterable) and all(isinstance(q, str) for q in sql_query))
        super().__init__(
            ctx,
            input_deps,
            output_name,
            output_path,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
        )
        self.sql_queries = [sql_query] if isinstance(sql_query, str) else list(sql_query)
        self.udfs = [ctx.create_duckdb_extension(path) for path in extension_paths or []] + [
            ctx.create_external_module(path) for path in udf_module_paths or []
        ]
        for udf in udfs or []:
            if isinstance(udf, UserDefinedFunction):
                name = ctx.create_function(udf.name, udf.func, udf.params, udf.return_type, udf.use_arrow_type)
            else:
                assert isinstance(udf, str), f"udf must be a string: {udf}"
                if udf in ctx.udfs:
                    name = udf
                elif udf.endswith(".duckdb_extension"):
                    name = ctx.create_duckdb_extension(udf)
                elif udf.endswith(".so"):
                    name = ctx.create_external_module(udf)
                else:
                    raise ValueError(f"invalid udf: {udf}")
            self.udfs.append(name)

        self.per_thread_output = per_thread_output
        self.materialize_output = materialize_output
        self.materialize_in_memory = materialize_in_memory
        self.batched_processing = batched_processing and len(input_deps) == 1
        self.enable_temp_directory = enable_temp_directory
        self.parquet_row_group_size = parquet_row_group_size or self.default_row_group_size
        self.parquet_dictionary_encoding = parquet_dictionary_encoding
        self.parquet_compression = parquet_compression
        self.parquet_compression_level = parquet_compression_level
        self.cpu_overcommit_ratio = cpu_overcommit_ratio
        self.memory_overcommit_ratio = memory_overcommit_ratio

    def __str__(self) -> str:
        return super().__str__() + f", sql_query=<{self.oneline_query[:100]}...>, udfs={self.udfs}, batched_processing={self.batched_processing}"

    @property
    def oneline_query(self) -> str:
        return "; ".join(" ".join(filter(None, map(str.strip, query.splitlines()))) for query in self.sql_queries)

    @Node.task_factory
    def create_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> SqlEngineTask:
        return self.spawn(
            runtime_ctx,
            input_deps,
            partition_infos,
            self.sql_queries,
            udfs=[self.ctx.udfs[name] for name in self.udfs],
            per_thread_output=self.per_thread_output,
            materialize_output=self.materialize_output,
            materialize_in_memory=self.materialize_in_memory,
            batched_processing=self.batched_processing,
            enable_temp_directory=self.enable_temp_directory,
            parquet_row_group_size=self.parquet_row_group_size,
            parquet_dictionary_encoding=self.parquet_dictionary_encoding,
            parquet_compression=self.parquet_compression,
            parquet_compression_level=self.parquet_compression_level,
            output_name=self.output_name,
            output_path=self.output_path,
            cpu_limit=self.cpu_limit,
            gpu_limit=self.gpu_limit,
            memory_limit=self.memory_limit,
            cpu_overcommit_ratio=self.cpu_overcommit_ratio,
            memory_overcommit_ratio=self.memory_overcommit_ratio,
        )

    def spawn(self, *args, **kwargs) -> SqlEngineTask:
        return SqlEngineTask(*args, **kwargs)

    @property
    def num_partitions(self) -> int:
        return self.input_deps[0].num_partitions


class UnionNode(Node):
    """
    Union two or more nodes into one flow of data.
    """

    def __init__(self, ctx: Context, input_deps: Tuple[Node, ...]):
        """
        Union two or more `input_deps` into one flow of data.

        Parameters
        ----------
        input_deps
            All input deps should have the same set of partition dimensions.
        """
        super().__init__(ctx, input_deps)


class RootNode(Node):
    """
    A virtual node that assembles multiple nodes and outputs nothing.
    """

    def __init__(self, ctx: Context, input_deps: Tuple[Node, ...]):
        """
        Assemble multiple nodes into a root node.
        """
        super().__init__(ctx, input_deps)


class ConsolidateNode(Node):
    """
    Consolidate partitions into larger ones.
    """

    def __init__(self, ctx: Context, input_dep: Node, dimensions: List[str]):
        """
        Effectively reduces the number of partitions without shuffling the data across the network.

        Parameters
        ----------
        dimensions
            Partitions would be grouped by these `dimensions` and consolidated into larger partitions.
        """
        assert isinstance(dimensions, Iterable), f"dimensions is not iterable: {dimensions}"
        assert all(isinstance(dim, str) for dim in dimensions), f"some dimensions are not strings: {dimensions}"
        super().__init__(ctx, [input_dep])
        self.dimensions = set(list(dimensions) + [PartitionInfo.toplevel_dimension])

    def __str__(self) -> str:
        return super().__str__() + f", dimensions={self.dimensions}"

    @Node.task_factory
    def create_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> MergeDataSetsTask:
        return MergeDataSetsTask(runtime_ctx, input_deps, partition_infos)


class PartitionNode(Node):
    """
    The base class for all partition nodes.
    """

    max_num_producer_tasks = 100
    max_card_of_producers_x_consumers = 4_096_000

    def __init__(
        self,
        ctx: Context,
        input_deps: Tuple[Node, ...],
        npartitions: int,
        dimension: str = None,
        nested: bool = False,
        output_name: str = None,
        output_path: str = None,
        cpu_limit: int = 1,
        memory_limit: Optional[int] = None,
    ) -> None:
        """
        Partition the outputs of `input_deps` into n partitions.

        Parameters
        ----------
        npartitions
            The dataset would be split and distributed to `npartitions` partitions.
        dimension
            The unique partition dimension. Required if this is a nested partition.
        nested, optional
            `npartitions` subpartitions are created in each existing partition of `input_deps` if true.

        Examples
        --------
        See unit tests in `test/test_partition.py`. For nested partition see `test_nested_partition`.
        Why nested partition? See **5.1 Partial Partitioning** of [Advanced partitioning techniques for massively distributed computation](https://dl.acm.org/doi/10.1145/2213836.2213839).
        """
        assert isinstance(npartitions, int), f"npartitions is not an integer: {npartitions}"
        assert dimension is None or re.match("[a-zA-Z0-9_]+", dimension), f"dimension has invalid format: {dimension}"
        assert not (nested and dimension is None), f"nested partition should have dimension"
        super().__init__(ctx, input_deps, output_name, output_path, cpu_limit, 0, memory_limit)
        self.npartitions = npartitions
        self.dimension = dimension if dimension is not None else PartitionInfo.default_dimension
        self.nested = nested

    def __str__(self) -> str:
        return super().__str__() + f", npartitions={self.npartitions}, dimension={self.dimension}, nested={self.nested}"

    @Node.task_factory
    def create_producer_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> PartitionProducerTask:
        raise NotImplementedError

    @Node.task_factory
    def create_consumer_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> PartitionConsumerTask:
        return PartitionConsumerTask(runtime_ctx, input_deps, partition_infos)

    @Node.task_factory
    def create_merge_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> MergeDataSetsTask:
        return MergeDataSetsTask(runtime_ctx, input_deps, partition_infos)

    @Node.task_factory
    def create_split_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> SplitDataSetTask:
        return SplitDataSetTask(runtime_ctx, input_deps, partition_infos)

    @property
    def num_partitions(self) -> int:
        return self.npartitions


class RepeatPartitionNode(PartitionNode):
    """
    Create a new partition dimension by repeating the `input_deps`. This is always a nested partition.
    """

    def __init__(
        self,
        ctx: Context,
        input_deps: Tuple[Node, ...],
        npartitions: int,
        dimension: str,
        cpu_limit: int = 1,
        memory_limit: Optional[int] = None,
    ) -> None:
        super().__init__(
            ctx,
            input_deps,
            npartitions,
            dimension,
            nested=True,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
        )

    @Node.task_factory
    def create_producer_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> RepeatPartitionProducerTask:
        return RepeatPartitionProducerTask(
            runtime_ctx,
            input_deps,
            partition_infos,
            self.npartitions,
            self.dimension,
            self.cpu_limit,
            self.memory_limit,
        )


class UserDefinedPartitionNode(PartitionNode):
    """
    Distribute the output files or rows of `input_deps` into n partitions based on user code.
    See unit test `test_user_defined_partition` in `test/test_partition.py`.
    """

    def __init__(
        self,
        ctx: Context,
        input_deps: Tuple[Node, ...],
        npartitions: int,
        dimension: str = None,
        nested: bool = False,
        cpu_limit: int = 1,
        memory_limit: Optional[int] = None,
    ) -> None:
        super().__init__(
            ctx,
            input_deps,
            npartitions,
            dimension,
            nested,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
        )

    @Node.task_factory
    def create_producer_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> UserDefinedPartitionProducerTask:
        return UserDefinedPartitionProducerTask(
            runtime_ctx,
            input_deps,
            partition_infos,
            self.npartitions,
            self.dimension,
            self.partition,
            self.cpu_limit,
            self.memory_limit,
        )

    def partition(self, runtime_ctx: RuntimeContext, dataset: DataSet) -> List[DataSet]:
        raise NotImplementedError


class UserPartitionedDataSourceNode(UserDefinedPartitionNode):
    max_num_producer_tasks = 1

    def __init__(self, ctx: Context, partitioned_datasets: List[DataSet], dimension: str = None) -> None:
        assert isinstance(partitioned_datasets, Iterable) and all(isinstance(dataset, DataSet) for dataset in partitioned_datasets)
        super().__init__(
            ctx,
            [DataSourceNode(ctx, dataset=None)],
            len(partitioned_datasets),
            dimension,
            nested=False,
        )
        self.partitioned_datasets = partitioned_datasets

    def partition(self, runtime_ctx: RuntimeContext, dataset: DataSet) -> List[DataSet]:
        return self.partitioned_datasets


class EvenlyDistributedPartitionNode(PartitionNode):
    """
    Evenly distribute the output files or rows of `input_deps` into n partitions.
    """

    max_num_producer_tasks = 1

    def __init__(
        self,
        ctx: Context,
        input_deps: Tuple[Node, ...],
        npartitions: int,
        dimension: str = None,
        nested: bool = False,
        *,
        partition_by_rows=False,
        random_shuffle=False,
        output_name: str = None,
        output_path: str = None,
        cpu_limit: int = 1,
        memory_limit: Optional[int] = None,
    ) -> None:
        """
        Evenly distribute the output files or rows of `input_deps` into n partitions.

        Parameters
        ----------
        partition_by_rows, optional
            Evenly distribute rows instead of input files into `npartitions` partitions, by default distribute by files.
        random_shuffle, optional
            Random shuffle the list of paths or parquet row groups (if `partition_by_rows=True`) in input datasets.
        """
        super().__init__(
            ctx,
            input_deps,
            npartitions,
            dimension,
            nested,
            output_name=output_name,
            output_path=output_path,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
        )
        self.partition_by_rows = partition_by_rows and npartitions > 1
        self.random_shuffle = random_shuffle

    def __str__(self) -> str:
        return super().__str__() + f", partition_by_rows={self.partition_by_rows}, random_shuffle={self.random_shuffle}"

    @Node.task_factory
    def create_producer_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ):
        return EvenlyDistributedPartitionProducerTask(
            runtime_ctx,
            input_deps,
            partition_infos,
            self.npartitions,
            self.dimension,
            self.partition_by_rows,
            self.random_shuffle,
            self.cpu_limit,
            self.memory_limit,
        )


class LoadPartitionedDataSetNode(PartitionNode):
    """
    Load existing partitioned dataset (only parquet files are supported).
    """

    max_num_producer_tasks = 10

    def __init__(
        self,
        ctx: Context,
        input_deps: Tuple[Node, ...],
        npartitions: int,
        dimension: str = None,
        nested: bool = False,
        data_partition_column: str = None,
        hive_partitioning: bool = False,
        cpu_limit: int = 1,
        memory_limit: Optional[int] = None,
    ) -> None:
        assert dimension or data_partition_column, f"Both 'dimension' and 'data_partition_column' are none or empty"
        super().__init__(
            ctx,
            input_deps,
            npartitions,
            dimension or data_partition_column,
            nested,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
        )
        self.data_partition_column = data_partition_column
        self.hive_partitioning = hive_partitioning

    def __str__(self) -> str:
        return super().__str__() + f", data_partition_column={self.data_partition_column}, hive_partitioning={self.hive_partitioning}"

    @Node.task_factory
    def create_producer_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ):
        return LoadPartitionedDataSetProducerTask(
            runtime_ctx,
            input_deps,
            partition_infos,
            self.npartitions,
            self.dimension,
            self.data_partition_column,
            self.hive_partitioning,
            self.cpu_limit,
            self.memory_limit,
        )


def DataSetPartitionNode(
    ctx: Context,
    input_deps: Tuple[Node, ...],
    npartitions: int,
    *,
    partition_by_rows=False,
    random_shuffle=False,
    data_partition_column=None,
):
    """
    Partition the outputs of `input_deps` into n partitions.

    Parameters
    ----------
    npartitions
        The number of partitions. The input files or rows would be evenly distributed to `npartitions` partitions.
    partition_by_rows, optional
        Evenly distribute rows instead of input files into `npartitions` partitions, by default distribute by files.
    random_shuffle, optional
        Random shuffle the list of paths or parquet row groups (if `partition_by_rows=True`) in input datasets.
    data_partition_column, optional
        Partition by files based on the partition keys stored in `data_partition_column` if specified.
        Default column name used by `HashPartitionNode` is `DATA_PARTITION_COLUMN_NAME`.

    Examples
    --------
    See unit test `test_load_partitioned_datasets` in `test/test_partition.py`.
    """
    assert not (partition_by_rows and data_partition_column), "partition_by_rows and data_partition_column cannot be set at the same time"
    if data_partition_column is None:
        partition_node = EvenlyDistributedPartitionNode(
            ctx,
            input_deps,
            npartitions,
            dimension=None,
            nested=False,
            partition_by_rows=partition_by_rows,
            random_shuffle=random_shuffle,
        )
        if npartitions == 1:
            return ConsolidateNode(ctx, partition_node, dimensions=[])
        else:
            return partition_node
    else:
        return LoadPartitionedDataSetNode(
            ctx,
            input_deps,
            npartitions,
            dimension=data_partition_column,
            nested=False,
            data_partition_column=data_partition_column,
            hive_partitioning=False,
        )


class HashPartitionNode(PartitionNode):
    """
    Partition the outputs of `input_deps` into n partitions based on the hash values of `hash_columns`.
    """

    default_cpu_limit = 1
    default_memory_limit = None
    default_data_partition_column = DATA_PARTITION_COLUMN_NAME
    default_engine_type = "duckdb"
    default_row_group_size = DEFAULT_ROW_GROUP_SIZE
    max_num_producer_tasks = 1000
    enable_resource_boost = True

    def __init__(
        self,
        ctx: Context,
        input_deps: Tuple[Node, ...],
        npartitions: int,
        hash_columns: List[str] = None,
        data_partition_column: str = None,
        *,
        dimension: str = None,
        nested: bool = False,
        engine_type: Literal["duckdb", "arrow"] = None,
        random_shuffle: bool = False,
        shuffle_only: bool = False,
        drop_partition_column: bool = False,
        use_parquet_writer: bool = False,
        hive_partitioning: bool = False,
        parquet_row_group_size: int = None,
        parquet_dictionary_encoding=False,
        parquet_compression="ZSTD",
        parquet_compression_level=3,
        output_name: str = None,
        output_path: str = None,
        cpu_limit: Optional[int] = None,
        memory_limit: Optional[int] = None,
    ) -> None:
        """
        Construct a HashPartitionNode. See :func:`Node.__init__` to find comments on other parameters.

        Parameters
        ----------
        npartitions
            The number of hash partitions. The number of generated parquet files would be proportional to `npartitions`.
        hash_columns
            The hash values are computed from `hash_columns`.
        data_partition_column, optional
            The name of column used to store partition keys.
        engine_type, optional
            The underlying query engine for hash partition.
        random_shuffle, optional
            Ignore `hash_columns` and shuffle each row to a random partition if true.
        shuffle_only, optional
            Ignore `hash_columns` and shuffle each row to the partition specified in `data_partition_column` if true.
        drop_partition_column, optional
            Exclude `data_partition_column` in output if true.
        use_parquet_writer, optional
            Convert partition data to arrow tables and append with parquet writer if true. This creates less number of
            intermediate files but makes partitioning slower.
        hive_partitioning, optional
            Use Hive partitioned write of duckdb if true.
        parquet_row_group_size, optional
            The number of rows stored in each row group of parquet file.
            Large row group size provides more opportunities to compress the data.
            Small row groups size could make filtering rows faster and achieve high concurrency.
            See https://duckdb.org/docs/data/parquet/tips.html#selecting-a-row_group_size.
        parquet_dictionary_encoding, optional
            Specify if we should use dictionary encoding in general or only for some columns.
            See `use_dictionary` in https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html.
        """
        assert not random_shuffle or not shuffle_only, f"random_shuffle and shuffle_only cannot be enabled at the same time"
        assert not shuffle_only or data_partition_column is not None, f"data_partition_column not specified for shuffle-only partitioning"
        assert data_partition_column is None or re.match(
            "[a-zA-Z0-9_]+", data_partition_column
        ), f"data_partition_column has invalid format: {data_partition_column}"
        assert engine_type in (
            None,
            "duckdb",
            "arrow",
        ), f"unknown query engine type: {engine_type}"
        data_partition_column = data_partition_column or self.default_data_partition_column
        super().__init__(
            ctx,
            input_deps,
            npartitions,
            dimension or data_partition_column,
            nested,
            output_name,
            output_path,
            cpu_limit or self.default_cpu_limit,
            memory_limit or self.default_memory_limit,
        )
        self.hash_columns = ["random()"] if random_shuffle else hash_columns
        self.data_partition_column = data_partition_column
        self.engine_type = engine_type or self.default_engine_type
        self.random_shuffle = random_shuffle
        self.shuffle_only = shuffle_only
        self.drop_partition_column = drop_partition_column
        self.use_parquet_writer = use_parquet_writer
        self.hive_partitioning = hive_partitioning and self.engine_type == "duckdb"
        self.parquet_row_group_size = parquet_row_group_size or self.default_row_group_size
        self.parquet_dictionary_encoding = parquet_dictionary_encoding
        self.parquet_compression = parquet_compression
        self.parquet_compression_level = parquet_compression_level

    def __str__(self) -> str:
        return (
            super().__str__()
            + f", hash_columns={self.hash_columns}, data_partition_column={self.data_partition_column}, engine_type={self.engine_type}, hive_partitioning={self.hive_partitioning}"
        )

    @Node.task_factory
    def create_producer_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> HashPartitionTask:
        return HashPartitionTask.create(
            self.engine_type,
            runtime_ctx,
            input_deps,
            partition_infos,
            self.npartitions,
            self.dimension,
            self.hash_columns,
            self.data_partition_column,
            self.random_shuffle,
            self.shuffle_only,
            self.drop_partition_column,
            self.use_parquet_writer,
            self.hive_partitioning,
            self.parquet_row_group_size,
            self.parquet_dictionary_encoding,
            self.parquet_compression,
            self.parquet_compression_level,
            self.output_name,
            self.output_path,
            self.cpu_limit,
            self.memory_limit,
        )


class ShuffleNode(HashPartitionNode):
    def __init__(
        self,
        ctx: Context,
        input_deps: Tuple[Node, ...],
        npartitions: int,
        data_partition_column: str = None,
        *,
        dimension: str = None,
        nested: bool = False,
        engine_type: Literal["duckdb", "arrow"] = None,
        use_parquet_writer: bool = False,
        hive_partitioning: bool = False,
        parquet_row_group_size: int = None,
        parquet_dictionary_encoding=False,
        parquet_compression="ZSTD",
        parquet_compression_level=3,
        output_name: str = None,
        output_path: str = None,
        cpu_limit: Optional[int] = None,
        memory_limit: Optional[int] = None,
    ) -> None:
        super().__init__(
            ctx,
            input_deps,
            npartitions,
            hash_columns=None,
            data_partition_column=data_partition_column,
            dimension=dimension,
            nested=nested,
            engine_type=engine_type,
            random_shuffle=False,
            shuffle_only=True,
            drop_partition_column=False,
            use_parquet_writer=use_parquet_writer,
            hive_partitioning=hive_partitioning,
            parquet_row_group_size=parquet_row_group_size,
            parquet_dictionary_encoding=parquet_dictionary_encoding,
            parquet_compression=parquet_compression,
            parquet_compression_level=parquet_compression_level,
            output_name=output_name,
            output_path=output_path,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
        )


class RangePartitionNode(PartitionNode):
    """
    Partition the outputs of `input_deps` into partitions defined by `split_points`. This node is not implemented yet.
    """

    def __init__(
        self,
        ctx: Context,
        input_deps: Tuple[Node, ...],
        split_points: List,
        dimension: str = None,
        nested: bool = False,
        output_name: str = None,
        output_path: str = None,
        cpu_limit: int = 16,
        memory_limit: int = 128 * GB,
    ) -> None:
        super().__init__(
            ctx,
            input_deps,
            len(split_points) + 1,
            dimension,
            nested,
            output_name,
            output_path,
            cpu_limit,
            memory_limit,
        )
        self.split_points = split_points


class ProjectionNode(Node):
    """
    Select columns from output of an input node.
    """

    def __init__(
        self,
        ctx: Context,
        input_dep: Node,
        columns: List[str] = None,
        generated_columns: List[Literal["filename", "file_row_number"]] = None,
        union_by_name=None,
    ) -> None:
        """
        Construct a ProjectNode to select only the `columns` from output of `input_dep`.

        Parameters
        ----------
        input_dep
            The input node whose output would be selected.
        columns, optional
            The columns to be selected or created. Select all columns if set to `None`.
        generated_columns
            Auto generated columns, supported values: `filename`, `file_row_number`.
        union_by_name, optional
            Unify the columns of different files by name (see https://duckdb.org/docs/data/multiple_files/combining_schemas#union-by-name).

        Examples
        --------
        First create an ArrowComputeNode to extract hosts from urls.

        .. code-block:: python

            class ParseUrl(ArrowComputeNode):
                def process(self, runtime_ctx: RuntimeContext, input_tables: List[arrow.Table]) -> arrow.Table:
                    assert input_tables[0].column_names == ["url"] # check url is the only column in table
                    urls, = input_tables[0].columns
                    hosts = [url.as_py().split("/", maxsplit=2)[0] for url in urls]
                    return arrow.Table.from_arrays([hosts, urls], names=["host", "url"])

        Suppose there are several columns in output of `data_partitions`,
        `ProjectionNode(..., ["url"])` selects the `url` column.
        Then only this column would be loaded into arrow table when feeding data to `ParseUrl`.

        .. code-block:: python

            urls_with_host = ParseUrl(ctx, (ProjectionNode(ctx, data_partitions, ["url"]),))
        """
        columns = columns or ["*"]
        generated_columns = generated_columns or []
        assert all(col in GENERATED_COLUMNS for col in generated_columns), f"invalid values found in generated columns: {generated_columns}"
        assert not (set(columns) & set(generated_columns)), f"columns {columns} and generated columns {generated_columns} share common columns"
        super().__init__(ctx, [input_dep])
        self.columns = columns
        self.generated_columns = generated_columns
        self.union_by_name = union_by_name

    def __str__(self) -> str:
        return super().__str__() + f", columns={self.columns}, generated_columns={self.generated_columns}, union_by_name={self.union_by_name}"

    @Node.task_factory
    def create_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> ProjectionTask:
        return ProjectionTask(
            runtime_ctx,
            input_deps,
            partition_infos,
            self.columns,
            self.generated_columns,
            self.union_by_name,
        )


class LimitNode(SqlEngineNode):
    """
    Limit the number of rows of the output of an input node.
    """

    def __init__(self, ctx: Context, input_dep: Node, limit: int) -> None:
        """
        Construct a LimitNode to limit the number of rows of the output of `input_dep`.

        Parameters
        ----------
        input_dep
            The input node whose output would be limited.
        limit
            The number of rows to be limited.
        """
        super().__init__(ctx, (input_dep,), f"select * from {{0}} limit {limit}")
        self.limit = limit

    def __str__(self) -> str:
        return super().__str__() + f", limit={self.limit}"

    @Node.task_factory
    def create_merge_task(
        self,
        runtime_ctx: RuntimeContext,
        input_deps: List[Task],
        partition_infos: List[PartitionInfo],
    ) -> MergeDataSetsTask:
        return MergeDataSetsTask(runtime_ctx, input_deps, partition_infos)


T = TypeVar("T")


class LogicalPlanVisitor(Generic[T]):
    """
    Visit the nodes of a logcial plan in depth-first order.
    """

    def visit(self, node: Node, depth: int = 0) -> T:
        """
        Visit a node depending on its type.
        If the method for the node type is not implemented, call `generic_visit`.
        """
        if isinstance(node, DataSourceNode):
            return self.visit_data_source_node(node, depth)
        elif isinstance(node, DataSinkNode):
            return self.visit_data_sink_node(node, depth)
        elif isinstance(node, RootNode):
            return self.visit_root_node(node, depth)
        elif isinstance(node, UnionNode):
            return self.visit_union_node(node, depth)
        elif isinstance(node, ConsolidateNode):
            return self.visit_consolidate_node(node, depth)
        elif isinstance(node, PartitionNode):
            return self.visit_partition_node(node, depth)
        elif isinstance(node, PythonScriptNode):
            return self.visit_python_script_node(node, depth)
        elif isinstance(node, ArrowComputeNode):
            return self.visit_arrow_compute_node(node, depth)
        elif isinstance(node, ArrowStreamNode):
            return self.visit_arrow_stream_node(node, depth)
        elif isinstance(node, LimitNode):
            return self.visit_limit_node(node, depth)
        elif isinstance(node, SqlEngineNode):
            return self.visit_query_engine_node(node, depth)
        elif isinstance(node, ProjectionNode):
            return self.visit_projection_node(node, depth)
        else:
            raise Exception(f"Unknown node type: {node}")

    def generic_visit(self, node: Node, depth: int) -> T:
        """This visitor calls visit() on all children of the node."""
        for dep in node.input_deps:
            self.visit(dep, depth + 1)

    def visit_data_source_node(self, node: DataSourceNode, depth: int) -> T:
        return self.generic_visit(node, depth)

    def visit_data_sink_node(self, node: DataSinkNode, depth: int) -> T:
        return self.generic_visit(node, depth)

    def visit_root_node(self, node: RootNode, depth: int) -> T:
        return self.generic_visit(node, depth)

    def visit_union_node(self, node: UnionNode, depth: int) -> T:
        return self.generic_visit(node, depth)

    def visit_consolidate_node(self, node: ConsolidateNode, depth: int) -> T:
        return self.generic_visit(node, depth)

    def visit_partition_node(self, node: PartitionNode, depth: int) -> T:
        return self.generic_visit(node, depth)

    def visit_python_script_node(self, node: PythonScriptNode, depth: int) -> T:
        return self.generic_visit(node, depth)

    def visit_arrow_compute_node(self, node: ArrowComputeNode, depth: int) -> T:
        return self.generic_visit(node, depth)

    def visit_arrow_stream_node(self, node: ArrowStreamNode, depth: int) -> T:
        return self.generic_visit(node, depth)

    def visit_query_engine_node(self, node: SqlEngineNode, depth: int) -> T:
        return self.generic_visit(node, depth)

    def visit_projection_node(self, node: ProjectionNode, depth: int) -> T:
        return self.generic_visit(node, depth)

    def visit_limit_node(self, node: LimitNode, depth: int) -> T:
        return self.generic_visit(node, depth)


class LogicalPlan(object):
    """
    The logical plan that defines a directed acyclic computation graph.
    """

    def __init__(self, ctx: Context, root_node: Node) -> None:
        self.ctx = ctx
        self.root_node = root_node

    def __str__(self) -> str:
        return self.explain_str()

    def explain_str(self) -> str:
        """
        Return a string that shows the structure of the logical plan.
        """
        visited = set()

        def to_str(node: Node, depth: int = 0) -> List[str]:
            lines = ["  " * depth + str(node) + ", file= " + node.location]
            if node.id in visited:
                return lines + ["  " * depth + " (omitted ...)"]
            visited.add(node.id)
            lines += ["  " * depth + f" | {name}: {stats}" for name, stats in node.perf_stats.items()]
            for dep in node.input_deps:
                lines.extend(to_str(dep, depth + 1))
            return lines

        return os.linesep.join(to_str(self.root_node))

    def graph(self) -> Digraph:
        """
        Return a graphviz graph that shows the structure of the logical plan.
        """
        dot = Digraph(comment="smallpond")
        for node in self.nodes.values():
            dot.node(str(node.id), repr(node))
            for dep in node.input_deps:
                dot.edge(str(dep.id), str(node.id))
        return dot

    @property
    def nodes(self) -> Dict[NodeId, Node]:
        nodes = {}

        def collect_nodes(node: Node):
            if node.id in nodes:
                return
            nodes[node.id] = node
            for dep in node.input_deps:
                collect_nodes(dep)

        collect_nodes(self.root_node)
        return nodes
