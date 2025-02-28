from __future__ import annotations

import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterator

import pandas as pd
import pyarrow as arrow
import ray
import ray.exceptions
from loguru import logger

from smallpond.execution.task import Task
from smallpond.io.filesystem import remove_path
from smallpond.logical.dataset import *
from smallpond.logical.node import *
from smallpond.logical.optimizer import Optimizer
from smallpond.logical.planner import Planner
from smallpond.session import SessionBase


class Session(SessionBase):
    # Extended session class with additional methods to create DataFrames.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._nodes: List[Node] = []

        self._node_to_tasks: Dict[Node, List[Task]] = {}
        """
        When a DataFrame is evaluated, the tasks of the logical plan are stored here.
        Subsequent DataFrames can reuse the tasks to avoid recomputation.
        """

    def read_csv(
        self, paths: Union[str, List[str]], schema: Dict[str, str], delim=","
    ) -> DataFrame:
        """
        Create a DataFrame from CSV files.
        """
        dataset = CsvDataSet(paths, OrderedDict(schema), delim)
        plan = DataSourceNode(self._ctx, dataset)
        return DataFrame(self, plan)

    def read_parquet(
        self,
        paths: Union[str, List[str]],
        recursive: bool = False,
        columns: Optional[List[str]] = None,
        union_by_name: bool = False,
    ) -> DataFrame:
        """
        Create a DataFrame from Parquet files.
        """
        dataset = ParquetDataSet(
            paths, columns=columns, union_by_name=union_by_name, recursive=recursive
        )
        plan = DataSourceNode(self._ctx, dataset)
        return DataFrame(self, plan)

    def read_json(
        self, paths: Union[str, List[str]], schema: Dict[str, str]
    ) -> DataFrame:
        """
        Create a DataFrame from JSON files.
        """
        dataset = JsonDataSet(paths, schema)
        plan = DataSourceNode(self._ctx, dataset)
        return DataFrame(self, plan)

    def from_items(self, items: List[Any]) -> DataFrame:
        """
        Create a DataFrame from a list of local Python objects.
        """

        assert isinstance(items, list), "items must be a list"
        assert len(items) > 0, "items must not be empty"
        if isinstance(items[0], dict):
            return self.from_arrow(arrow.Table.from_pylist(items))
        else:
            return self.from_arrow(arrow.table({"item": items}))

    def from_pandas(self, df: pd.DataFrame) -> DataFrame:
        """
        Create a DataFrame from a pandas DataFrame.
        """
        plan = DataSourceNode(self._ctx, PandasDataSet(df))
        return DataFrame(self, plan)

    def from_arrow(self, table: arrow.Table) -> DataFrame:
        """
        Create a DataFrame from a pyarrow Table.
        """
        plan = DataSourceNode(self._ctx, ArrowTableDataSet(table))
        return DataFrame(self, plan)

    def partial_sql(self, query: str, *inputs: DataFrame, **kwargs) -> DataFrame:
        """
        Execute a SQL query on each partition of the input DataFrames.

        The query can contain placeholder `{0}`, `{1}`, etc. for the input DataFrames.
        If multiple DataFrames are provided, they must have the same number of partitions.

        Examples
        --------
        Join two datasets. You need to make sure the join key is correctly partitioned.

        .. code-block::

            a = sp.read_parquet("a/*.parquet").repartition(10, hash_by="id")
            b = sp.read_parquet("b/*.parquet").repartition(10, hash_by="id")
            c = sp.partial_sql("select * from {0} join {1} on a.id = b.id", a, b)
        """

        plan = SqlEngineNode(
            self._ctx, tuple(input.plan for input in inputs), query, **kwargs
        )
        recompute = any(input.need_recompute for input in inputs)
        return DataFrame(self, plan, recompute=recompute)

    def wait(self, *dfs: DataFrame):
        """
        Wait for all DataFrames to be computed.

        Example
        -------
        This can be used to wait for multiple outputs from a pipeline:

        .. code-block::

            df = sp.read_parquet("input/*.parquet")
            output1 = df.write_parquet("output1")
            output2 = df.map("col1, col2").write_parquet("output2")
            sp.wait(output1, output2)
        """
        ray.get([task.run_on_ray() for df in dfs for task in df._get_or_create_tasks()])

    def graph(self) -> Digraph:
        """
        Get the DataFrame graph.
        """
        dot = Digraph(comment="SmallPond")
        for node in self._nodes:
            dot.node(str(node.id), repr(node))
            for dep in node.input_deps:
                dot.edge(str(dep.id), str(node.id))
        return dot

    def shutdown(self):
        """
        Shutdown the session.
        """
        # prevent shutdown from being called multiple times
        if hasattr(self, "_shutdown_called"):
            return
        self._shutdown_called = True

        # log status
        finished = self._all_tasks_finished()
        with open(self._runtime_ctx.job_status_path, "a") as fout:
            status = "success" if finished else "failure"
            fout.write(f"{status}@{datetime.now():%Y-%m-%d-%H-%M-%S}\n")

        # clean up runtime directories if success
        if finished:
            logger.info("all tasks are finished, cleaning up")
            self._runtime_ctx.cleanup(remove_output_root=self.config.remove_output_root)
        else:
            logger.warning("tasks are not finished!")

        super().shutdown()

    def _summarize_task(self) -> Tuple[int, int]:
        """
        Return the total number of tasks and the number of tasks that are finished.
        """
        dataset_refs = [
            task._dataset_ref
            for tasks in self._node_to_tasks.values()
            for task in tasks
            if task._dataset_ref is not None
        ]
        ready_tasks, _ = ray.wait(
            dataset_refs, num_returns=len(dataset_refs), timeout=0, fetch_local=False
        )
        return len(dataset_refs), len(ready_tasks)

    def _all_tasks_finished(self) -> bool:
        """
        Check if all tasks are finished.
        """
        dataset_refs = [
            task._dataset_ref
            for tasks in self._node_to_tasks.values()
            for task in tasks
        ]
        try:
            ray.get(dataset_refs, timeout=0)
        except Exception:
            # GetTimeoutError is raised if any task is not finished
            # RuntimeError is raised if any task failed
            return False
        return True


class DataFrame:
    """
    A distributed data collection. It represents a 2 dimensional table of rows and columns.

    Internally, it's a wrapper around a `Node` and a `Session` required for execution.
    """

    def __init__(self, session: Session, plan: Node, recompute: bool = False):
        self.session = session
        self.plan = plan
        self.optimized_plan: Optional[Node] = None
        self.need_recompute = recompute
        """Whether to recompute the data regardless of whether it's already computed."""

        session._nodes.append(plan)

    def __str__(self) -> str:
        return repr(self.plan)

    def _get_or_create_tasks(self) -> List[Task]:
        """
        Get or create tasks to compute the data.
        """
        # optimize the plan
        if self.optimized_plan is None:
            logger.info(f"optimizing\n{LogicalPlan(self.session._ctx, self.plan)}")
            self.optimized_plan = Optimizer(
                exclude_nodes=set(self.session._node_to_tasks.keys())
            ).visit(self.plan)
            logger.info(
                f"optimized\n{LogicalPlan(self.session._ctx, self.optimized_plan)}"
            )
        # return the tasks if already created
        if tasks := self.session._node_to_tasks.get(self.optimized_plan):
            return tasks

        # remove all completed task files if recompute is needed
        if self.need_recompute:
            remove_path(
                os.path.join(
                    self.session._runtime_ctx.completed_task_dir,
                    str(self.optimized_plan.id),
                )
            )
            logger.info(f"cleared all results of {self.optimized_plan!r}")

        # create tasks for the optimized plan
        planner = Planner(self.session._runtime_ctx)
        # let planner update self.session._node_to_tasks
        planner.node_to_tasks = self.session._node_to_tasks
        return planner.visit(self.optimized_plan)

    def is_computed(self) -> bool:
        """
        Check if the data is ready on disk.
        """
        if tasks := self.session._node_to_tasks.get(self.plan):
            _, unready_tasks = ray.wait(tasks, timeout=0)
            return len(unready_tasks) == 0
        return False

    def compute(self) -> None:
        """
        Compute the data.

        This operation will trigger execution of the lazy transformations performed on this DataFrame.
        """
        self._compute()

    def _compute(self) -> List[DataSet]:
        """
        Compute the data and return the datasets.
        """
        for retry_count in range(3):
            try:
                return ray.get(
                    [task.run_on_ray() for task in self._get_or_create_tasks()]
                )
            except ray.exceptions.RuntimeEnvSetupError as e:
                # XXX: Ray may raise this error when a worker is interrupted.
                #      ```
                #      ray.exceptions.RuntimeEnvSetupError: Failed to set up runtime environment.
                #      Failed to create runtime env for job 01000000, status = IOError:
                #      on_read bad version, maybe there are some network problems, will fail the request.
                #      ```
                #      This is a bug of Ray and has been fixed in Ray 2.24: <https://github.com/ray-project/ray/pull/45513>
                #      However, since Ray dropped support for Python 3.8 since 2.11, we can not upgrade Ray.
                #      So we catch this error and retry by ourselves.
                logger.error(f"found ray RuntimeEnvSetupError, retrying...\n{e}")
                time.sleep(10 << retry_count)
        raise RuntimeError("Failed to compute data after 3 retries")

    # operations

    def recompute(self) -> DataFrame:
        """
        Always recompute the data regardless of whether it's already computed.

        Examples
        --------
        Modify the code as follows and rerun:

        .. code-block:: diff

            - df = input.select('a')
            + df = input.select('b').recompute()

        The result of `input` can be reused.
        """
        self.need_recompute = True
        return self

    def repartition(
        self,
        npartitions: int,
        hash_by: Union[str, List[str], None] = None,
        by: Optional[str] = None,
        by_rows: bool = False,
        **kwargs,
    ) -> DataFrame:
        """
        Repartition the data into the given number of partitions.

        Parameters
        ----------
        npartitions
            The dataset would be split and distributed to `npartitions` partitions.
            If not specified, the number of partitions would be the default partition size of the context.
        hash_by, optional
            If specified, the dataset would be repartitioned by the hash of the given columns.
        by, optional
            If specified, the dataset would be repartitioned by the given column.
        by_rows, optional
            If specified, the dataset would be repartitioned by rows instead of by files.

        Examples
        --------
        .. code-block::

            df = df.repartition(10)                 # evenly distributed
            df = df.repartition(10, by_rows=True)   # evenly distributed by rows
            df = df.repartition(10, hash_by='host') # hash partitioned
            df = df.repartition(10, by='bucket')    # partitioned by column
        """
        if by is not None:
            assert hash_by is None, "cannot specify both by and hash_by"
            plan = ShuffleNode(
                self.session._ctx,
                (self.plan,),
                npartitions,
                data_partition_column=by,
                **kwargs,
            )
        elif hash_by is not None:
            hash_columns = [hash_by] if isinstance(hash_by, str) else hash_by
            plan = HashPartitionNode(
                self.session._ctx, (self.plan,), npartitions, hash_columns, **kwargs
            )
        else:
            plan = EvenlyDistributedPartitionNode(
                self.session._ctx,
                (self.plan,),
                npartitions,
                partition_by_rows=by_rows,
                **kwargs,
            )
        return DataFrame(self.session, plan, recompute=self.need_recompute)

    def random_shuffle(self, **kwargs) -> DataFrame:
        """
        Randomly shuffle all rows globally.
        """

        repartition = HashPartitionNode(
            self.session._ctx,
            (self.plan,),
            self.plan.num_partitions,
            random_shuffle=True,
            **kwargs,
        )
        plan = SqlEngineNode(
            self.session._ctx,
            (repartition,),
            r"select * from {0} order by random()",
            **kwargs,
        )
        return DataFrame(self.session, plan, recompute=self.need_recompute)

    def partial_sort(self, by: Union[str, List[str]], **kwargs) -> DataFrame:
        """
        Sort rows by the given columns in each partition.

        Parameters
        ----------
        by
            A column or a list of columns to sort by.

        Examples
        --------
        .. code-block::

            df = df.partial_sort(by='a')
            df = df.partial_sort(by=['a', 'b desc'])
        """

        by = [by] if isinstance(by, str) else by
        plan = SqlEngineNode(
            self.session._ctx,
            (self.plan,),
            f"select * from {{0}} order by {', '.join(by)}",
            **kwargs,
        )
        return DataFrame(self.session, plan, recompute=self.need_recompute)

    def filter(
        self, sql_or_func: Union[str, Callable[[Dict[str, Any]], bool]], **kwargs
    ) -> DataFrame:
        """
        Filter out rows that don't satisfy the given predicate.

        Parameters
        ----------
        sql_or_func
            A SQL expression or a predicate function.
            For functions, it should take a dictionary of columns as input and returns a boolean.
            SQL expression is preferred as it's more efficient.

        Examples
        --------
        .. code-block::

            df = df.filter('a > 1')
            df = df.filter(lambda r: r['a'] > 1)
        """
        if isinstance(sql := sql_or_func, str):
            plan = SqlEngineNode(
                self.session._ctx,
                (self.plan,),
                f"select * from {{0}} where ({sql})",
                **kwargs,
            )
        elif isinstance(func := sql_or_func, Callable):

            def process_func(_runtime_ctx, tables: List[arrow.Table]) -> arrow.Table:
                table = tables[0]
                return table.filter([func(row) for row in table.to_pylist()])

            plan = ArrowBatchNode(
                self.session._ctx, (self.plan,), process_func=process_func, **kwargs
            )
        else:
            raise ValueError(
                "condition must be a SQL expression or a predicate function"
            )
        return DataFrame(self.session, plan, recompute=self.need_recompute)

    def map(
        self,
        sql_or_func: Union[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
        *,
        schema: Optional[arrow.Schema] = None,
        **kwargs,
    ) -> DataFrame:
        """
        Apply a function to each row.

        Parameters
        ----------
        sql_or_func
            A SQL expression or a function to apply to each row.
            For functions, it should take a dictionary of columns as input and returns a dictionary of columns.
            SQL expression is preferred as it's more efficient.
        schema, optional
            The schema of the output DataFrame.
            If not passed, will be inferred from the first row of the mapping values.
        udfs, optional
            A list of user defined functions to be referenced in the SQL expression.

        Examples
        --------
        .. code-block::

            df = df.map('a, b')
            df = df.map('a + b as c')
            df = df.map(lambda row: {'c': row['a'] + row['b']})


        Use user-defined functions in SQL expression:

        .. code-block::

            @udf(params=[UDFType.INT, UDFType.INT], return_type=UDFType.INT)
            def gcd(a: int, b: int) -> int:
                while b:
                    a, b = b, a % b
                return a
            # load python udf
            df = df.map('gcd(a, b)', udfs=[gcd])

            # load udf from duckdb extension
            df = df.map('gcd(a, b)', udfs=['path/to/udf.duckdb_extension'])

        """
        if isinstance(sql := sql_or_func, str):
            plan = SqlEngineNode(
                self.session._ctx, (self.plan,), f"select {sql} from {{0}}", **kwargs
            )
        elif isinstance(func := sql_or_func, Callable):

            def process_func(_runtime_ctx, tables: List[arrow.Table]) -> arrow.Table:
                output_rows = [func(row) for row in tables[0].to_pylist()]
                return arrow.Table.from_pylist(output_rows, schema=schema)

            plan = ArrowBatchNode(
                self.session._ctx, (self.plan,), process_func=process_func, **kwargs
            )
        else:
            raise ValueError(f"must be a SQL expression or a function: {sql_or_func!r}")
        return DataFrame(self.session, plan, recompute=self.need_recompute)

    def flat_map(
        self,
        sql_or_func: Union[str, Callable[[Dict[str, Any]], List[Dict[str, Any]]]],
        *,
        schema: Optional[arrow.Schema] = None,
        **kwargs,
    ) -> DataFrame:
        """
        Apply a function to each row and flatten the result.

        Parameters
        ----------
        sql_or_func
            A SQL expression or a function to apply to each row.
            For functions, it should take a dictionary of columns as input and returns a list of dictionaries.
            SQL expression is preferred as it's more efficient.
        schema, optional
            The schema of the output DataFrame.
            If not passed, will be inferred from the first row of the mapping values.

        Examples
        --------
        .. code-block::

            df = df.flat_map('unnest(array[a, b]) as c')
            df = df.flat_map(lambda row: [{'c': row['a']}, {'c': row['b']}])
        """
        if isinstance(sql := sql_or_func, str):

            plan = SqlEngineNode(
                self.session._ctx, (self.plan,), f"select {sql} from {{0}}", **kwargs
            )
        elif isinstance(func := sql_or_func, Callable):

            def process_func(_runtime_ctx, tables: List[arrow.Table]) -> arrow.Table:
                output_rows = [
                    item for row in tables[0].to_pylist() for item in func(row)
                ]
                return arrow.Table.from_pylist(output_rows, schema=schema)

            plan = ArrowBatchNode(
                self.session._ctx, (self.plan,), process_func=process_func, **kwargs
            )
        else:
            raise ValueError(f"must be a SQL expression or a function: {sql_or_func!r}")
        return DataFrame(self.session, plan, recompute=self.need_recompute)

    def map_batches(
        self,
        func: Callable[[arrow.Table], arrow.Table],
        *,
        batch_size: int = 122880,
        streaming: bool = False,
        **kwargs,
    ) -> DataFrame:
        """
        Apply the given function to batches of data.

        Parameters
        ----------
        func
            A function or a callable class to apply to each batch of data.
            It should take a `arrow.Table` as input and returns a `arrow.Table`.
        batch_size, optional
            The number of rows in each batch. Defaults to 122880.
        streaming, optional
            If true, the function takes an iterator of `arrow.Table` as input and yields a streaming of `arrow.Table` as output.
            i.e. func: Callable[[Iterator[arrow.Table]], Iterator[arrow.Table]]
            Defaults to false.
        """

        if streaming:

            def process_func(
                _runtime_ctx, readers: List[arrow.RecordBatchReader]
            ) -> Iterator[arrow.Table]:
                tables = map(
                    lambda batch: arrow.Table.from_batches([batch]), readers[0]
                )
                return func(tables)

            plan = ArrowStreamNode(
                self.session._ctx,
                (self.plan,),
                process_func=process_func,
                streaming_batch_size=batch_size,
                **kwargs,
            )
        else:

            def process_func(_runtime_ctx, tables: List[arrow.Table]) -> arrow.Table:
                return func(tables[0])

            plan = ArrowBatchNode(
                self.session._ctx,
                (self.plan,),
                process_func=process_func,
                streaming_batch_size=batch_size,
                **kwargs,
            )
        return DataFrame(self.session, plan, recompute=self.need_recompute)

    def limit(self, limit: int) -> DataFrame:
        """
        Limit the number of rows to the given number.

        Unlike `take`, this method doesn't trigger execution.
        """
        plan = LimitNode(self.session._ctx, self.plan, limit)
        return DataFrame(self.session, plan, recompute=self.need_recompute)

    def write_parquet(self, path: str) -> None:
        """
        Write data to a series of parquet files under the given path.

        This is a blocking operation. See :func:`write_parquet_lazy` for a non-blocking version.

        Examples
        --------
        .. code-block::

            df.write_parquet('output')
        """
        self.write_parquet_lazy(path).compute()

    def write_parquet_lazy(self, path: str) -> DataFrame:
        """
        Write data to a series of parquet files under the given path.

        This is a non-blocking operation. See :func:`write_parquet` for a blocking version.

        Examples
        --------
        .. code-block::

            o1 = df.write_parquet_lazy('output1')
            o2 = df.write_parquet_lazy('output2')
            sp.wait(o1, o2)
        """

        plan = DataSinkNode(
            self.session._ctx, (self.plan,), os.path.abspath(path), type="link_or_copy"
        )
        return DataFrame(self.session, plan, recompute=self.need_recompute)

    # inspection

    def count(self) -> int:
        """
        Count the number of rows.

        If this dataframe consists of more than a read, or if the row count can't be determined from
        the metadata provided by the datasource, then this operation will trigger execution of the
        lazy transformations performed on this dataframe.
        """
        datasets = self._compute()
        # FIXME: don't use ThreadPoolExecutor because duckdb results will be mixed up
        return sum(dataset.num_rows for dataset in datasets)

    def take(self, limit: int) -> List[Dict[str, Any]]:
        """
        Return up to `limit` rows.

        This operation will trigger execution of the lazy transformations performed on this DataFrame.
        """
        if self.is_computed() or isinstance(self.plan, DataSourceNode):
            datasets = self._compute()
        else:
            datasets = self.limit(limit)._compute()
        rows = []
        for dataset in datasets:
            for batch in dataset.to_batch_reader():
                rows.extend(batch.to_pylist())
                if len(rows) >= limit:
                    return rows[:limit]
        return rows

    def take_all(self) -> List[Dict[str, Any]]:
        """
        Return all rows.

        This operation will trigger execution of the lazy transformations performed on this DataFrame.
        """
        datasets = self._compute()
        rows = []
        for dataset in datasets:
            for batch in dataset.to_batch_reader():
                rows.extend(batch.to_pylist())
        return rows

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame.

        This operation will trigger execution of the lazy transformations performed on this DataFrame.
        """
        datasets = self._compute()
        with ThreadPoolExecutor() as pool:
            return pd.concat(pool.map(lambda dataset: dataset.to_pandas(), datasets))

    def to_arrow(self) -> arrow.Table:
        """
        Convert to an arrow Table.

        This operation will trigger execution of the lazy transformations performed on this DataFrame.
        """
        datasets = self._compute()
        with ThreadPoolExecutor() as pool:
            return arrow.concat_tables(
                pool.map(lambda dataset: dataset.to_arrow_table(), datasets)
            )
