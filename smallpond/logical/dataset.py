import copy
import functools
import glob
import os.path
import random
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Union

import duckdb
import fsspec
import pandas as pd
import pyarrow as arrow
import pyarrow.parquet as parquet
from loguru import logger

from smallpond.common import (
    DEFAULT_BATCH_SIZE,
    GB,
    PARQUET_METADATA_KEY_PREFIX,
    split_into_rows,
)
from smallpond.io.arrow import (
    RowRange,
    build_batch_reader_from_files,
    dump_to_parquet_files,
    load_from_parquet_files,
)
from smallpond.logical.udf import UDFContext

magic_check = re.compile(r"([*?]|\[.*\])")
magic_check_bytes = re.compile(rb"([*?]|\[.*\])")


def has_magic(s):
    if isinstance(s, bytes):
        match = magic_check_bytes.search(s)
    else:
        match = magic_check.search(s)
    return match is not None


class DataSet(object):
    """
    The base class for all datasets.
    """

    __slots__ = (
        "paths",
        "root_dir",
        "recursive",
        "columns",
        "__dict__",
        "_union_by_name",
        "_resolved_paths",
        "_absolute_paths",
        "_resolved_num_rows",
    )

    def __init__(
        self,
        paths: Union[str, List[str]],
        root_dir: Optional[str] = "",
        recursive=False,
        columns: Optional[List[str]] = None,
        union_by_name=False,
    ) -> None:
        """
        Construct a dataset from a list of paths.

        Parameters
        ----------
        paths
            A path or a list of paths or path patterns.
            e.g. `['data/100.parquet', '/datasetA/*.parquet']`.
        root_dir, optional
            Relative paths in `paths` would be resolved under `root_dir` if specified.
        recursive, optional
            Resolve path patterns recursively if true.
        columns, optional
            Only load the specified columns if not None.
        union_by_name, optional
            Unify the columns of different files by name (see https://duckdb.org/docs/data/multiple_files/combining_schemas#union-by-name).
        """
        self.paths = [paths] if isinstance(paths, str) else paths
        "The paths to the dataset files."
        self.root_dir = os.path.abspath(root_dir) if root_dir is not None else None
        "The root directory of paths."
        self.recursive = recursive
        "Whether to resolve path patterns recursively."
        self.columns = columns
        "The columns to load from the dataset files."
        self._union_by_name = union_by_name
        self._resolved_paths: List[str] = None
        self._absolute_paths: List[str] = None
        self._resolved_num_rows: int = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: paths[{len(self.paths)}]={self.paths[:3]}...', root_dir={self.root_dir}, columns={self.columns}"

    __repr__ = __str__

    @property
    def _resolved_path_str(self) -> str:
        return ", ".join(map(lambda x: f"'{x}'", self.resolved_paths))

    @property
    def _column_str(self) -> str:
        """
        A column string used in SQL select clause.
        """
        return ", ".join(self.columns) if self.columns else "*"

    @property
    def union_by_name(self) -> bool:
        """
        Whether to unify the columns of different files by name.
        """
        return self._union_by_name or self.columns is not None

    @property
    def udfs(self) -> List[UDFContext]:
        return []

    @staticmethod
    def merge(datasets: "List[DataSet]") -> "DataSet":
        """
        Merge multiple datasets into a single dataset.
        """
        raise NotImplementedError

    def reset(
        self,
        paths: Optional[List[str]] = None,
        root_dir: Optional[str] = "",
        recursive=None,
    ) -> None:
        """
        Reset the dataset with new paths, root_dir, and recursive flag.
        """
        self.partition_by_files.cache_clear()
        self.paths = paths or []
        self.root_dir = os.path.abspath(root_dir) if root_dir is not None else None
        self.recursive = recursive if recursive is not None else self.recursive
        self._resolved_paths = None
        self._absolute_paths = None
        self._resolved_num_rows = None

    @property
    def num_files(self) -> int:
        """
        The number of files in the dataset.
        """
        return len(self.resolved_paths)

    @property
    def num_rows(self) -> int:
        """
        The number of rows in the dataset.
        """
        if self._resolved_num_rows is None:
            sql_query = f"select count(*) from {self.sql_query_fragment()}"
            row = duckdb.sql(sql_query).fetchone()
            assert row is not None, "no rows returned"
            self._resolved_num_rows = row[0]
        return self._resolved_num_rows

    @property
    def empty(self) -> bool:
        """
        Whether the dataset is empty.
        """
        if self._resolved_paths is not None:
            return len(self._resolved_paths) == 0
        for path in self.paths:
            if has_magic(path):
                if any(
                    glob.iglob(
                        os.path.join(self.root_dir or "", path),
                        recursive=self.recursive,
                    )
                ):
                    return False
            else:
                return False
        return True

    @property
    def resolved_paths(self) -> List[str]:
        """
        An ordered list of absolute paths of files.
        File patterns are expanded to absolute paths.

        Example::
        >>> DataSet(['data/100.parquet', '/datasetA/*.parquet']).resolved_paths
        ['/datasetA/1.parquet', '/datasetA/2.parquet', '/home/user/data/100.parquet']
        """
        if self._resolved_paths is None:
            resolved_paths = []
            wildcard_paths = []
            for path in self.absolute_paths:
                if has_magic(path):
                    wildcard_paths.append(path)
                else:
                    resolved_paths.append(path)
            if wildcard_paths:
                if len(wildcard_paths) == 1:
                    expanded_paths = glob.glob(wildcard_paths[0], recursive=self.recursive)
                else:
                    logger.debug(
                        "resolving {} paths with wildcards in {}",
                        len(wildcard_paths),
                        self,
                    )
                    with ThreadPoolExecutor(min(32, len(wildcard_paths))) as pool:
                        expanded_paths = [
                            p
                            for paths in pool.map(
                                lambda p: glob.glob(p, recursive=self.recursive),
                                wildcard_paths,
                            )
                            for p in paths
                        ]
                resolved_paths.extend(expanded_paths)
                logger.debug(
                    "resolved {} files from {} wildcard path(s) in {}",
                    len(expanded_paths),
                    len(wildcard_paths),
                    self,
                )
            self._resolved_paths = sorted(resolved_paths)
        return self._resolved_paths

    @property
    def absolute_paths(self) -> List[str]:
        """
        An ordered list of absolute paths of the given file patterns.

        Example::
        >>> DataSet(['data/100.parquet', '/datasetA/*.parquet']).absolute_paths
        ['/datasetA/*.parquet', '/home/user/data/100.parquet']
        """
        if self._absolute_paths is None:
            if self.root_dir is None:
                self._absolute_paths = sorted(self.paths)
            else:
                self._absolute_paths = [os.path.join(self.root_dir, p) for p in sorted(self.paths)]
        return self._absolute_paths

    def sql_query_fragment(
        self,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> str:
        """
        Return a sql fragment that represents the dataset.
        """
        raise NotImplementedError

    def log(self, num_rows=200):
        """
        Log the dataset to the logger.
        """
        import pandas as pd

        pd.set_option("display.max_columns", None)  # Show all columns
        pd.set_option("display.max_rows", None)  # Optionally show all rows
        pd.set_option("display.max_colwidth", None)  # No truncation of column contents
        pd.set_option("display.expand_frame_repr", False)  # Do not wrap rows
        logger.info("{} ->\n{}", self, self.to_pandas().head(num_rows))

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the dataset to a pandas dataframe.
        """
        return self.to_arrow_table().to_pandas()

    def to_arrow_table(
        self,
        max_workers: int = 16,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> arrow.Table:
        """
        Load the dataset to an arrow table.

        Parameters
        ----------
        max_workers, optional
            The maximum number of worker threads to use. Default to 16.
        filesystem, optional
            If provided, use the filesystem to load the dataset.
        conn, optional
            A duckdb connection. If provided, use duckdb to load the dataset.
        """
        sql_query = f"select {self._column_str} from {self.sql_query_fragment(filesystem, conn)}"
        if conn is not None:
            return conn.sql(sql_query).fetch_arrow_table()
        else:
            return duckdb.sql(sql_query).fetch_arrow_table()

    def to_batch_reader(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> arrow.RecordBatchReader:
        """
        Return an arrow record batch reader to read the dataset.

        Parameters
        ----------
        batch_size, optional
            The record batch size. Default to 122880.
        filesystem, optional
            If provided, use the filesystem to load the dataset.
        conn, optional
            A duckdb connection. If provided, use duckdb to load the dataset.
        """
        sql_query = f"select {self._column_str} from {self.sql_query_fragment(filesystem, conn)}"
        if conn is not None:
            return conn.sql(sql_query).fetch_arrow_reader(batch_size)
        else:
            return duckdb.sql(sql_query).fetch_arrow_reader(batch_size)

    def _init_file_partitions(self, npartition: int) -> "List[DataSet]":
        """
        Return `npartition` empty datasets.
        """
        file_partitions = []
        for _ in range(npartition):
            empty_dataset = copy.copy(self)
            empty_dataset.reset()
            file_partitions.append(empty_dataset)
        return file_partitions

    @functools.lru_cache
    def partition_by_files(self, npartition: int, random_shuffle: bool = False) -> "List[DataSet]":
        """
        Evenly split into `npartition` datasets by files.
        """
        assert npartition > 0, f"npartition has negative value: {npartition}"
        if npartition > self.num_files:
            logger.debug(f"number of partitions {npartition} is greater than the number of files {self.num_files}")

        resolved_paths = random.sample(self.resolved_paths, len(self.resolved_paths)) if random_shuffle else self.resolved_paths
        evenly_split_groups = split_into_rows(resolved_paths, npartition)
        num_paths_in_groups = list(map(len, evenly_split_groups))

        file_partitions = self._init_file_partitions(npartition)
        for i, paths in enumerate(evenly_split_groups):
            file_partitions[i].reset(paths, None)

        logger.debug(
            f"created {npartition} file partitions (min #files: {min(num_paths_in_groups)}, max #files: {max(num_paths_in_groups)}, avg #files: {sum(num_paths_in_groups)/len(num_paths_in_groups):.3f}) from {self}"
        )
        return random.sample(file_partitions, len(file_partitions)) if random_shuffle else file_partitions


class PartitionedDataSet(DataSet):
    """
    A dataset that is partitioned into multiple datasets.
    """

    __slots__ = ("datasets",)

    def __init__(self, datasets: List[DataSet]) -> None:
        assert len(datasets) > 0, "no dataset given"
        self.datasets = datasets
        absolute_paths = [p for dataset in datasets for p in dataset.absolute_paths]
        super().__init__(
            absolute_paths,
            datasets[0].root_dir,
            datasets[0].recursive,
            datasets[0].columns,
            datasets[0].union_by_name,
        )

    def __getitem__(self, key: int) -> DataSet:
        return self.datasets[key]

    @property
    def udfs(self) -> List[UDFContext]:
        return [udf for dataset in self.datasets for udf in dataset.udfs]

    @staticmethod
    def merge(datasets: "List[PartitionedDataSet]") -> DataSet:
        # merge partitioned datasets results in an unpartitioned dataset
        assert all(isinstance(dataset, PartitionedDataSet) for dataset in datasets)
        datasets = [d for dataset in datasets for d in dataset]
        return datasets[0].merge(datasets)


class FileSet(DataSet):
    """
    A set of files.
    """

    @staticmethod
    def merge(datasets: "List[FileSet]") -> "FileSet":
        assert all(isinstance(dataset, FileSet) for dataset in datasets)
        absolute_paths = [p for dataset in datasets for p in dataset.absolute_paths]
        return FileSet(absolute_paths)

    def to_arrow_table(
        self,
        max_workers=16,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> arrow.Table:
        return arrow.table([self.resolved_paths], names=["resolved_paths"])

    def to_batch_reader(
        self,
        batch_size=DEFAULT_BATCH_SIZE,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> arrow.RecordBatchReader:
        return self.to_arrow_table().to_reader(batch_size)


class CsvDataSet(DataSet):
    """
    A set of csv files.
    """

    __slots__ = (
        "schema",
        "delim",
        "max_line_size",
        "parallel",
        "header",
    )

    def __init__(
        self,
        paths: List[str],
        schema: Dict[str, str],
        delim=",",
        max_line_size: Optional[int] = None,
        parallel=True,
        header=False,
        root_dir: Optional[str] = "",
        recursive=False,
        columns: Optional[List[str]] = None,
        union_by_name=False,
    ) -> None:
        super().__init__(paths, root_dir, recursive, columns, union_by_name)
        assert isinstance(schema, OrderedDict), f"type of csv schema is not OrderedDict: {type(schema)}"
        self.schema = schema
        self.delim = delim
        self.max_line_size = max_line_size
        self.parallel = parallel
        self.header = header

    @staticmethod
    def merge(datasets: "List[CsvDataSet]") -> "CsvDataSet":
        assert all(isinstance(dataset, CsvDataSet) for dataset in datasets)
        absolute_paths = [p for dataset in datasets for p in dataset.absolute_paths]
        return CsvDataSet(
            absolute_paths,
            datasets[0].schema,
            datasets[0].delim,
            datasets[0].max_line_size,
            datasets[0].parallel,
            recursive=any(dataset.recursive for dataset in datasets),
            columns=datasets[0].columns,
            union_by_name=any(dataset.union_by_name for dataset in datasets),
        )

    def sql_query_fragment(
        self,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> str:
        schema_str = ", ".join(map(lambda x: f"'{x[0]}': '{x[1]}'", self.schema.items()))
        max_line_size_str = f"max_line_size={self.max_line_size}, " if self.max_line_size is not None else ""
        return (
            f"( select {self._column_str} from read_csv([ {self._resolved_path_str} ], delim='{self.delim}', columns={{ {schema_str} }}, header={self.header}, "
            f"{max_line_size_str} parallel={self.parallel}, union_by_name={self.union_by_name}) )"
        )


class JsonDataSet(DataSet):
    """
    A set of json files.
    """

    __slots__ = (
        "schema",
        "format",
        "max_object_size",
    )

    def __init__(
        self,
        paths: List[str],
        schema: Dict[str, str],
        format="newline_delimited",
        max_object_size=1 * GB,
        root_dir: Optional[str] = "",
        recursive=False,
        columns: Optional[List[str]] = None,
        union_by_name=False,
    ) -> None:
        super().__init__(paths, root_dir, recursive, columns, union_by_name)
        self.schema = schema
        self.format = format
        self.max_object_size = max_object_size

    @staticmethod
    def merge(datasets: "List[JsonDataSet]") -> "JsonDataSet":
        assert all(isinstance(dataset, JsonDataSet) for dataset in datasets)
        absolute_paths = [p for dataset in datasets for p in dataset.absolute_paths]
        return JsonDataSet(
            absolute_paths,
            datasets[0].schema,
            datasets[0].format,
            datasets[0].max_object_size,
            recursive=any(dataset.recursive for dataset in datasets),
            columns=datasets[0].columns,
            union_by_name=any(dataset.union_by_name for dataset in datasets),
        )

    def sql_query_fragment(
        self,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> str:
        schema_str = ", ".join(map(lambda x: f"'{x[0]}': '{x[1]}'", self.schema.items()))
        return (
            f"( select {self._column_str} from read_json([ {self._resolved_path_str} ], format='{self.format}', columns={{ {schema_str} }}, "
            f"maximum_object_size={self.max_object_size}, union_by_name={self.union_by_name}) )"
        )


class ParquetDataSet(DataSet):
    """
    A set of parquet files.
    """

    __slots__ = (
        "generated_columns",
        "_resolved_row_ranges",
    )

    def __init__(
        self,
        paths: List[str],
        root_dir: Optional[str] = "",
        recursive=False,
        columns: Optional[List[str]] = None,
        generated_columns: Optional[List[str]] = None,
        union_by_name=False,
    ) -> None:
        super().__init__(paths, root_dir, recursive, columns, union_by_name)
        self.generated_columns = generated_columns or []
        "Generated columns of DuckDB `read_parquet` function. e.g. `file_name`, `file_row_number`."
        self._resolved_row_ranges: List[RowRange] = None

    def __str__(self) -> str:
        s = super().__str__() + f", generated_columns={self.generated_columns}"
        if self._resolved_row_ranges:
            s += f", resolved_row_ranges[{len(self._resolved_row_ranges)}]={self._resolved_row_ranges[:3]}..."
        return s

    __repr__ = __str__

    @property
    def _column_str(self) -> str:
        if not self.columns:
            return "*"
        if "*" in self.columns:
            return ", ".join(self.columns)
        return ", ".join(self.columns + self.generated_columns)

    @staticmethod
    def merge(datasets: "List[ParquetDataSet]") -> "ParquetDataSet":
        assert all(isinstance(dataset, ParquetDataSet) for dataset in datasets)
        dataset = ParquetDataSet(
            paths=[p for dataset in datasets for p in dataset.absolute_paths],
            root_dir=None,
            recursive=any(dataset.recursive for dataset in datasets),
            columns=datasets[0].columns,
            generated_columns=datasets[0].generated_columns,
            union_by_name=any(dataset.union_by_name for dataset in datasets),
        )
        # merge row ranges if any dataset has resolved row ranges
        if any(dataset._resolved_row_ranges is not None for dataset in datasets):
            dataset._resolved_row_ranges = [row_range for dataset in datasets for row_range in dataset.resolved_row_ranges]
        return dataset

    @staticmethod
    def create_from(table: arrow.Table, output_dir: str, filename: str = "data"):
        dump_to_parquet_files(table, output_dir, filename)
        return ParquetDataSet([os.path.join(output_dir, f"{filename}*.parquet")])

    def reset(
        self,
        paths: Optional[List[str]] = None,
        root_dir: Optional[str] = "",
        recursive=None,
    ) -> None:
        """
        NOTE: all row ranges will be cleared. DO NOT call this if you want to keep partial files.
        """
        super().reset(paths, root_dir, recursive)
        self._resolved_row_ranges = None
        self.partition_by_files.cache_clear()
        self.partition_by_rows.cache_clear()
        self.partition_by_size.cache_clear()

    @property
    def resolved_row_ranges(self) -> List[RowRange]:
        """
        Return row ranges for each parquet file.
        """
        if self._resolved_row_ranges is None:
            if len(self.resolved_paths) == 0:
                self._resolved_row_ranges = []
            else:

                def resolve_row_range(path: str) -> RowRange:
                    # read parquet metadata to get number of rows
                    metadata = parquet.read_metadata(path)
                    num_rows = metadata.num_rows
                    uncompressed_data_size = sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))
                    return RowRange(
                        path,
                        data_size=uncompressed_data_size,
                        file_num_rows=num_rows,
                        begin=0,
                        end=num_rows,
                    )

                with ThreadPoolExecutor(max_workers=min(32, len(self.resolved_paths))) as pool:
                    self._resolved_row_ranges = list(pool.map(resolve_row_range, self.resolved_paths))
        return self._resolved_row_ranges

    @property
    def num_rows(self) -> int:
        if self._resolved_num_rows is None:
            self._resolved_num_rows = sum(row_range.num_rows for row_range in self.resolved_row_ranges)
        return self._resolved_num_rows

    @property
    def empty(self) -> bool:
        # this method should be quick. do not resolve row ranges.
        if self._resolved_num_rows is not None or self._resolved_row_ranges is not None:
            return self.num_rows == 0
        return super().empty

    @property
    def estimated_data_size(self) -> int:
        """
        Return the estimated data size in bytes.
        """
        return sum(row_range.estimated_data_size for row_range in self.resolved_row_ranges)

    def sql_query_fragment(
        self,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> str:
        extra_parameters = "".join(f", {col}=true" for col in self.generated_columns) if self.generated_columns else ""
        parquet_file_queries = []
        full_row_ranges = []

        for row_range in self.resolved_row_ranges:
            path = filesystem.unstrip_protocol(row_range.path) if filesystem else row_range.path
            if row_range.num_rows == row_range.file_num_rows:
                full_row_ranges.append(row_range)
            else:
                sql_query = f"""
          select {self._column_str}
          from read_parquet('{path}' {extra_parameters}, file_row_number=true)
          where file_row_number between {row_range.begin} and {row_range.end - 1}
        """
                if "file_row_number" not in self.generated_columns:
                    sql_query = f"select columns(c -> c != 'file_row_number') from ( {sql_query} )"
                parquet_file_queries.append(sql_query)

        # NOTE: prefer:     read_parquet([path1, path2, ...])
        #       instead of: read_parquet(path1) union all read_parquet(path2) union all ...
        #       for performance
        if full_row_ranges:
            # XXX: duckdb uses the first file as the estimated cardinality of `read_parquet`
            #      to prevent incorrect estimation, we move the largest file to the head
            largest_index = max(
                range(len(full_row_ranges)),
                key=lambda i: full_row_ranges[i].file_num_rows,
            )
            full_row_ranges[0], full_row_ranges[largest_index] = (
                full_row_ranges[largest_index],
                full_row_ranges[0],
            )
            parquet_file_str = ",\n            ".join(map(lambda x: f"'{x.path}'", full_row_ranges))
            parquet_file_queries.insert(
                0,
                f"""
          select {self._column_str}
          from read_parquet([
            {parquet_file_str}
          ], union_by_name={self.union_by_name} {extra_parameters})
      """,
            )

        union_op = " union all by name " if self.union_by_name else " union all "
        return f"( {union_op.join(parquet_file_queries)} )"

    def to_arrow_table(
        self,
        max_workers=16,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> arrow.Table:
        if conn is not None:
            return super().to_arrow_table(max_workers, filesystem, conn)

        tables = []
        if self.resolved_row_ranges:
            tables.append(load_from_parquet_files(self.resolved_row_ranges, self.columns, max_workers, filesystem))
        return arrow.concat_tables(tables)

    def to_batch_reader(
        self,
        batch_size=DEFAULT_BATCH_SIZE,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> arrow.RecordBatchReader:
        if conn is not None:
            return super().to_batch_reader(batch_size, filesystem, conn)

        return build_batch_reader_from_files(
            self.resolved_row_ranges,
            columns=self.columns,
            batch_size=batch_size,
            filesystem=filesystem,
        )

    @functools.lru_cache
    def partition_by_files(self, npartition: int, random_shuffle: bool = False) -> "List[ParquetDataSet]":
        if self._resolved_row_ranges is not None:
            return self.partition_by_rows(npartition, random_shuffle)
        else:
            return super().partition_by_files(npartition, random_shuffle)

    @functools.lru_cache
    def partition_by_rows(self, npartition: int, random_shuffle: bool = False) -> "List[ParquetDataSet]":
        """
        Evenly split the dataset into `npartition` partitions by rows.
        If `random_shuffle` is True, shuffle the files before partitioning.
        """
        assert npartition > 0, f"npartition has negative value: {npartition}"

        resolved_row_ranges = self.resolved_row_ranges
        resolved_row_ranges = random.sample(resolved_row_ranges, len(resolved_row_ranges)) if random_shuffle else resolved_row_ranges

        def create_dataset(row_ranges: List[RowRange]) -> ParquetDataSet:
            row_ranges = sorted(row_ranges, key=lambda x: x.path)
            resolved_paths = [x.path for x in row_ranges]
            dataset = ParquetDataSet(
                resolved_paths,
                columns=self.columns,
                generated_columns=self.generated_columns,
                union_by_name=self.union_by_name,
            )
            dataset._resolved_paths = resolved_paths
            dataset._resolved_row_ranges = row_ranges
            return dataset

        return [create_dataset(row_ranges) for row_ranges in RowRange.partition_by_rows(resolved_row_ranges, npartition)]

    @functools.lru_cache
    def partition_by_size(self, max_partition_size: int) -> "List[ParquetDataSet]":
        """
        Split the dataset into multiple partitions so that each partition has at most `max_partition_size` bytes.
        """
        if self.empty:
            return []
        estimated_data_size = sum(row_range.estimated_data_size for row_range in self.resolved_row_ranges)
        npartition = estimated_data_size // max_partition_size + 1
        return self.partition_by_rows(npartition)

    @staticmethod
    def _read_partition_key(path: str, data_partition_column: str, hive_partitioning: bool) -> int:
        """
        Get the partition key of the parquet file.

        Examples
        --------
        ```
        >>> ParquetDataSet._read_partition_key("output/000.parquet", "key", hive_partitioning=False)
        1
        >>> ParquetDataSet._read_partition_key("output/key=1/000.parquet", "key", hive_partitioning=True)
        1
        ```
        """

        def parse_partition_key(key: str):
            try:
                return int(key)
            except ValueError:
                logger.error(f"cannot parse partition key '{data_partition_column}' of {path} from: {key}")
                raise

        if hive_partitioning:
            path_part_prefix = data_partition_column + "="
            for part in path.split(os.path.sep):
                if part.startswith(path_part_prefix):
                    return parse_partition_key(part[len(path_part_prefix) :])
            raise RuntimeError(f"cannot extract hive partition key '{data_partition_column}' from path: {path}")

        with parquet.ParquetFile(path) as file:
            kv_metadata = file.schema_arrow.metadata or file.metadata.metadata
            if kv_metadata is not None:
                for key, val in kv_metadata.items():
                    key, val = key.decode("utf-8"), val.decode("utf-8")
                    if key == PARQUET_METADATA_KEY_PREFIX + data_partition_column:
                        return parse_partition_key(val)
            if file.metadata.num_rows == 0:
                logger.warning(f"cannot read partition keys from empty parquet file: {path}")
                return None
            for batch in file.iter_batches(batch_size=128, columns=[data_partition_column], use_threads=False):
                assert data_partition_column in batch.column_names, f"cannot find column '{data_partition_column}' in {batch.column_names}"
                assert batch.num_columns == 1, f"unexpected num of columns: {batch.column_names}"
                uniq_partition_keys = set(batch.columns[0].to_pylist())
                assert uniq_partition_keys and len(uniq_partition_keys) == 1, f"partition keys found in {path} not unique: {uniq_partition_keys}"
                return uniq_partition_keys.pop()

    def load_partitioned_datasets(self, npartition: int, data_partition_column: str, hive_partitioning=False) -> "List[ParquetDataSet]":
        """
        Split the dataset into a list of partitioned datasets.
        """
        assert npartition > 0, f"npartition has negative value: {npartition}"
        if npartition > self.num_files:
            logger.debug(f"number of partitions {npartition} is greater than the number of files {self.num_files}")

        file_partitions: List[ParquetDataSet] = self._init_file_partitions(npartition)
        for dataset in file_partitions:
            # elements will be appended later
            dataset._absolute_paths = []
            dataset._resolved_paths = []
            dataset._resolved_row_ranges = []

        if not self.resolved_paths:
            logger.debug(f"create {npartition} empty data partitions from {self}")
            return file_partitions

        with ThreadPoolExecutor(min(32, len(self.resolved_paths))) as pool:
            partition_keys = pool.map(
                lambda path: ParquetDataSet._read_partition_key(path, data_partition_column, hive_partitioning),
                self.resolved_paths,
            )

        for row_range, partition_key in zip(self.resolved_row_ranges, partition_keys):
            if partition_key is not None:
                assert 0 <= partition_key <= npartition, f"invalid partition key {partition_key} found in {row_range.path}"
                dataset = file_partitions[partition_key]
                dataset.paths.append(row_range.path)
                dataset._absolute_paths.append(row_range.path)
                dataset._resolved_paths.append(row_range.path)
                dataset._resolved_row_ranges.append(row_range)

        logger.debug(f"loaded {npartition} data partitions from {self}")
        return file_partitions

    def remove_empty_files(self) -> None:
        """
        Remove empty parquet files from the dataset.
        """
        new_row_ranges = [row_range for row_range in self.resolved_row_ranges if row_range.num_rows > 0]
        if len(new_row_ranges) == 0:
            # keep at least one file to avoid empty dataset
            new_row_ranges = self.resolved_row_ranges[:1]
        if len(new_row_ranges) == len(self.resolved_row_ranges):
            # no empty files found
            return
        logger.info(f"removed {len(self.resolved_row_ranges) - len(new_row_ranges)}/{len(self.resolved_row_ranges)} empty parquet files from {self}")
        self._resolved_row_ranges = new_row_ranges
        self._resolved_paths = [row_range.path for row_range in new_row_ranges]
        self._absolute_paths = self._resolved_paths
        self.paths = self._resolved_paths


class SqlQueryDataSet(DataSet):
    """
    The result of a sql query.
    """

    __slots__ = (
        "sql_query",
        "query_builder",
    )

    def __init__(
        self,
        sql_query: str,
        query_builder: Callable[[duckdb.DuckDBPyConnection, fsspec.AbstractFileSystem], str] = None,
    ) -> None:
        super().__init__([])
        self.sql_query = sql_query
        self.query_builder = query_builder

    @property
    def num_rows(self) -> int:
        num_rows = duckdb.sql(f"select count(*) as num_rows from {self.sql_query_fragment()}").fetchall()
        return num_rows[0][0]

    def sql_query_fragment(
        self,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> str:
        sql_query = self.sql_query if self.query_builder is None else self.query_builder(conn, filesystem)
        return f"( {sql_query} )"


class ArrowTableDataSet(DataSet):
    """
    An arrow table.
    """

    def __init__(self, table: arrow.Table) -> None:
        super().__init__([])
        self.table = copy.deepcopy(table)

    def to_arrow_table(
        self,
        max_workers=16,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> arrow.Table:
        return self.table

    def to_batch_reader(
        self,
        batch_size=DEFAULT_BATCH_SIZE,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> arrow.RecordBatchReader:
        return self.table.to_reader(batch_size)

    def sql_query_fragment(
        self,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> str:
        name = f"arrow_table_{id(self.table)}"
        self.table.to_pandas().to_sql(name, conn, index=False)
        return f"( select * from {name} )"


class PandasDataSet(DataSet):
    """
    A pandas dataframe.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__([])
        self.df = df

    def to_pandas(self) -> pd.DataFrame:
        return self.df

    def to_arrow_table(
        self,
        max_workers=16,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> arrow.Table:
        return arrow.Table.from_pandas(self.df)

    def to_batch_reader(
        self,
        batch_size=DEFAULT_BATCH_SIZE,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> arrow.RecordBatchReader:
        return self.to_arrow_table().to_reader(batch_size)

    def sql_query_fragment(
        self,
        filesystem: fsspec.AbstractFileSystem = None,
        conn: duckdb.DuckDBPyConnection = None,
    ) -> str:
        name = f"pandas_table_{id(self.df)}"
        self.df.to_sql(name, conn, index=False)
        return f"( select * from {name} )"
