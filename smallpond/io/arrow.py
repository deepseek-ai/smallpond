import copy
import math
import os.path
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import fsspec
import pyarrow as arrow
import pyarrow.parquet as parquet
from loguru import logger

from smallpond.common import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ROW_GROUP_BYTES,
    DEFAULT_ROW_GROUP_SIZE,
    MAX_PARQUET_FILE_BYTES,
    MB,
    split_into_rows,
)


@dataclass
class RowRange:
    """A range of rows in a file."""

    path: str
    """Path to the file."""
    data_size: int
    """The uncompressed data size in bytes."""
    file_num_rows: int
    """The number of rows in the file."""
    begin: int
    """Index of first row in the file."""
    end: int
    """Index of last row + 1 in the file."""

    @property
    def num_rows(self) -> int:
        """The number of rows in the range."""
        return self.end - self.begin

    @property
    def estimated_data_size(self) -> int:
        """The estimated uncompressed data size in bytes."""
        return self.data_size * self.num_rows // self.file_num_rows if self.file_num_rows > 0 else 0

    def take(self, num_rows: int) -> "RowRange":
        """
        Take `num_rows` rows from the range.
        NOTE: this function modifies the current row range.
        """
        num_rows = min(num_rows, self.num_rows)
        head = copy.copy(self)
        head.end = head.begin + num_rows
        self.begin += num_rows
        return head

    @staticmethod
    def partition_by_rows(row_ranges: List["RowRange"], npartition: int) -> List[List["RowRange"]]:
        """Evenly split a list of row ranges into `npartition` partitions."""
        # NOTE: `row_ranges` should not be modified by this function
        row_ranges = copy.deepcopy(row_ranges)
        num_rows: int = sum(row_range.num_rows for row_range in row_ranges)
        num_partitions: int = npartition
        row_range_partitions: List[List[RowRange]] = []
        while num_partitions:
            rows_in_partition = (num_rows + num_partitions - 1) // num_partitions
            num_rows -= rows_in_partition
            num_partitions -= 1
            row_ranges_in_partition = []
            while rows_in_partition:
                current_range = row_ranges[0]
                if current_range.num_rows == 0:
                    row_ranges.pop(0)
                    continue
                taken_range = current_range.take(rows_in_partition)
                row_ranges_in_partition.append(taken_range)
                rows_in_partition -= taken_range.num_rows
            row_range_partitions.append(row_ranges_in_partition)
        assert num_rows == 0 and num_partitions == 0
        return row_range_partitions


def convert_type_to_large(type_: arrow.DataType) -> arrow.DataType:
    """
    Convert all string and binary types to large types recursively.
    """
    # Since arrow uses 32-bit signed offsets for string and binary types, convert all string and binary columns
    # to large_string and large_binary to avoid offset overflow, see https://issues.apache.org/jira/browse/ARROW-17828.
    if arrow.types.is_string(type_):
        return arrow.large_string()
    elif arrow.types.is_binary(type_):
        return arrow.large_binary()
    elif isinstance(type_, arrow.ListType):
        return arrow.list_(convert_type_to_large(type_.value_type))
    elif isinstance(type_, arrow.StructType):
        return arrow.struct(
            [
                arrow.field(
                    field.name,
                    convert_type_to_large(field.type),
                    nullable=field.nullable,
                )
                for field in type_
            ]
        )
    elif isinstance(type_, arrow.MapType):
        return arrow.map_(
            convert_type_to_large(type_.key_type),
            convert_type_to_large(type_.item_type),
        )
    else:
        return type_


def convert_types_to_large_string(schema: arrow.Schema) -> arrow.Schema:
    """
    Convert all string and binary types to large types in the schema.
    """
    new_fields = []
    for field in schema:
        new_type = convert_type_to_large(field.type)
        new_field = arrow.field(field.name, new_type, nullable=field.nullable, metadata=field.metadata)
        new_fields.append(new_field)
    return arrow.schema(new_fields, metadata=schema.metadata)


def cast_columns_to_large_string(table: arrow.Table) -> arrow.Table:
    schema = convert_types_to_large_string(table.schema)
    return table.cast(schema)


def filter_schema(
    schema: arrow.Schema,
    included_cols: Optional[List[str]] = None,
    excluded_cols: Optional[List[str]] = None,
):
    assert included_cols is None or excluded_cols is None
    if included_cols is None and excluded_cols is None:
        return schema
    if included_cols is not None:
        fields = [schema.field(col_name) for col_name in included_cols]
    if excluded_cols is not None:
        fields = [schema.field(col_name) for col_name in schema.names if col_name not in excluded_cols]
    return arrow.schema(fields, metadata=schema.metadata)


def _iter_record_batches(
    file: parquet.ParquetFile,
    columns: List[str],
    offset: int,
    length: int,
    batch_size: int,
) -> Iterable[arrow.RecordBatch]:
    """
    Read record batches from a range of a parquet file.
    """
    current_offset = 0
    required_l, required_r = offset, offset + length

    for batch in file.iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
        current_l, current_r = current_offset, current_offset + batch.num_rows
        # check if intersection is null
        if current_r <= required_l:
            pass
        elif current_l >= required_r:
            break
        else:
            intersection_l = max(required_l, current_l)
            intersection_r = min(required_r, current_r)
            trimmed = batch.slice(intersection_l - current_offset, intersection_r - intersection_l)
            assert (
                trimmed.num_rows == intersection_r - intersection_l
            ), f"trimmed.num_rows {trimmed.num_rows} != batch_length {intersection_r - intersection_l}"
            yield cast_columns_to_large_string(trimmed)
        current_offset += batch.num_rows


def build_batch_reader_from_files(
    paths_or_ranges: Union[List[str], List[RowRange]],
    *,
    columns: Optional[List[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batch_byte_size: Optional[int] = None,
    filesystem: fsspec.AbstractFileSystem = None,
) -> arrow.RecordBatchReader:
    assert len(paths_or_ranges) > 0, "paths_or_ranges must be a non-empty list"
    schema = _read_schema_from_file(paths_or_ranges[0], columns, filesystem)
    iterator = _iter_record_batches_from_files(paths_or_ranges, columns, batch_size, max_batch_byte_size, filesystem)
    return arrow.RecordBatchReader.from_batches(schema, iterator)


def _read_schema_from_file(
    path_or_range: Union[str, RowRange],
    columns: Optional[List[str]] = None,
    filesystem: fsspec.AbstractFileSystem = None,
) -> arrow.Schema:
    path = path_or_range.path if isinstance(path_or_range, RowRange) else path_or_range
    schema = parquet.read_schema(filesystem.unstrip_protocol(path) if filesystem else path, filesystem=filesystem)
    if columns is not None:
        assert all(
            c in schema.names for c in columns
        ), f"""some of {columns} cannot be found in schema of {path}:
  {schema}

  The following query can help to find files with missing columns:
    duckdb-dev -c "select * from ( select file_name, list(name) as column_names, list_filter({columns}, c -> not list_contains(column_names, c)) as missing_columns FROM parquet_schema(['{os.path.join(os.path.dirname(path), '*.parquet')}']) group by file_name ) where len(missing_columns) > 0"
  """
        schema = filter_schema(schema, columns)
    return convert_types_to_large_string(schema)


def _iter_record_batches_from_files(
    paths_or_ranges: Union[List[str], List[RowRange]],
    columns: Optional[List[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batch_byte_size: Optional[int] = None,
    filesystem: fsspec.AbstractFileSystem = None,
) -> Iterable[arrow.RecordBatch]:
    """
    Build a batch reader from a list of row ranges.
    """
    buffered_batches = []
    buffered_rows = 0
    buffered_bytes = 0

    def combine_buffered_batches(
        batches: List[arrow.RecordBatch],
    ) -> Iterable[arrow.RecordBatch]:
        table = arrow.Table.from_batches(batches)
        yield from table.combine_chunks().to_batches(batch_size)

    for path_or_range in paths_or_ranges:
        path = path_or_range.path if isinstance(path_or_range, RowRange) else path_or_range
        with parquet.ParquetFile(
            filesystem.unstrip_protocol(path) if filesystem else path,
            buffer_size=16 * MB,
            filesystem=filesystem,
        ) as file:
            if isinstance(path_or_range, RowRange):
                offset, length = path_or_range.begin, path_or_range.num_rows
            else:
                offset, length = 0, file.metadata.num_rows
            for batch in _iter_record_batches(file, columns, offset, length, batch_size):
                batch_size_exceeded = batch.num_rows + buffered_rows >= batch_size
                batch_byte_size_exceeded = max_batch_byte_size is not None and batch.nbytes + buffered_bytes >= max_batch_byte_size
                if not batch_size_exceeded and not batch_byte_size_exceeded:
                    buffered_batches.append(batch)
                    buffered_rows += batch.num_rows
                    buffered_bytes += batch.nbytes
                else:
                    if batch_size_exceeded:
                        buffered_batches.append(batch.slice(0, batch_size - buffered_rows))
                        batch = batch.slice(batch_size - buffered_rows)
                    if buffered_batches:
                        yield from combine_buffered_batches(buffered_batches)
                    buffered_batches = [batch]
                    buffered_rows = batch.num_rows
                    buffered_bytes = batch.nbytes

    if buffered_batches:
        yield from combine_buffered_batches(buffered_batches)


def read_parquet_files_into_table(
    paths_or_ranges: Union[List[str], List[RowRange]],
    columns: List[str] = None,
    filesystem: fsspec.AbstractFileSystem = None,
) -> arrow.Table:
    batch_reader = build_batch_reader_from_files(paths_or_ranges, columns=columns, filesystem=filesystem)
    return batch_reader.read_all()


def load_from_parquet_files(
    paths_or_ranges: Union[List[str], List[RowRange]],
    columns: List[str] = None,
    max_workers: int = 16,
    filesystem: fsspec.AbstractFileSystem = None,
) -> arrow.Table:
    start_time = time.time()
    assert len(paths_or_ranges) > 0, "paths_or_ranges must be a non-empty list"
    paths = [path_or_range.path if isinstance(path_or_range, RowRange) else path_or_range for path_or_range in paths_or_ranges]
    total_compressed_size = sum(
        (path_or_range.data_size if isinstance(path_or_range, RowRange) else os.path.getsize(path_or_range)) for path_or_range in paths_or_ranges
    )
    logger.debug(f"loading {len(paths)} parquet files (compressed size: {total_compressed_size/MB:.3f}MB): {paths[:3]}...")
    num_workers = min(len(paths), max_workers)
    with ThreadPoolExecutor(num_workers) as pool:
        running_works = [
            pool.submit(read_parquet_files_into_table, batch, columns, filesystem) for batch in split_into_rows(paths_or_ranges, num_workers)
        ]
        tables = [work.result() for work in running_works]
        logger.debug(f"collected {len(tables)} tables from: {paths[:3]}... (elapsed: {time.time() - start_time:.3f} secs)")
        return arrow.concat_tables(tables)


def parquet_write_table(table, where, filesystem: fsspec.AbstractFileSystem = None, **write_table_args) -> int:
    if filesystem is not None:
        return parquet.write_table(
            table,
            where=(filesystem.unstrip_protocol(where) if filesystem else where),
            filesystem=filesystem,
            **write_table_args,
        )
    with open(where, "wb", buffering=32 * MB) as file:
        return parquet.write_table(table, where=file, **write_table_args)


def dump_to_parquet_files(
    table: arrow.Table,
    output_dir: str,
    filename: str = "data",
    compression="ZSTD",
    compression_level=3,
    row_group_size=DEFAULT_ROW_GROUP_SIZE,
    row_group_bytes=DEFAULT_ROW_GROUP_BYTES,
    use_dictionary=False,
    max_workers=16,
    filesystem: fsspec.AbstractFileSystem = None,
) -> bool:
    table = cast_columns_to_large_string(table)
    if table.num_rows == 0:
        logger.warning(f"creating empty parquet file in {output_dir}")
        parquet_write_table(
            table,
            os.path.join(output_dir, f"{filename}-0.parquet"),
            compression=compression,
            row_group_size=row_group_size,
        )
        return True

    start_time = time.time()
    avg_row_size = max(1, table.nbytes // table.num_rows)
    row_group_size = min(row_group_bytes // avg_row_size, row_group_size)
    logger.debug(
        f"dumping arrow table ({table.nbytes/MB:.3f}MB, {table.num_rows} rows) to {output_dir}, avg row size: {avg_row_size}, row group size: {row_group_size}"
    )

    batches = table.to_batches(max_chunksize=row_group_size)
    num_workers = min(len(batches), max_workers)
    num_tables = max(math.ceil(table.nbytes / MAX_PARQUET_FILE_BYTES), num_workers)
    logger.debug(f"evenly distributed {len(batches)} batches into {num_tables} files")
    tables = [arrow.Table.from_batches(batch, table.schema) for batch in split_into_rows(batches, num_tables)]
    assert sum(t.num_rows for t in tables) == table.num_rows

    logger.debug(f"writing {len(tables)} files to {output_dir}")
    with ThreadPoolExecutor(num_workers) as pool:
        running_works = [
            pool.submit(
                parquet_write_table,
                table=table,
                where=os.path.join(output_dir, f"{filename}-{i}.parquet"),
                use_dictionary=use_dictionary,
                compression=compression,
                compression_level=compression_level,
                row_group_size=row_group_size,
                write_batch_size=max(16 * 1024, row_group_size // 8),
                data_page_size=max(64 * MB, row_group_bytes // 8),
                filesystem=filesystem,
            )
            for i, table in enumerate(tables)
        ]
        assert all(work.result() or True for work in running_works)

    logger.debug(f"finished writing {len(tables)} files to {output_dir} (elapsed: {time.time() - start_time:.3f} secs)")
    return True
