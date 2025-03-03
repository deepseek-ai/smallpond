import itertools
import math
import sys
from typing import Dict, List, TypeVar

import numpy as np

from smallpond.logical.udf import *

KB = 1024
MB = 1024 * KB
GB = 1024 * MB
TB = 1024 * GB

DEFAULT_MAX_RETRY_COUNT = 5
DEFAULT_MAX_FAIL_COUNT = 3
# duckdb default row group size https://duckdb.org/docs/data/parquet/tips#selecting-a-row_group_size
MAX_ROW_GROUP_SIZE = 10 * 1024 * 1024
MAX_ROW_GROUP_BYTES = 2 * GB
MAX_NUM_ROW_GROUPS = 256
MAX_PARQUET_FILE_BYTES = 8 * GB
DEFAULT_ROW_GROUP_SIZE = 122880
DEFAULT_ROW_GROUP_BYTES = DEFAULT_ROW_GROUP_SIZE * 4 * KB
DEFAULT_BATCH_SIZE = 122880
RAND_SEED_BYTE_LEN = 128
DATA_PARTITION_COLUMN_NAME = "__data_partition__"
PARQUET_METADATA_KEY_PREFIX = "SMALLPOND:"
INPUT_VIEW_PREFIX = "__input"
GENERATED_COLUMNS = ("filename", "file_row_number")


def pytest_running():
    return "pytest" in sys.modules


def clamp_value(val, minval, maxval):
    return max(minval, min(val, maxval))


def clamp_row_group_size(val, minval=DEFAULT_ROW_GROUP_SIZE, maxval=MAX_ROW_GROUP_SIZE):
    return clamp_value(val, minval, maxval)


def clamp_row_group_bytes(val, minval=DEFAULT_ROW_GROUP_BYTES, maxval=MAX_ROW_GROUP_BYTES):
    return clamp_value(val, minval, maxval)


class SmallpondError(Exception):
    """Base class for all errors in smallpond."""


class InjectedFault(SmallpondError):
    pass


class OutOfMemory(SmallpondError):
    pass


class NonzeroExitCode(SmallpondError):
    pass


K = TypeVar("K")
V = TypeVar("V")


def first_value_in_dict(d: Dict[K, V]) -> V:
    return next(iter(d.values())) if d else None


def split_into_cols(items: List[V], npartitions: int) -> List[List[V]]:
    none = object()
    chunks = [items[i : i + npartitions] for i in range(0, len(items), npartitions)]
    return [[x for x in col if x is not none] for col in itertools.zip_longest([none] * npartitions, *chunks, fillvalue=none)]


def split_into_rows(items: List[V], npartitions: int) -> List[List[V]]:
    """
    Evenly split items into npartitions.

    Example::
    >>> split_into_rows(list(range(10)), 3)
    [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    """
    split_idxs = np.array_split(np.arange(len(items)), npartitions)
    return [[items[i] for i in idxs] for idxs in split_idxs]


def get_nth_partition(items: List[V], n: int, npartitions: int) -> List[V]:
    num_items = len(items)
    large_partition_size = (num_items + npartitions - 1) // npartitions
    small_partition_size = num_items // npartitions
    num_large_partitions = num_items - small_partition_size * npartitions
    if n < num_large_partitions:
        start = n * large_partition_size
        items_in_partition = items[start : start + large_partition_size]
    else:
        start = large_partition_size * num_large_partitions + (n - num_large_partitions) * small_partition_size
        items_in_partition = items[start : start + small_partition_size]
    return items_in_partition


def next_power_of_two(x) -> int:
    return 2 ** (x - 1).bit_length()


def round_up(x, align_size=MB) -> int:
    return math.ceil(x / align_size) * align_size
