# Test the correctness of file system read and write.
#
# This script runs multiple tasks to write and read data to/from the file system.
# Each task writes to an individual file in the given directory.
# Then it reads the data back and verifies the correctness.

import argparse
import glob
import logging
import os
import random
import time
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import numpy as np

import smallpond
from smallpond.dataframe import Session


def fswrite(
    path: str,
    size: int,
    blocksize: Union[int, Tuple[int, int]],
) -> Dict[str, Any]:
    t0 = time.time()
    with open(path, "wb") as f:
        for start, length in iter_io_slice(0, size, blocksize):
            logging.debug(f"writing {length} bytes at offset {start}")
            f.write(generate_data(start, length))
    t1 = time.time()
    elapsed = t1 - t0
    logging.info(f"write done: {path} in {elapsed:.2f}s")
    return {
        "path": path,
        "size": size,
        "elapsed(s)": elapsed,
        "throughput(MB/s)": size / elapsed / 1024 / 1024,
    }


def fsread(
    path: str,
    blocksize: Union[int, Tuple[int, int]],
    randread: bool,
) -> Dict[str, Any]:
    t0 = time.time()
    with open(path, "rb") as f:
        size = os.path.getsize(path)
        slices = list(iter_io_slice(0, size, blocksize))
        if randread:
            random.shuffle(slices)
        for start, length in slices:
            logging.debug(f"reading {length} bytes at offset {start}")
            f.seek(start)
            data = f.read(length)
            expected_data = generate_data(start, length)
            check_data(data, expected_data, start)
    t1 = time.time()
    elapsed = t1 - t0
    logging.info(f"read done: {path} in {elapsed:.2f}s")
    return {
        "path": path,
        "size": size,
        "elapsed(s)": elapsed,
        "throughput(MB/s)": size / elapsed / 1024 / 1024,
    }


def check_data(actual: bytes, expected: bytes, offset: int) -> None:
    """
    Check if the expected data matches the actual data. Raise an error if there is a mismatch.
    """
    if expected == actual:
        return
    # find the first mismatch
    index = next(
        (i for i, (b1, b2) in enumerate(zip(actual, expected)) if b1 != b2),
        min(len(actual), len(expected)),
    )
    expected = expected[index : index + 16]
    actual = actual[index : index + 16]
    raise ValueError(f"Data mismatch at offset {offset + index}.\nexpect: {expected}\nactual: {actual}")


def generate_data(offset: int, length: int) -> bytes:
    """
    Generate data for the slice [offset, offset + length).
    The full data is a repeated sequence of [0x00000000, 0x00000001, ..., 0xffffffff] in little-endian.
    """
    istart = offset // 4
    iend = (offset + length + 3) // 4
    return np.arange(istart, iend).astype(np.uint32).tobytes()[offset % 4 : offset % 4 + length]


def iter_io_slice(offset: int, length: int, block_size: Union[int, Tuple[int, int]]) -> Iterator[Tuple[int, int]]:
    """
    Generate the IO (offset, size) for the slice [offset, offset + length) with the given block size.
    `block_size` can be an integer or a range [start, end]. If a range is provided, the IO size will be randomly selected from the range.
    """
    start = offset
    end = offset + length
    while start < end:
        if isinstance(block_size, int):
            size = block_size
        else:
            smin, smax = block_size
            size = random.randint(smin, smax)
        size = min(size, end - start)
        yield (start, size)
        start += size


def size_str_to_bytes(size_str: str) -> int:
    """
    Parse size string to bytes.
    e.g. 1k -> 1024, 1M -> 1024^2, 1G -> 1024^3, 1T -> 1024^4
    """
    if size_str.endswith("k"):
        return int(size_str[:-1]) * 1024
    elif size_str.endswith("M"):
        return int(size_str[:-1]) * 1024 * 1024
    elif size_str.endswith("G"):
        return int(size_str[:-1]) * 1024 * 1024 * 1024
    elif size_str.endswith("T"):
        return int(size_str[:-1]) * 1024 * 1024 * 1024 * 1024
    else:
        return int(size_str)


def fstest(
    sp: Session,
    input_path: Optional[str],
    output_path: Optional[str],
    size: Optional[str],
    npartitions: int,
    blocksize: Optional[str] = "4k",
    blocksize_range: Optional[str] = None,
    randread: bool = False,
) -> None:
    # preprocess arguments
    if output_path is not None and size is None:
        raise ValueError("--size is required if --output_path is provided")
    if size is not None:
        size = size_str_to_bytes(size)
    if blocksize_range is not None:
        start, end = blocksize_range.split("-")
        blocksize = (size_str_to_bytes(start), size_str_to_bytes(end))
    elif blocksize is not None:
        blocksize = size_str_to_bytes(blocksize)
    else:
        raise ValueError("either --blocksize or --blocksize_range must be provided")

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        df = sp.from_items([{"path": os.path.join(output_path, f"{i}")} for i in range(npartitions)])
        df = df.repartition(npartitions, by_rows=True)
        stats = df.map(lambda x: fswrite(x["path"], size, blocksize)).to_pandas()
        logging.info(f"write stats:\n{stats}")

    if input_path is not None:
        paths = list(glob.glob(input_path))
        df = sp.from_items([{"path": path} for path in paths])
        df = df.repartition(len(paths), by_rows=True)
        stats = df.map(lambda x: fsread(x["path"], blocksize, randread)).to_pandas()
        logging.info(f"read stats:\n{stats}")


if __name__ == "__main__":
    """
    Example usage:
    - write only:
        python example/fstest.py -o 'fstest' -j 8 -s 1G
    - read only:
        python example/fstest.py -i 'fstest/*'
    - write and then read:
        python example/fstest.py -o 'fstest' -j 8 -s 1G -i 'fstest/*'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", type=str, help="The output path to write data to.")
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        help="The input path to read data from. If -o is provided, this is ignored.",
    )
    parser.add_argument("-j", "--npartitions", type=int, help="The number of parallel jobs", default=10)
    parser.add_argument(
        "-s",
        "--size",
        type=str,
        help="The size for each file. Required if -o is provided.",
    )
    parser.add_argument("-bs", "--blocksize", type=str, help="Block size", default="4k")
    parser.add_argument(
        "-bsrange",
        "--blocksize_range",
        type=str,
        help="A range of I/O block sizes. e.g. 4k-128k",
    )
    parser.add_argument(
        "-randread",
        "--randread",
        action="store_true",
        help="Whether to read data randomly",
        default=False,
    )
    args = parser.parse_args()

    sp = smallpond.init()
    fstest(sp, **vars(args))
