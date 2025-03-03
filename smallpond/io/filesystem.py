import io
import os
import shutil
import tempfile
import time
from typing import Any

import cloudpickle
import zstandard as zstd
from loguru import logger

from smallpond.common import MB

HF3FS_MOUNT_POINT_PREFIX = "/hf3fs"
HF3FS_FSSPEC_PROTOCOL = "hf3fs://"


def on_hf3fs(path: str):
    return path.startswith(HF3FS_MOUNT_POINT_PREFIX)


def extract_hf3fs_mount_point(path: str):
    return os.path.join("/", *path.split("/")[1:3]) if on_hf3fs(path) else None


def remove_path(path: str):
    realpath = os.path.realpath(path)

    if os.path.islink(path):
        logger.debug(f"removing link: {path}")
        os.unlink(path)

    if not os.path.exists(realpath):
        return

    logger.debug(f"removing path: {realpath}")
    if on_hf3fs(realpath):
        try:
            link = os.path.join(
                extract_hf3fs_mount_point(realpath),
                "3fs-virt/rm-rf",
                f"{os.path.basename(realpath)}-{time.time_ns()}",
            )
            os.symlink(realpath, link)
            return
        except Exception as ex:
            logger.opt(exception=ex).debug(f"fast recursive remove failed, fall back to shutil.rmtree('{realpath}')")
    shutil.rmtree(realpath, ignore_errors=True)


def find_mount_point(path: str):
    path = os.path.abspath(path)
    while not os.path.ismount(path):
        path = os.path.dirname(path)
    return path


def dump(obj: Any, path: str, buffering=2 * MB, atomic_write=False) -> int:
    """
    Dump an object to a file.

    Args:
      obj: The object to dump.
      path: The path to the file to dump the object to.
      buffering: The buffering size.
      atomic_write: Whether to atomically write the file.

    Returns:
      The size of the file.
    """

    def get_pickle_trace(obj):
        try:
            import dill
            import dill.detect
        except ImportError:
            return None, None
        pickle_trace = io.StringIO()
        pickle_error = None
        with dill.detect.trace(pickle_trace):
            try:
                dill.dumps(obj, recurse=True)
            except Exception as ex:
                pickle_error = ex
        return pickle_trace.getvalue(), pickle_error

    def write_to_file(fout):
        with zstd.ZstdCompressor().stream_writer(fout, closefd=False) as zstd_writer:
            try:
                cloudpickle.dump(obj, zstd_writer)
            except zstd.ZstdError as ex:
                raise
            except Exception as ex:
                trace_str, trace_err = get_pickle_trace(obj)
                logger.opt(exception=ex).error(f"pickle trace of {repr(obj)}:{os.linesep}{trace_str}")
                if trace_err is None:
                    raise
                else:
                    raise trace_err from ex
            logger.trace("{} saved to {}", repr(obj), path)

    size = 0

    if atomic_write:
        directory, filename = os.path.split(path)
        with tempfile.NamedTemporaryFile("wb", buffering=buffering, dir=directory, prefix=filename, delete=False) as fout:
            write_to_file(fout)
            fout.seek(0, os.SEEK_END)
            size = fout.tell()
            os.rename(fout.name, path)
    else:
        with open(path, "wb", buffering=buffering) as fout:
            write_to_file(fout)
            fout.seek(0, os.SEEK_END)
            size = fout.tell()

    if size >= buffering:
        logger.debug(f"created a large pickle file ({size/MB:.3f}MB): {path}")
    return size


def load(path: str, buffering=2 * MB) -> Any:
    """
    Load an object from a file.
    """
    with open(path, "rb", buffering=buffering) as fin:
        with zstd.ZstdDecompressor().stream_reader(fin) as zstd_reader:
            obj = cloudpickle.load(zstd_reader)
            logger.trace("{} loaded from {}", repr(obj), path)
            return obj
