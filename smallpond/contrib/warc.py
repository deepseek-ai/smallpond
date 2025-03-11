import string
import sys
import unicodedata
from pathlib import PurePath
from typing import Iterable, List, Tuple
from urllib.parse import urlparse

import pyarrow as arrow
import zstandard as zstd
from bs4 import BeautifulSoup
from loguru import logger
from warcio import ArchiveIterator

from smallpond.common import MB
from smallpond.execution.task import RuntimeContext
from smallpond.io.arrow import dump_to_parquet_files
from smallpond.logical.dataset import DataSet
from smallpond.logical.node import ArrowStreamNode, PythonScriptNode


class ImportWarcFiles(PythonScriptNode):

    schema = arrow.schema(
        [
            arrow.field("url", arrow.string()),
            arrow.field("domain", arrow.string()),
            arrow.field("date", arrow.string()),
            arrow.field("content", arrow.binary()),
        ]
    )

    def import_warc_file(self, warc_path: PurePath, parquet_path: PurePath) -> Tuple[int, int]:
        total_size = 0
        docs = []

        with open(warc_path, "rb") as warc_file:
            zstd_reader = zstd.ZstdDecompressor().stream_reader(warc_file, read_size=16 * MB)
            for record in ArchiveIterator(zstd_reader):
                if record.rec_type == "response":
                    url = record.rec_headers.get_header("WARC-Target-URI")
                    domain = urlparse(url).netloc
                    date = record.rec_headers.get_header("WARC-Date")
                    content = record.content_stream().read()
                    total_size += len(content)
                    docs.append((url, domain, date, content))

            table = arrow.Table.from_arrays([arrow.array(column) for column in zip(*docs)], schema=self.schema)
            dump_to_parquet_files(table, parquet_path.parent, parquet_path.name)
            return len(docs), total_size

    def process(
        self,
        runtime_ctx: RuntimeContext,
        input_datasets: List[DataSet],
        output_path: str,
    ) -> bool:
        warc_paths = [PurePath(warc_path) for dataset in input_datasets for warc_path in dataset.resolved_paths]
        parquet_paths = [
            PurePath(output_path) / f"data{path_index}-{PurePath(warc_path.name).with_suffix('.parquet')}"
            for path_index, warc_path in enumerate(warc_paths)
        ]

        logger.info(f"importing web pages from {len(warc_paths)} warc files...")
        for warc_path, parquet_path in zip(warc_paths, parquet_paths):
            try:
                doc_count, total_size = self.import_warc_file(warc_path, parquet_path)
                logger.info(f"imported {doc_count} web pages ({total_size/MB:.3f}MB) from file '{warc_path}' to '{parquet_path}'")
            except Exception as ex:
                logger.opt(exception=ex).error(f"failed to import web pages from file '{warc_path}'")
                return False
        return True


class ExtractHtmlBody(ArrowStreamNode):

    unicode_punctuation = "".join(chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P"))
    separator_str = string.whitespace + string.punctuation + unicode_punctuation
    translator = str.maketrans(separator_str, " " * len(separator_str))

    schema = arrow.schema(
        [
            arrow.field("url", arrow.string()),
            arrow.field("domain", arrow.string()),
            arrow.field("date", arrow.string()),
            arrow.field("tokens", arrow.list_(arrow.string())),
        ]
    )

    def split_string(self, s: str):
        return s.translate(self.translator).split()

    def extract_tokens(self, url: arrow.string, content: arrow.binary) -> List[str]:
        tokens = []
        try:
            doc = BeautifulSoup(content.as_py(), "lxml")
            # if doc.title is not None and doc.title.string is not None:
            #   tokens.extend(self.split_string(doc.title.string.lower()))
            tokens.extend(self.split_string(doc.get_text(" ", strip=True).lower()))
            return tokens
        except Exception as ex:
            logger.opt(exception=ex).error(f"failed to extract tokens from {url.as_py()}")
            return []

    def process(self, runtime_ctx: RuntimeContext, input_readers: List[arrow.RecordBatchReader]) -> Iterable[arrow.Table]:
        for batch in input_readers[0]:
            urls, domains, dates, contents = batch.columns
            doc_tokens = []
            try:
                for i, (url, content) in enumerate(zip(urls, contents)):
                    tokens = self.extract_tokens(url, content)
                    logger.info(f"#{i}/{len(urls)} extracted {len(tokens)} tokens from {url}")
                    doc_tokens.append(tokens)
                yield arrow.Table.from_arrays([urls, domains, dates, arrow.array(doc_tokens)], schema=self.schema)
            except Exception as ex:
                logger.opt(exception=ex).error(f"failed to extract tokens")
                break
