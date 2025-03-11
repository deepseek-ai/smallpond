"""High-Level URL Sorting Example using DataFrame API
==========================================

This script demonstrates the simplified DataFrame API approach to URL sorting in Smallpond.
It shows how to process and sort URLs using a high-level, pandas-like interface that
automatically handles distributed execution.

Key Features:
    - Uses smallpond.init() for simplified setup
    - Supports both local and distributed execution via Ray
    - Partitioned sorting for better scalability:
        * First partitions data by host (hash_by="host")
        * Then sorts within each partition (partial_sort)
    - For global sorting across all partitions, use:
        df.map("SELECT * FROM {0} ORDER BY column")
"""
import argparse
from typing import List

import smallpond
from smallpond.dataframe import Session



def sort_mock_urls_v2(sp: Session, input_paths: List[str], output_path: str, npartitions: int):
    dataset = sp.read_csv(input_paths, schema={"urlstr": "varchar", "valstr": "varchar"}, delim=r"\t").repartition(npartitions)
    #Creates Dataframe of host, url, payload. Uses DuckDB SQL syntax to transform.
    urls = dataset.map(
        """
    split_part(urlstr, '/', 1) as host,
    split_part(urlstr, ' ', 1) as url,
    from_base64(valstr) AS payload
  """
    )
    urls = urls.repartition(npartitions, hash_by="host")
    #Sorts each partition independently, sorting done locally in each partition, not globla partition
    #For Global sorting use SQL like df.map("""SELECT * FROM {0} ORDER BY sort_column""")
    sorted_urls = urls.partial_sort(by=["host"])
    sorted_urls.write_parquet(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_paths", nargs="+", default=["tests/data/mock_urls/*.tsv"])
    parser.add_argument("-o", "--output_path", type=str, default="sort_mock_urls")
    parser.add_argument("-n", "--npartitions", type=int, default=10)
    args = parser.parse_args()

    sp = smallpond.init()
    sort_mock_urls_v2(sp, **vars(args))
