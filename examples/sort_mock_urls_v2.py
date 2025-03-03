import argparse
from typing import List

import smallpond
from smallpond.dataframe import Session


def sort_mock_urls_v2(sp: Session, input_paths: List[str], output_path: str, npartitions: int):
    dataset = sp.read_csv(input_paths, schema={"urlstr": "varchar", "valstr": "varchar"}, delim=r"\t").repartition(npartitions)
    urls = dataset.map(
        """
    split_part(urlstr, '/', 1) as host,
    split_part(urlstr, ' ', 1) as url,
    from_base64(valstr) AS payload
  """
    )
    urls = urls.repartition(npartitions, hash_by="host")
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
