from typing import Iterator, List

import pandas as pd
import pyarrow as pa
import pytest

from smallpond.dataframe import Session


def test_pandas(sp: Session):
    pandas_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = sp.from_pandas(pandas_df)
    assert df.to_pandas().equals(pandas_df)


def test_arrow(sp: Session):
    arrow_table = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = sp.from_arrow(arrow_table)
    assert df.to_arrow() == arrow_table


def test_items(sp: Session):
    df = sp.from_items([1, 2, 3])
    assert df.take_all() == [{"item": 1}, {"item": 2}, {"item": 3}]
    df = sp.from_items([{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}])
    assert df.take_all() == [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]


def test_csv(sp: Session):
    df = sp.read_csv(
        "tests/data/mock_urls/*.tsv",
        schema={"urlstr": "varchar", "valstr": "varchar"},
        delim=r"\t",
    )
    assert df.count() == 1000


def test_parquet(sp: Session):
    df = sp.read_parquet("tests/data/mock_urls/*.parquet")
    assert df.count() == 1000


def test_take(sp: Session):
    df = sp.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    assert df.take(2) == [{"a": 1, "b": 4}, {"a": 2, "b": 5}]
    assert df.take_all() == [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]


def test_map(sp: Session):
    df = sp.from_arrow(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}))
    df1 = df.map("a + b as c")
    assert df1.to_arrow() == pa.table({"c": [5, 7, 9]})
    df2 = df.map(lambda r: {"c": r["a"] + r["b"]})
    assert df2.to_arrow() == pa.table({"c": [5, 7, 9]})

    # user need to specify the schema if can not be inferred from the mapping values
    df3 = df.map(
        lambda r: {"c": None if r["a"] == 1 else r["a"] + r["b"]},
        schema=pa.schema([("c", pa.int64())]),
    )
    assert df3.to_arrow() == pa.table({"c": pa.array([None, 7, 9], type=pa.int64())})


def test_flat_map(sp: Session):
    df = sp.from_arrow(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}))
    df1 = df.flat_map(lambda r: [{"c": r["a"]}, {"c": r["b"]}])
    assert df1.to_arrow() == pa.table({"c": [1, 4, 2, 5, 3, 6]})
    df2 = df.flat_map("unnest(array[a, b]) as c")
    assert df2.to_arrow() == pa.table({"c": [1, 4, 2, 5, 3, 6]})

    # user need to specify the schema if can not be inferred from the mapping values
    df3 = df.flat_map(lambda r: [{"c": None}], schema=pa.schema([("c", pa.int64())]))
    assert df3.to_arrow() == pa.table(
        {"c": pa.array([None, None, None], type=pa.int64())}
    )


def test_map_batches(sp: Session):
    df = sp.read_parquet("tests/data/mock_urls/*.parquet")
    df = df.map_batches(
        lambda batch: pa.table({"num_rows": [batch.num_rows]}),
        batch_size=350,
    )
    assert df.take_all() == [{"num_rows": 350}, {"num_rows": 350}, {"num_rows": 300}]


def test_map_batches_streaming(sp: Session):
    df = sp.read_parquet("tests/data/mock_urls/*.parquet")

    def batched2(tables: Iterator[pa.Table]) -> Iterator[pa.Table]:
        # same as itertools.pairwise
        num_rows = 0
        count = 0
        for batch in tables:
            num_rows += batch.num_rows
            count += 1
            if count == 2:
                yield pa.table({"num_rows": [num_rows]})
                num_rows = 0
                count = 0
        if count > 0:
            yield pa.table({"num_rows": [num_rows]})

    df = df.map_batches(
        batched2,
        batch_size=350,
        streaming=True,
    )
    assert df.take_all() == [{"num_rows": 700}, {"num_rows": 300}]


def test_filter(sp: Session):
    df = sp.from_arrow(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}))
    df1 = df.filter("a > 1")
    assert df1.to_arrow() == pa.table({"a": [2, 3], "b": [5, 6]})
    df2 = df.filter(lambda r: r["a"] > 1)
    assert df2.to_arrow() == pa.table({"a": [2, 3], "b": [5, 6]})


def test_random_shuffle(sp: Session):
    df = sp.from_items(list(range(1000))).repartition(10, by_rows=True)
    df = df.random_shuffle()
    shuffled = [d["item"] for d in df.take_all()]
    assert sorted(shuffled) == list(range(1000))

    def count_inversions(arr: List[int]) -> int:
        return sum(
            sum(1 for j in range(i + 1, len(arr)) if arr[i] > arr[j])
            for i in range(len(arr))
        )

    # check the shuffle is random enough
    # the expected number of inversions is n*(n-1)/4 = 249750
    assert 220000 <= count_inversions(shuffled) <= 280000


def test_partition_by(sp: Session):
    df = sp.from_items(list(range(1000))).repartition(10, by="item % 10")
    df = df.map("min(item % 10) as min, max(item % 10) as max")
    assert df.take_all() == [{"min": i, "max": i} for i in range(10)]


def test_partition_by_key_out_of_range(sp: Session):
    df = sp.from_items(list(range(1000))).repartition(10, by="item % 11")
    try:
        df.to_arrow()
    except Exception as ex:
        assert "partition key 10 is out of range 0-9" in str(ex)
    else:
        assert False, "expected exception"


def test_partition_by_hash(sp: Session):
    df = sp.from_items(list(range(1000))).repartition(10, hash_by="item")
    items = [d["item"] for d in df.take_all()]
    assert sorted(items) == list(range(1000))


def test_count(sp: Session):
    df = sp.from_items([1, 2, 3])
    assert df.count() == 3


def test_limit(sp: Session):
    df = sp.from_items(list(range(1000))).repartition(10, by_rows=True)
    assert df.limit(2).count() == 2


@pytest.mark.skip(reason="limit can not be pushed down to sql node for now")
@pytest.mark.timeout(10)
def test_limit_large(sp: Session):
    # limit will be fused with the previous select
    # otherwise, it will be timeout
    df = sp.partial_sql("select * from range(1000000000)")
    assert df.limit(2).count() == 2


def test_partial_sql(sp: Session):
    # no input deps
    df = sp.partial_sql("select * from range(3)")
    assert df.to_arrow() == pa.table({"range": [0, 1, 2]})

    # join
    df1 = sp.from_arrow(pa.table({"id1": [1, 2, 3], "val1": ["a", "b", "c"]}))
    df2 = sp.from_arrow(pa.table({"id2": [1, 2, 3], "val2": ["d", "e", "f"]}))
    joined = sp.partial_sql(
        "select id1, val1, val2 from {0} join {1} on id1 = id2", df1, df2
    )
    assert joined.to_arrow() == pa.table(
        {"id1": [1, 2, 3], "val1": ["a", "b", "c"], "val2": ["d", "e", "f"]},
        schema=pa.schema(
            [
                ("id1", pa.int64()),
                ("val1", pa.large_string()),
                ("val2", pa.large_string()),
            ]
        ),
    )


def test_error_message(sp: Session):
    df = sp.from_items([1, 2, 3])
    df = sp.partial_sql("select a,, from {0}", df)
    try:
        df.to_arrow()
    except Exception as ex:
        # sql query should be in the exception message
        assert "select a,, from" in str(ex)
    else:
        assert False, "expected exception"


def test_unpicklable_task_exception(sp: Session):
    from loguru import logger

    df = sp.from_items([1, 2, 3])
    try:
        df.map(lambda x: logger.info("use outside logger")).to_arrow()
    except Exception as ex:
        assert "Can't pickle task" in str(ex)
        assert (
            "HINT: DO NOT use externally imported loguru logger in your task. Please import it within the task."
            in str(ex)
        )
    else:
        assert False, "expected exception"


def test_log(sp: Session):
    df = sp.from_items([1, 2, 3])

    def log_record(x):
        import logging
        import sys

        from loguru import logger

        print("stdout")
        print("stderr", file=sys.stderr)
        logger.info("loguru")
        logging.info("logging")
        return x

    df.map(log_record).to_arrow()

    # TODO: check logs should be see in the log file
    # FIXME: logs in unit test are not written to the log file
    #        because we share the same ray instance for all tests
