import base64
import glob
import os
import random
import string
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock


def generate_url_and_domain() -> Tuple[str, str]:
    domain_part = "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 15)))
    tld = random.choice(["com", "net", "org", "cn", "edu", "gov", "co", "io"])
    domain = f"www.{domain_part}.{tld}"

    path_segments = []
    for _ in range(random.randint(1, 3)):
        segment = "".join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(3, 10)))
        path_segments.append(segment)
    path = "/" + "/".join(path_segments)

    protocol = random.choice(["http", "https"])

    if random.random() < 0.3:
        path += random.choice([".html", ".php", ".htm", ".aspx"])

    return f"{protocol}://{domain}{path}", domain


def generate_random_date() -> str:
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end = datetime(2023, 12, 31, tzinfo=timezone.utc)
    delta = end - start
    random_date = start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))
    return random_date.strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_content() -> bytes:
    target_length = random.randint(1000, 100000) if random.random() < 0.8 else random.randint(100000, 1000000)
    before = b"<!DOCTYPE html><html><head><title>Random Page</title></head><body>"
    after = b"</body></html>"
    total_before_after = len(before) + len(after)

    fill_length = max(target_length - total_before_after, 0)
    filler = "".join(random.choices(string.printable, k=fill_length)).encode("ascii")[:fill_length]

    return before + filler + after


def generate_arrow_parquet(path: str, num_rows=100):
    data = []
    for _ in range(num_rows):
        url, domain = generate_url_and_domain()
        date = generate_random_date()
        content = generate_content()

        data.append({"url": url, "domain": domain, "date": date, "content": content})

    df = pd.DataFrame(data)
    df.to_parquet(path, engine="pyarrow")


def generate_arrow_files(output_dir: str, num_files=10):
    os.makedirs(output_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=10) as executor:
        executor.map(
            generate_arrow_parquet,
            [f"{output_dir}/data{i}.parquet" for i in range(num_files)],
        )


def concat_arrow_files(input_dir: str, output_dir: str, repeat: int = 10):
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, "*.parquet"))
    table = pa.concat_tables([pa.parquet.read_table(file) for file in files] * repeat)
    pq.write_table(table, os.path.join(output_dir, "large_array.parquet"))


def generate_random_string(length: int) -> str:
    """Generate a random string of a specified length"""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_random_url() -> str:
    """Generate a random URL"""
    path = generate_random_string(random.randint(10, 20))
    return f"com.{random.randint(10000, 999999)}.{random.randint(100, 9999)}/{path}.html"


def generate_random_data() -> str:
    """Generate random data"""
    url = generate_random_url()
    content = generate_random_string(random.randint(50, 100))
    encoded_content = base64.b64encode(content.encode()).decode()
    return f"{url}\t{encoded_content}"


def generate_url_parquet(path: str, num_rows=100):
    """Generate a parquet file with a specified number of random data lines"""
    data = []
    for _ in range(num_rows):
        url = generate_random_url()
        host = url.split("/")[0]
        data.append({"host": host, "url": url})

    df = pd.DataFrame(data)
    df.to_parquet(path, engine="pyarrow")


def generate_url_parquet_files(output_dir: str, num_files: int = 10):
    """Generate multiple parquet files with a specified number of random data lines"""
    os.makedirs(output_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=10) as executor:
        executor.map(
            generate_url_parquet,
            [f"{output_dir}/urls{i}.parquet" for i in range(num_files)],
        )


def generate_url_tsv_files(output_dir: str, num_files: int = 10, lines_per_file: int = 100):
    """Generate multiple files, each containing a specified number of random data lines"""
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_files):
        with open(f"{output_dir}/urls{i}.tsv", "w") as f:
            for _ in range(lines_per_file):
                f.write(generate_random_data() + "\n")


def generate_long_path_list(path: str, num_lines: int = 1048576):
    """Generate a list of long paths"""
    with open(path, "w", buffering=16 * 1024 * 1024) as f:
        for i in range(num_lines):
            path = os.path.abspath(f"tests/data/arrow/data{i % 10}.parquet")
            f.write(f"{path}\n")


def generate_data(path: str = "tests/data"):
    """
    Generate all data for testing.
    """
    os.makedirs(path, exist_ok=True)
    try:
        with FileLock(path + "/data.lock"):
            print("Generating data...")
            if not os.path.exists(path + "/mock_urls"):
                generate_url_tsv_files(output_dir=path + "/mock_urls", num_files=10, lines_per_file=100)
                generate_url_parquet_files(output_dir=path + "/mock_urls", num_files=10)
            if not os.path.exists(path + "/arrow"):
                generate_arrow_files(output_dir=path + "/arrow", num_files=10)
            if not os.path.exists(path + "/large_array"):
                concat_arrow_files(input_dir=path + "/arrow", output_dir=path + "/large_array")
            if not os.path.exists(path + "/long_path_list.txt"):
                generate_long_path_list(path=path + "/long_path_list.txt")
    except Exception as e:
        print(f"Error generating data: {e}")


if __name__ == "__main__":
    generate_data()
