smallpond
=========

Smallpond is a lightweight distributed data processing framework.
It uses `duckdb`_ as the compute engine and stores data in `parquet`_ format on a distributed file system (e.g. `3FS`_).

.. _duckdb: https://duckdb.org/
.. _parquet: https://parquet.apache.org/
.. _3FS: https://github.com/deepseek-ai/3fs

Why smallpond?
--------------

- **Performance**: Smallpond uses DuckDB to deliver native-level performance for efficient data processing.
- **Scalability**: Leverages high-performance distributed file systems for intermediate storage, enabling PB-scale data handling without memory bottlenecks.
- **Simplicity**: No long-running services or complex dependencies, making it easy to deploy and maintain.

.. toctree::
   :maxdepth: 1

   getstarted
   architecture
   internals

.. toctree::
   :maxdepth: 3

   api
