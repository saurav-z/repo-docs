# ClickHouse Connect: High-Performance Python Connectivity

**Seamlessly connect Python, Pandas, and Apache Superset to ClickHouse with the lightning-fast ClickHouse Connect driver.**

[View the ClickHouse Connect Repository on GitHub](https://github.com/ClickHouse/clickhouse-connect)

## Key Features

*   **Pandas DataFrames:** Seamlessly work with both NumPy and Arrow-backed Pandas DataFrames.
*   **Numpy Arrays:** Directly integrate with NumPy arrays for efficient data manipulation.
*   **PyArrow Tables:** Leverage the power of Apache Arrow with PyArrow table support.
*   **Polars DataFrames:** Supports Polars DataFrames for high-performance data processing.
*   **Superset Connector:** Native integration for easy connection to Apache Superset.
*   **SQLAlchemy Core:** Provides a lightweight SQLAlchemy dialect for Core-based queries, including `SELECT` with `JOIN`s and lightweight `DELETE` statements.

## Installation

Get started quickly with a simple `pip` command:

```bash
pip install clickhouse-connect
```

**Compatibility:**  ClickHouse Connect is compatible with Python 3.9 and higher. We regularly test against Python versions 3.9 through 3.13.

## Apache Superset Integration

ClickHouse Connect offers a fully integrated experience with Apache Superset. To connect, use either the provided connection dialog within Superset or a SQLAlchemy DSN string in the format:

```
clickhousedb://{username}:{password}@{host}:{port}
```

## SQLAlchemy Implementation (Core)

ClickHouse Connect includes a streamlined SQLAlchemy dialect, ideal for compatibility with Superset and SQLAlchemy Core.  It's designed for basic query execution and supports `SELECT` statements with `JOIN`s, and lightweight `DELETE` statements.  Note that it does not include full ORM support.

## Asynchronous (asyncio) Support

ClickHouse Connect offers an `asyncio` wrapper for asynchronous operations.  For an example, see the `run_async.py` example.

## Documentation

For comprehensive documentation, visit the official ClickHouse documentation: [ClickHouse Docs](https://clickhouse.com/docs/integrations/python)