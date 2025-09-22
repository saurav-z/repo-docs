# ClickHouse Connect: The High-Performance Python Driver for ClickHouse

**Connect Python, Pandas, and Superset to ClickHouse with unparalleled speed and efficiency using ClickHouse Connect!**

[View the original repository on GitHub](https://github.com/ClickHouse/clickhouse-connect)

## Key Features

ClickHouse Connect offers a streamlined and high-performance approach to connecting to ClickHouse, with support for:

*   **Pandas DataFrames:** Seamlessly work with both numpy and arrow-backed Pandas DataFrames.
*   **Numpy Arrays:** Direct integration for efficient data transfer.
*   **PyArrow Tables:** Leverage the power of PyArrow for data processing.
*   **Polars DataFrames:** Optimized support for Polars DataFrames.
*   **Superset Connector:** Effortless integration with Apache Superset for data visualization.
*   **SQLAlchemy Core:** Execute select queries, joins, and lightweight delete operations.

## Installation

Get started quickly with a simple pip install:

```bash
pip install clickhouse-connect
```

**Requirements:**

*   Python 3.9 or higher.  We test against Python 3.9 through 3.13.

## Superset Connectivity

ClickHouse Connect is fully integrated with Apache Superset.  Connect to your ClickHouse data sources using either:

*   The standard Superset connection dialog.
*   A SQLAlchemy DSN in the format: `clickhousedb://{username}:{password}@{host}:{port}`.

## SQLAlchemy Implementation

ClickHouse Connect includes a lightweight SQLAlchemy dialect designed for compatibility with Superset and SQLAlchemy Core.  It provides:

*   Basic query execution via SQLAlchemy Core.
*   `SELECT` queries including `JOIN`s.
*   Lightweight `DELETE` statements.

**Important Note:** This implementation is not a full SQLAlchemy dialect.  It is primarily designed for Core-based applications and Superset and may not be suitable for complex applications relying on full ORM or advanced dialect features.

## Asynchronous Support

ClickHouse Connect includes an asynchronous wrapper, allowing you to utilize the client within an `asyncio` environment.  See the example in the [run_async example](./examples/run_async.py) for more details.

## Complete Documentation

For comprehensive documentation, please visit the [ClickHouse Docs](https://clickhouse.com/docs/integrations/python).