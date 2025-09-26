# ClickHouse Connect: High-Performance Python Driver for Seamless ClickHouse Integration

**Effortlessly connect Python, Pandas, and Superset to ClickHouse with ClickHouse Connect, a powerful and fast database driver.**

[Explore the ClickHouse Connect Repository on GitHub](https://github.com/ClickHouse/clickhouse-connect)

## Key Features

*   **Broad Data Format Support:** Directly work with various data formats, including:
    *   Pandas DataFrames (NumPy and Arrow-backed)
    *   NumPy Arrays
    *   PyArrow Tables
    *   Polars DataFrames
*   **Superset Integration:**  Seamlessly connects to Apache Superset for robust data visualization and exploration.
*   **SQLAlchemy Compatibility:** Offers a lightweight SQLAlchemy dialect for basic query execution, including `SELECT` queries with `JOIN`s and lightweight `DELETE` statements.
*   **Asynchronous Support:** Includes an `asyncio` wrapper for asynchronous operations, enhancing performance in concurrent environments.

## Installation

Install ClickHouse Connect easily using pip:

```bash
pip install clickhouse-connect
```

**Requirements:** ClickHouse Connect requires Python 3.9 or higher. It is officially tested against Python versions 3.9 through 3.13.

## Superset Connectivity

ClickHouse Connect is fully integrated with Apache Superset.  Connect to your ClickHouse data sources using the provided connection dialog or a SQLAlchemy DSN: `clickhousedb://{username}:{password}@{host}:{port}`.

## SQLAlchemy Implementation

ClickHouse Connect includes a lightweight SQLAlchemy dialect specifically designed for compatibility with Superset and SQLAlchemy Core.  This implementation provides essential features such as executing basic queries, `SELECT` queries with `JOIN`s, and lightweight `DELETE` statements.  Note that it does not include ORM support and may not be suitable for advanced SQLAlchemy applications.

## Asynchronous Support

Leverage the power of asynchronous programming with ClickHouse Connect. The library provides an `asyncio` wrapper. Refer to the [run_async example](./examples/run_async.py) for detailed usage.

## Documentation

For complete documentation, please visit the [ClickHouse Docs](https://clickhouse.com/docs/integrations/python).