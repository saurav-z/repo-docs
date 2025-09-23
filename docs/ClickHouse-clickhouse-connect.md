# ClickHouse Connect: High-Performance Python Driver for ClickHouse

**Seamlessly connect your Python applications to ClickHouse with ClickHouse Connect, offering blazing-fast data transfer and robust integration capabilities.**

[View the original repository on GitHub](https://github.com/ClickHouse/clickhouse-connect)

## Key Features

*   **Pandas DataFrame Support:** Efficiently handle data with support for both NumPy and Arrow-backed DataFrames.
*   **Array and Table Compatibility:** Direct integration with NumPy arrays and PyArrow tables for streamlined data processing.
*   **Polars DataFrame Support:** Leverage the speed of Polars DataFrames with direct integration.
*   **Superset Integration:** Effortlessly connect to Apache Superset for powerful data visualization and exploration.
*   **SQLAlchemy Core Compatibility:** Utilize a lightweight SQLAlchemy dialect for basic query execution, including SELECTs, JOINs, and lightweight DELETEs.
*   **Asyncio Support:** Leverage the benefits of asynchronous programming with a built-in async wrapper.

## Installation

Install ClickHouse Connect using pip:

```bash
pip install clickhouse-connect
```

**Requirements:**

*   Python 3.9 or higher. We test against Python versions 3.9 through 3.13.

## Superset Connectivity

ClickHouse Connect is fully integrated with Apache Superset.  When creating a Superset Data Source, use either the provided connection dialog or a SQLAlchemy DSN in the format `clickhousedb://{username}:{password}@{host}:{port}`.

## SQLAlchemy Implementation

The lightweight SQLAlchemy dialect implementation supports:

*   Basic query execution with SQLAlchemy Core
*   `SELECT` queries with `JOIN`s
*   Lightweight `DELETE` statements

**Note:** This implementation is primarily focused on Superset and SQLAlchemy Core compatibility. It does not include ORM support and may not be suitable for complex SQLAlchemy applications.

## Asyncio Support

ClickHouse Connect offers an asynchronous wrapper to enable client usage within an `asyncio` environment. Explore the [run\_async example](./examples/run_async.py) for a practical demonstration.

## Documentation

Find comprehensive documentation and examples on the official [ClickHouse Documentation](https://clickhouse.com/docs/integrations/python) site.