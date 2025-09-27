# ClickHouse Connect: The Fastest Way to Connect Python to ClickHouse

Unlock the power of ClickHouse with ClickHouse Connect, a high-performance driver designed for seamless integration with Python, Pandas, Superset, and more!  ([See the original repo](https://github.com/ClickHouse/clickhouse-connect))

## Key Features

*   **High-Performance Data Transfer:** Optimized for speed and efficiency when working with ClickHouse.
*   **Pandas DataFrame Support:** Directly read and write data using NumPy and Arrow-backed DataFrames.
*   **PyArrow Integration:** Seamlessly work with PyArrow Tables.
*   **Polars DataFrame Compatibility:** Integrates with Polars DataFrames for high-speed data manipulation.
*   **Superset Connector:**  Fully compatible and integrated with Apache Superset for data visualization and exploration.
*   **SQLAlchemy Core Support:**  Provides basic query execution, including `SELECT` statements with `JOIN`s and lightweight `DELETE` operations, ideal for Superset and Core-based applications.
*   **Asyncio Support:** Offers an async wrapper, enabling you to use the client within an `asyncio` environment.

## Installation

Get started quickly with a simple `pip` command:

```bash
pip install clickhouse-connect
```

**Requirements:**

*   Python 3.9 or higher.  We actively test against Python 3.9 through 3.13.

## Superset Integration

ClickHouse Connect offers full compatibility with Apache Superset.  To connect, use either the Superset connection dialog or a SQLAlchemy DSN of the form: `clickhousedb://{username}:{password}@{host}:{port}`.

## SQLAlchemy Implementation (Core)

ClickHouse Connect provides a lightweight SQLAlchemy dialect implementation, perfect for Superset and Core usage:

*   **Supported Features:**
    *   Basic query execution via SQLAlchemy Core
    *   `SELECT` queries with `JOIN`s
    *   Lightweight `DELETE` statements

**Important Note:**  This implementation focuses on Core compatibility and is not a full SQLAlchemy dialect. It does *not* support ORM features and may not be suitable for complex SQLAlchemy applications.

## Asyncio Support

Leverage the power of asynchronous programming with ClickHouse Connect. See the `run_async` example in the `examples` directory of the original repository.

## Documentation

For comprehensive documentation and usage guides, please visit the official [ClickHouse Docs](https://clickhouse.com/docs/integrations/python).