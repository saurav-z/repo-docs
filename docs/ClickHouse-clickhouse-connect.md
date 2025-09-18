# ClickHouse Connect: The Fastest Way to Connect Python to ClickHouse 🚀

ClickHouse Connect is a high-performance Python driver designed for seamless integration with the ClickHouse database, making data access and analysis faster and more efficient than ever before.

[View the original repository on GitHub](https://github.com/ClickHouse/clickhouse-connect)

## Key Features

*   **Pandas Integration:** Directly load and manipulate data using Pandas DataFrames (both NumPy and Arrow-backed).
*   **Array and Table Support:** Native support for NumPy arrays and PyArrow tables.
*   **Polars Compatibility:** Works seamlessly with Polars DataFrames for high-performance data processing.
*   **Superset Connector:** Effortlessly connect ClickHouse to Apache Superset for data visualization and exploration.
*   **SQLAlchemy Core Support:** Provides a lightweight SQLAlchemy dialect for basic SQL operations.
*   **Asyncio Support:** Fully supports asynchronous operations for improved performance in concurrent environments.

## Installation

Installing ClickHouse Connect is simple:

```bash
pip install clickhouse-connect
```

**Requirements:**

*   Python 3.9 or higher

## Superset Integration

ClickHouse Connect is fully integrated with Apache Superset, enabling you to easily connect and visualize data. Use the built-in connection dialog within Superset or configure a SQLAlchemy DSN in the format: `clickhousedb://{username}:{password}@{host}:{port}`.

*Note:* For older Superset versions prior to v2.1.0, use clickhouse-connect v0.5.25.

## SQLAlchemy Implementation

ClickHouse Connect offers a lightweight SQLAlchemy dialect, ideal for:

*   Executing basic queries with SQLAlchemy Core.
*   Performing `SELECT` queries with `JOIN` operations.
*   Executing lightweight `DELETE` statements.

**Important:** This implementation is primarily focused on Superset compatibility and does not provide full ORM support or advanced dialect functionality.

## Asynchronous Operations

ClickHouse Connect includes an `asyncio` wrapper, allowing you to leverage asynchronous operations.  See the [run_async example](./examples/run_async.py) for usage details.

## Documentation

For comprehensive documentation and usage examples, please refer to the official ClickHouse documentation: [ClickHouse Docs](https://clickhouse.com/docs/integrations/python)