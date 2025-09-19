# ClickHouse Connect: Unleash the Power of ClickHouse in Python

**ClickHouse Connect** is a high-performance Python driver that seamlessly connects to ClickHouse, empowering you to work with your data using familiar tools like Pandas, Superset, and SQLAlchemy.  For the latest updates and information, visit the [original repository](https://github.com/ClickHouse/clickhouse-connect).

## Key Features:

*   **Pandas Integration:** Directly read and write data with Pandas DataFrames (supports NumPy and Arrow-backed DataFrames).
*   **PyArrow & Polars Support:**  Effortlessly work with PyArrow Tables and Polars DataFrames.
*   **Superset Connector:**  Seamlessly integrate ClickHouse with Apache Superset for powerful data visualization and analysis.
*   **SQLAlchemy Core Compatibility:** Execute SQL queries using SQLAlchemy Core for basic SELECTs, JOINs, and lightweight DELETEs.
*   **Asynchronous Operations:**  Leverage asyncio for non-blocking database interactions.

## Installation:

Get started in seconds:

```bash
pip install clickhouse-connect
```

**Requirements:**  ClickHouse Connect requires Python 3.9 or higher. We rigorously test against Python versions 3.9 to 3.13.

## Superset Integration

ClickHouse Connect is fully compatible with Apache Superset.  Configure your Superset data source using either the connection dialog or a SQLAlchemy DSN in the format:  `clickhousedb://{username}:{password}@{host}:{port}`.

## SQLAlchemy Core Implementation

ClickHouse Connect includes a lightweight SQLAlchemy dialect tailored for Superset and SQLAlchemy Core compatibility.  Key features include:

*   Execution of basic SQL queries using SQLAlchemy Core.
*   Support for `SELECT` queries, including `JOIN`s.
*   Lightweight `DELETE` statements.

**Note:** This implementation is focused on Core compatibility and does not offer full ORM support or extensive dialect features.

## Asynchronous Support

The library provides an async wrapper, which allows you to integrate it into `asyncio` environments. Find an example usage in the [run_async example](./examples/run_async.py).

## Documentation

Comprehensive documentation is available on the official [ClickHouse Docs](https://clickhouse.com/docs/integrations/python).