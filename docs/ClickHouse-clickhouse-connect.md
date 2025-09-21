## ClickHouse Connect: Supercharge Your Python Data Interactions with ClickHouse

**Effortlessly connect your Python applications to ClickHouse with ClickHouse Connect, a high-performance driver designed for speed and compatibility.**  [Explore the original repository](https://github.com/ClickHouse/clickhouse-connect)

### Key Features

*   **Seamless Data Integration:** Easily work with popular data structures like Pandas DataFrames (NumPy and Arrow-backed), NumPy Arrays, PyArrow Tables, and Polars DataFrames.
*   **Superset Integration:** Direct connectivity to Apache Superset, making it easy to visualize and analyze your ClickHouse data.
*   **SQLAlchemy Core Support:** Utilize a lightweight SQLAlchemy dialect for core operations such as SELECT queries with JOINs and lightweight DELETE statements.
*   **Asynchronous Support:** Leverage asynchronous capabilities for improved performance in asyncio environments.

### Installation

Get started in seconds:

```bash
pip install clickhouse-connect
```

**Requirements:**

*   Python 3.9 or higher (officially tested up to 3.13)

### Superset Connectivity

ClickHouse Connect offers robust integration with Apache Superset:

*   **Simplified Setup:** Use the connection dialog or a SQLAlchemy DSN (`clickhousedb://{username}:{password}@{host}:{port}`) to connect.
*   **Compatibility:**  Works seamlessly with recent Superset versions as the engine spec is incorporated directly into Superset (version 2.1.0 and later). For earlier Superset versions, use clickhouse-connect v0.5.25.

### SQLAlchemy Implementation

ClickHouse Connect's SQLAlchemy dialect provides essential functionality:

*   **Core Compatibility:** Supports basic query execution using SQLAlchemy Core.
*   **SELECT and JOIN Operations:**  Allows for selecting data and joining tables.
*   **Lightweight DELETE:** Enables DELETE statements.
*   **Limitations:** The implementation is not a full SQLAlchemy dialect, so ORM support and advanced dialect features are not included.

### Asyncio Support

Take advantage of asynchronous operations with the `asyncio` wrapper for improved performance.  See the [run\_async example](./examples/run_async.py) for guidance.

### Complete Documentation

For comprehensive information, refer to the official documentation: [ClickHouse Docs](https://clickhouse.com/docs/integrations/python)