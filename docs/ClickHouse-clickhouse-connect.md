## ClickHouse Connect: High-Performance Python Driver for ClickHouse

**Unlock the power of ClickHouse with ClickHouse Connect, a high-speed, versatile Python driver designed for seamless data integration and analysis.**  ([Original Repo](https://github.com/ClickHouse/clickhouse-connect))

This Python library offers a fast and efficient way to connect to and interact with your ClickHouse databases.

**Key Features:**

*   **Pandas Integration:** Load and process data directly from Pandas DataFrames (supports NumPy and Arrow-backed DataFrames).
*   **Numpy Array Support:** Directly work with NumPy arrays for efficient numerical computations.
*   **PyArrow Compatibility:** Leverage the power of PyArrow tables for optimized data handling.
*   **Polars DataFrame Support:** Seamlessly integrate with Polars DataFrames for high-performance data manipulation.
*   **Superset Connector:**  Effortlessly connect ClickHouse to Apache Superset for powerful data visualization and exploration.
*   **SQLAlchemy Core Support:** Utilize a lightweight SQLAlchemy dialect for basic querying, including `SELECT` with `JOIN`s and lightweight `DELETE` statements (designed for Superset and Core functionality; ORM is *not* supported).
*   **Asyncio Support:** Enables asynchronous operations for improved performance in `asyncio` environments.

### Installation

Get started with ClickHouse Connect easily:

```bash
pip install clickhouse-connect
```

**Requirements:**

*   Python 3.9 or higher. We test with Python 3.9 through 3.13.

### Superset Connectivity

ClickHouse Connect integrates perfectly with Apache Superset. Use either the built-in connection dialog or a SQLAlchemy DSN:

*   **DSN Format:** `clickhousedb://{username}:{password}@{host}:{port}`

***Note:*** If you have issues connecting to *earlier* Superset versions, use clickhouse-connect v0.5.25.

### SQLAlchemy Implementation Details

ClickHouse Connect provides a lightweight SQLAlchemy dialect focused on Core support.  This implementation supports:

*   Basic query execution via SQLAlchemy Core.
*   `SELECT` queries with `JOIN`s.
*   Lightweight `DELETE` statements.

*Note:  ORM support is *not* included.*  This implementation is well-suited for Superset and SQLAlchemy Core applications, but may lack functionality for more complex ORM-based applications or those requiring advanced dialect features.

### Asyncio Support

ClickHouse Connect offers full `asyncio` support.  Check out the [run\_async example](./examples/run_async.py) for details on how to use the client in an asynchronous environment.

### Documentation

For comprehensive documentation, please visit:
[ClickHouse Docs](https://clickhouse.com/docs/integrations/python)