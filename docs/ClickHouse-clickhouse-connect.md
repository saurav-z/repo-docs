# ClickHouse Connect: Your High-Performance Python Bridge to ClickHouse

**Connect to ClickHouse seamlessly with Python, Pandas, Superset, and SQLAlchemy using ClickHouse Connect, a high-performance driver built for speed and compatibility.**

ClickHouse Connect is a robust Python library designed to facilitate efficient data transfer and interaction with ClickHouse databases. It leverages the ClickHouse HTTP interface for maximum compatibility and offers a range of features for various data science and engineering workflows.

**Key Features:**

*   **Pandas Integration:** Directly load and export data using Pandas DataFrames (including NumPy and Arrow-backed).
*   **Array Support:** Seamlessly work with NumPy arrays.
*   **PyArrow Compatibility:** Import and export data using PyArrow Tables.
*   **Polars DataFrame Support:** Enables data exchange with Polars DataFrames.
*   **Superset Connector:** Integrate directly with Apache Superset for easy data visualization and analysis.
*   **SQLAlchemy Core Support:** Offers a lightweight SQLAlchemy dialect for core functionalities like `SELECT` queries, `JOIN`s, and lightweight `DELETE` statements.

**Why Choose ClickHouse Connect?**

*   **Performance:** Optimized for speed and efficiency when interacting with your ClickHouse data.
*   **Flexibility:** Works with a wide range of data structures, including Pandas DataFrames, NumPy arrays, and PyArrow Tables.
*   **Compatibility:** Built on the ClickHouse HTTP interface for broad support across ClickHouse versions.
*   **Superset Integration:** Designed specifically to work with Superset, making data visualization and analysis easier.
*   **Asyncio Support:** Provides an async wrapper for usage in `asyncio` environments.

**Installation**

Install ClickHouse Connect with pip:

```bash
pip install clickhouse-connect
```

**Requirements:**

*   Python 3.9 or higher
*   We officially test against Python 3.9 through 3.13.

**Superset Integration**

ClickHouse Connect seamlessly integrates with Apache Superset. For versions of Superset v2.1.0 and later, no special configuration is needed.  To connect, either use the provided connection dialog within Superset, or specify a SQLAlchemy DSN in the format: `clickhousedb://{username}:{password}@{host}:{port}`.

**SQLAlchemy Support**

ClickHouse Connect includes a streamlined SQLAlchemy dialect supporting core functionalities like `SELECT` queries, `JOIN`s, and lightweight `DELETE` statements. This allows you to leverage SQLAlchemy Core for specific tasks.  Note that full ORM support and advanced dialect functionality are not provided.

**Asyncio Support**

The library offers an async wrapper, enabling you to use the client within an `asyncio` environment. See the [run\_async example](./examples/run_async.py) for more details.

**Documentation**

For complete documentation and usage examples, please visit the [official ClickHouse documentation](https://clickhouse.com/docs/integrations/python).

**Contribute & Learn More**

Explore the source code and contribute to the project on GitHub: [ClickHouse/clickhouse-connect](https://github.com/ClickHouse/clickhouse-connect)