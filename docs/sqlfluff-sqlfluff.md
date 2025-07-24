[![SQLFluff](https://raw.githubusercontent.com/sqlfluff/sqlfluff/main/images/sqlfluff-wide.png)](https://github.com/sqlfluff/sqlfluff)

# SQLFluff: The SQL Linter for Humans

SQLFluff is a powerful, dialect-flexible, and configurable SQL linter that helps you write clean, consistent, and error-free SQL. Perfect for ELT applications, dbt, and Jinja templating, SQLFluff automatically fixes most linting errors, saving you time and effort.

[![PyPi Version](https://img.shields.io/pypi/v/sqlfluff.svg?style=flat-square&logo=PyPi)](https://pypi.org/project/sqlfluff/)
[![PyPi License](https://img.shields.io/pypi/l/sqlfluff.svg?style=flat-square)](https://pypi.org/project/sqlfluff/)
[![PyPi Python Versions](https://img.shields.io/pypi/pyversions/sqlfluff.svg?style=flat-square)](https://pypi.org/project/sqlfluff/)
[![PyPi Status](https://img.shields.io/pypi/status/sqlfluff.svg?style=flat-square)](https://pypi.org/project/sqlfluff/)
[![PyPi Downloads](https://img.shields.io/pypi/dm/sqlfluff?style=flat-square)](https://pypi.org/project/sqlfluff/)

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/sqlfluff/sqlfluff/.github/workflows/ci-tests.yml?logo=github&style=flat-square)](https://github.com/sqlfluff/sqlfluff/actions/workflows/ci-tests.yml?query=branch%3Amain)
[![ReadTheDocs](https://img.shields.io/readthedocs/sqlfluff?style=flat-square&logo=Read%20the%20Docs)](https://sqlfluff.readthedocs.io)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![Docker Pulls](https://img.shields.io/docker/pulls/sqlfluff/sqlfluff?logo=docker&style=flat-square)](https://hub.docker.com/r/sqlfluff/sqlfluff)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20SQLFluff%20Guru-006BFF?style=flat-square)](https://gurubase.io/g/sqlfluff)

## Key Features

*   **Dialect Flexibility:** Supports numerous SQL dialects, including ANSI SQL, BigQuery, Snowflake, and many more.
*   **Configurable Rules:** Customize linting rules to match your team's coding style and project requirements.
*   **Automatic Fixing:** Automatically fix most linting errors with a single command, saving time and effort.
*   **Template Support:** Works seamlessly with Jinja, dbt, and other templating languages.
*   **VS Code Extension:** Integrated VS Code extension for real-time linting and fixing within your IDE.
*   **Docker Image:** Easily integrate SQLFluff into your CI/CD pipelines with the official Docker image.
*   **Comprehensive Documentation:** Detailed documentation with CLI usage and rule reference.

## Supported SQL Dialects

SQLFluff supports a wide range of SQL dialects:

*   ANSI SQL
*   Athena
*   BigQuery
*   ClickHouse
*   Databricks
*   Db2
*   Doris
*   DuckDB
*   Exasol
*   Greenplum
*   Hive
*   Impala
*   MariaDB
*   Materialize
*   MySQL
*   Oracle
*   PostgreSQL (Postgres)
*   Redshift
*   Snowflake
*   SOQL
*   SparkSQL
*   SQLite
*   StarRocks
*   Teradata
*   Transact-SQL (T-SQL)
*   Trino
*   Vertica

We are constantly expanding our dialect support, and contributions are welcome!

## Supported Templates

SQLFluff supports the following templating languages:

*   Jinja (Jinja2)
*   SQL placeholders
*   Python format strings
*   dbt (requires plugin)

## Getting Started

Install SQLFluff using pip:

```bash
pip install sqlfluff
```

Lint your SQL files:

```bash
sqlfluff lint your_file.sql --dialect <your_dialect>
```

Fix linting errors:

```bash
sqlfluff fix your_file.sql --dialect <your_dialect>
```

Explore the [SQLFluff docs](https://docs.sqlfluff.com/en/stable/) for detailed usage instructions and rule configurations.

## Resources

*   **Documentation:** [docs.sqlfluff.com](https://docs.sqlfluff.com/en/stable/)
*   **VS Code Extension:** [SQLFluff VS Code Extension](https://marketplace.visualstudio.com/items?itemName=dorzey.vscode-sqlfluff)
*   **Docker Image:** [Official SQLFluff Docker Image](https://hub.docker.com/r/sqlfluff/sqlfluff)
*   **Online Playground:** [SQLFluff Online](https://online.sqlfluff.com/)
*   **Slack Community:** [Join the SQLFluff Slack](https://join.slack.com/t/sqlfluff/shared_invite/zt-2qtu36kdt-OS4iONPbQ3aCz2DIbYJdWg)
*   **Twitter:** [@SQLFluff](https://twitter.com/SQLFluff)

## Contribute

We welcome contributions! Find open issues on [GitHub](https://github.com/sqlfluff/sqlfluff/issues) and review the [contributing guide](CONTRIBUTING.md).

## Sponsors

<img src="images/datacoves.png" alt="Datacoves" width="150"/><br>
The turnkey analytics stack, find out more at [Datacoves.com](https://datacoves.com/).

---
**[Back to the top](https://github.com/sqlfluff/sqlfluff)**