<!-- SQLFluff Banner -->
![SQLFluff](https://raw.githubusercontent.com/sqlfluff/sqlfluff/main/images/sqlfluff-wide.png)

# SQLFluff: The SQL Linter for Humans ✨

**SQLFluff** is a powerful, dialect-flexible SQL linter and auto-formatter that helps you write clean, consistent, and error-free SQL, improving code quality and streamlining your workflow. [Explore the original repository](https://github.com/sqlfluff/sqlfluff).

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

*   **Dialect Flexibility:** Supports a wide range of SQL dialects, with the ability to easily add more.
*   **Automated Fixes:** Auto-formats most linting errors, saving you time and effort.
*   **Template Support:** Works seamlessly with Jinja, dbt, and other templating engines.
*   **Configurable:** Customize linting rules to match your team's coding style.
*   **Integrations:** Includes a VS Code extension for convenient use within your IDE, as well as a Docker image.

## Supported SQL Dialects

SQLFluff offers robust support for numerous SQL dialects, including:

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

If your specific dialect isn't listed, [raise an issue](https://github.com/sqlfluff/sqlfluff/issues) to request support, or better yet, contribute!

## Supported Templates

Enhance your SQL code with SQLFluff's templating support:

*   Jinja
*   SQL placeholders
*   Python format strings
*   dbt (requires plugin)

## Getting Started

1.  **Installation:**
    ```bash
    pip install sqlfluff
    ```
2.  **Linting:**
    ```bash
    echo "  SELECT a  +  b FROM tbl;  " > test.sql
    sqlfluff lint test.sql --dialect ansi
    ```

3.  **Fixing:**
   ```bash
   sqlfluff fix test.sql --dialect ansi
   ```

Explore the [CLI usage](https://docs.sqlfluff.com/en/stable/perma/cli.html) and [rules reference](https://docs.sqlfluff.com/en/stable/perma/rules.html) in the full documentation.  You can also use the [**Official SQLFluff Docker Image**](https://hub.docker.com/r/sqlfluff/sqlfluff) or the [**SQLFluff online**](https://online.sqlfluff.com/).

## Resources

*   **Documentation:** [docs.sqlfluff.com](https://docs.sqlfluff.com/en/stable/)
*   **Releases:** [Releases](https://github.com/sqlfluff/sqlfluff/releases) (Monthly Releases)
*   **VS Code Extension:** [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=dorzey.vscode-sqlfluff)
*   **Slack Community:** [Join us on Slack](https://join.slack.com/t/sqlfluff/shared_invite/zt-2qtu36kdt-OS4iONPbQ3aCz2DIbYJdWg)
*   **Twitter:** [@SQLFluff](https://twitter.com/SQLFluff)
*   **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)

## Sponsors

<img src="images/datacoves.png" alt="Datacoves" width="150"/><br>
The turnkey analytics stack, find out more at [Datacoves.com](https://datacoves.com/).