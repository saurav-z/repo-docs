![SQLFluff](https://raw.githubusercontent.com/sqlfluff/sqlfluff/main/images/sqlfluff-wide.png)

# SQLFluff: The SQL Linter for Humans

**SQLFluff is a dialect-flexible and configurable SQL linter that helps you write clean, consistent, and error-free SQL code.**  Improve your SQL code quality and maintainability with this powerful tool.  Get started today and [visit the SQLFluff repository](https://github.com/sqlfluff/sqlfluff).

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

*   **Dialect Flexibility:** Supports a wide range of SQL dialects.
*   **Configurable Rules:** Customize linting rules to match your style.
*   **Automatic Fixes:** Auto-fix most linting errors to save time.
*   **Template Support:** Integrates with Jinja, dbt, and other templating languages.
*   **VS Code Extension:**  Seamlessly integrate SQLFluff into your workflow.

## Supported SQL Dialects

SQLFluff supports a large and growing list of SQL dialects, including:

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

(See the original README for full dialect descriptions.)

## Templates Supported

*   Jinja (Jinja2)
*   SQL placeholders (e.g. SQLAlchemy parameters)
*   Python format strings
*   dbt (requires plugin)

## Getting Started

Install and use SQLFluff with a simple `pip install sqlfluff` and then:

```bash
$ echo "  SELECT a  +  b FROM tbl;  " > test.sql
$ sqlfluff lint test.sql --dialect ansi
```

(See the original README for full Getting Started instructions and an example.)

For full CLI usage and rules reference, see the [SQLFluff docs](https://docs.sqlfluff.com/en/stable/). You can also use the [Official SQLFluff Docker Image](https://hub.docker.com/r/sqlfluff/sqlfluff) or [SQLFluff online](https://online.sqlfluff.com/).

## Additional Resources

*   **Documentation:** [docs.sqlfluff.com](https://docs.sqlfluff.com/en/stable/)
*   **VS Code Extension:**
    *   [GitHub Repository](https://github.com/sqlfluff/vscode-sqlfluff)
    *   [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=dorzey.vscode-sqlfluff)
*   **Releases:** [Releases](https://github.com/sqlfluff/sqlfluff/releases)
*   **Changelog:** [CHANGELOG.md](CHANGELOG.md)
*   **Slack:** [SQLFluff on Slack](https://join.slack.com/t/sqlfluff/shared_invite/zt-2qtu36kdt-OS4iONPbQ3aCz2DIbYJdWg)
*   **Twitter:** [SQLFluff on Twitter @SQLFluff](https://twitter.com/SQLFluff)
*   **Contributing:** [Contributing Guide](CONTRIBUTING.md)

## Sponsors

<img src="images/datacoves.png" alt="Datacoves" width="150"/><br>
The turnkey analytics stack, find out more at [Datacoves.com](https://datacoves.com/).