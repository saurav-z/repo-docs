![SQLFluff](https://raw.githubusercontent.com/sqlfluff/sqlfluff/main/images/sqlfluff-wide.png)

# SQLFluff: The SQL Linter for Humans - Improve Your SQL Code Today

**SQLFluff**, a dialect-flexible and configurable SQL linter, is designed to help you write cleaner, more consistent, and error-free SQL code, making it ideal for ELT applications and dbt projects. Check out the original repository [here](https://github.com/sqlfluff/sqlfluff).

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

*   **Dialect Flexibility:** Supports a wide range of SQL dialects, including ANSI, BigQuery, PostgreSQL, MySQL, Snowflake, and many more.
*   **Configurable Rules:** Customize linting rules to match your team's coding style and preferences.
*   **Auto-Fixing:** Automatically fixes most linting errors, saving you time and effort.
*   **Template Support:** Integrates with Jinja, dbt, and other templating engines.
*   **VS Code Extension:** Enhance your development workflow with the official VS Code extension.
*   **Easy to Integrate:** Simple installation and integration with your existing CI/CD pipelines.

## Supported SQL Dialects

SQLFluff supports a wide variety of SQL dialects, including:

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

## Supported Templates

SQLFluff supports the following templating languages:

*   Jinja (Jinja2)
*   SQL placeholders (e.g. SQLAlchemy parameters)
*   Python format strings
*   dbt (requires plugin)

## Getting Started

Install SQLFluff using pip:

```bash
pip install sqlfluff
```

Lint your SQL files:

```bash
sqlfluff lint your_file.sql --dialect <dialect>
```

Fix linting errors:

```bash
sqlfluff fix your_file.sql --dialect <dialect>
```

For detailed instructions, CLI usage, and rule references, see the [SQLFluff documentation](https://docs.sqlfluff.com/en/stable/).

## Resources

*   **Documentation:** [docs.sqlfluff.com](https://docs.sqlfluff.com/en/stable/)
*   **Releases:** [Releases](https://github.com/sqlfluff/sqlfluff/releases)
*   **SQLFluff on Slack:** [Join our Slack community](https://join.slack.com/t/sqlfluff/shared_invite/zt-2qtu36kdt-OS4iONPbQ3aCz2DIbYJdWg)
*   **SQLFluff on Twitter:** [@SQLFluff](https://twitter.com/SQLFluff)
*   **VS Code Extension:** [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=dorzey.vscode-sqlfluff)
*   **Docker Image:** [Official SQLFluff Docker Image](https://hub.docker.com/r/sqlfluff/sqlfluff)
*   **Online Playgroud:** [SQLFluff online](https://online.sqlfluff.com/)
*   **Contributing:** [Contributing Guide](CONTRIBUTING.md)
*   **Architecture:** [Architecture](https://docs.sqlfluff.com/en/stable/perma/architecture.html)
*   **Release notes:** [Release notes](https://docs.sqlfluff.com/en/latest/perma/releasenotes.html)
*   **Changelog:** [CHANGELOG.md](CHANGELOG.md)

## Contribute

We welcome contributions! Check out the [open issues](https://github.com/sqlfluff/sqlfluff/issues) and our [contributing guidelines](CONTRIBUTING.md).

## Sponsors

<img src="images/datacoves.png" alt="Datacoves" width="150"/><br>
The turnkey analytics stack, find out more at [Datacoves.com](https://datacoves.com/).