<!-- SQLFluff Banner -->
![SQLFluff](https://raw.githubusercontent.com/sqlfluff/sqlfluff/main/images/sqlfluff-wide.png)

# SQLFluff: The SQL Linter for Humans 🚀

SQLFluff is a powerful, dialect-flexible SQL linter that helps you write clean, consistent, and error-free SQL code. Improve your SQL code quality and efficiency with automatic fixes and customization!

[Visit the original repo on GitHub](https://github.com/sqlfluff/sqlfluff)

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

*   **Dialect Flexibility:** Supports a wide range of SQL dialects, including ANSI SQL, BigQuery, PostgreSQL, MySQL, Snowflake, and many more.
*   **Automatic Fixing:**  SQLFluff automatically fixes most linting errors, saving you time and effort.
*   **Configurable:** Customize linting rules to match your specific coding style and project requirements.
*   **Template Support:**  Works seamlessly with Jinja, dbt, and other templating languages.
*   **VS Code Extension:**  Integrates directly into VS Code for real-time linting and code quality checks.

## Supported SQL Dialects

SQLFluff offers extensive support for the following SQL dialects:

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
*   PostgreSQL
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

  *   *(...and more!)*

We encourage contributions for adding support for more dialects - see the [Contributing](#contributing) section below.

## Templates Supported

Enhance your SQL code with support for templating languages:

*   Jinja (Jinja2)
*   SQL placeholders (e.g. SQLAlchemy parameters)
*   Python format strings
*   dbt (requires plugin)

## VS Code Extension

Integrate SQLFluff directly into your VS Code workflow:

*   [Github Repository](https://github.com/sqlfluff/vscode-sqlfluff)
*   [Extension in VS Code marketplace](https://marketplace.visualstudio.com/items?itemName=dorzey.vscode-sqlfluff)

## Getting Started

Get started with SQLFluff by installing and linting your SQL files.

```bash
$ pip install sqlfluff
$ echo "  SELECT a  +  b FROM tbl;  " > test.sql
$ sqlfluff lint test.sql --dialect ansi
# (Output from the original README)
```

You can also use the official [**SQLFluff Docker Image**](https://hub.docker.com/r/sqlfluff/sqlfluff) or try it out with [**SQLFluff online**](https://online.sqlfluff.com/).  For detailed CLI usage, refer to the [SQLFluff docs](https://docs.sqlfluff.com/en/stable/).

## Documentation

Find comprehensive documentation and guides at [docs.sqlfluff.com](https://docs.sqlfluff.com/en/stable/).

## Releases

SQLFluff follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).  Refer to the [release notes](https://docs.sqlfluff.com/en/latest/perma/releasenotes.html) for details on breaking changes. Check the [changelog](CHANGELOG.md) for more. New releases are made monthly. See [Releases](https://github.com/sqlfluff/sqlfluff/releases).

## Community

*   **Slack:** Join our fast-growing community [on Slack](https://join.slack.com/t/sqlfluff/shared_invite/zt-2qtu36kdt-OS4iONPbQ3aCz2DIbYJdWg)
*   **Twitter:** Follow us [on Twitter @SQLFluff](https://twitter.com/SQLFluff)

## Contributing

We welcome contributions!  Check out the [open issues on GitHub](https://github.com/sqlfluff/sqlfluff/issues) and see the guide to [contributing](CONTRIBUTING.md).  Learn more about SQLFluff's architecture at [docs.sqlfluff.com/en/latest/perma/architecture.html](https://docs.sqlfluff.com/en/latest/perma/architecture.html).

## Sponsors

<img src="images/datacoves.png" alt="Datacoves" width="150"/><br>
The turnkey analytics stack, find out more at [Datacoves.com](https://datacoves.com/).