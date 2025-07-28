# SQLFluff: The SQL Linter for Humans - Improve Your SQL Code Today!

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

![SQLFluff](https://raw.githubusercontent.com/sqlfluff/sqlfluff/main/images/sqlfluff-wide.png)

**SQLFluff** is a powerful, dialect-flexible, and configurable SQL linter designed to help you write cleaner, more consistent, and error-free SQL code.  Built with [ELT](https://www.techtarget.com/searchdatamanagement/definition/Extract-Load-Transform-ELT) applications and dbt in mind, SQLFluff automatically fixes most linting errors, freeing you to focus on building great SQL. Check out the [original repo](https://github.com/sqlfluff/sqlfluff) for more details.

**Key Features:**

*   **Dialect Flexibility:** Supports a wide range of SQL dialects.
*   **Configurable:** Highly customizable to fit your coding style.
*   **Auto-Fixing:** Automatically corrects most linting errors.
*   **Template Support:** Integrates with Jinja, dbt, and other templating engines.
*   **VS Code Extension:** Provides seamless linting within your VS Code environment.

## Table of Contents

1.  [Dialects Supported](#dialects-supported)
2.  [Templates Supported](#templates-supported)
3.  [VS Code Extension](#vs-code-extension)
4.  [Getting Started](#getting-started)
5.  [Documentation](#documentation)
6.  [Releases](#releases)
7.  [SQLFluff on Slack](#sqlfluff-on-slack)
8.  [SQLFluff on Twitter](#sqlfluff-on-twitter)
9.  [Contributing](#contributing)
10. [Sponsors](#sponsors)

## Dialects Supported

SQLFluff offers broad support for various SQL dialects, allowing you to maintain code quality across different database platforms.  Currently supported dialects include:

*   ANSI SQL
*   Athena
*   BigQuery
*   ClickHouse
*   Databricks (with Unity Catalog)
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

We continuously strive to expand dialect support.  Please [submit issues](https://github.com/sqlfluff/sqlfluff/issues) to request or upvote missing dialects or features!

## Templates Supported

SQLFluff enhances SQL's modularity through templating support, crucial for modern data engineering workflows. SQLFluff supports:

*   Jinja (Jinja2)
*   SQL placeholders (e.g., SQLAlchemy)
*   Python format strings
*   dbt (requires plugin)

## VS Code Extension

Improve your SQL coding experience directly within VS Code:

*   [VS Code Extension](https://marketplace.visualstudio.com/items?itemName=dorzey.vscode-sqlfluff)
*   [GitHub Repository](https://github.com/sqlfluff/vscode-sqlfluff)

## Getting Started

Install SQLFluff and begin linting your SQL code:

```bash
pip install sqlfluff
echo "  SELECT a  +  b FROM tbl;  " > test.sql
sqlfluff lint test.sql --dialect ansi
```

Or use the [Official SQLFluff Docker Image](https://hub.docker.com/r/sqlfluff/sqlfluff) or [SQLFluff online](https://online.sqlfluff.com/).

For comprehensive CLI usage and rule details, explore the [SQLFluff docs](https://docs.sqlfluff.com/en/stable/).

## Documentation

Find detailed documentation at [docs.sqlfluff.com](https://docs.sqlfluff.com/en/stable/).  Help improve the documentation by submitting [issues](https://github.com/sqlfluff/sqlfluff/issues) or pull requests.

## Releases

SQLFluff adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), ensuring stable releases.  Refer to the [release notes](https://docs.sqlfluff.com/en/latest/perma/releasenotes.html) and [changelog](CHANGELOG.md) for breaking changes and migration guidance. Monthly releases. Check out [Releases](https://github.com/sqlfluff/sqlfluff/releases).

## SQLFluff on Slack

Join our active community on [Slack](https://join.slack.com/t/sqlfluff/shared_invite/zt-2qtu36kdt-OS4iONPbQ3aCz2DIbYJdWg)!

## SQLFluff on Twitter

Stay updated on SQLFluff news: follow us [on Twitter @SQLFluff](https://twitter.com/SQLFluff).

## Contributing

We welcome contributions! Review the [open issues](https://github.com/sqlfluff/sqlfluff/issues) or the [contributing guide](CONTRIBUTING.md). Learn about the architecture [here](https://docs.sqlfluff.com/en/latest/perma/architecture.html).

## Sponsors

<img src="images/datacoves.png" alt="Datacoves" width="150"/><br>
Discover the turnkey analytics stack at [Datacoves.com](https://datacoves.com/).