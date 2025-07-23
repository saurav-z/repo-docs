![SQLFluff](https://raw.githubusercontent.com/sqlfluff/sqlfluff/main/images/sqlfluff-wide.png)

# SQLFluff: The SQL Linter for Humans - Improve Your SQL Code Quality

**SQLFluff** is the ultimate SQL linter, helping you write cleaner, more consistent, and error-free SQL code.  This dialect-flexible and configurable tool is designed with ELT applications, dbt, and Jinja templating in mind, making it an essential tool for any SQL developer.  [See the original repository](https://github.com/sqlfluff/sqlfluff)

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

**Key Features:**

*   **Dialect Flexibility:** Supports a wide range of SQL dialects.
*   **Configurable:** Customize linting rules to match your style.
*   **Auto-Fixing:** Automatically fixes most linting errors, saving you time.
*   **Template Support:** Works seamlessly with Jinja, dbt, and other templating languages.
*   **VS Code Extension:** Integrated extension for convenient linting within your IDE.
*   **Docker Image:** Easy deployment with the official Docker image.
*   **Community Driven:** Thriving community on Slack and Twitter.

## Dialects Supported

SQLFluff supports a comprehensive list of SQL dialects, ensuring broad compatibility.

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
*   PostgreSQL (aka Postgres)
*   Redshift
*   Snowflake
*   SOQL
*   SparkSQL
*   SQLite
*   StarRocks
*   Teradata
*   Transact-SQL (aka T-SQL)
*   Trino
*   Vertica

We are continuously working to expand dialect support.  Please [raise issues](https://github.com/sqlfluff/sqlfluff/issues) to request new dialects or upvote existing ones.  Contributions are welcome!

## Templates Supported

SQLFluff enhances SQL's capabilities by supporting various templating languages.

*   Jinja (aka Jinja2)
*   SQL placeholders (e.g. SQLAlchemy parameters)
*   Python format strings
*   dbt (requires plugin)

## VS Code Extension

Improve your workflow with the SQLFluff VS Code extension.

*   [Github Repository](https://github.com/sqlfluff/vscode-sqlfluff)
*   [Extension in VS Code marketplace](https://marketplace.visualstudio.com/items?itemName=dorzey.vscode-sqlfluff)

## Getting Started

Get up and running quickly with SQLFluff.

1.  **Install:** `pip install sqlfluff`
2.  **Lint your SQL:** `sqlfluff lint your_file.sql --dialect ansi`
3.  **Fix your SQL:** `sqlfluff fix your_file.sql --dialect ansi`

Explore the [CLI usage](https://docs.sqlfluff.com/en/stable/perma/cli.html) and [rules reference](https://docs.sqlfluff.com/en/stable/perma/rules.html) in the [SQLFluff docs](https://docs.sqlfluff.com/en/stable/) for more details.

## Documentation

Access comprehensive documentation at [docs.sqlfluff.com](https://docs.sqlfluff.com/en/stable/) for detailed information and guidance.  Help us improve the documentation by submitting [issues](https://github.com/sqlfluff/sqlfluff/issues) or pull requests.

## Releases

SQLFluff follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).  Find release notes and details on breaking changes in the [release notes](https://docs.sqlfluff.com/en/latest/perma/releasenotes.html) and [changelog](CHANGELOG.md). New releases are made monthly.

## Community

Join the SQLFluff community and connect with other users.

*   **Slack:** [Join us on Slack](https://join.slack.com/t/sqlfluff/shared_invite/zt-2qtu36kdt-OS4iONPbQ3aCz2DIbYJdWg)
*   **Twitter:** Follow us [@SQLFluff](https://twitter.com/SQLFluff)

## Contributing

Contribute to SQLFluff and help improve the project.

*   Explore the [architecture](https://docs.sqlfluff.com/en/latest/perma/architecture.html) to understand the project's structure.
*   Review [open issues on GitHub](https://github.com/sqlfluff/sqlfluff/issues) to find areas to contribute.
*   See the [contributing guide](CONTRIBUTING.md) for details.

## Sponsors

<img src="images/datacoves.png" alt="Datacoves" width="150"/><br>
The turnkey analytics stack, find out more at [Datacoves.com](https://datacoves.com/).