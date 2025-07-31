![SQLFluff](https://raw.githubusercontent.com/sqlfluff/sqlfluff/main/images/sqlfluff-wide.png)

# SQLFluff: The SQL Linter for Humans - Improve Your SQL Code Quality

**SQLFluff** is the dialect-flexible and configurable SQL linter that helps you write clean, consistent, and error-free SQL code.  [Check out the original repo!](https://github.com/sqlfluff/sqlfluff)

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
*   **Configurable Rules:** Customize linting rules to match your coding style.
*   **Automated Fixing:** Automatically fixes most linting errors for faster development.
*   **Template Support:** Integrates seamlessly with Jinja, dbt, and more.
*   **VS Code Extension:** Enhance your VS Code experience with the official extension.
*   **Integration Friendly:**  Perfect for ELT pipelines and dbt projects.

## Supported Dialects

SQLFluff offers extensive dialect support, including:

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

**Contribute:**  We encourage contributions to expand dialect support.

## Supported Templates

SQLFluff supports templating for enhanced modularity:

*   Jinja (Jinja2)
*   SQL placeholders
*   Python format strings
*   dbt (requires plugin)

**Suggest a new template:**  Raise an issue to request support for additional templating languages.

## Getting Started

1.  **Install:**  `pip install sqlfluff`
2.  **Lint:**  `sqlfluff lint <your_sql_file.sql> --dialect <your_dialect>`
3.  **Fix:**  `sqlfluff fix <your_sql_file.sql> --dialect <your_dialect>`

You can also use the [**Official SQLFluff Docker Image**](https://hub.docker.com/r/sqlfluff/sqlfluff) or try it out [**online**](https://online.sqlfluff.com/).

For detailed usage, refer to the [CLI documentation](https://docs.sqlfluff.com/en/stable/perma/cli.html) and [rules reference](https://docs.sqlfluff.com/en/stable/perma/rules.html) in the [SQLFluff docs](https://docs.sqlfluff.com/en/stable/).

## VS Code Extension

Enhance your SQLFluff experience with the official VS Code extension:

*   [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=dorzey.vscode-sqlfluff)
*   [GitHub Repository](https://github.com/sqlfluff/vscode-sqlfluff)

## Documentation

Comprehensive documentation is available at [docs.sqlfluff.com](https://docs.sqlfluff.com/en/stable/).  Contribute by submitting [issues](https://github.com/sqlfluff/sqlfluff/issues) or pull requests to improve the documentation.

## Releases

SQLFluff follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).  Find release notes and the [changelog](CHANGELOG.md) for more details.  New releases are made monthly.

## Community

*   **Slack:** Join our fast-growing community [on Slack](https://join.slack.com/t/sqlfluff/shared_invite/zt-2qtu36kdt-OS4iONPbQ3aCz2DIbYJdWg).
*   **Twitter:** Follow us [on Twitter @SQLFluff](https://twitter.com/SQLFluff).

## Contributing

We welcome contributions! Explore the [open issues](https://github.com/sqlfluff/sqlfluff/issues) and consult the [contributing guide](CONTRIBUTING.md).  Learn more about the project's architecture [here](https://docs.sqlfluff.com/en/stable/perma/architecture.html).

## Sponsors

<img src="images/datacoves.png" alt="Datacoves" width="150"/><br>
The turnkey analytics stack, find out more at [Datacoves.com](https://datacoves.com/).