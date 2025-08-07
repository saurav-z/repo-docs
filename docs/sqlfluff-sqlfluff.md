![SQLFluff](https://raw.githubusercontent.com/sqlfluff/sqlfluff/main/images/sqlfluff-wide.png)

# SQLFluff: The SQL Linter for Humans 🚀

**Tired of messy SQL code? SQLFluff is a dialect-flexible and configurable SQL linter that helps you write clean, consistent, and maintainable SQL, supporting automatic fixing of most linting errors.**

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

*   **Dialect Flexibility:** Supports a wide range of SQL dialects, including ANSI, BigQuery, PostgreSQL, Snowflake, and many more.
*   **Configurable Rules:** Customize linting rules to fit your specific coding style and project requirements.
*   **Automatic Fixing:**  Auto-fix most linting errors, saving you time and effort.
*   **Template Support:**  Works seamlessly with Jinja, dbt, and other templating languages.
*   **VS Code Extension:**  Integrates directly into VS Code for real-time linting and error highlighting.
*   **Easy Integration:** Compatible with ELT applications, Jinja templating, and dbt.

## Getting Started

Install SQLFluff using pip:

```bash
pip install sqlfluff
```

Then, lint your SQL files:

```bash
sqlfluff lint test.sql --dialect <your_dialect>
```

You can also automatically fix issues:

```bash
sqlfluff fix test.sql --dialect <your_dialect>
```

For more details, see the [full CLI usage](https://docs.sqlfluff.com/en/stable/perma/cli.html) and [rules reference](https://docs.sqlfluff.com/en/stable/perma/rules.html) in the SQLFluff documentation.

## Supported SQL Dialects

SQLFluff offers robust support for a wide range of SQL dialects:

*   ANSI SQL
*   [Athena](https://aws.amazon.com/athena/)
*   [BigQuery](https://cloud.google.com/bigquery/)
*   [ClickHouse](https://clickhouse.com/)
*   [Databricks](https://databricks.com/)
*   [Db2](https://www.ibm.com/analytics/db2)
*   [Doris](https://doris.apache.org/)
*   [DuckDB](https://duckdb.org/)
*   [Exasol](https://www.exasol.com/)
*   [Greenplum](https://greenplum.org/)
*   [Hive](https://hive.apache.org/)
*   [Impala](https://impala.apache.org/)
*   [MariaDB](https://www.mariadb.com/)
*   [Materialize](https://materialize.com/)
*   [MySQL](https://www.mysql.com/)
*   [Oracle](https://docs.oracle.com/en/database/oracle/oracle-database/21/sqlrf/index.html)
*   [PostgreSQL](https://www.postgresql.org/) (aka Postgres)
*   [Redshift](https://docs.aws.amazon.com/redshift/index.html)
*   [Snowflake](https://www.snowflake.com/)
*   [SOQL](https://developer.salesforce.com/docs/atlas.en-us.soql_sosl.meta/soql_sosl/sforce_api_calls_soql.htm)
*   [SparkSQL](https://spark.apache.org/docs/latest/)
*   [SQLite](https://www.sqlite.org/)
*   [StarRocks](https://www.starrocks.io)
*   [Teradata](https://www.teradata.com/)
*   [Transact-SQL](https://docs.microsoft.com/en-us/sql/t-sql/language-reference) (aka T-SQL)
*   [Trino](https://trino.io/)
*   [Vertica](https://www.vertica.com/)

We are continuously working to expand dialect support.  Please [submit an issue](https://github.com/sqlfluff/sqlfluff/issues) or upvote existing ones to request support for a missing dialect.  Contributions are welcome!

## Template Support

SQLFluff supports the following templating languages:

*   [Jinja](https://jinja.palletsprojects.com/)
*   SQL placeholders
*   [Python format strings](https://docs.python.org/3/library/string.html#format-string-syntax)
*   [dbt](https://www.getdbt.com/) (requires plugin)

## VS Code Extension

Enhance your SQL workflow with the official VS Code extension:

*   [Github Repository](https://github.com/sqlfluff/vscode-sqlfluff)
*   [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=dorzey.vscode-sqlfluff)

## Documentation

Comprehensive documentation is available at [docs.sqlfluff.com](https://docs.sqlfluff.com/en/stable/).  Help us improve the documentation by submitting [issues](https://github.com/sqlfluff/sqlfluff/issues) or pull requests.

## Releases & Changelog

SQLFluff follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).  Check the [release notes](https://docs.sqlfluff.com/en/latest/perma/releasenotes.html) for details on breaking changes and the [changelog](CHANGELOG.md) for a complete history.  New releases are made monthly. Visit [Releases](https://github.com/sqlfluff/sqlfluff/releases) for more information.

## Community

Join our community!

*   **Slack:** [Join our Slack channel](https://join.slack.com/t/sqlfluff/shared_invite/zt-2qtu36kdt-OS4iONPbQ3aCz2DIbYJdWg)
*   **Twitter:** Follow us [on Twitter @SQLFluff](https://twitter.com/SQLFluff)

## Contributing

We welcome contributions!  Check out the [open issues on GitHub](https://github.com/sqlfluff/sqlfluff/issues) and the [contributing guide](CONTRIBUTING.md).  Learn more about the project's architecture in the [architecture documentation](https://docs.sqlfluff.com/en/latest/perma/architecture.html).

## Sponsors

<img src="images/datacoves.png" alt="Datacoves" width="150"/><br>
Thanks to our sponsor, Datacoves.  Learn more at [Datacoves.com](https://datacoves.com/).

[Back to Top](#sqlfluff-the-sql-linter-for-humans-🚀)
[Go to Original Repo](https://github.com/sqlfluff/sqlfluff)