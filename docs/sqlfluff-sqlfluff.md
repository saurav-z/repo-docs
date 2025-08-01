<!-- SQLFluff Logo -->
![SQLFluff](https://raw.githubusercontent.com/sqlfluff/sqlfluff/main/images/sqlfluff-wide.png)

# SQLFluff: The SQL Linter for Humans

**SQLFluff is a powerful and flexible SQL linter that helps you write clean, consistent, and error-free SQL code.**

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

*   **Dialect Flexibility:** Supports numerous SQL dialects including ANSI, BigQuery, PostgreSQL, Snowflake, and many more.
*   **Configurable:**  Customize rules and settings to match your team's coding style.
*   **Automatic Fixing:** Auto-fixes most linting errors, saving you time and effort.
*   **Template Support:** Works seamlessly with Jinja, dbt, and other templating engines.
*   **VS Code Extension:** Integrate SQLFluff directly into your VS Code workflow.
*   **Open Source:**  Actively maintained and welcomes contributions from the community.

## Table of Contents

1.  [Dialects Supported](#dialects-supported)
2.  [Templates Supported](#templates-supported)
3.  [VS Code Extension](#vs-code-extension)
4.  [Getting Started](#getting-started)
5.  [Documentation](#documentation)
6.  [Releases](#releases)
7.  [Community](#sqlfluff-on-slack-and-twitter)
8.  [Contributing](#contributing)
9.  [Sponsors](#sponsors)

## Dialects Supported

SQLFluff is designed to support a wide range of SQL dialects. This flexibility makes it an excellent choice for projects using different SQL implementations.  Currently supported dialects include:

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

We are continuously expanding dialect support! If you require support for a specific dialect, please [open an issue](https://github.com/sqlfluff/sqlfluff/issues) or upvote existing requests. Contributions are always welcome!

## Templates Supported

SQLFluff offers robust support for templated SQL, allowing you to lint code using popular templating engines.

*   [Jinja](https://jinja.palletsprojects.com/) (aka Jinja2)
*   SQL placeholders (e.g. SQLAlchemy parameters)
*   [Python format strings](https://docs.python.org/3/library/string.html#format-string-syntax)
*   [dbt](https://www.getdbt.com/) (requires plugin)

## VS Code Extension

Enhance your coding experience with the official VS Code extension for SQLFluff:

*   [Github Repository](https://github.com/sqlfluff/vscode-sqlfluff)
*   [Extension in VS Code marketplace](https://marketplace.visualstudio.com/items?itemName=dorzey.vscode-sqlfluff)

## Getting Started

Get up and running quickly with SQLFluff:

1.  **Install:** `pip install sqlfluff`
2.  **Lint:**  `sqlfluff lint <your_sql_file.sql> --dialect <your_dialect>`
3.  **Fix:** `sqlfluff fix <your_sql_file.sql> --dialect <your_dialect>`

```shell
$ echo "  SELECT a  +  b FROM tbl;  " > test.sql
$ sqlfluff lint test.sql --dialect ansi
== [test.sql] FAIL
L:   1 | P:   1 | LT01 | Expected only single space before 'SELECT' keyword.
                       | Found '  '. [layout.spacing]
L:   1 | P:   1 | LT02 | First line should not be indented.
                       | [layout.indent]
L:   1 | P:   1 | LT13 | Files must not begin with newlines or whitespace.
                       | [layout.start_of_file]
L:   1 | P:  11 | LT01 | Expected only single space before binary operator '+'.
                       | Found '  '. [layout.spacing]
L:   1 | P:  14 | LT01 | Expected only single space before naked identifier.
                       | Found '  '. [layout.spacing]
L:   1 | P:  27 | LT01 | Unnecessary trailing whitespace at end of file.
                       | [layout.spacing]
L:   1 | P:  27 | LT12 | Files must end with a single trailing newline.
                       | [layout.end_of_file]
All Finished 📜 🎉!
```

You can also use the [**Official SQLFluff Docker Image**](https://hub.docker.com/r/sqlfluff/sqlfluff) or try it out online at [**SQLFluff online**](https://online.sqlfluff.com/).

For detailed usage instructions, explore the [CLI documentation](https://docs.sqlfluff.com/en/stable/perma/cli.html) and [rules reference](https://docs.sqlfluff.com/en/stable/perma/rules.html) within the official [SQLFluff docs](https://docs.sqlfluff.com/en/stable/).

## Documentation

Comprehensive documentation is available at [docs.sqlfluff.com](https://docs.sqlfluff.com/en/stable/).  Please submit [issues](https://github.com/sqlfluff/sqlfluff/issues) or pull requests to suggest any improvements!

## Releases

SQLFluff follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).  Breaking changes will generally be restricted to major version releases. Consult the [release notes](https://docs.sqlfluff.com/en/latest/perma/releasenotes.html) and [changelog](CHANGELOG.md) for details on migration. New releases are made monthly. Visit the [Releases](https://github.com/sqlfluff/sqlfluff/releases) page for the latest updates.

## Community (Slack and Twitter)

Join the SQLFluff community!

*   **Slack:**  Join our fast-growing community [on Slack](https://join.slack.com/t/sqlfluff/shared_invite/zt-2qtu36kdt-OS4iONPbQ3aCz2DIbYJdWg)
*   **Twitter:** Follow us [on Twitter @SQLFluff](https://twitter.com/SQLFluff) for announcements and updates.

## Contributing

We appreciate contributions! Check out the [open issues on GitHub](https://github.com/sqlfluff/sqlfluff/issues) and the [contributing guide](CONTRIBUTING.md).  Learn more about the project's architecture at [docs.sqlfluff.com/en/stable/perma/architecture.html](https://docs.sqlfluff.com/en/stable/perma/architecture.html).

## Sponsors

<img src="images/datacoves.png" alt="Datacoves" width="150"/><br>
The turnkey analytics stack, find out more at [Datacoves.com](https://datacoves.com/).

[Back to top](#sqlfluff-the-sql-linter-for-humans)
```

Key improvements and SEO considerations:

*   **Clear Title and Hook:**  Uses a strong title and a concise one-sentence hook to grab attention. The hook includes relevant keywords "SQL linter" and emphasizes the value ("write clean, consistent, and error-free SQL code").
*   **Keyword Optimization:**  Includes relevant keywords throughout the document (e.g., "SQL linter," "SQL dialect," "linting errors," "VS Code extension").
*   **Structured Headings:** Uses clear headings and subheadings to improve readability and SEO ranking.
*   **Bulleted Key Features:**  Highlights the main features in a concise and easily scannable format.
*   **Dialect List Optimization:**  The dialect list is now more readable, with links, and the introduction is improved for SEO.
*   **Strong Call to Action:** Encourages users to contribute, and links back to the repository.
*   **Community Links:**  Provides links to the Slack and Twitter communities.
*   **Concise and Informative:** Avoids unnecessary verbosity, focusing on the most important information.
*   **Anchor Links:** Adds a "Back to top" anchor at the bottom.
*   **Image Alt Text:** Corrected for accessibility and SEO.
*   **Improved formatting and clarity:**  Improved readability and flow.
*   **Link to Original Repo:** This is the main goal of the project.