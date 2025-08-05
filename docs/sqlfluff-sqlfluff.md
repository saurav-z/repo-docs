![SQLFluff](https://raw.githubusercontent.com/sqlfluff/sqlfluff/main/images/sqlfluff-wide.png)

# SQLFluff: The SQL Linter for Humans

**SQLFluff** is a powerful, dialect-flexible, and configurable SQL linter that helps you write clean, consistent, and error-free SQL code. Check out the original repository [here](https://github.com/sqlfluff/sqlfluff).

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

## Key Features:

*   **Dialect Flexibility:** Supports a wide range of SQL dialects, including ANSI, BigQuery, PostgreSQL, MySQL, Snowflake, and many more.
*   **Configurable Rules:** Customize linting rules to match your team's coding style.
*   **Automatic Fixes:** Auto-fixes most linting errors, saving you time and effort.
*   **Template Support:** Integrates with Jinja, dbt, and other templating engines.
*   **VS Code Extension:** Provides seamless integration within VS Code for real-time linting and fixing.
*   **Modular Design:** Designed with ELT applications in mind and also works with Jinja templating and dbt.
*   **Easy to Use:** Simple installation and command-line interface (CLI).

## Dialects Supported

SQLFluff supports a wide range of SQL dialects.  Here's a list of currently supported dialects:

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

We aim to make it easy to expand on the support of these dialects and also
add other, currently unsupported, dialects. Please [raise issues](https://github.com/sqlfluff/sqlfluff/issues)
(or upvote any existing issues) to let us know of demand for missing support.

Pull requests from those that know the missing syntax or dialects are especially
welcomed and are the question way for you to get support added. We are happy
to work with any potential contributors on this to help them add this support.
Please raise an issue first for any large feature change to ensure it is a good
fit for this project before spending time on this work.

## Templates Supported

SQLFluff supports the following templates:

*   Jinja (aka Jinja2)
*   SQL placeholders (e.g. SQLAlchemy parameters)
*   Python format strings
*   dbt (requires plugin)

Again, please raise issues if you wish to support more templating languages/syntaxes.

## VS Code Extension

We also have a VS Code extension:

*   [Github Repository](https://github.com/sqlfluff/vscode-sqlfluff)
*   [Extension in VS Code marketplace](https://marketplace.visualstudio.com/items?itemName=dorzey.vscode-sqlfluff)

## Getting Started

Get started with SQLFluff in seconds by installing the package and running `sqlfluff lint` or `sqlfluff fix`.

```shell
$ pip install sqlfluff
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

You can also use the [**Official SQLFluff Docker Image**](https://hub.docker.com/r/sqlfluff/sqlfluff) or have a play using [**SQLFluff online**](https://online.sqlfluff.com/).

For full [CLI usage](https://docs.sqlfluff.com/en/stable/perma/cli.html) and [rules reference](https://docs.sqlfluff.com/en/stable/perma/rules.html), see [the SQLFluff docs](https://docs.sqlfluff.com/en/stable/).

## Documentation

Explore comprehensive documentation at [docs.sqlfluff.com](https://docs.sqlfluff.com/en/stable/) to learn more about SQLFluff.

## Releases

SQLFluff follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and releases new versions monthly.  Find out more in the [release notes](https://docs.sqlfluff.com/en/latest/perma/releasenotes.html) and [changelog](CHANGELOG.md).

## Community

*   **Slack:** Join our fast-growing community [on Slack](https://join.slack.com/t/sqlfluff/shared_invite/zt-2qtu36kdt-OS4iONPbQ3aCz2DIbYJdWg).
*   **Twitter:** Follow us [on Twitter @SQLFluff](https://twitter.com/SQLFluff) for updates and announcements.

## Contributing

Contribute to the project!  Check out the [open issues on GitHub](https://github.com/sqlfluff/sqlfluff/issues) and the [contributing guide](CONTRIBUTING.md).

## Sponsors

<img src="images/datacoves.png" alt="Datacoves" width="150"/><br>
The turnkey analytics stack, find out more at [Datacoves.com](https://datacoves.com/).