![SQLFluff](https://raw.githubusercontent.com/sqlfluff/sqlfluff/main/images/sqlfluff-wide.png)

# SQLFluff: The SQL Linter for Humans 🧑‍💻

**SQLFluff** is a powerful, dialect-flexible, and configurable SQL linter that helps you write clean, consistent, and error-free SQL code.  Improve your SQL code today!

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

*   **Dialect-Flexible:** Supports a wide range of SQL dialects, including ANSI, BigQuery, PostgreSQL, MySQL, Snowflake, and many more.
*   **Configurable:** Customize linting rules to match your team's coding style.
*   **Auto-Fixing:** Automatically fixes most linting errors, saving you time and effort.
*   **Template Support:** Works seamlessly with Jinja, dbt, and other templating languages.
*   **VS Code Extension:**  Integrates directly into your VS Code environment for real-time linting and error highlighting.
*   **ELT Focused:** Designed with Extract, Load, Transform (ELT) applications and dbt in mind.
*   **Docker Support:** Easily integrate SQLFluff into your CI/CD pipelines with our official Docker image.

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

SQLFluff is designed to support a wide array of SQL dialects. We aim to make it easy to expand on the support of these dialects and also add other, currently unsupported, dialects. Please [raise issues](https://github.com/sqlfluff/sqlfluff/issues)
(or upvote any existing issues) to let us know of demand for missing support.

**SQLFluff** currently supports the following SQL dialects (though perhaps not in full):

*   ANSI SQL
*   [Athena](https://aws.amazon.com/athena/)
*   [BigQuery](https://cloud.google.com/bigquery/)
*   [ClickHouse](https://clickhouse.com/)
*   [Databricks](https://databricks.com/) (note: this extends the `sparksql` dialect with
    [Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/index.html) syntax).
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

Pull requests from those that know the missing syntax or dialects are especially welcomed and are the question way for you to get support added. We are happy
to work with any potential contributors on this to help them add this support.
Please raise an issue first for any large feature change to ensure it is a good
fit for this project before spending time on this work.

## Templates Supported

SQLFluff supports templating to introduce flexibility and reusability.

**SQLFluff** supports the following templates:

*   [Jinja](https://jinja.palletsprojects.com/) (aka Jinja2)
*   SQL placeholders (e.g. SQLAlchemy parameters)
*   [Python format strings](https://docs.python.org/3/library/string.html#format-string-syntax)
*   [dbt](https://www.getdbt.com/) (requires plugin)

Again, please raise issues if you wish to support more templating languages/syntaxes.

## VS Code Extension

Enhance your SQLFluff experience with our VS Code extension:

*   [Github Repository](https://github.com/sqlfluff/vscode-sqlfluff)
*   [Extension in VS Code marketplace](https://marketplace.visualstudio.com/items?itemName=dorzey.vscode-sqlfluff)

# Getting Started

Install and start linting or fixing your SQL with just a few commands:

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

You can also leverage the [**Official SQLFluff Docker Image**](https://hub.docker.com/r/sqlfluff/sqlfluff)
or try it out [**SQLFluff online**](https://online.sqlfluff.com/).

For comprehensive [CLI usage](https://docs.sqlfluff.com/en/stable/perma/cli.html) and
[rules reference](https://docs.sqlfluff.com/en/stable/perma/rules.html), explore
[the SQLFluff docs](https://docs.sqlfluff.com/en/stable/).

# Documentation

Access the full documentation at [docs.sqlfluff.com](https://docs.sqlfluff.com/en/stable/).
For any additions, corrections, or clarifications, please submit
[issues](https://github.com/sqlfluff/sqlfluff/issues) or pull requests.

# Releases

**SQLFluff** follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
with breaking changes reserved for major releases. For details on breaking changes and migration, see our
[release notes](https://docs.sqlfluff.com/en/latest/perma/releasenotes.html) and the
[changelog](CHANGELOG.md).

New releases are made monthly. Find out more at
[Releases](https://github.com/sqlfluff/sqlfluff/releases).

# SQLFluff on Slack

Join our fast-growing community
[on Slack](https://join.slack.com/t/sqlfluff/shared_invite/zt-2qtu36kdt-OS4iONPbQ3aCz2DIbYJdWg).

# SQLFluff on Twitter

Stay up-to-date by following us [on Twitter @SQLFluff](https://twitter.com/SQLFluff) for announcements
and other related posts.

# Contributing

We are grateful to all our [contributors](https://github.com/sqlfluff/sqlfluff/graphs/contributors).

If you want to understand more about the architecture of **SQLFluff**, you can
find [more here](https://docs.sqlfluff.com/en/latest/perma/architecture.html).

If you would like to contribute, check out the
[open issues on GitHub](https://github.com/sqlfluff/sqlfluff/issues) and the guide to [contributing](CONTRIBUTING.md).

# Sponsors

<img src="images/datacoves.png" alt="Datacoves" width="150"/><br>
The turnkey analytics stack, find out more at [Datacoves.com](https://datacoves.com/).

[Back to Top](#sqlfluff-the-sql-linter-for-humans-%F0%9F%91%A8%E2%80%8D%F0%9F%92%BB)
```

Key improvements and SEO considerations:

*   **Clear Headline:**  Strong headline that includes the primary keyword "SQL linter."
*   **One-Sentence Hook:** Provides a concise and compelling introduction.
*   **Keyword Optimization:**  Uses keywords like "SQL linter," "SQL code," "linting," "dialect-flexible," and specific SQL dialects naturally throughout the text.
*   **Bulleted Key Features:** Makes the benefits of SQLFluff immediately visible.
*   **Well-Organized Structure:**  Uses headings, subheadings, and clear sections for readability.
*   **Internal Linking:** Includes a "Back to Top" link for better navigation.
*   **Call to Action:** Encourages users to contribute.
*   **Removed redundant links:** Optimized the existing links.
*   **Concise language:** Simplified phrases and wording for better readability.