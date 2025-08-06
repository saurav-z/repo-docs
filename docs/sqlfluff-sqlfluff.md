<!-- SQLFluff Banner -->
![SQLFluff](https://raw.githubusercontent.com/sqlfluff/sqlfluff/main/images/sqlfluff-wide.png)

# SQLFluff: The SQL Linter for Humans 🤖✨

SQLFluff is a dialect-flexible and configurable SQL linter that helps you write cleaner, more consistent, and error-free SQL code.  Get started improving your SQL today! [Check out the original repository](https://github.com/sqlfluff/sqlfluff).

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

*   **Dialect Flexibility:** Supports numerous SQL dialects, including ANSI, BigQuery, PostgreSQL, Snowflake, and many more.  Easily adapt to your specific SQL environment.
*   **Configurable:** Customize linting rules to match your team's coding style and preferences.
*   **Auto-Fixing:** Automatically resolves many linting errors, saving you time and effort.
*   **Template Support:** Works seamlessly with Jinja, dbt, and other templating languages to lint SQL code with dynamic content.
*   **VS Code Extension:** Integrates directly into VS Code for real-time linting and feedback.
*   **Open Source:** Actively maintained, and welcomes contributions from the community.

## Supported Dialects

SQLFluff supports a wide range of SQL dialects.  The following is a non-exhaustive list:

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

*(Note: Dialect support is constantly evolving.  Check the documentation for the most up-to-date list.)*

## Template Support

SQLFluff supports templating to improve code modularity and reusability:

*   Jinja (Jinja2)
*   SQL placeholders (e.g., SQLAlchemy parameters)
*   Python format strings
*   dbt (requires plugin)

## VS Code Extension

Enhance your SQL coding experience with the official SQLFluff VS Code extension:

*   [GitHub Repository](https://github.com/sqlfluff/vscode-sqlfluff)
*   [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=dorzey.vscode-sqlfluff)

## Getting Started

Install SQLFluff and start linting!

```bash
pip install sqlfluff
echo "  SELECT a  +  b FROM tbl;  " > test.sql
sqlfluff lint test.sql --dialect ansi
```

Alternatively, use the [**Official SQLFluff Docker Image**](https://hub.docker.com/r/sqlfluff/sqlfluff) or try it online at [**SQLFluff online**](https://online.sqlfluff.com/).

For detailed CLI usage and rules, see the [SQLFluff documentation](https://docs.sqlfluff.com/en/stable/).

## Documentation

Comprehensive documentation is available at [docs.sqlfluff.com](https://docs.sqlfluff.com/en/stable/).

## Releases

SQLFluff follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and releases new versions monthly. See the [release notes](https://docs.sqlfluff.com/en/latest/perma/releasenotes.html) and [changelog](CHANGELOG.md) for details.

## Community

*   **Slack:** Join our growing community [on Slack](https://join.slack.com/t/sqlfluff/shared_invite/zt-2qtu36kdt-OS4iONPbQ3aCz2DIbYJdWg)
*   **Twitter:** Follow us for updates [@SQLFluff](https://twitter.com/SQLFluff)

## Contributing

We welcome contributions! Check out the [open issues on GitHub](https://github.com/sqlfluff/sqlfluff/issues) and our [contributing guide](CONTRIBUTING.md). Learn more about the architecture [here](https://docs.sqlfluff.com/en/latest/perma/architecture.html).

## Sponsors

<img src="images/datacoves.png" alt="Datacoves" width="150"/><br>
The turnkey analytics stack, find out more at [Datacoves.com](https://datacoves.com/).
```
Key improvements and SEO considerations:

*   **Compelling Title and Hook:**  Replaced the basic heading with a more engaging title ("SQLFluff: The SQL Linter for Humans 🤖✨") and a concise, benefit-driven introduction. This makes it more likely to be read.
*   **Clear Headings:** Organized the content with clear, descriptive headings for better readability and SEO.
*   **Bulleted Key Features:** Highlights the main benefits of using SQLFluff in an easy-to-scan format.  This is much more effective than a paragraph.
*   **Keyword Optimization:** Used relevant keywords like "SQL linter," "SQL code," "SQL dialect," "linting," and "auto-fix" throughout the description.  This improves search engine rankings.
*   **Dialect and Template Sections:** Focused on the variety of SQL dialects supported and templating options, because the user is likely looking for support for their specific flavor of SQL.
*   **Call to Action:** Encouraged the user to get started right away.
*   **Concise Language:**  Simplified language and removed unnecessary jargon for better readability.
*   **Community and Contribution Emphasis:** Highlighted the community aspect and how to contribute.
*   **Consistent Formatting:** Used Markdown consistently for a clean and professional look.
*   **Updated README badges**