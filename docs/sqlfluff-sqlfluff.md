![SQLFluff](https://raw.githubusercontent.com/sqlfluff/sqlfluff/main/images/sqlfluff-wide.png)

# SQLFluff: The SQL Linter for Humans

**SQLFluff** is a powerful SQL linter that helps you write clean, consistent, and error-free SQL code.

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

*   **Dialect-Flexible:** Supports a wide range of SQL dialects, including ANSI, BigQuery, Snowflake, PostgreSQL, and many more.
*   **Configurable:** Customize linting rules to match your team's coding style.
*   **Auto-Fixing:** Automatically corrects most linting errors, saving you time and effort.
*   **Template Support:** Works seamlessly with Jinja, dbt, and other templating languages.
*   **VS Code Extension:** Integrates directly with VS Code for real-time linting.
*   **Easy to Use:** Simple installation and intuitive command-line interface.
*   **ELT & dbt Friendly:** Designed with ELT applications and dbt projects in mind.

## Supported Dialects

SQLFluff supports numerous SQL dialects to ensure code quality across various database platforms.  See the [original repository](https://github.com/sqlfluff/sqlfluff) for an up-to-date list and details on dialect support.

## Supported Templates

SQLFluff supports various templating languages, allowing for modular and reusable SQL code.  See the [original repository](https://github.com/sqlfluff/sqlfluff) for an up-to-date list and details on template support.

## Getting Started

To start using SQLFluff, simply install it using pip and then lint or fix your SQL files:

```bash
pip install sqlfluff
sqlfluff lint your_file.sql --dialect <your_dialect>
sqlfluff fix your_file.sql --dialect <your_dialect>
```

For detailed usage, see the [SQLFluff documentation](https://docs.sqlfluff.com/en/stable/).

## Resources

*   [Documentation](https://docs.sqlfluff.com/en/stable/)
*   [VS Code Extension](https://marketplace.visualstudio.com/items?itemName=dorzey.vscode-sqlfluff)
*   [SQLFluff on Slack](https://join.slack.com/t/sqlfluff/shared_invite/zt-2qtu36kdt-OS4iONPbQ3aCz2DIbYJdWg)
*   [SQLFluff on Twitter](https://twitter.com/SQLFluff)
*   [SQLFluff Docker Image](https://hub.docker.com/r/sqlfluff/sqlfluff)
*   [SQLFluff Online](https://online.sqlfluff.com/)

## Contributing

We welcome contributions!  Check out the [SQLFluff GitHub repository](https://github.com/sqlfluff/sqlfluff) for open issues and guidelines on how to contribute.

## Sponsors

<img src="images/datacoves.png" alt="Datacoves" width="150"/><br>
The turnkey analytics stack, find out more at [Datacoves.com](https://datacoves.com/).