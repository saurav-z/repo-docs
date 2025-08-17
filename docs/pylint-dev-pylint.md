# Pylint: Static Code Analysis for Python

**Pylint helps you write cleaner, more maintainable Python code by identifying potential errors and enforcing coding standards.**  [View the Pylint Repository on GitHub](https://github.com/pylint-dev/pylint)

## Key Features:

*   **Static Analysis:** Analyzes your code without execution, detecting errors, and enforcing style guidelines.
*   **Error Detection:** Identifies potential bugs, unused variables, and other common coding issues.
*   **Coding Standard Enforcement:** Enforces PEP 8 and other coding standards to ensure consistency.
*   **Code Smell Detection:**  Highlights areas of code that may indicate design problems or maintainability issues.
*   **Customization:** Highly configurable with plugins for extending functionality.
*   **Integration:** Seamlessly integrates with popular IDEs and editors.
*   **Inference Engine:** Pylint's advanced inference engine, based on astroid, allows for more accurate analysis, even in dynamically typed code.

## Installation

Install Pylint using pip:

```bash
pip install pylint
```

For spell-checking support, install with the `spelling` extra:

```bash
pip install pylint[spelling]
```

## Why Choose Pylint?

Pylint's thoroughness, including its powerful inference capabilities, sets it apart from other linters.  It offers a comprehensive set of checks, including opinionated ones that you can enable or disable based on your project's needs.

## How to Use Pylint

Start by running `pylint` on your Python files or directories. It's recommended to start with the `--errors-only` flag and disable convention and refactor messages with `--disable=C,R`, progressively enabling more checks as you improve your code quality. Pylint is highly configurable, allowing you to customize its behavior to fit your project's specific requirements.

## Advised Linters

For a comprehensive linting and code quality workflow, consider using Pylint alongside these tools:

*   **Ruff:** Very fast linter with auto-fix and a large number of checks.
*   **Flake8:** Framework to implement custom checks.
*   **Mypy, Pyright / Pylance, Pyre:** Static typing checkers.
*   **Bandit:** Security-focused checks.
*   **Black and isort:** Auto-formatters.
*   **Autoflake:** Removes unused imports and variables.
*   **Pyupgrade:** Upgrades to newer Python syntax.
*   **Pydocstringformatter:** Enforces PEP 257 docstring formatting.

## Additional Tools

Pylint ships with these useful utilities:

*   **Pyreverse:** Generates package and class diagrams.
*   **Symilar:** Detects duplicate code.

## Contributing

We welcome contributions!  Please review the [Contributor Guides](https://pylint.readthedocs.io/en/latest/development_guide/contribute.html) and [Code of Conduct](https://github.com/pylint-dev/pylint/blob/main/CODE_OF_CONDUCT.md).  You can also [create issues](https://pylint.readthedocs.io/en/latest/contact.html#bug-reports-feedback) for bugs or feature requests.

## License

Pylint is licensed under the [GPLv2](https://github.com/pylint-dev/pylint/blob/main/LICENSE).
The icon files are licensed under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

## Support

For support, please see the [contact information](https://pylint.readthedocs.io/en/latest/contact.html).

## Tidelift Subscription

Professional support for pylint is available as part of the [Tidelift Subscription](https://tidelift.com/subscription/pkg/pypi-pylint?utm_source=pypi-pylint&utm_medium=referral&utm_campaign=readme).