# Pylint: Your Go-To Static Code Analyzer for Python

Pylint is a powerful and customizable static code analyzer that helps you write cleaner, more maintainable, and error-free Python code.  [Visit the original repository](https://github.com/pylint-dev/pylint).

**Key Features:**

*   **Static Analysis:** Analyzes your Python code without executing it, identifying potential errors, style issues, and code smells.
*   **Coding Standard Enforcement:** Enforces a coding standard, making your code more consistent and readable.
*   **Customizable:** Highly configurable to meet your specific project needs, allowing you to define your own rules and checks through plugins.
*   **Inference Engine:** Advanced inference engine that can understand code with complex structures, uncovering hidden issues.
*   **Integration:** Easily integrates with popular editors and IDEs.
*   **Refactoring Suggestions:** Provides suggestions on how to improve your code's structure and readability.
*   **Additional Tools:** Includes tools like `pyreverse` (for generating diagrams) and `symilar` (for finding duplicate code).
*   **Extensive Plugin Ecosystem:** Supports plugins for popular frameworks and libraries such as Django and Pydantic, expanding its capabilities.

## Installation

Install Pylint using pip:

```bash
pip install pylint
```

For spell checking, install with:

```bash
pip install pylint[spelling]
```

## Why Choose Pylint?

Pylint differentiates itself by its deep code analysis capabilities, making it more thorough than other linters.  It can detect errors even in dynamically typed code.

## How to Use Pylint

Start by using the `--errors-only` flag and `--disable=C,R` to focus on critical issues and then progressively enable more checks as you improve your code.

## Advised Linters Alongside Pylint

Enhance your development workflow with these additional tools:

*   **Ruff:** Extremely fast linter with built-in auto-fix.
*   **Flake8:** A framework for creating custom checks.
*   **Mypy/Pyright/Pylance/Pyre:** Static type checkers.
*   **Bandit:** Security-focused code analysis.
*   **Black/Isort:** Auto-formatters.
*   **Autoflake:** Removes unused imports and variables.
*   **Pyupgrade:** Upgrades to newer Python syntax.
*   **Pydocstringformatter:** Automatically formats docstrings.

## Contributing

Contributions of all kinds are welcome! Please review the [Contributor Guides](https://pylint.readthedocs.io/en/latest/development_guide/contribute.html) and [Code of Conduct](https://github.com/pylint-dev/pylint/blob/main/CODE_OF_CONDUCT.md).

## Show Your Usage

Add the Pylint badge to your README:

```
[![linting-pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
```

Learn more about adding a badge [in the documentation](https://pylint.readthedocs.io/en/latest/user_guide/installation/badge.html).

## License

Pylint is primarily licensed under the [GPLv2](https://github.com/pylint-dev/pylint/blob/main/LICENSE).  The icon files are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Support

For support, please check the [contact information](https://pylint.readthedocs.io/en/latest/contact.html).  Professional support is available through the [Tidelift Subscription](https://tidelift.com/subscription/pkg/pypi-pylint?utm_source=pypi-pylint&utm_medium=referral&utm_campaign=readme).