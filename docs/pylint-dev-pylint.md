# Pylint: Static Code Analysis for Python 

**Pylint is the go-to static analysis tool for Python, helping you write cleaner, more maintainable, and error-free code.**  Find the original repo [here](https://github.com/pylint-dev/pylint).

## Key Features

*   **Error Detection:** Identifies potential errors in your Python code, such as undefined variables, missing arguments, and more.
*   **Coding Standard Enforcement:** Enforces a coding style to help maintain consistency and readability.
*   **Code Smell Detection:** Flags "code smells" that may indicate design problems or areas for improvement.
*   **Customizable:** Highly configurable to fit your project's specific needs, with the ability to write plugins.
*   **Inference Engine:**  Its internal code representation (astroid) infers actual values of nodes allowing it to find more issues.
*   **Integration:** Seamlessly integrates with most editors and IDEs.
*   **Plugin Ecosystem:** Offers an ecosystem of existing plugins for popular frameworks and libraries.
*   **Additional Tools:** Includes `pyreverse` (diagram generator) and `symilar` (duplicate code finder).

## Installation

Install Pylint using pip:

```bash
pip install pylint
```

For spell checking (requires `enchant` C library):

```bash
pip install pylint[spelling]
```

## How to Use Pylint

Pylint helps you identify issues in your code by:

*   Starting with the `--errors-only` flag to focus on critical errors.
*   Disabling convention and refactor messages using `--disable=C,R`.
*   Progressively re-evaluating and re-enabling messages as needed.

## Advised Linters Alongside Pylint

To further enhance your code quality and development workflow, consider using these tools alongside Pylint:

*   **Ruff:** Extremely fast linter and formatter with many checks.
*   **Flake8:** A framework for custom checks using AST.
*   **Mypy, Pyright/Pylance, Pyre:** Typing checkers.
*   **Bandit:** Security-focused checks.
*   **Black, isort:** Auto-formatters.
*   **Autoflake:** Removes unused imports and variables.
*   **Pyupgrade:** Updates to newer Python syntax.
*   **Pydocstringformatter:** Automated PEP257 formatting.

## Contributing

We welcome contributions!  Whether it's documentation updates, code improvements, or bug reports, your help is greatly appreciated.  See the [Contributor Guides](https://pylint.readthedocs.io/en/latest/development_guide/contribute.html) for more information.

## License

Pylint is licensed under the [GPLv2](https://github.com/pylint-dev/pylint/blob/main/LICENSE).

## Support

For support and contact information, please visit the [contact page](https://pylint.readthedocs.io/en/latest/contact.html).