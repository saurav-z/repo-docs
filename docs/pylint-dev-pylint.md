# Pylint: Static Code Analysis for Python

**Pylint is a powerful and highly configurable static code analyzer for Python, helping you write cleaner, more maintainable, and error-free code.**  [Get started with Pylint!](https://github.com/pylint-dev/pylint)

## Key Features

*   **Error Detection:** Identifies common programming errors, such as undefined variables, missing docstrings, and incorrect method calls.
*   **Coding Standard Enforcement:** Enforces a coding style based on the PEP 8 style guide (customizable) or other standards.
*   **Code Smell Detection:** Flags potential code smells, such as overly complex functions or duplicated code, to improve code quality.
*   **Refactoring Suggestions:** Offers suggestions on how to improve your code's structure and readability.
*   **Highly Configurable:** Customize Pylint's behavior through configuration files to suit your project's specific needs.
*   **Plugin Ecosystem:** Extensible through plugins for checking specific frameworks and third-party libraries like Django and Pydantic.
*   **Detailed Reporting:** Provides comprehensive reports with error messages, warnings, and code analysis statistics.
*   **Inference Engine:** Pylint's inference engine is not trusting your typing and is inferring the actual values of nodes using its internal code representation (astroid).
*   **Integration:** Integrates with most editors and IDEs.

## Installation

Install Pylint using pip:

```bash
pip install pylint
```

For spelling checks (requires `enchant`):

```bash
pip install pylint[spelling]
```

For more information on installation, see the [Pylint documentation](https://pylint.readthedocs.io/en/latest/user_guide/installation/index.html).

## How Pylint Differs

Pylint's inference engine allows it to find more issues even if your code isn't fully typed. It's more thorough than other linters, including many customizable checks.

## Usage Tips

*   Start with `--errors-only` to focus on critical issues.
*   Disable convention and refactor messages initially ( `--disable=C,R`) and re-enable them as needed.
*   Use plugins to support third-party libraries.

## Tools to Use Alongside Pylint

*   [Ruff](https://github.com/astral-sh/ruff)
*   [Flake8](https://github.com/PyCQA/flake8)
*   [Mypy](https://github.com/python/mypy)
*   [Pyright](https://github.com/microsoft/pyright) / [Pylance](https://github.com/microsoft/pyright)
*   [Pyre](https://github.com/facebook/pyre-check)
*   [Bandit](https://github.com/PyCQA/bandit)
*   [Black](https://github.com/psf/black)
*   [Isort](https://pycqa.github.io/isort/)
*   [Autoflake](https://github.com/myint/autoflake)
*   [Pyupgrade](https://github.com/asottile/pyupgrade)
*   [Pydocstringformatter](https://github.com/DanielNoord/pydocstringformatter)

## Additional Tools

Pylint includes:

*   **Pyreverse:** Generate package and class diagrams.
*   **Symilar:** Find duplicate code.

## Contributing

Contributions are welcome! Check out the [Contributor Guides](https://pylint.readthedocs.io/en/latest/development_guide/contribute.html) for details.

## License

Pylint is primarily licensed under GPLv2.