# Pylint: Static Code Analysis for Python

**Pylint is a powerful static code analyzer for Python that helps you write cleaner, more maintainable code by detecting errors, enforcing coding standards, and suggesting improvements.**  You can find the original repo [here](https://github.com/pylint-dev/pylint).

## Key Features

*   **Error Detection:** Identifies common coding errors, potential bugs, and style issues.
*   **Coding Standard Enforcement:** Enforces a configurable coding style, ensuring consistency across your codebase.
*   **Code Smell Detection:** Detects "code smells" (indicators of deeper problems) and suggests refactoring opportunities.
*   **Highly Configurable:**  Allows extensive customization through configuration files and plugins to fit your project's needs.
*   **Detailed Analysis:**  Performs deep code analysis, including inference of variable types, to catch subtle issues.
*   **Plugin Ecosystem:** Extends functionality with plugins for popular frameworks and libraries.
*   **Additional Tools:** Includes `pyreverse` (diagram generation) and `symilar` (duplicate code finder).

## Installation

Install Pylint using pip:

```bash
pip install pylint
```

For spell checking (requires `enchant`):

```bash
pip install pylint[spelling]
```

Pylint integrates with many IDEs and editors.  More information in the [documentation](https://pylint.readthedocs.io/en/latest/user_guide/installation/index.html).

## Why Choose Pylint?

Pylint goes beyond simple linting, providing a deeper understanding of your code's structure and behavior. Its inference engine helps catch issues that other linters might miss. While Pylint may be slower than some alternatives, it offers a more thorough and in-depth analysis, leading to higher code quality.

## How to Use Pylint Effectively

Start by running Pylint with the `--errors-only` flag to focus on critical issues.  Then, gradually disable less important messages (e.g., using `--disable=C,R`) and re-enable them as your coding standards evolve.  Leverage configuration files and plugins to tailor Pylint to your project's specific requirements.

## Recommended Linters and Tools

Consider using these tools alongside Pylint for comprehensive code quality checks:

*   [Ruff](https://github.com/astral-sh/ruff)
*   [Flake8](https://github.com/PyCQA/flake8)
*   [Mypy](https://github.com/python/mypy)
*   [Pyright](https://github.com/microsoft/pyright) / [Pylance](https://github.com/microsoft/pyright) / [Pyre](https://github.com/facebook/pyre-check)
*   [Bandit](https://github.com/PyCQA/bandit)
*   [Black](https://github.com/psf/black)
*   [Isort](https://pycqa.github.io/isort/)
*   [Autoflake](https://github.com/myint/autoflake)
*   [Pyupgrade](https://github.com/asottile/pyupgrade)
*   [Pydocstringformatter](https://github.com/DanielNoord/pydocstringformatter)

## Additional Tools Included

*   **pyreverse:** Generates UML diagrams of your Python code (package and class diagrams).
*   **symilar:**  Finds duplicate code within your project.

## Contributing

We welcome contributions! Please review the [code of conduct](https://github.com/pylint-dev/pylint/blob/main/CODE_OF_CONDUCT.md) and the [Contributor Guides](https://pylint.readthedocs.io/en/latest/development_guide/contribute.html) for details on how to contribute.  You can also report bugs or request features on the [contact page](https://pylint.readthedocs.io/en/latest/contact.html#bug-reports-feedback).

## Show Your Usage

Add this badge to your README:

```
![Linting with Pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)
```

## License

Pylint is primarily licensed under the [GPLv2](https://github.com/pylint-dev/pylint/blob/main/LICENSE). The icon files are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Support

For support, please check the [contact information](https://pylint.readthedocs.io/en/latest/contact.html).
Professional support is available as part of the [Tidelift Subscription](https://tidelift.com/subscription/pkg/pypi-pylint?utm_source=pypi-pylint&utm_medium=referral&utm_campaign=readme).