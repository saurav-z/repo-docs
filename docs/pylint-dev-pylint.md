# Pylint: Static Code Analysis for Python

**Pylint is a powerful static code analyzer that helps you write cleaner, more maintainable Python code.**  Check out the original repository [here](https://github.com/pylint-dev/pylint).

## Key Features:

*   **Static Analysis:** Analyzes your Python code without executing it, identifying potential errors, style violations, and code smells.
*   **Coding Standard Enforcement:** Enforces a coding standard, ensuring consistency and readability throughout your codebase.
*   **Customizable Configuration:** Highly configurable, allowing you to tailor Pylint to your project's specific needs.
*   **Plugin Support:** Extensible with plugins for popular frameworks and libraries.
*   **Inference Engine:** Advanced inference capabilities (via Astroid) to detect issues even in dynamically typed code, offering more thorough analysis.
*   **Integration:** Seamlessly integrates with most IDEs and code editors.
*   **Additional Tools:** Includes `pyreverse` for generating diagrams and `symilar` for finding duplicate code.
*   **Auto-Fix Capabilities:** Integrates with tools like `ruff` to provide auto-fix functionality.

## Installation

Install Pylint using pip:

```bash
pip install pylint
```

For spell checking support (requires `enchant`):

```bash
pip install pylint[spelling]
```

## What Differentiates Pylint?

Pylint's robust inference engine sets it apart, enabling in-depth analysis, even in code that isn't fully typed. While this may cause a slightly slower performance, Pylint's comprehensive approach allows it to catch a wider range of issues. It offers a vast array of checks, including opinionated ones, that can be customized through configuration.

## How to Use Pylint

*   Start by using the `--errors-only` flag to focus on critical issues.
*   Disable less critical messages using `--disable=C,R` (convention and refactor messages).
*   Progressively re-enable messages as your project evolves.
*   Explore available plugins for third-party libraries.

## Advised Linters

Pylint works well alongside these tools:

*   [Ruff](https://github.com/astral-sh/ruff)
*   [Flake8](https://github.com/PyCQA/flake8)
*   [Mypy](https://github.com/python/mypy)
*   [Pyright](https://github.com/microsoft/pyright)
*   [Pyre](https://github.com/facebook/pyre-check)
*   [Bandit](https://github.com/PyCQA/bandit)
*   [Black](https://github.com/psf/black)
*   [Isort](https://pycqa.github.io/isort/)
*   [Autoflake](https://github.com/myint/autoflake)
*   [Pyupgrade](https://github.com/asottile/pyupgrade)
*   [Pydocstringformatter](https://github.com/DanielNoord/pydocstringformatter)

## Additional Tools

*   **Pyreverse:** Generates package and class diagrams.
*   **Symilar:** Identifies duplicate code.

## Contributing

Contributions of all kinds are welcome! Please refer to the [Contributor Guides](https://pylint.readthedocs.io/en/latest/development_guide/contribute.html) and the [Code of Conduct](https://github.com/pylint-dev/pylint/blob/main/CODE_OF_CONDUCT.md).

## Show Your Usage

Display the Pylint badge in your README:

```
![Pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)
```

## License

Pylint is licensed under the [GPLv2](https://github.com/pylint-dev/pylint/blob/main/LICENSE).

## Support

For support, please refer to [the contact information](https://pylint.readthedocs.io/en/latest/contact.html).