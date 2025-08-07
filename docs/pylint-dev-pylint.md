# Pylint: Static Code Analysis for Python

**Pylint is a powerful static code analyzer that helps you write cleaner, more maintainable Python code.** Check out the original repository at [https://github.com/pylint-dev/pylint](https://github.com/pylint-dev/pylint).

## Key Features

*   **Static Analysis:** Analyzes your Python code without running it, identifying potential errors, style issues, and code smells.
*   **Enforcement of Coding Standards:** Enforces a coding standard (like PEP 8) and helps maintain consistency across your codebase.
*   **Code Smell Detection:** Identifies code smells, suggesting areas for refactoring and improvement.
*   **Highly Configurable:** Customizable rules and plugins allow you to tailor Pylint to your project's specific needs.
*   **Advanced Inference:**  Leverages its internal code representation (astroid) for more thorough analysis, even in partially typed codebases.
*   **Integration:** Easily integrates with most editors and IDEs.
*   **Additional Tools:** Includes `pyreverse` (for generating diagrams) and `symilar` (for duplicate code detection).

## Installation

Install Pylint using pip:

```bash
pip install pylint
```

For spell-checking functionality, install with the `spelling` extra:

```bash
pip install pylint[spelling]
```

## What Differentiates Pylint?

Pylint stands out with its advanced inference capabilities, which allow it to understand your code's behavior more deeply than other linters.  It's known for its thoroughness, offering a wider range of checks and highly configurable options, making it a comprehensive tool for code quality.

## How to Use Pylint Effectively

*   Start with the `--errors-only` flag to focus on critical issues.
*   Disable convention and refactor messages with `--disable=C,R` initially to reduce noise.
*   Progressively re-evaluate and re-enable messages as your project's priorities evolve.
*   Explore plugins for extended support of popular frameworks and third-party libraries.

## Advised Linters to Use Alongside Pylint

Enhance your code quality workflow with these complementary tools:

*   **Ruff:** A fast linter with built-in auto-fix.
*   **Flake8:** A framework for custom checks.
*   **Mypy, Pyright / Pylance, Pyre:**  For static typing.
*   **Bandit:** For security checks.
*   **Black & isort:** For auto-formatting.
*   **Autoflake:**  For removing unused imports.
*   **Pyupgrade:** For upgrading Python syntax.
*   **Pydocstringformatter:**  For automated docstring formatting.

## Contributing

We welcome all contributions! Check out our [Contributor Guides](https://pylint.readthedocs.io/en/latest/development_guide/contribute.html) for details on how to get involved.

## License

Pylint is primarily licensed under the `GPLv2 <https://github.com/pylint-dev/pylint/blob/main/LICENSE>`_.