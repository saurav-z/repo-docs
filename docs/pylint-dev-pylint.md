# Pylint: Static Code Analysis for Python

**Pylint is a powerful static code analyzer for Python that helps you write cleaner, more maintainable, and bug-free code.**

[Go to the Pylint Repository](https://github.com/pylint-dev/pylint)

## Key Features:

*   **Static Code Analysis:** Analyzes your Python code without running it, identifying potential errors, style issues, and code smells.
*   **Coding Standard Enforcement:** Enforces a customizable coding style, promoting consistency across your projects.
*   **Error Detection:** Detects a wide range of errors, from simple typos to complex logic flaws, using astroid for comprehensive code representation and inference.
*   **Code Smell Detection:** Identifies code that could be improved, such as overly complex functions or duplicated code.
*   **Highly Configurable:** Offers extensive configuration options to tailor Pylint to your specific needs and preferences, and allows you to write custom plugins.
*   **Integration with IDEs and Editors:** Seamlessly integrates with popular IDEs and editors for real-time feedback and issue highlighting.
*   **Additional Tools:** Includes tools like `pyreverse` (for generating UML diagrams) and `symilar` (for finding duplicate code).

## Installation

Install Pylint using pip:

```bash
pip install pylint
```

For spell checking (requires `enchant` and potentially the enchant C library):

```bash
pip install pylint[spelling]
```

Detailed installation instructions can be found in the [Pylint documentation](https://pylint.readthedocs.io/en/latest/user_guide/installation/index.html).

## How to Use Pylint

Start by running Pylint on your Python code:

```bash
pylint your_module.py
```

Pylint's output will highlight any issues it finds. Start with the `--errors-only` flag, then disable convention and refactor messages with `--disable=C,R` and progressively re-evaluate and re-enable messages as your priorities evolve.
Pylint is highly configurable and permits to write plugins in order to add your own checks.

## What Differentiates Pylint?

Pylint utilizes its internal code representation (astroid) to infer the actual values of nodes, making it more thorough than other linters. This helps to identify more issues, even in partially typed code. Pylint is designed to be thorough and is not afraid of being slow than other linters.

## Advised linters alongside pylint

Projects that you might want to use alongside pylint include ruff (**really** fast, with builtin auto-fix and a large number of checks taken from popular linters, but implemented in rust) or flake8 (a framework to implement your own checks in python using ast directly), mypy, pyright / pylance or pyre (typing checks), bandit (security oriented checks), black and isort (auto-formatting), autoflake (automated removal of unused imports or variables), pyupgrade (automated upgrade to newer python syntax) and pydocstringformatter (automated pep257).

## Contributing

We welcome contributions! Please review the [code of conduct](https://github.com/pylint-dev/pylint/blob/main/CODE_OF_CONDUCT.md) and the [contributor guide](https://pylint.readthedocs.io/en/latest/development_guide/contribute.html) for guidance.

## License

Pylint is licensed under the [GPLv2](https://github.com/pylint-dev/pylint/blob/main/LICENSE), with icon files under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

## Support

For support and contact information, please see the [Pylint documentation](https://pylint.readthedocs.io/en/latest/contact.html).