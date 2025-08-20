# Cookiecutter PyPackage: Jumpstart Your Python Package Development

**Quickly and easily create a well-structured Python package with Cookiecutter using this powerful template.**

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

**[View the source code on GitHub](https://github.com/audreyfeldroy/cookiecutter-pypackage/)**

## Key Features

This Cookiecutter template provides a solid foundation for your Python package, offering the following benefits:

*   **Automated Testing:** Built-in testing setup using `pytest` for reliable code quality.
*   **CI/CD with GitHub Actions:** Seamless integration with GitHub Actions for automated testing across multiple Python versions (3.10, 3.11, 3.12, and 3.13).
*   **Simplified Package Releases:** Optional automated release to PyPI upon pushing a new tag to your main branch.
*   **Command-Line Interface (CLI) Ready:**  Includes a command-line interface framework using Typer, allowing for easy CLI creation.

## Getting Started

### Prerequisites

Ensure you have Cookiecutter installed:

```bash
pip install -U cookiecutter
```

### Generate Your Python Package Project

Use the template to generate your project:

```bash
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
```

### Next Steps

1.  **Repository Creation:** Create a new repository and upload your generated project code.
2.  **PyPI Registration:** Register your project on PyPI ([https://pypi.org/](https://pypi.org/)).
3.  **Documentation with Read the Docs (Optional):** Add your repository to Read the Docs ([https://readthedocs.org/](https://readthedocs.org/)) and enable the Read the Docs service hook for automated documentation building.
4.  **Release Your Package:** Release your package by creating and pushing a new tag to your main branch.

## Customization and Further Exploration

This template is designed to be a starting point. Feel free to customize it to fit your specific project needs.

### Alternatives

*   **Fork and Adapt:**  Create your own version by forking the repository.
*   **Explore Similar Templates:**  Discover other templates by exploring the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) of this repository.
*   **Contribute:**  Submit pull requests to improve this template if you have suggestions.