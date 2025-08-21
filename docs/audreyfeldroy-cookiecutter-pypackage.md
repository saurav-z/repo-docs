# Cookiecutter PyPackage: Jumpstart Your Python Project Development

**Quickly and easily scaffold a well-structured and production-ready Python package with Cookiecutter PyPackage, saving you valuable time and effort.**

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

*   **GitHub Repository:** [https://github.com/audreyfeldroy/cookiecutter-pypackage/](https://github.com/audreyfeldroy/cookiecutter-pypackage/)
*   **License:** MIT
*   **Discord:** [https://discord.gg/PWXJr3upUE](https://discord.gg/PWXJr3upUE)

## Key Features

Cookiecutter PyPackage provides a robust foundation for your Python project, including:

*   **Automated Testing:** Integrated testing setup with pytest for reliable code quality.
*   **CI/CD with GitHub Actions:** Ready-to-use GitHub Actions workflows to test your package across multiple Python versions (3.10, 3.11, 3.12, and 3.13).
*   **Automated PyPI Releases (Optional):**  Easily publish new versions of your package to PyPI upon pushing new tags.
*   **Command-Line Interface:**  Supports command-line interface creation using Typer.

## Getting Started

### Prerequisites

*   Python installed
*   [Cookiecutter](https://github.com/cookiecutter/cookiecutter)

### Installation

Ensure you have the latest version of Cookiecutter installed:

```bash
pip install -U cookiecutter
```

### Project Generation

Create your Python package project using Cookiecutter:

```bash
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
```

### Next Steps

After generating your project, consider these steps:

*   Create a new repository and commit your generated project to it.
*   [Register](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) your project on PyPI.
*   Integrate your project with [Read the Docs](https://readthedocs.io/) for automated documentation generation.
*   Release your package by pushing a new tag to your main branch.

## Customization and Contribution

This template is designed to be a starting point.  Feel free to:

### Fork and Adapt

If you have specific needs, consider forking this repository to customize it to your preferences.

### Explore Alternatives

Check out other forks to see different configurations and options via the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members).

### Contribute

Small, focused pull requests that enhance the template are welcomed.