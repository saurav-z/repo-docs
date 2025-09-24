# Cookiecutter PyPackage: Jumpstart Your Python Package Development

**Quickly and easily create a production-ready Python package with a robust foundation using Cookiecutter PyPackage.**

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

This is a [Cookiecutter](https://github.com/cookiecutter/cookiecutter) template designed to streamline the creation of Python packages, providing a solid base for your projects. Visit the original repository on GitHub: [https://github.com/audreyfeldroy/cookiecutter-pypackage/](https://github.com/audreyfeldroy/cookiecutter-pypackage/)

## Key Features

*   **Automated Testing:** Integrated testing setup with pytest for reliable code quality.
*   **CI/CD with GitHub Actions:** Automated testing across multiple Python versions (3.10, 3.11, 3.12, and 3.13) using GitHub Actions.
*   **Automated PyPI Release:** (Optional) Automatically release your package to [PyPI](https://pypi.python.org/pypi) upon tagging and pushing to the main branch.
*   **Command-Line Interface:** Built-in command-line interface support using Typer.

## Getting Started

### Prerequisites

Make sure you have Cookiecutter installed:

```bash
pip install -U cookiecutter
```

### Generate Your Python Package

Use Cookiecutter to generate your project:

```bash
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
```

### Next Steps

After running Cookiecutter, follow these steps to set up your project:

*   Create a repository for your new project.
*   [Register](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) your project with PyPI.
*   Set up documentation hosting using [Read the Docs](https://readthedocs.io/).
*   Release your package by creating a new tag and pushing it to your main branch.

## Customization and Alternatives

### Tailor the Template

This template is designed to be a starting point. Feel free to:

*   **Fork the template:** Create your own customized version to suit your specific needs.
*   **Explore forks:** Discover other variations and adaptations of this template via the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) of the repository.
*   **Contribute:** Submit pull requests to the original repository to improve it.