# Cookiecutter PyPackage: Jumpstart Your Python Project

**Quickly and easily create a well-structured Python package project with the Cookiecutter PyPackage template.** ([Original Repository](https://github.com/audreyfeldroy/cookiecutter-pypackage))

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

## Key Features

*   **Automated Testing:** Integrated testing setup with pytest to ensure code quality.
*   **CI/CD with GitHub Actions:** Seamlessly test your package across multiple Python versions (3.10, 3.11, 3.12, and 3.13) using GitHub Actions.
*   **Automated PyPI Releases (Optional):** Configure automatic releases to PyPI upon new tag pushes to the master branch.
*   **Command-Line Interface:** Built-in support for creating command-line interfaces using Typer.
*   **MIT License:** The template is released under an MIT license.

## Getting Started

### Prerequisites

*   Python 3.6+
*   [Cookiecutter](https://github.com/cookiecutter/cookiecutter)

### Installation

Install Cookiecutter:

```bash
pip install -U cookiecutter
```

### Project Generation

Generate your Python package project:

```bash
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
```

Follow the prompts to configure your new project. After generation, you can:

*   Create a Git repository for your project.
*   [Register](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) your project on PyPI (if you intend to publish your package).
*   Set up documentation with [Read the Docs](https://readthedocs.io/) and enable the service hook.
*   Release your package by tagging and pushing to the master branch.

## Customization and Further Development

### Forking and Customization

This template is designed to be a starting point.  Feel free to fork this repository and modify it to suit your specific project needs.

### Exploring Alternatives

*   Explore the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) of this repository to find similar and forked templates.

### Contributing

Small, focused pull requests that enhance the template are welcome.