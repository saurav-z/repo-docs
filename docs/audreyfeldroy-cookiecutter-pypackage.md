# Create Production-Ready Python Packages with Cookiecutter-PyPackage

Easily generate a well-structured Python package project with this powerful Cookiecutter template. For the original template, visit the GitHub repository: [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage).

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

*   **License:** MIT
*   **Discord:** [https://discord.gg/PWXJr3upUE](https://discord.gg/PWXJr3upUE)

## Key Features

*   **Automated Testing:** Integrated testing setup using pytest for robust code validation.
*   **CI/CD with GitHub Actions:** Seamless testing across multiple Python versions (3.10, 3.11, 3.12, and 3.13) using GitHub Actions.
*   **Automated PyPI Releases:**  Optional automatic release to PyPI upon new tag pushes.
*   **Command Line Interface (CLI):**  Built-in CLI functionality using Typer for easy interaction.

## Getting Started

Follow these simple steps to create your Python package project:

### 1. Install Cookiecutter

Make sure you have Cookiecutter installed:

```bash
pip install -U cookiecutter
```

### 2. Generate Your Project

Use the Cookiecutter template to create your Python package project:

```bash
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
```

### 3.  Next Steps

After generation, complete the following:

*   Create a repository for your project.
*   Register your project with PyPI (see [packaging tutorial](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives)).
*   Consider integrating with [Read the Docs](https://readthedocs.io/) for documentation.
*   Release your package by pushing a new tag to your main branch.

## Customization and Contribution

### Fork or Create Your Own Template

Customize the setup to fit your needs by forking the project or creating your own Cookiecutter template.

### Explore Alternatives

Discover other options within the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) of this repository.

### Contribute

Feel free to submit pull requests to improve the template.