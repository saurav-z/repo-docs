# Cookiecutter PyPackage: Jumpstart Your Python Project

**Quickly and easily generate a well-structured Python package with the Cookiecutter PyPackage template!** This template provides a solid foundation for your Python project, saving you time and effort on setup and configuration.  Check out the original project on GitHub:  [https://github.com/audreyfeldroy/cookiecutter-pypackage/](https://github.com/audreyfeldroy/cookiecutter-pypackage/)

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

*   **Free Software:** MIT License
*   **Discord:** [https://discord.gg/PWXJr3upUE](https://discord.gg/PWXJr3upUE)

## Key Features

This template streamlines the development process with several essential features:

*   **Automated Testing:**  Includes a robust testing setup with `pytest` for reliable code quality.
*   **GitHub Actions Integration:**  Seamlessly integrates with GitHub Actions for automated testing across multiple Python versions (3.10, 3.11, 3.12, and 3.13), ensuring broad compatibility.
*   **Simplified PyPI Release (Optional):**  Supports automatic releases to [PyPI](https://pypi.python.org/pypi) upon pushing a new tag to the main branch.
*   **Command Line Interface (CLI):**  Provides a foundation for building a command-line interface using `Typer`.

## Getting Started: Quickstart Guide

Follow these simple steps to get your Python package project up and running:

1.  **Install Cookiecutter:**  If you haven't already, install the latest version of Cookiecutter:

    ```bash
    pip install -U cookiecutter
    ```

2.  **Generate Your Project:**  Use Cookiecutter to create your Python package project from this template:

    ```bash
    cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
    ```

3.  **Project Setup & Release:**

    *   Create a new repository for your project and push your code.
    *   [Register](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) your project on PyPI.
    *   Add the repository to your [Read the Docs](https://readthedocs.io/) account and activate the Read the Docs service hook (optional).
    *   Release your package by pushing a new tag to the main branch.

## Customization and Further Development

This template is designed to be a starting point, adaptable to your specific needs.

### Fork or Create Your Own Template

If you have unique requirements, the best approach is to fork this template. Alternatively, you can create your own cookiecutter template from scratch.

### Explore Similar Projects

Browse the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) of this repository for inspiration and ideas.

### Contribute

Small, focused pull requests that enhance the packaging experience are welcome!