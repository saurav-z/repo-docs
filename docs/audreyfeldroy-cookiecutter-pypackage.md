# Cookiecutter PyPackage: Jumpstart Your Python Package Development

Easily create well-structured and production-ready Python packages with the **Cookiecutter PyPackage** template. ([Original Repository](https://github.com/audreyfeldroy/cookiecutter-pypackage))

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

## Key Features

*   **Automated Testing:** Integrated testing setup with pytest to ensure code quality.
*   **CI/CD with GitHub Actions:**  Pre-configured GitHub Actions for testing across multiple Python versions (3.10, 3.11, 3.12, and 3.13), streamlining continuous integration.
*   **Automated PyPI Release (Optional):**  Automated publishing to PyPI upon pushing a new tag to your main branch, simplifying package distribution.
*   **Command-Line Interface (CLI):** Includes a command-line interface using Typer, making your package user-friendly.

## Quickstart Guide

Get started in minutes by following these simple steps:

1.  **Install Cookiecutter:**  Ensure you have the latest version of Cookiecutter installed.

    ```bash
    pip install -U cookiecutter
    ```

2.  **Generate Your Python Package Project:**  Use Cookiecutter to create your project from the template.

    ```bash
    cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
    ```

3.  **Follow the remaining steps:**

    *   Create a repository for your project (e.g., on GitHub).
    *   Register your project with PyPI to enable publishing.
    *   Add the repository to your Read the Docs account and enable the service hook for documentation hosting.
    *   Release your package by pushing a new tag to your main branch.

## Customization and Further Development

This template is designed to be a solid foundation for your Python package.  Feel free to adapt it to your specific needs.

### Forking and Customization

We encourage you to fork this project to create your own customized template. You can then modify it to include your preferred settings and features.

### Alternatives and Collaboration

Explore the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) of this repository to discover other related templates. If you have suggestions for improving this template, you are also welcome to submit a pull request.