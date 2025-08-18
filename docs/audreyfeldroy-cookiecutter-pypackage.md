# Build Python Packages Quickly with Cookiecutter PyPackage

**Kickstart your Python project with a robust and feature-rich template, designed for streamlined development and effortless deployment.** This template, available on [GitHub](https://github.com/audreyfeldroy/cookiecutter-pypackage), provides a solid foundation for your next Python package.

## Key Features

*   **Automated Testing:**  Integrated testing setup using pytest, ensuring code quality.
*   **Comprehensive Testing Matrix:**  GitHub Actions configured for testing across Python 3.10, 3.11, 3.12, and 3.13.
*   **Automated PyPI Releases (Optional):**  Automate your releases by pushing a new tag to master for automatic deployment to PyPI.
*   **Command-Line Interface:**  Easily create a command-line interface (CLI) using Typer.

## Quickstart Guide

Get started in minutes:

1.  **Install Cookiecutter:** (if you haven't already)

    ```bash
    pip install -U cookiecutter
    ```

2.  **Generate Your Python Package Project:**

    ```bash
    cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
    ```

3.  **Post-Generation Steps:**

    *   Create a repository for your project and initialize it with the generated code.
    *   [Register](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) your project with PyPI.
    *   (Optional) Add your repo to [Read the Docs](https://readthedocs.io/) and enable the service hook for automated documentation.
    *   Release your package by simply pushing a new tag to master.

## Customization and Contributions

This template is designed to be a starting point. Feel free to:

### Fork and Customize

Create your own version tailored to your specific needs by forking the [repository](https://github.com/audreyfeldroy/cookiecutter-pypackage/).

### Explore Similar Templates

Discover other forks and templates for inspiration. See the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) of this repository.

### Contribute

Small, focused pull requests are welcome!