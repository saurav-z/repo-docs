# Create Production-Ready Python Packages Easily with cookiecutter-pypackage

**Jumpstart your Python project with a robust and customizable template that streamlines your development workflow.** Developed by Audrey Feldroy, the `cookiecutter-pypackage` ([original repo](https://github.com/audreyfeldroy/cookiecutter-pypackage)) is a powerful Cookiecutter template designed to help you quickly create well-structured and production-ready Python packages.

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

## Key Features

*   **Automated Testing:** Integrated testing setup with pytest for reliable code validation.
*   **CI/CD with GitHub Actions:** Includes a pre-configured GitHub Actions workflow to test your package across multiple Python versions (3.10, 3.11, 3.12, and 3.13).
*   **Automated PyPI Releases (Optional):**  Easily configure automatic releases to [PyPI](https://pypi.python.org/pypi) upon pushing a new tag to your main branch.
*   **Command Line Interface (CLI) with Typer:**  Simplifies the creation of command-line interfaces using Typer.

## Quick Start

Get started in minutes by following these simple steps:

1.  **Install Cookiecutter:** If you don't have it already, install the latest version:

    ```bash
    pip install -U cookiecutter
    ```

2.  **Generate Your Package:** Use Cookiecutter to create your project from this template:

    ```bash
    cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
    ```

3.  **Customize and Deploy:**
    *   Create a new repository for your project and upload your code.
    *   Register your project with [PyPI](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives).
    *   Integrate with [Read the Docs](https://readthedocs.io/) for automated documentation.
    *   Release new package versions by creating and pushing tags to your main branch.

## Further Customization

This template is designed to be a solid foundation, but you have several options to tailor it to your specific needs:

*   **Fork and Modify:** Create your own version of the template to customize the setup to match your preferred practices.
*   **Explore Forks:** Browse the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) of this repo for inspiration and alternative configurations.
*   **Contribute:** Submit pull requests if you have small, focused improvements that would enhance the template.

**Learn more and contribute at the original repository: [https://github.com/audreyfeldroy/cookiecutter-pypackage/](https://github.com/audreyfeldroy/cookiecutter-pypackage/)**