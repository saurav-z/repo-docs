# Create Production-Ready Python Packages Quickly with Cookiecutter PyPackage

Easily bootstrap your Python project with a well-structured and automated setup using Cookiecutter PyPackage. This template provides a solid foundation for building and distributing your Python packages, saving you time and effort.

**[View the original repository on GitHub](https://github.com/audreyfeldroy/cookiecutter-pypackage/)**

## Key Features

*   **Automated Testing:** Integrated with pytest for robust unit and integration testing.
*   **GitHub Actions CI/CD:** Pre-configured GitHub Actions workflows for testing across multiple Python versions (3.10, 3.11, 3.12, and 3.13).
*   **Automated PyPI Release (Optional):**  Easily publish your package to PyPI upon tagging a new release.
*   **Command Line Interface (CLI) with Typer:**  Generate a CLI for your package using Typer for easy interaction.
*   **MIT License:**  Free and open-source under the MIT license.

## Quickstart Guide

Get your Python package project up and running in minutes:

1.  **Install Cookiecutter:**

    ```bash
    pip install -U cookiecutter
    ```

2.  **Generate your project:**

    ```bash
    cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
    ```

3.  **Follow these steps to deploy your package:**

    *   Create a repository for your project.
    *   Register your project on PyPI.
    *   Consider integrating Read the Docs for documentation and enabling the service hook.
    *   Release your package by tagging and pushing to the master branch.

## Customization and Contributions

This template is designed to be a starting point.

### Fork or Create Your Own

If you have specific requirements or preferences, feel free to fork this template or create your own. This approach provides flexibility.

### Explore Related Templates

Discover other Cookiecutter templates and forks to see how others have customized their package setups. Check the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) for inspiration.

### Contribute

Small, targeted pull requests that improve the template are welcome!