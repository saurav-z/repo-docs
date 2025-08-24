# Cookiecutter PyPackage: Jumpstart Your Python Project with a Robust Template

**Quickly and easily scaffold a production-ready Python package with this powerful Cookiecutter template.**

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

**[View the original repository on GitHub](https://github.com/audreyfeldroy/cookiecutter-pypackage)**

This Cookiecutter template provides a solid foundation for your Python package, streamlining development and deployment.

## Key Features

*   **Simplified Testing:** Integrated testing setup with pytest for reliable code validation.
*   **Automated Testing with GitHub Actions:**  Seamlessly test across multiple Python versions (3.10, 3.11, 3.12, and 3.13) with GitHub Actions.
*   **Automated PyPI Releases:** Optional auto-release functionality to [PyPI](https://pypi.python.org/pypi) upon new tag pushes.
*   **Command-Line Interface:** Built-in command-line interface using Typer for easy project interaction.

## Getting Started:  Create Your Python Package

Follow these simple steps to begin:

1.  **Install Cookiecutter:** Ensure you have the latest version of Cookiecutter installed:

    ```bash
    pip install -U cookiecutter
    ```

2.  **Generate Your Project:** Use Cookiecutter to create your new package:

    ```bash
    cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
    ```

3.  **Next Steps:**

    *   Create a repository for your package (e.g., on GitHub).
    *   [Register](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) your project with PyPI.
    *   Consider integrating with [Read the Docs](https://readthedocs.io/) for documentation hosting.
    *   Release your package by pushing a new tag to your main branch.

## Customization and Contribution

### Forking and Customization

This template is designed to be a starting point.  Feel free to fork it and tailor it to your specific needs and preferences.

### Exploring Alternatives

Consider exploring other similar templates or forks for inspiration. See the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) for related projects.

### Contributing

Small, focused pull requests are welcome if they improve the template and the overall packaging experience.