# Cookiecutter PyPackage: Jumpstart Your Python Project Development

**Quickly and easily create production-ready Python packages with a robust, pre-configured template.** This template streamlines your project setup, allowing you to focus on writing code.

**Original Repository:** [https://github.com/audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage/)

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

*   **Discord:** [https://discord.gg/PWXJr3upUE](https://discord.gg/PWXJr3upUE)
*   **License:** MIT License

## Key Features

*   **Automated Testing:** Integrated testing setup with `pytest` to ensure code quality.
*   **CI/CD with GitHub Actions:** Pre-configured GitHub Actions for automated testing across multiple Python versions (3.10, 3.11, 3.12, and 3.13).
*   **Simplified PyPI Releases (Optional):**  Automated releases to PyPI upon pushing new tags to your main branch.
*   **Command-Line Interface (CLI):**  Uses Typer to generate a command line interface.

## Getting Started

### Prerequisites

Make sure you have Cookiecutter installed:

```bash
pip install -U cookiecutter
```

### Generate Your Project

Use Cookiecutter to create your Python package project:

```bash
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
```

### Next Steps

1.  **Create a Repository:**  Create a new repository on GitHub (or your preferred platform) and push your generated project code.
2.  **Register with PyPI:**  Register your project on [PyPI](https://pypi.python.org/pypi) if you plan to distribute your package.
3.  **Optional: Documentation with Read the Docs:**  Add your repo to [Read the Docs](https://readthedocs.io/) and configure the Read the Docs service hook for automated documentation builds.
4.  **Release Your Package:**  Release a new version of your package by pushing a new tag to your main branch.

## Customization and Contribution

### Not Quite Right?

This template is designed to be a starting point, and you have several options to customize it:

*   **Fork and Modify:**  Fork the repository and customize it to suit your specific needs.
*   **Explore Similar Templates:** Browse the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) of this project for inspiration.
*   **Submit a Pull Request:**  Contribute improvements directly by submitting a pull request, especially for small, focused changes.