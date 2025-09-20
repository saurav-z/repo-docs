# Cookiecutter PyPackage: Kickstart Your Python Project with Ease

Tired of repetitive project setup? Cookiecutter PyPackage provides a streamlined, customizable template to quickly bootstrap a well-structured Python package.  ([Original Repository](https://github.com/audreyfeldroy/cookiecutter-pypackage))

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

**Key Features:**

*   **Automated Testing:** Built-in testing setup using pytest for reliable code quality.
*   **CI/CD with GitHub Actions:**  Effortlessly test your package across multiple Python versions (3.10, 3.11, 3.12, and 3.13) with GitHub Actions.
*   **Automated PyPI Releases (Optional):**  Configure automatic releases to PyPI upon pushing a new tag, simplifying your deployment process.
*   **Command-Line Interface (CLI) Ready:**  Includes support for building a CLI using Typer, enabling user-friendly interaction with your package.
*   **MIT License:**  Open-source and freely usable under the MIT license.
*   **Discord Community:** Connect with other users and get support on the Discord server: [https://discord.gg/PWXJr3upUE](https://discord.gg/PWXJr3upUE)

## Getting Started

Follow these simple steps to generate your Python package project:

**1. Install Cookiecutter:**

```bash
pip install -U cookiecutter
```

**2. Generate Your Project:**

```bash
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
```

**3. Post-Generation Steps:**

*   Create a repository for your new project (e.g., on GitHub).
*   (Optional) [Register](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) your project on PyPI.
*   (Optional) Integrate with [Read the Docs](https://readthedocs.io/) for documentation and activate the Read the Docs service hook.
*   Release your package by pushing a new tag to the main branch.

## Customization and Contributions

This template is designed to be a starting point.  Feel free to:

*   **Fork and Customize:** Create your own version to suit your specific project needs.
*   **Explore Forks:** Discover variations and enhancements within the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) of this repository.
*   **Contribute:**  Submit pull requests for improvements that enhance the core template.