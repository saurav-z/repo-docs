# Cookiecutter PyPackage: Jumpstart Your Python Package Development

Quickly create a well-structured Python package with automated testing, CI/CD, and a command-line interface using this powerful Cookiecutter template. ([View the original repo](https://github.com/audreyfeldroy/cookiecutter-pypackage/))

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

**Key Features:**

*   **Automated Testing:**  Includes a robust testing setup with pytest for reliable code.
*   **GitHub Actions Integration:**  Effortlessly test your package across multiple Python versions (3.10, 3.11, 3.12, and 3.13) using GitHub Actions.
*   **Automated PyPI Releases (Optional):**  Configure your package for automatic releases to [PyPI](https://pypi.python.org/pypi) upon new tag pushes.
*   **Command-Line Interface (CLI):**  Built-in support for creating a command-line interface using Typer.

## Getting Started

### Prerequisites

Make sure you have Cookiecutter installed:

```bash
pip install -U cookiecutter
```

### Generate Your Package

Use Cookiecutter to generate your new Python package project:

```bash
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
```

### Next Steps

After generating your project, consider these steps:

*   **Repository Setup:** Create a new repository (e.g., on GitHub) and push your generated project files there.
*   **PyPI Registration:** [Register](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) your package on PyPI.
*   **Documentation (Optional):**  Integrate with [Read the Docs](https://readthedocs.io/) for automated documentation generation; add the repo to your account and enable the service hook.
*   **Release:** Release your package by pushing a new tag to your main branch (e.g., `git tag 1.0.0` then `git push origin 1.0.0`).

## Customization and Contribution

### Fork and Customize

If you have specific needs, feel free to fork this template and tailor it to your preferences.

### Explore Alternatives

Discover related projects and templates by exploring the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) of this repository.

### Contribute

Small, targeted pull requests are welcome to improve this template.

---

*   Discord: [https://discord.gg/PWXJr3upUE](https://discord.gg/PWXJr3upUE)
*   Free software: MIT license