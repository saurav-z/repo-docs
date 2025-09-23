# Cookiecutter PyPackage: Jumpstart Your Python Project with a Robust Template

**Quickly and easily create a well-structured, production-ready Python package with the `cookiecutter-pypackage` template.**

[Cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage/) provides a comprehensive starting point for your Python projects, saving you time and effort on boilerplate setup.

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

## Key Features

*   **Automated Testing:** Integrated with `pytest` for easy and reliable testing.
*   **CI/CD with GitHub Actions:**  Pre-configured for testing across multiple Python versions (3.10, 3.11, 3.12, and 3.13).
*   **Automated PyPI Releases:**  Optional automatic publishing to [PyPI](https://pypi.python.org/pypi) upon new tag pushes.
*   **Command-Line Interface:**  Built-in support for command-line applications using `Typer`.
*   **MIT License:**  Free and open-source under the MIT license.
*   **Community Support:** Join the discussion on [Discord](https://discord.gg/PWXJr3upUE).

## Getting Started

### Prerequisites

*   [Cookiecutter](https://github.com/cookiecutter/cookiecutter)

### Installation

Install the latest Cookiecutter:

```bash
pip install -U cookiecutter
```

### Project Generation

Generate your Python package project:

```bash
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
```

### Next Steps

After generating your project:

1.  Create a new repository for your project and push the generated code.
2.  [Register](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) your project with PyPI (if you intend to publish).
3.  Add your repository to your [Read the Docs](https://readthedocs.io/) account and enable the Read the Docs service hook for documentation.
4.  Release your package by creating and pushing a new tag to your main branch.

## Customization and Alternatives

### Forking and Customization

Feel free to fork this template to tailor it to your specific needs.  This is the recommended approach for significant customization.

### Exploring Alternatives

*   Browse the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) of this repository for other related templates.

### Contributing

Small, targeted pull requests that improve the template are welcome.