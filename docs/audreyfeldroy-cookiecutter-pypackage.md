# Create Production-Ready Python Packages Quickly with cookiecutter-pypackage

**Kickstart your Python project with a pre-configured, best-practice template for rapid development and deployment.** This cookiecutter template by Audrey Feldroy ([original repo](https://github.com/audreyfeldroy/cookiecutter-pypackage/)) provides a solid foundation for building and releasing high-quality Python packages.

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

## Key Features

*   **Robust Testing:** Includes a pre-configured testing setup using pytest.
*   **Automated Testing with GitHub Actions:** Easily test your package across multiple Python versions (3.10, 3.11, 3.12, and 3.13) using GitHub Actions.
*   **Automated PyPI Releases:** Optionally configure automatic releases to [PyPI](https://pypi.python.org/pypi) upon pushing new tags.
*   **Command-Line Interface (CLI):**  Built-in support for creating command-line interfaces using Typer.

## Getting Started

### Prerequisites

Ensure you have Cookiecutter installed:

```bash
pip install -U cookiecutter
```

### Project Generation

Use Cookiecutter to create your Python package project from the template:

```bash
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
```

### Next Steps

1.  **Create a Repository:**  Initialize a Git repository and push your project to a platform like GitHub.
2.  **Register on PyPI (Optional):** If you plan to publish your package, [register your project](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) with PyPI.
3.  **Document with Read the Docs (Optional):** Integrate with [Read the Docs](https://readthedocs.io/) for automated documentation generation.  Enable the Read the Docs service hook in your repository.
4.  **Release Your Package:**  Release your package by pushing a new tag to your main branch (e.g., `master`).

## Customization and Contribution

### Fork or Create Your Own Template

If this template doesn't perfectly fit your needs, consider forking it or creating your own custom Cookiecutter template.

### Explore Similar Templates

Discover other related templates by exploring the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) of this repository.

### Contribute

Small, focused pull requests that enhance the core functionality or improve the user experience are welcome.

## Resources

*   **Original Repository:**  [https://github.com/audreyfeldroy/cookiecutter-pypackage/](https://github.com/audreyfeldroy/cookiecutter-pypackage/)
*   **Discord:** [https://discord.gg/PWXJr3upUE](https://discord.gg/PWXJr3upUE)
*   **License:** MIT License