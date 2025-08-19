# Cookiecutter PyPackage: Jumpstart Your Python Project

**Quickly and easily create a well-structured Python package with this powerful Cookiecutter template.**  Find the original repository [here](https://github.com/audreyfeldroy/cookiecutter-pypackage/).

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

## Key Features

*   **Automated Testing:** Integrated testing setup with pytest for reliable code quality.
*   **Comprehensive Testing with GitHub Actions:** Easily test your package across multiple Python versions (3.10, 3.11, 3.12, and 3.13).
*   **Automated PyPI Releases (Optional):** Configure automatic releases to PyPI upon pushing a new tag.
*   **Command-Line Interface Ready:**  Leverages Typer for easy command-line interface creation.

## Getting Started

### Prerequisites

Ensure you have Cookiecutter installed:

```bash
pip install -U cookiecutter
```

### Generate Your Python Package

Use Cookiecutter to create your project:

```bash
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
```

### Next Steps

After running Cookiecutter, follow these steps to finalize your package setup:

*   Create a repository for your new package and place the generated code within.
*   [Register](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) your project on PyPI.
*   Integrate your project with [Read the Docs](https://readthedocs.io/) and enable the service hook for automated documentation.
*   Release your package to PyPI by creating and pushing a new tag to your main branch.

## Customization and Contribution

This template is designed to be adaptable to your specific needs.

### Fork and Customize

Feel free to fork this template and modify it to create your own tailored Python package structure.

### Explore Alternatives

Consider exploring similar Cookiecutter templates for inspiration, and browse forks of this project via the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) views.

### Contribute

We welcome pull requests! If you have improvements to suggest, especially those that enhance the core functionality or streamline the packaging process, please submit them.