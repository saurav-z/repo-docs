# Cookiecutter PyPackage: Jumpstart Your Python Project with a Ready-Made Template

Quickly and easily create a well-structured Python package project with the power of Cookiecutter. Get started with this template and save time on boilerplate setup.

[Original Repository](https://github.com/audreyfeldroy/cookiecutter-pypackage)

## Key Features

*   **Automated Testing:** Integrated testing setup using pytest for robust code quality.
*   **CI/CD with GitHub Actions:** Seamlessly test your package across multiple Python versions (3.10, 3.11, 3.12, and 3.13) using GitHub Actions.
*   **Automated PyPI Release (Optional):** Automatically publish your package to [PyPI](https://pypi.python.org/pypi) upon pushing new tags to your main branch.
*   **Command-Line Interface (CLI) Ready:** Built-in support for creating a command-line interface using Typer.

## Getting Started

### Prerequisites

*   [Cookiecutter](https://github.com/cookiecutter/cookiecutter)

### Installation

Install the latest Cookiecutter:

```bash
pip install -U cookiecutter
```

### Usage

Generate your Python package project:

```bash
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
```

After project generation:

1.  Create a repository for your project.
2.  [Register](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) your project with PyPI.
3.  (Optional) Add your repository to your [Read the Docs](https://readthedocs.io/) account and enable the Read the Docs service hook for automated documentation.
4.  Release your package by creating a new tag and pushing it to your main branch.

## Customization and Further Development

This template is designed to be a solid starting point.

### Forking & Creating Your Own Template

For extensive customization, consider forking this template or creating your own Cookiecutter template tailored to your specific needs.

### Exploring Similar Projects

Discover other projects and forks to explore various approaches and ideas. See the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) for this repo.

### Contributing

Small and targeted pull requests that enhance the user experience are welcome.