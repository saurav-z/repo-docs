# Create Python Packages Quickly with Cookiecutter PyPackage

**Kickstart your Python project with a professional, well-structured foundation using Cookiecutter PyPackage!** This template provides a robust starting point for building and distributing your Python packages efficiently.  Check out the original repository [here](https://github.com/audreyfeldroy/cookiecutter-pypackage/).

## Key Features

*   **Automated Testing:** Integrated testing setup with pytest to ensure code quality.
*   **Comprehensive CI/CD:** GitHub Actions pre-configured to test across multiple Python versions (3.10, 3.11, 3.12, and 3.13).
*   **Automated PyPI Release (Optional):** Easily release new versions to PyPI upon pushing a new tag to your main branch.
*   **Command-Line Interface (CLI) with Typer:** Provides a user-friendly CLI using Typer for easy interaction with your package.

## Getting Started

### Prerequisites

*   [Cookiecutter](https://github.com/cookiecutter/cookiecutter)

### Installation

Install the latest Cookiecutter:

```bash
pip install -U cookiecutter
```

### Project Generation

Generate your Python package project using the template:

```bash
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
```

## Next Steps

1.  Create a repository for your new project (e.g., on GitHub).
2.  [Register](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) your project with PyPI.
3.  Consider integrating your documentation with [Read the Docs](https://readthedocs.io/) for easy hosting.
4.  Release your package by pushing a new tag to your main branch.

## Customization and Contribution

### Forking/Creating Your Own

Feel free to fork this template and customize it to your specific needs.  Create your own template to reflect your preferred development practices.

### Exploring Alternatives

*   Explore forks and related projects to discover alternative configurations and ideas (see the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members)).

### Contributing

Small, targeted pull requests are welcome if they improve the template or packaging experience.