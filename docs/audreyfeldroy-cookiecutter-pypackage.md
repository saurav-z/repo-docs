# Create Production-Ready Python Packages Quickly with Cookiecutter-PyPackage

Tired of repetitive setup when starting new Python projects? **Cookiecutter-PyPackage streamlines Python package creation with a pre-configured, best-practice template.**

**[View the original repository on GitHub](https://github.com/audreyfeldroy/cookiecutter-pypackage/)**

## Key Features

*   **Automated Testing:** Ready-to-use testing setup with pytest.
*   **GitHub Actions CI/CD:** Seamless testing across multiple Python versions (3.10, 3.11, 3.12, 3.13) using GitHub Actions.
*   **Automated PyPI Publishing:** Optional auto-release to PyPI upon new tag creation.
*   **Command-Line Interface (CLI):** Built-in command-line interface support using Typer.
*   **MIT License:** Free and open-source software, allowing you to use, modify, and distribute the code.
*   **Discord Community:** Join the [Discord](https://discord.gg/PWXJr3upUE) to discuss this tool and Python packaging.

## Getting Started

### Prerequisites

*   Python (3.7 or higher recommended)
*   [Cookiecutter](https://github.com/cookiecutter/cookiecutter) (Install if you haven't already.)

### Installation

First, install Cookiecutter:

```bash
pip install -U cookiecutter
```

### Project Generation

Use Cookiecutter to generate your Python package:

```bash
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
```

After generating the project, follow these steps:

1.  Create a Git repository and initialize the project in it.
2.  [Register](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) your project with PyPI.
3.  Optional: Integrate with [Read the Docs](https://readthedocs.io/) and activate the service hook for documentation.
4.  Release your package by pushing a new tag to master.

## Customization & Contribution

### Fork or Create Your Own

Feel free to adapt this template to your specific needs by forking the repository or creating your own from scratch.

### Explore Similar Templates

Discover alternative templates to get inspiration by browsing the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) of this repository.

### Contribute

Small and atomic pull requests are welcome to improve this template.