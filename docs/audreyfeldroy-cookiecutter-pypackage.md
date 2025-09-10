# Cookiecutter PyPackage: Jumpstart Your Python Project with a Robust Template

Quickly and easily bootstrap your Python project with the **Cookiecutter PyPackage** template, designed to streamline development and deployment. Check out the original repo [here](https://github.com/audreyfeldroy/cookiecutter-pypackage/).

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

## Key Features

*   **Automated Testing:** Integrated testing setup using `pytest` ensures code quality.
*   **CI/CD with GitHub Actions:**  Seamlessly test your project across multiple Python versions (3.10, 3.11, 3.12, and 3.13) with GitHub Actions.
*   **Automated PyPI Releases (Optional):**  Configure your project to automatically release new versions to [PyPI](https://pypi.python.org/pypi) upon tagging.
*   **Command-Line Interface (CLI):**  Generate a CLI using Typer for user-friendly interaction.
*   **MIT License:**  Benefit from the permissive MIT license, allowing for flexible use and modification.
*   **Discord Community:** Connect with other users in the Discord community [here](https://discord.gg/PWXJr3upUE).

## Getting Started

### Prerequisites

*   Ensure you have [Cookiecutter](https://github.com/cookiecutter/cookiecutter) installed. If not, install it using:

```bash
pip install -U cookiecutter
```

### Generate Your Python Package

1.  Run the following command to generate your project using the template:

```bash
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
```

2.  Follow the prompts to configure your project.

3.  After generation, you can:

    *   Create a repository (e.g., on GitHub) and place your project files there.
    *   [Register](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) your package with PyPI.
    *   Consider setting up documentation on [Read the Docs](https://readthedocs.io/) and enabling the Read the Docs service hook.
    *   Release new versions of your package by pushing tags to your main branch (e.g., `main` or `master`).

## Customization and Further Development

### Fork and Customize

Feel free to fork this template to create your own tailored version.

### Explore Alternatives

Consider exploring forks and related templates for inspiration.  See the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members).

### Contribute

Small, focused pull requests are welcomed to improve the template.