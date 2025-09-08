# Create Python Packages Quickly with cookiecutter-pypackage

Tired of setting up Python package boilerplate? **cookiecutter-pypackage** is a powerful template that streamlines the creation of well-structured, production-ready Python packages.  See the original repo at [https://github.com/audreyfeldroy/cookiecutter-pypackage/](https://github.com/audreyfeldroy/cookiecutter-pypackage/).

[![PyPI version](https://img.shields.io/pypi/v/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)
[![PyPI downloads](https://img.shields.io/pypi/dm/cookiecutter-pypackage.svg)](https://pypi.python.org/pypi/cookiecutter-pypackage)

*   **GitHub Repo:** [https://github.com/audreyfeldroy/cookiecutter-pypackage/](https://github.com/audreyfeldroy/cookiecutter-pypackage/)
*   **License:** MIT
*   **Discord:** [https://discord.gg/PWXJr3upUE](https://discord.gg/PWXJr3upUE)

## Key Features

*   **Automated Testing:** Includes a robust testing setup using pytest.
*   **CI/CD with GitHub Actions:**  Easily test your package across multiple Python versions (3.10, 3.11, 3.12, and 3.13).
*   **Automated PyPI Releases (Optional):**  Configure automatic releases to [PyPI](https://pypi.python.org/pypi) upon new tag pushes.
*   **Command-Line Interface (CLI) with Typer:** Quickly build command-line tools.

## Getting Started

1.  **Install Cookiecutter:**  Make sure you have Cookiecutter installed:

    ```bash
    pip install -U cookiecutter
    ```

2.  **Generate Your Package:**  Use the template to create your Python package project:

    ```bash
    cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
    ```

3.  **Next Steps:**

    *   Create a GitHub repository for your new project.
    *   [Register](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) your project with PyPI.
    *   Consider integrating your package with [Read the Docs](https://readthedocs.io/) for automatic documentation generation.
    *   Release new versions by pushing new tags to your main branch.

## Customization and Contributing

### Fork or Create Your Own

If you need a setup that's a bit different, feel free to:

*   Fork this repository.
*   Create your own cookiecutter template based on this one.

### Explore Alternatives

Check out the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) on GitHub to see related projects and learn from other approaches.

### Submit Pull Requests

Small, targeted pull requests that enhance the core functionality are welcome!