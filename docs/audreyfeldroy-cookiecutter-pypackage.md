# Create Python Packages Quickly with cookiecutter-pypackage

**Jumpstart your Python project with cookiecutter-pypackage, a robust template for creating production-ready, well-structured Python packages.** This template provides a solid foundation for your project, streamlining development and deployment.

[View the original repository on GitHub](https://github.com/audreyfeldroy/cookiecutter-pypackage)

## Key Features

*   **Automated Testing:** Integrated testing setup with pytest.
*   **CI/CD with GitHub Actions:**  Effortless testing across multiple Python versions (3.10, 3.11, 3.12, and 3.13).
*   **Automated PyPI Releases:**  Optional automatic releases to PyPI upon tagging and pushing to master.
*   **Command-Line Interface:** Includes support for creating a command-line interface using Typer.
*   **MIT License:** Free and open-source, allowing you to use, modify, and distribute the software.

## Getting Started

Follow these simple steps to generate your Python package project:

1.  **Install Cookiecutter:** If you don't have it already, install the latest Cookiecutter version:

    ```bash
    pip install -U cookiecutter
    ```

2.  **Generate Your Project:** Use the template to create your project:

    ```bash
    cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
    ```

3.  **Post-Generation Steps:**
    *   Create a repository for your project and upload your generated code.
    *   Register your project with PyPI ([Packaging Projects Tutorial](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives)).
    *   Set up documentation using Read the Docs (optional).
    *   Release your package by creating and pushing a new tag to your main branch (e.g., `git tag 0.1.0` followed by `git push origin 0.1.0`).

## Customization and Contribution

This template is designed to be a starting point.  Feel free to customize it to suit your project's needs.

### Fork or Create Your Own

Adapt the template to your specific project requirements by forking it.

### Explore Alternatives

Browse other forks of this project on the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) for inspiration and alternative approaches.

### Contribute

If you have improvements or bug fixes, submit a pull request.