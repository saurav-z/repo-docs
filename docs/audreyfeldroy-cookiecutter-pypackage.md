# Cookiecutter PyPackage: Jumpstart Your Python Project with a Production-Ready Template

Quickly create a well-structured and easily deployable Python package with **Cookiecutter PyPackage**, a powerful template designed to streamline your development workflow.  (Original repo: [https://github.com/audreyfeldroy/cookiecutter-pypackage/](https://github.com/audreyfeldroy/cookiecutter-pypackage/))

## Key Features for Efficient Python Package Development

This template offers a robust foundation for your Python projects, including:

*   **Automated Testing:**  Pre-configured with `pytest` for comprehensive unit and integration testing.
*   **CI/CD with GitHub Actions:**  Ready-to-use GitHub Actions workflows for testing across multiple Python versions (3.10, 3.11, 3.12, and 3.13).
*   **Seamless PyPI Release:**  Optional auto-release to [PyPI](https://pypi.python.org/pypi) upon tagging and pushing to the main branch.
*   **Command-Line Interface (CLI) with Typer:**  Easily build a command-line interface for your package using Typer.

## Getting Started Quickly

Follow these simple steps to generate your Python package project:

1.  **Install Cookiecutter:** If you don't have it already, install the latest version:

    ```bash
    pip install -U cookiecutter
    ```

2.  **Generate Your Project:**  Use Cookiecutter with the template repository:

    ```bash
    cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
    ```

3.  **Project Setup:** After running the cookiecutter command, you'll need to:

    *   Create a Git repository for your new project and push the generated code there.
    *   Register your project with PyPI (if you intend to publish your package).
    *   Consider setting up documentation with Read the Docs.
    *   Release your package by pushing a new tag to the main branch (e.g., `git tag 1.0.0` then `git push --tags`).

## Customization and Contributions

This template is designed to be a starting point. You have several options to tailor it to your needs:

### Fork and Customize

Create your own version by forking the repository. This allows you to make significant changes and adapt the template to your specific preferences.

### Explore Alternatives

Browse the [network](https://github.com/audreyfeldroy/cookiecutter-pypackage/network) and [family tree](https://github.com/audreyfeldroy/cookiecutter-pypackage/network/members) to discover similar templates and explore different approaches.

### Contribute Back

If you have small, targeted improvements that would benefit the community, feel free to submit a pull request.

## Resources

*   **GitHub Repository:** [https://github.com/audreyfeldroy/cookiecutter-pypackage/](https://github.com/audreyfeldroy/cookiecutter-pypackage/)
*   **Discord:** [https://discord.gg/PWXJr3upUE](https://discord.gg/PWXJr3upUE)
*   **License:** MIT