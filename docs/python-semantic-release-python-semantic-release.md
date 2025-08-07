# Python Semantic Release: Automate Your Releases with SemVer and Commit Message Conventions

**Effortlessly automate your Python project releases, adhering to Semantic Versioning (SemVer) and commit message conventions.**  This tool simplifies the release process, saving you time and ensuring consistency.

[View the original repository on GitHub](https://github.com/python-semantic-release/python-semantic-release)

## Key Features

*   **Automated Versioning:** Automatically determine the next version number based on your commit history, adhering to SemVer (Major.Minor.Patch).
*   **Commit Message Parsing:** Uses commit message conventions to determine the type of change (e.g., feature, bug fix, breaking change) and increment the version accordingly.
*   **Simplified Release Process:**  Streamlines the release workflow, making it easier to publish new versions of your Python packages.
*   **Integration with CI/CD:** Seamlessly integrates with continuous integration and continuous delivery pipelines.
*   **Documentation:** Comprehensive documentation available at [python-semantic-release.readthedocs.io](https://python-semantic-release.readthedocs.io/en/stable/).

## GitHub Action

The Python Semantic Release GitHub Action automates the versioning and release process within your GitHub workflows. It executes the command `semantic-release version` using `python-semantic-release`.

Find detailed usage information and examples in the [GitHub Actions section](https://python-semantic-release.readthedocs.io/en/stable/configuration/automatic-releases/github-actions.html) of the documentation.