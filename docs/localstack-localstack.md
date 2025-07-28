[![LocalStack](https://raw.githubusercontent.com/localstack/localstack/main/docs/localstack-readme-banner.svg)](https://github.com/localstack/localstack)

# LocalStack: Develop and Test AWS Applications Locally

LocalStack is a powerful cloud service emulator that lets you develop and test your AWS applications locally, without needing a live AWS connection.

**[View the original repository on GitHub](https://github.com/localstack/localstack)**

**Key Features:**

*   **Local AWS Development:** Run and test AWS applications (Lambdas, S3, DynamoDB, etc.) on your local machine.
*   **Accelerated Testing:** Speed up your development and testing workflows by eliminating the need for remote cloud resources.
*   **Comprehensive Service Support:** Supports a growing number of AWS services. Check out the [Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) for details.
*   **CLI and Docker Integration:** Easy to install and run via the LocalStack CLI, Docker, Docker Compose, or Helm.
*   **Pro Version:** Unlock advanced features and expanded API support with [LocalStack Pro](https://localstack.cloud/pricing).

**Key Benefits:**

*   **Faster Development Cycles:** Test changes immediately without deployment delays.
*   **Reduced Cloud Costs:** Minimize expenses by developing and testing locally.
*   **Improved Security:** Develop and test AWS applications offline and within your secure network.
*   **Enhanced Developer Experience:** A streamlined workflow with a full local AWS emulation.

**Quickstart**

1.  **Install the LocalStack CLI:** Choose your preferred method from the Installation section below.
2.  **Start LocalStack:**

    ```bash
    localstack start -d
    ```

3.  **Interact with Services:** Use the `awslocal` CLI (install separately) to interact with emulated AWS services (e.g., `awslocal sqs create-queue --queue-name sample-queue`).

**Installation**

*   **LocalStack CLI:** Simplifies starting and managing the LocalStack Docker container.
    *   **Homebrew (macOS/Linux):** `brew install localstack/tap/localstack-cli`
    *   **Binary Download (macOS, Linux, Windows):** Download from [GitHub Releases](https://github.com/localstack/localstack-cli/releases/latest)
    *   **PyPI (macOS, Linux, Windows):** `python3 -m pip install localstack`
*   **Docker:**
    *   Ensure you have a working Docker environment.
    *   Run: `docker run -d -p 4566:4566 localstack/localstack`
*   **Docker Compose:** See the [Docker Compose documentation](https://docs.localstack.cloud/getting-started/installation/#docker-compose).
*   **Helm:** See the [Helm documentation](https://docs.localstack.cloud/getting-started/installation/#helm).

**Usage**

Explore our comprehensive documentation for detailed information on how to use LocalStack:

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

**UI Clients**
*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

**Releases**

*   View the complete list of changes in each release: [GitHub Releases](https://github.com/localstack/localstack/releases)
*   Extended release notes: [Changelog](https://docs.localstack.cloud/references/changelog/)

**Get Involved**

*   [Contributing Guide](docs/CONTRIBUTING.md)
*   [Development Environment Setup](docs/development-environment-setup/README.md)
*   [Open Issues](https://github.com/localstack/localstack/issues)

**Connect with the LocalStack Community**

*   [LocalStack Slack Community](https://localstack.cloud/contact/)
*   [GitHub Issue Tracker](https://github.com/localstack/localstack/issues)

**Contributors**

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

**Backers**

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

**Sponsors**

<a href="https://opencollective.com/localstack/sponsor/0/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/0/avatar.svg"></a>
<a href="https://opencollective.com/localstack/sponsor/1/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/1/avatar.svg"></a>
<a href="https://opencollective.com/localstack/sponsor/2/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/2/avatar.svg"></a>
<a href="https://opencollective.com/localstack/sponsor/3/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/3/avatar.svg"></a>
<a href="https://opencollective.com/localstack/sponsor/4/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/4/avatar.svg"></a>
<a href="https://opencollective.com/localstack/sponsor/5/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/5/avatar.svg"></a>
<a href="https://opencollective.com/localstack/sponsor/6/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/6/avatar.svg"></a>
<a href="https://opencollective.com/localstack/sponsor/7/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/7/avatar.svg"></a>
<a href="https://opencollective.com/localstack/sponsor/8/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/8/avatar.svg"></a>
<a href="https://opencollective.com/localstack/sponsor/9/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/9/avatar.svg"></a>

**License**

[Apache License, Version 2.0](LICENSE.txt) and [End-User License Agreement (EULA)](docs/end_user_license_agreement).