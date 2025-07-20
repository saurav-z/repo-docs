# LocalStack: Develop and Test AWS Applications Locally 🚀

LocalStack is a powerful cloud service emulator that allows you to develop and test your AWS applications locally, accelerating your development workflow without relying on a remote cloud provider.  [Back to the original repo](https://github.com/localstack/localstack)

<p align="center">
  <img src="https://raw.githubusercontent.com/localstack/localstack/main/docs/localstack-readme-banner.svg" alt="LocalStack - A fully functional local cloud stack">
</p>

<p align="center">
  <a href="https://github.com/localstack/localstack/actions/workflows/aws-main.yml?query=branch%3Amain"><img alt="GitHub Actions" src="https://github.com/localstack/localstack/actions/workflows/aws-main.yml/badge.svg?branch=main"></a>
  <a href="https://coveralls.io/github/localstack/localstack?branch=main"><img alt="Coverage Status" src="https://coveralls.io/repos/github/localstack/localstack/badge.svg?branch=main"></a>
  <a href="https://pypi.org/project/localstack/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/localstack?color=blue"></a>
  <a href="https://hub.docker.com/r/localstack/localstack"><img alt="Docker Pulls" src="https://img.shields.io/docker/pulls/localstack/localstack"></a>
  <a href="https://pypi.org/project/localstack"><img alt="PyPi downloads" src="https://static.pepy.tech/badge/localstack"></a>
  <a href="#backers"><img alt="Backers on Open Collective" src="https://opencollective.com/localstack/backers/badge.svg"></a>
  <a href="#sponsors"><img alt="Sponsors on Open Collective" src="https://opencollective.com/localstack/sponsors/badge.svg"></a>
  <a href="https://img.shields.io/pypi/l/localstack.svg"><img alt="PyPI License" src="https://img.shields.io/pypi/l/localstack.svg"></a>
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
  <a href="https://bsky.app/profile/localstack.cloud"><img alt="Bluesky" src="https://img.shields.io/badge/bluesky-Follow-blue?logo=bluesky"></a>
</p>

---

## Key Features:

*   **Local AWS Cloud Emulation:**  Run a wide range of AWS services locally, including Lambda, S3, DynamoDB, SQS, SNS, and more.
*   **Accelerated Development & Testing:**  Speed up your development cycle by testing your applications locally without incurring cloud costs.
*   **Comprehensive Service Coverage:** Supports a growing number of AWS services.  Check [feature coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/).
*   **Multiple Integration Options:** Integrate with LocalStack using the CLI, Docker, Docker Compose, or Helm.
*   **Pro Version:** Explore advanced features and additional APIs with the [LocalStack Pro version](https://localstack.cloud/pricing).
*   **GUI Options:** Interact with LocalStack through a variety of UI clients [Web App](https://app.localstack.cloud), [Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/), and [Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)
*   **CI/CD Integration:** Easily integrate LocalStack into your CI/CD pipelines for automated testing.

## Installation

Choose your preferred installation method:

### LocalStack CLI (Recommended)

*   **Brew (macOS/Linux):**
    ```bash
    brew install localstack/tap/localstack-cli
    ```
*   **Binary Download (macOS, Linux, Windows):** Download the latest release from [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and add it to your PATH.
*   **PyPI (macOS, Linux, Windows):**
    ```bash
    python3 -m pip install localstack
    ```
    Install `awslocal` CLI separately for interacting with local AWS services (see documentation).

### Docker

*   Refer to the [Docker Installation Guide](https://docs.localstack.cloud/getting-started/installation/#docker).

### Docker Compose

*   Refer to the [Docker Compose Installation Guide](https://docs.localstack.cloud/getting-started/installation/#docker-compose).

### Helm

*   Refer to the [Helm Installation Guide](https://docs.localstack.cloud/getting-started/installation/#helm).

## Quickstart

1.  **Start LocalStack:**
    ```bash
    localstack start -d
    ```

2.  **Check Service Status:**
    ```bash
    localstack status services
    ```

3.  **Use an AWS Service (SQS Example):**
    ```bash
    awslocal sqs create-queue --queue-name sample-queue
    ```

For more details, see [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and `awslocal` CLI usage.

## Usage

*   **Configuration:** [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   **CI/CD:** [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   **Integrations:** [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   **Tools:** [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   **Understanding:** [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   **FAQ:** [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

## Releases

Review the [GitHub releases](https://github.com/localstack/localstack/releases) and [changelog](https://docs.localstack.cloud/references/changelog/) for detailed release notes.

## Contributing

*   Read the [contributing guide](docs/CONTRIBUTING.md).
*   Set up your [development environment](docs/development-environment-setup/README.md).
*   Browse the [issue tracker](https://github.com/localstack/localstack/issues) and contribute.

## Get in Touch

*   Report issues: [GitHub Issues](https://github.com/localstack/localstack/issues/new/choose)
*   Upvote feature requests: [Feature Requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   Ask for support: [Help and Support](https://docs.localstack.cloud/getting-started/help-and-support/)
*   Discuss development: [LocalStack Slack Community](https://localstack.cloud/contact/)

### Contributors

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

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

## License

Released under the Apache License, Version 2.0 ([LICENSE](LICENSE.txt)).  See the [End-User License Agreement (EULA)](docs/end_user_license_agreement).