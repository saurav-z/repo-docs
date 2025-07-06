# LocalStack: Your Local AWS Cloud Development & Testing Framework

**LocalStack empowers developers to build, test, and deploy AWS applications locally, accelerating development cycles.** ([Visit the original repo](https://github.com/localstack/localstack))

[![GitHub Actions](https://github.com/localstack/localstack/actions/workflows/aws-main.yml/badge.svg?branch=master)](https://github.com/localstack/localstack/actions/workflows/aws-main.yml?query=branch%3Amaster)
[![Coverage Status](https://coveralls.io/repos/github/localstack/localstack/badge.svg?branch=master)](https://coveralls.io/github/localstack/localstack?branch=master)
[![PyPI Version](https://img.shields.io/pypi/v/localstack?color=blue)](https://pypi.org/project/localstack/)
[![Docker Pulls](https://img.shields.io/docker/pulls/localstack/localstack)](https://hub.docker.com/r/localstack/localstack)
[![PyPi downloads](https://static.pepy.tech/badge/localstack)](https://pypi.org/project/localstack)
[![Backers on Open Collective](https://opencollective.com/localstack/backers/badge.svg)](https://opencollective.com/localstack/backers/badge.svg)
[![Sponsors on Open Collective](https://opencollective.com/localstack/sponsors/badge.svg)](https://opencollective.com/localstack/sponsors/badge.svg)
[![PyPI License](https://img.shields.io/pypi/l/localstack.svg)](https://img.shields.io/pypi/l/localstack.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/localstack)

---

## Key Features

*   **Local AWS Cloud Emulation:** Run a comprehensive set of AWS services locally, including Lambda, S3, DynamoDB, SQS, SNS, and many more.
*   **Accelerated Development:** Test and debug your AWS applications without deploying to the cloud, significantly speeding up your development workflow.
*   **Simplified Testing:** Effortlessly integrate LocalStack into your CI/CD pipelines for automated testing of your AWS infrastructure and applications.
*   **Cross-Platform Compatibility:** Supports Docker, Docker Compose, Helm, and the LocalStack CLI, offering flexibility in how you manage your local cloud environment.
*   **Extensive Service Coverage:**  Supports a wide range of AWS services and APIs, with the Pro version providing even more advanced features and API support.

## Installation

Choose your preferred method to install the LocalStack CLI:

### 1. LocalStack CLI (Recommended)

*   **Homebrew (macOS/Linux):**
    ```bash
    brew install localstack/tap/localstack-cli
    ```
*   **Binary Download (macOS, Linux, Windows):** Download the latest release from the [LocalStack CLI releases](https://github.com/localstack/localstack-cli/releases/latest).

*   **PyPI (macOS, Linux, Windows):**
    ```bash
    python3 -m pip install localstack
    ```

**Note:** After installing the LocalStack CLI, you will also need to install the `awslocal` CLI separately to interact with local AWS services.  See the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal) for details.

### 2. Docker

*   Refer to the official [Docker installation guide](https://docs.localstack.cloud/getting-started/installation/#docker).

### 3. Docker Compose

*   Refer to the official [Docker Compose installation guide](https://docs.localstack.cloud/getting-started/installation/#docker-compose).

### 4. Helm

*   Refer to the official [Helm installation guide](https://docs.localstack.cloud/getting-started/installation/#helm).

## Quickstart

Get started quickly using the LocalStack CLI:

```bash
% localstack start -d
```

Check service status:

```bash
% localstack status services
```

Create an SQS queue:

```bash
% awslocal sqs create-queue --queue-name sample-queue
```

## Running

Run LocalStack using:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage & Resources

Explore the following resources to master LocalStack:

*   [Documentation](https://docs.localstack.cloud)
*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

## UI Clients

Interact with LocalStack through the following UI clients:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

See the [GitHub releases](https://github.com/localstack/localstack/releases) and the [changelog](https://docs.localstack.cloud/references/changelog/) for detailed release information.

## Contributing

Contribute to LocalStack:

*   Read the [contributing guide](docs/CONTRIBUTING.md).
*   Set up your [development environment](docs/development-environment-setup/README.md).
*   Find and [open issues](https://github.com/localstack/localstack/issues).

## Get in Touch

*   Report [issues](https://github.com/localstack/localstack/issues/new/choose).
*   Upvote [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+).
*   Ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/).
*   Discuss local cloud development in the [LocalStack Slack Community](https://localstack.cloud/contact/) and the [GitHub Issue tracker](https://github.com/localstack/localstack/issues).

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

Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)).  By using this software, you agree to the [End-User License Agreement (EULA)](docs/end_user_license_agreement).