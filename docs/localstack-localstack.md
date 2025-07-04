# LocalStack: Develop and Test AWS Applications Locally

**LocalStack empowers developers to build, test, and deploy AWS applications locally, eliminating the need for a remote cloud provider, and is a leading open-source cloud service emulator. ([View on GitHub](https://github.com/localstack/localstack))**

[![GitHub Actions](https://github.com/localstack/localstack/actions/workflows/aws-main.yml/badge.svg?branch=master)](https://github.com/localstack/localstack/actions/workflows/aws-main.yml?query=branch%3Amaster)
[![Coverage Status](https://coveralls.io/repos/github/localstack/localstack/badge.svg?branch=master)](https://coveralls.io/github/localstack/localstack?branch=master)
[![PyPI Version](https://img.shields.io/pypi/v/localstack?color=blue)](https://pypi.org/project/localstack/)
[![Docker Pulls](https://img.shields.io/docker/pulls/localstack/localstack)](https://hub.docker.com/r/localstack/localstack)
[![PyPi downloads](https://static.pepy.tech/badge/localstack)](https://pypi.org/project/localstack)
[![Backers on Open Collective](https://opencollective.com/localstack/backers/badge.svg)](https://opencollective.com/localstack/backers)
[![Sponsors on Open Collective](https://opencollective.com/localstack/sponsors/badge.svg)](https://opencollective.com/localstack/sponsors)
[![PyPI License](https://img.shields.io/pypi/l/localstack.svg)](https://img.shields.io/pypi/l/localstack.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/localstack)

---

## Key Features

*   **Local Cloud Development:** Run your AWS applications and serverless functions (like Lambdas) entirely on your local machine.
*   **Comprehensive AWS Service Support:** Emulates a wide range of AWS services, including Lambda, S3, DynamoDB, SQS, SNS, and more.  [See Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) for details.
*   **Fast Development Cycle:**  Significantly speeds up testing and development workflows by eliminating the need for remote cloud interactions.
*   **Integration with Popular Tools:** Works seamlessly with AWS CLI, SDKs, and popular infrastructure-as-code tools like Terraform and CDK.
*   **Flexible Installation:** Install via CLI, Docker, Docker Compose, or Helm.
*   **Pro Version:** Enhance your development with the [Pro version](https://localstack.cloud/pricing) for additional APIs and advanced features.

## Getting Started

### Install

Choose your preferred method:

*   **LocalStack CLI (Recommended):** Start and manage LocalStack directly from your command line.

    *   **Homebrew (macOS/Linux):** `brew install localstack/tap/localstack-cli`
    *   **Binary Download (macOS/Linux/Windows):** Download from [LocalStack CLI Releases](https://github.com/localstack/localstack-cli/releases/latest) and add to your `PATH`.
    *   **PyPI (macOS/Linux/Windows):** `python3 -m pip install localstack`

*   **Docker:** [Docker Installation](https://docs.localstack.cloud/getting-started/installation/#docker)
*   **Docker Compose:** [Docker Compose Installation](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   **Helm:** [Helm Installation](https://docs.localstack.cloud/getting-started/installation/#helm)

To interact with the local AWS services using the `awslocal` CLI, install it separately according to the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

### Quickstart

1.  **Start LocalStack:**  `localstack start -d`
2.  **Check Service Status:** `localstack status services`
3.  **Use AWS Services (e.g., SQS):**
    ```shell
    awslocal sqs create-queue --queue-name sample-queue
    ```
    (See [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) for more examples.)

## Usage

*   [Documentation](https://docs.localstack.cloud)
    *   [Configuration](https://docs.localstack.cloud/references/configuration/)
    *   [CI Integration](https://docs.localstack.cloud/user-guide/ci/)
    *   [Integrations](https://docs.localstack.cloud/user-guide/integrations/)
    *   [Tools](https://docs.localstack.cloud/user-guide/tools/)
    *   [References](https://docs.localstack.cloud/references/)
    *   [FAQ](https://docs.localstack.cloud/getting-started/faq/)

*   **GUI Clients:**
    *   [LocalStack Web Application](https://app.localstack.cloud)
    *   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
    *   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

Find details of each release in the [GitHub releases](https://github.com/localstack/localstack/releases) or the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contribute

*   [Contributing Guide](docs/CONTRIBUTING.md)
*   [Development Environment Setup](docs/development-environment-setup/README.md)
*   [Open Issues](https://github.com/localstack/localstack/issues)

## Get in Touch

*   Report 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose)
*   Upvote 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   Ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/)
*   Discuss local cloud development in the [LocalStack Slack Community](https://localstack.cloud/contact/) and [GitHub Issue tracker](https://github.com/localstack/localstack/issues).

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

Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)). By using this software, you agree to the [End-User License Agreement (EULA)](docs/end_user_license_agreement).