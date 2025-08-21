[![LocalStack](https://raw.githubusercontent.com/localstack/localstack/main/docs/localstack-readme-banner.svg)](https://github.com/localstack/localstack)

# LocalStack: Your Local Cloud for AWS Development and Testing

**LocalStack empowers developers to build, test, and deploy AWS applications locally, accelerating development and eliminating cloud costs.**

[GitHub Actions](https://github.com/localstack/localstack/actions/workflows/aws-main.yml?query=branch%3Amain) | [Coverage Status](https://coveralls.io/github/localstack/localstack?branch=main) | [PyPI Version](https://pypi.org/project/localstack/) | [Docker Pulls](https://hub.docker.com/r/localstack/localstack) | [PyPi Downloads](https://static.pepy.tech/badge/localstack) | [Backers](https://opencollective.com/localstack/backers/badge.svg) | [Sponsors](https://opencollective.com/localstack/sponsors/badge.svg) | [PyPI License](https://img.shields.io/pypi/l/localstack.svg) | [Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg) | [Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json) | [Bluesky](https://bsky.app/profile/localstack.cloud)

**Key Features:**

*   **Local AWS Emulation:** Run and test your AWS applications locally, simulating a wide range of AWS services like Lambda, S3, DynamoDB, and more.
*   **Accelerated Development:** Speed up your development workflow by eliminating the need to deploy to the cloud for testing and debugging.
*   **Cost Savings:** Reduce your cloud costs by developing and testing your applications offline or in your CI/CD environment.
*   **Comprehensive Service Coverage:**  Supports a growing number of AWS services, including Lambda, S3, DynamoDB, Kinesis, SQS, SNS, and many more.
*   **Easy Integration:** Integrates seamlessly with various tools and frameworks, including the AWS CLI, SDKs, and popular CI/CD systems.

**[View the latest release (v4.7.0)](https://blog.localstack.cloud/localstack-for-aws-release-v-4-7-0/)**

**Get Started:**

*   [Overview](#overview)
*   [Install](#install)
*   [Quickstart](#quickstart)
*   [Running](#running)
*   [Usage](#usage)
*   [Releases](#releases)
*   [Contributing](#contributing)

---

## Overview

[LocalStack](https://localstack.cloud) is a cloud service emulator designed to run in a single container, enabling developers to build, test, and debug AWS applications locally.  This allows for rapid iteration and eliminates the need to connect to a remote cloud provider during development.  It's perfect for testing complex CDK applications, Terraform configurations, or simply learning about AWS services without incurring cloud costs.

LocalStack supports a wide range of AWS services, with the [Pro version](https://localstack.cloud/pricing) offering even more features and APIs.  See the [Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) page for a complete list of supported services.  Explore [User Guides](https://docs.localstack.cloud/user-guide/) for additional features and insights.

## Install

The quickest way to get started is with the LocalStack CLI. Ensure you have Docker installed.

### Brew (macOS or Linux with Homebrew)

```bash
brew install localstack/tap/localstack-cli
```

### Binary Download (macOS, Linux, Windows)

*   Download the latest release from [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest).
*   Extract to a directory in your `PATH`.

    ```bash
    sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin  # Example for macOS/Linux
    ```

### PyPI (macOS, Linux, Windows)

```bash
python3 -m pip install localstack
```

Install `awslocal` CLI separately for interacting with AWS services.  See the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

> **Important:** Avoid using `sudo` or running as `root`.

## Quickstart

Start LocalStack in Docker:

```bash
% localstack start -d
```

Check service status:

```bash
% localstack status services
```

Use SQS:

```shell
% awslocal sqs create-queue --queue-name sample-queue
```

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and `awslocal` CLI.

## Running

Run LocalStack using these methods:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Explore our comprehensive documentation for detailed usage instructions:

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

GUI options:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

See the [GitHub releases](https://github.com/localstack/localstack/releases) for complete release details and the [changelog](https://docs.localstack.cloud/references/changelog/) for extended release notes.

## Contributing

Join the LocalStack community:

*   Read the [contributing guide](docs/CONTRIBUTING.md).
*   Explore the [development environment setup guide](docs/development-environment-setup/README.md).
*   Review [open issues](https://github.com/localstack/localstack/issues).

## Get in Touch

Report issues, request features, ask support questions, and discuss local cloud development.

*   [LocalStack Slack Community](https://localstack.cloud/contact/)
*   [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

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

[Apache License 2.0](LICENSE.txt) - [End-User License Agreement (EULA)](docs/end_user_license_agreement).

**[Return to the LocalStack Repository](https://github.com/localstack/localstack)**