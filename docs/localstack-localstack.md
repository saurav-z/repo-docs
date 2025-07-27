# LocalStack: Develop and Test AWS Applications Locally

**LocalStack** empowers developers to build and test AWS applications locally, accelerating development and reducing cloud costs; learn more on the original repo: [https://github.com/localstack/localstack](https://github.com/localstack/localstack).

[![GitHub Actions](https://github.com/localstack/localstack/actions/workflows/aws-main.yml/badge.svg?branch=main)](https://github.com/localstack/localstack/actions/workflows/aws-main.yml?query=branch%3Amain)
[![Coverage Status](https://coveralls.io/repos/github/localstack/localstack/badge.svg?branch=main)](https://coveralls.io/github/localstack/localstack?branch=main)
[![PyPI Version](https://img.shields.io/pypi/v/localstack?color=blue)](https://pypi.org/project/localstack/)
[![Docker Pulls](https://img.shields.io/docker/pulls/localstack/localstack)](https://hub.docker.com/r/localstack/localstack)
[![PyPi downloads](https://static.pepy.tech/badge/localstack)](https://pypi.org/project/localstack)
[![Backers on Open Collective](https://opencollective.com/localstack/backers/badge.svg)](https://opencollective.com/localstack)
[![Sponsors on Open Collective](https://opencollective.com/localstack/sponsors/badge.svg)](https://opencollective.com/localstack)
[![PyPI License](https://img.shields.io/pypi/l/localstack.svg)](https://img.shields.io/pypi/l/localstack.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Bluesky](https://img.shields.io/badge/bluesky-Follow-blue?logo=bluesky)](https://bsky.app/profile/localstack.cloud)

---

## Key Features:

*   **Local AWS Cloud Emulation:** Run your AWS applications and Lambdas entirely on your local machine.
*   **Fast Development & Testing:** Speed up your development and testing workflows.
*   **Cost-Effective:** Reduce costs by developing and testing locally without using remote cloud resources.
*   **Comprehensive Service Support:** Supports a wide range of AWS services, including Lambda, S3, DynamoDB, and more.
*   **Easy Integration:** Seamlessly integrates with various CI/CD pipelines and development environments.
*   **CLI & GUI Tools:** Offers a command-line interface and graphical user interfaces (web application, desktop app, Docker extension) for streamlined management and interaction.
*   **Open Source & Pro Version:**  Provides a powerful open-source version and a Pro version with expanded API coverage and advanced features.

---

## Overview

LocalStack is a cloud service emulator designed to run within a single container on your local machine or in your CI environment.  It allows you to develop and test AWS applications and Lambdas without connecting to a remote cloud provider.  Whether you're working on complex CDK applications, Terraform configurations, or just starting to learn about AWS services, LocalStack simplifies your testing and development workflow.

LocalStack supports many AWS services like Lambda, S3, DynamoDB, Kinesis, SQS, and SNS. The [Pro version](https://localstack.cloud/pricing) offers additional APIs and advanced features. See the [Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) page for a complete list of supported APIs.  Explore the [User Guides](https://docs.localstack.cloud/user-guide/) for more information.

## Installation

Get started with LocalStack using the LocalStack CLI. Ensure you have a functional [`docker` environment](https://docs.docker.com/get-docker/) installed.

### Brew (macOS or Linux with Homebrew)

```bash
brew install localstack/tap/localstack-cli
```

### Binary Download (macOS, Linux, Windows)

1.  Visit [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the latest release for your platform.
2.  Extract the archive to a directory in your `PATH` variable:
    -   macOS/Linux: `sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin`

### PyPI (macOS, Linux, Windows)

```bash
python3 -m pip install localstack
```

Install the `awslocal` CLI separately for interacting with local AWS services; see the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

> **Important**: Install and run LocalStack as a local, non-root user; avoid using `sudo`.  On macOS High Sierra, install with `pip install --user localstack`.

## Quickstart

Run LocalStack in a Docker container:

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

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and the `awslocal` CLI.

## Running Options

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Explore the [documentation](https://docs.localstack.cloud) for:

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

Use these UI clients:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

View the complete list of changes in the [GitHub releases](https://github.com/localstack/localstack/releases) and the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

*   Read the [contributing guide](docs/CONTRIBUTING.md).
*   Review the [development environment setup guide](docs/development-environment-setup/README.md).
*   Explore the codebase and [open issues](https://github.com/localstack/localstack/issues).

## Get in Touch

Report 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose), upvote 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+), ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/), or 🗣️ discuss local cloud development:

*   [LocalStack Slack Community](https://localstack.cloud/contact/)
*   [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

### Contributors

[Contributors' graph](https://github.com/localstack/localstack/graphs/contributors)

### Backers

[Backers' graph](https://opencollective.com/localstack#backers)

### Sponsors

[Sponsors' graph](https://opencollective.com/localstack#sponsor)

## License

Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)) and [End-User License Agreement (EULA)](docs/end_user_license_agreement).
```
Key improvements:

*   **SEO Optimization:**  Included relevant keywords like "AWS," "local," "cloud," "development," "testing," and service names.
*   **Clear Headings:**  Used clear and descriptive headings for better readability and SEO.
*   **Bulleted Key Features:**  Emphasized key features for quick understanding.
*   **Concise Hook:** The opening sentence is a clear value proposition.
*   **Structured Content:** Improved organization of sections for easier navigation.
*   **Actionable Instructions:** Kept the installation and quickstart instructions but made them more scannable.
*   **Links to Docs:**  Included links to important documentation and resources.
*   **Concise Summaries:** Kept the text concise while providing the essential information.
*   **Removed Redundancy:**  Combined similar sections.
*   **Improved Formatting:**  Used bold text and lists to improve readability.
*   **Alt Text:** Added `alt` text to image badges.
*   **Simplified Installation:**  Simplified and made the installation process easier to understand.