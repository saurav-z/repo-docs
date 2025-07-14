# LocalStack: Your Local Cloud for AWS Development and Testing

**Develop and test your AWS applications locally with LocalStack, a fully functional cloud service emulator!**  [Visit the LocalStack Repository](https://github.com/localstack/localstack)

[![GitHub Actions](https://github.com/localstack/localstack/actions/workflows/aws-main.yml/badge.svg?branch=master)](https://github.com/localstack/localstack/actions/workflows/aws-main.yml?query=branch%3Amaster)
[![Coverage Status](https://coveralls.io/repos/github/localstack/localstack/badge.svg?branch=master)](https://coveralls.io/github/localstack/localstack?branch=master)
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

## Key Features

*   **Local AWS Emulation:** Run your AWS applications and Lambdas on your local machine without connecting to a remote cloud provider.
*   **Comprehensive Service Support:** Supports a wide range of AWS services including Lambda, S3, DynamoDB, Kinesis, SQS, SNS, and many more.
*   **Accelerated Development:** Speeds up your testing and development workflow, making it easier to test complex CDK applications and Terraform configurations.
*   **Pro Version:** Access additional APIs and advanced features with the [Pro version](https://localstack.cloud/pricing).
*   **Multiple Installation Options:** Easy installation via CLI, Docker, Docker Compose, and Helm.
*   **GUI Support:** Integrate with GUI clients for a user-friendly experience.

## Overview

LocalStack is a cloud service emulator that allows you to develop and test your AWS applications locally.  It runs in a single container, enabling you to simulate various AWS services on your laptop or in your CI environment. This eliminates the need to connect to a remote cloud provider during development and testing, saving time and resources. Whether you're a beginner learning AWS or an experienced developer, LocalStack streamlines your workflow.

LocalStack supports a growing number of AWS services, making it a versatile tool for various development scenarios. For a complete list of supported APIs, check the [feature coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) page.

## Install

Get started with LocalStack using the [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli) for easy management of your Docker container.

### Brew (macOS or Linux with Homebrew)

```bash
brew install localstack/tap/localstack-cli
```

### Binary download (macOS, Linux, Windows)

1.  Download the latest release from [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest).
2.  Extract the archive to a directory in your `PATH`.
    - For macOS/Linux: `sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin`

### PyPI (macOS, Linux, Windows)

```bash
python3 -m pip install localstack
```

**Note:** Install `awslocal` CLI separately to interact with local AWS services. See the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal) for guidance.

## Quickstart

Get up and running with LocalStack quickly:

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

Refer to the [LocalStack AWS services documentation](https://docs.localstack.cloud/references/coverage/) for more information.

## Running

Choose your preferred method:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Explore the documentation for detailed usage:

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

Use GUI clients:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

View complete release information in the [GitHub releases](https://github.com/localstack/localstack/releases) and the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

Contribute to LocalStack:

*   Read the [contributing guide](docs/CONTRIBUTING.md).
*   Set up your [development environment](docs/development-environment-setup/README.md).
*   Explore the [open issues](https://github.com/localstack/localstack/issues).

## Get in touch

Contact the LocalStack team for:

*   🐞 [Issues](https://github.com/localstack/localstack/issues/new/choose)
*   👍 [Feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   🙋🏽 [Support questions](https://docs.localstack.cloud/getting-started/help-and-support/)
*   🗣️ Discussions: [LocalStack Slack Community](https://localstack.cloud/contact/) or [GitHub Issue tracker](https://github.com/localstack/localstack/issues)

### Contributors

Thanks to all contributors!

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

Support the project by becoming a backer on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

Become a sponsor on [Open Collective](https://opencollective.com/localstack#sponsor).

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

LocalStack is released under the [Apache License, Version 2.0](LICENSE.txt). Read the [End-User License Agreement (EULA)](docs/end_user_license_agreement).