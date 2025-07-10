# LocalStack: Your Local Cloud for AWS Development and Testing

Develop and test your AWS applications locally with LocalStack, a powerful cloud service emulator, streamlining your development workflow.  [Visit the original repository](https://github.com/localstack/localstack).

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

**Key Features:**

*   **Local AWS Emulation:** Run AWS services like Lambda, S3, DynamoDB, and more, locally.
*   **Simplified Testing:** Test your applications without connecting to the cloud.
*   **Faster Development:** Speed up your development cycle by eliminating remote dependencies.
*   **Comprehensive Service Support:** Supports a growing list of AWS services, with more available in the [Pro version](https://localstack.cloud/pricing).
*   **Multiple Installation Options:** Install via CLI, Docker, Docker Compose, or Helm.
*   **Rich Documentation:**  Extensive [documentation](https://docs.localstack.cloud) and [feature coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) pages available.

---

## Overview

LocalStack ([localstack.cloud](https://localstack.cloud)) is a cloud service emulator that runs in a single container, allowing you to develop and test your AWS applications locally, efficiently and without the need for a remote cloud provider.  Whether you're testing complex CDK applications or Terraform configurations, or just beginning to learn about AWS services, LocalStack helps speed up and simplify your testing and development workflow.

## Install

Choose your preferred installation method:

### Install via LocalStack CLI

The quickest way to get started with LocalStack is by using the LocalStack CLI. Ensure that your machine has a functional [`docker` environment](https://docs.docker.com/get-docker/) installed before proceeding.

#### Brew (macOS or Linux with Homebrew)

Install the LocalStack CLI through our [official LocalStack Brew Tap](https://github.com/localstack/homebrew-tap):

```bash
brew install localstack/tap/localstack-cli
```

#### Binary download (macOS, Linux, Windows)

If Brew is not installed on your machine, you can download the pre-built LocalStack CLI binary directly:

- Visit [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the latest release for your platform.
- Extract the downloaded archive to a directory included in your `PATH` variable:
    -   For macOS/Linux, use the command: `sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin`

#### PyPI (macOS, Linux, Windows)

LocalStack is developed using Python. To install the LocalStack CLI using `pip`, run the following command:

```bash
python3 -m pip install localstack
```

The `localstack-cli` installation enables you to run the Docker image containing the LocalStack runtime. To interact with the local AWS services, you need to install the `awslocal` CLI separately. For installation guidelines, refer to the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

> **Important**: Do not use `sudo` or run as `root` user. LocalStack must be installed and started entirely under a local non-root user. If you have problems with permissions in macOS High Sierra, install with `pip install --user localstack`

## Quickstart

Get started with LocalStack inside a Docker container:

```bash
 % localstack start -d
```

Check service statuses:

```bash
% localstack status services
```

Use SQS:

```shell
% awslocal sqs create-queue --queue-name sample-queue
```

For more details on using LocalStack AWS services, consult the [documentation](https://docs.localstack.cloud/references/coverage/).

## Running

You can run LocalStack using:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Explore our detailed [documentation](https://docs.localstack.cloud) for comprehensive guidance:

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

View the complete list of changes in each release at [GitHub releases](https://github.com/localstack/localstack/releases) and [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

Contribute to LocalStack by:

*   Reading the [contributing guide](docs/CONTRIBUTING.md).
*   Setting up your [development environment](docs/development-environment-setup/README.md).
*   Exploring the codebase and opening [issues](https://github.com/localstack/localstack/issues).

## Get in touch

Connect with the LocalStack team:

*   Report 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose)
*   Upvote 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   Ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/)
*   Discuss local cloud development:
    *   [LocalStack Slack Community](https://localstack.cloud/contact/)
    *   [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

### Contributors

Thank you to all contributors:

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

Support the project on [Open Collective](https://opencollective.com/localstack#backer):

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

Become a sponsor on [Open Collective](https://opencollective.com/localstack#sponsor):

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

Copyright (c) 2017-2025 LocalStack maintainers and contributors.

Copyright (c) 2016 Atlassian and others.

This version of LocalStack is released under the Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)). By downloading and using this software you agree to the [End-User License Agreement (EULA)](docs/end_user_license_agreement).