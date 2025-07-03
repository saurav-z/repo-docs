# LocalStack: Develop and Test AWS Applications Locally

[LocalStack](https://github.com/localstack/localstack) empowers developers to build, test, and debug cloud applications locally, without the need for a real cloud provider.

<p align="center">
  <img src="https://raw.githubusercontent.com/localstack/localstack/master/docs/localstack-readme-banner.svg" alt="LocalStack - A fully functional local cloud stack">
</p>

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

*   **Local AWS Cloud Emulation:** Simulate a wide array of AWS services locally, including Lambda, S3, DynamoDB, SQS, SNS, and more.
*   **Fast Development Cycle:** Test and debug your cloud applications locally, significantly reducing development time and costs.
*   **Comprehensive Service Coverage:** Supports a growing list of AWS services, with the [Pro version](https://localstack.cloud/pricing) offering even more features and APIs.
*   **Integration with Popular Tools:** Seamlessly integrates with AWS CLI (`awslocal`), popular IDEs, and CI/CD pipelines.
*   **Multiple Installation Options:** Install via CLI, Docker, Docker Compose, Helm, or PyPI for flexibility.
*   **Detailed Documentation & Resources:** Extensive [documentation](https://docs.localstack.cloud) and [user guides](https://docs.localstack.cloud/user-guide/) to get you started.

---

## Table of Contents

*   [Overview](#overview)
*   [Install](#install)
    *   [Brew (macOS or Linux with Homebrew)](#brew-macos-or-linux-with-homebrew)
    *   [Binary download (macOS, Linux, Windows)](#binary-download-macos-linux-windows)
    *   [PyPI (macOS, Linux, Windows)](#pypi-macos-linux-windows)
*   [Quickstart](#quickstart)
*   [Running](#running)
*   [Usage](#usage)
*   [Releases](#releases)
*   [Contributing](#contributing)
*   [Get in touch](#get-in-touch)
    *   [Contributors](#contributors)
    *   [Backers](#backers)
    *   [Sponsors](#sponsors)
*   [License](#license)

---

## Overview

[LocalStack](https://localstack.cloud) is a powerful cloud service emulator that allows you to develop and test your AWS applications entirely on your local machine. This enables rapid prototyping, faster feedback loops, and reduced cloud costs. It runs in a single container, making it easy to integrate into your development workflow and CI/CD pipelines. From testing complex CDK applications and Terraform configurations to learning and experimenting with AWS services, LocalStack simplifies the entire process.

LocalStack supports numerous AWS services, and the [Pro version](https://localstack.cloud/pricing) extends the functionality further. Explore the [feature coverage page](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) for a complete list of supported APIs.

## Install

Choose your preferred installation method:

### Brew (macOS or Linux with Homebrew)

```bash
brew install localstack/tap/localstack-cli
```

### Binary download (macOS, Linux, Windows)

1.  Visit [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the latest release for your platform.
2.  Extract the downloaded archive to a directory included in your `PATH` variable (e.g., `/usr/local/bin`).

### PyPI (macOS, Linux, Windows)

```bash
python3 -m pip install localstack
```

Install the `awslocal` CLI separately for interacting with the local AWS services. Refer to the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

> **Important**: Install and run LocalStack as a local non-root user, without `sudo`. Use `pip install --user localstack` on macOS High Sierra if you encounter permissions issues.

## Quickstart

Start LocalStack with:

```bash
% localstack start -d
```

Check service status:

```bash
% localstack status services
```

Create an SQS queue:

```shell
% awslocal sqs create-queue --queue-name sample-queue
```

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and using them with the `awslocal` CLI.

## Running

Run LocalStack using:

-   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
-   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
-   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
-   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Explore the comprehensive [documentation](https://docs.localstack.cloud) for detailed guides:

-   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
-   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
-   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
-   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
-   [Understanding LocalStack](https://docs.localstack.cloud/references/)
-   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

Use GUI clients for easier interaction:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

View the complete list of changes in each release on [GitHub releases](https://github.com/localstack/localstack/releases) and the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

Contribute to LocalStack by:

-   Reading the [contributing guide](docs/CONTRIBUTING.md).
-   Setting up your [development environment](docs/development-environment-setup/README.md).
-   Exploring the [issues](https://github.com/localstack/localstack/issues).

We appreciate all contributions.

## Get in touch

Connect with the LocalStack Team for:

-   Reporting [issues](https://github.com/localstack/localstack/issues/new/choose)
-   Upvoting [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
-   Asking [support questions](https://docs.localstack.cloud/getting-started/help-and-support/)
-   Discussing local cloud development:

    *   [LocalStack Slack Community](https://localstack.cloud/contact/)
    *   [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

### Contributors

Thank you to all contributors!

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

Become a backer on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

Support the project by sponsoring on [Open Collective](https://opencollective.com/localstack#sponsor).

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

Released under the Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)) and the [End-User License Agreement (EULA)](docs/end_user_license_agreement).