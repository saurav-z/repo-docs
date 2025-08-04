# LocalStack: Develop and Test AWS Applications Locally

**LocalStack lets you emulate AWS cloud services on your local machine, making it easier to develop, test, and debug your cloud applications.**

[<img src="https://raw.githubusercontent.com/localstack/localstack/main/docs/localstack-readme-banner.svg" alt="LocalStack Banner" width="100%">](https://github.com/localstack/localstack)

[![GitHub Actions](https://github.com/localstack/localstack/actions/workflows/aws-main.yml/badge.svg?branch=main)](https://github.com/localstack/localstack/actions/workflows/aws-main.yml?query=branch%3Amain)
[![Coverage Status](https://coveralls.io/repos/github/localstack/localstack/badge.svg?branch=main)](https://coveralls.io/github/localstack/localstack?branch=main)
[![PyPI Version](https://img.shields.io/pypi/v/localstack?color=blue)](https://pypi.org/project/localstack/)
[![Docker Pulls](https://img.shields.io/docker/pulls/localstack/localstack)](https://hub.docker.com/r/localstack/localstack)
[![PyPi downloads](https://static.pepy.tech/badge/localstack)](https://pypi.org/project/localstack)
[![Backers on Open Collective](https://opencollective.com/localstack/backers/badge.svg)](https://opencollective.com/localstack#backers)
[![Sponsors on Open Collective](https://opencollective.com/localstack/sponsors/badge.svg)](https://opencollective.com/localstack#sponsors)
[![PyPI License](https://img.shields.io/pypi/l/localstack.svg)](https://img.shields.io/pypi/l/localstack.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Bluesky](https://img.shields.io/badge/bluesky-Follow-blue?logo=bluesky)](https://bsky.app/profile/localstack.cloud)

---

## Key Features

*   **Local Cloud Development:** Develop and test AWS applications locally without connecting to the cloud.
*   **AWS Service Support:** Supports a wide range of AWS services like Lambda, S3, DynamoDB, SQS, SNS, and more.
*   **Fast Development & Testing:** Accelerates testing of CDK applications, Terraform configurations, and AWS service learning.
*   **Pro Version:** Offers advanced features and expanded API support.
*   **Multiple Installation Options:** Available via CLI, Docker, Docker Compose, Helm, and PyPI.
*   **GUI Clients:** Integrates with web applications, desktop applications, and Docker extensions for a seamless user experience.

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

[LocalStack](https://localstack.cloud) is a powerful cloud service emulator that runs in a single container, designed to replicate the AWS cloud environment on your local machine or within your CI/CD pipeline. This allows you to build, test, and debug AWS applications, serverless functions, and infrastructure code without requiring a connection to the actual AWS cloud. It is an invaluable tool for developers aiming to improve development workflows and reduce cloud-related costs.

LocalStack supports a wide array of AWS services, including essential components like Lambda, S3, DynamoDB, Kinesis, SQS, and SNS. For advanced users and extensive feature sets, explore the [Pro version of LocalStack](https://localstack.cloud/pricing) which offers expanded APIs and enhanced functionalities. Explore the full list of supported services on the [☑️ Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) page.

Check out the [User Guides](https://docs.localstack.cloud/user-guide/) for more in-depth information and advanced usage.

## Install

Get started with LocalStack by choosing one of the installation methods below:

### Brew (macOS or Linux with Homebrew)

Install the LocalStack CLI using the official tap:

```bash
brew install localstack/tap/localstack-cli
```

### Binary download (macOS, Linux, Windows)

If you do not use Homebrew, download the CLI binary:

1.  Go to [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the correct binary for your operating system.
2.  Extract the archive to a directory included in your `PATH`.

    *   For macOS/Linux: `sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin`

### PyPI (macOS, Linux, Windows)

Install the LocalStack CLI using `pip`:

```bash
python3 -m pip install localstack
```

Install `awslocal` CLI separately by consulting the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

> **Important:** Avoid using `sudo` or running as the `root` user during the installation and startup of LocalStack.

## Quickstart

Start LocalStack in a Docker container:

```bash
% localstack start -d
```

Check the status of services:

```bash
% localstack status services
```

Create a queue in SQS:

```shell
% awslocal sqs create-queue --queue-name sample-queue
```

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and how to use them with the `awslocal` CLI.

## Running

You can run LocalStack using the following methods:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

To get started with LocalStack, check out our documentation at:

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

You can use the following UI clients to use LocalStack:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

Find release details at [GitHub releases](https://github.com/localstack/localstack/releases) and the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

Contribute to LocalStack:

*   Read our [contributing guide](docs/CONTRIBUTING.md).
*   Set up your [development environment](docs/development-environment-setup/README.md).
*   Explore the codebase and [open issues](https://github.com/localstack/localstack/issues).

We value all contributions and feedback.

## Get in touch

Contact the LocalStack Team to report [issues](https://github.com/localstack/localstack/issues/new/choose), upvote [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+), ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/), or discuss local cloud development:

*   [LocalStack Slack Community](https://localstack.cloud/contact/)
*   [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

### Contributors

See the contributors to this project:

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

Become a backer on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

Support the project as a sponsor on [Open Collective](https://opencollective.com/localstack#sponsor):

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

Released under the Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)).
By using the software, you agree to the [End-User License Agreement (EULA)](docs/end_user_license_agreement).