# LocalStack: Develop and Test AWS Applications Locally

**LocalStack empowers developers to build, test, and debug AWS applications offline by providing a fully functional local cloud environment.**  [Go to the original repository](https://github.com/localstack/localstack).

<p align="center">
  <img src="https://raw.githubusercontent.com/localstack/localstack/main/docs/localstack-readme-banner.svg" alt="LocalStack - A fully functional local cloud stack">
</p>

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

## Key Features

*   **Local AWS Cloud:** Emulates a wide range of AWS services locally.
*   **Offline Development:** Develop and test applications without an internet connection.
*   **Faster Development Cycles:** Speed up testing and debugging by eliminating cloud deployment delays.
*   **Supports Popular Services:** Includes AWS Lambda, S3, DynamoDB, Kinesis, SQS, SNS, and more.
*   **Integration Friendly:** Works with popular tools like the AWS CLI, Terraform, and CDK.
*   **Pro Version Available:** Offers additional APIs and advanced features in the [Pro version](https://localstack.cloud/pricing).

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

LocalStack simplifies cloud development and testing by providing a local AWS cloud emulator. Develop and test your applications and Lambdas without connecting to a remote cloud provider. This speeds up and simplifies testing complex CDK applications or Terraform configurations.

LocalStack supports many AWS services, with the [Pro version](https://localstack.cloud/pricing) providing more APIs and advanced features. A comprehensive list of supported APIs can be found on the [Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) page.

Explore LocalStack's [User Guides](https://docs.localstack.cloud/user-guide/) for more details.

## Install

The quickest way to get started with LocalStack is by using the LocalStack CLI. Ensure you have Docker installed on your machine.

### Brew (macOS or Linux with Homebrew)

Install the LocalStack CLI using the official LocalStack Brew Tap:

```bash
brew install localstack/tap/localstack-cli
```

### Binary download (macOS, Linux, Windows)

If you don't have Brew:

1.  Download the latest release from [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest).
2.  Extract the archive to a directory in your `PATH`.
    -   macOS/Linux example: `sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin`

### PyPI (macOS, Linux, Windows)

Install the LocalStack CLI using `pip`:

```bash
python3 -m pip install localstack
```

Install `awslocal` separately for interacting with the local AWS services; see the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

> **Important:** Do not run as `root`.  Install and start LocalStack under a local non-root user.  If you have permission issues on macOS High Sierra, install with `pip install --user localstack`.

## Quickstart

Start LocalStack:

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

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and using them with LocalStack's `awslocal` CLI.

## Running

Run LocalStack using:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Get started with our [documentation](https://docs.localstack.cloud).

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

Use the following UI clients:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

See the complete list of changes at [GitHub releases](https://github.com/localstack/localstack/releases). Extended release notes are in the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

Contribute to LocalStack:

*   Read the [contributing guide](docs/CONTRIBUTING.md).
*   Set up your [development environment](docs/development-environment-setup/README.md).
*   Browse the codebase and [open issues](https://github.com/localstack/localstack/issues).

Your contributions and feedback are highly valued.

## Get in touch

*   Report issues: [issue tracker](https://github.com/localstack/localstack/issues/new/choose)
*   Upvote feature requests: [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   Ask for support: [support questions](https://docs.localstack.cloud/getting-started/help-and-support/)
*   Discuss local cloud development:
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

Copyright (c) 2017-2025 LocalStack maintainers and contributors.
Copyright (c) 2016 Atlassian and others.

Released under the Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)). By using this software, you agree to the [End-User License Agreement (EULA)](docs/end_user_license_agreement).