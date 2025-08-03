# LocalStack: Develop and Test AWS Applications Locally

**LocalStack enables developers to build and test AWS applications locally, without needing a remote cloud provider.** ([View on GitHub](https://github.com/localstack/localstack))

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

*   **Local AWS Cloud Emulation:** Mimic AWS services locally, including Lambda, S3, DynamoDB, Kinesis, SQS, and SNS.
*   **Local Development & Testing:** Develop and test AWS applications on your local machine, streamlining your development workflow.
*   **CI/CD Integration:** Seamlessly integrate LocalStack into your CI/CD pipelines for automated testing.
*   **Broad Service Support:** Supports a wide range of AWS services, with the Pro version expanding capabilities.
*   **Multiple Installation Options:** Easy to install via CLI, Docker, Docker Compose, or Helm.
*   **User-Friendly Tools:** Provides a Web UI, desktop application, and Docker extension for easier management.

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
*   [Get in Touch](#get-in-touch)
    *   [Contributors](#contributors)
    *   [Backers](#backers)
    *   [Sponsors](#sponsors)
*   [License](#license)

---

## Overview

[LocalStack](https://localstack.cloud) is a powerful cloud service emulator that allows you to develop and test your AWS applications locally in a single container. This eliminates the need for a remote cloud provider during the development and testing phases, making it easier and faster to build and iterate on your AWS applications, including complex CDK applications, Terraform configurations, and AWS Lambda functions.  LocalStack supports a growing list of AWS services, including Lambda, S3, DynamoDB, Kinesis, SQS, SNS, and more. Explore the [Pro version](https://localstack.cloud/pricing) for more advanced features and APIs. Check out the [Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) page for a comprehensive list of supported APIs. Additionally, LocalStack offers helpful [User Guides](https://docs.localstack.cloud/user-guide/) to enhance your experience.

## Install

Get started quickly by installing the LocalStack CLI. Ensure your machine has a functional [`docker` environment](https://docs.docker.com/get-docker/) installed.

### Brew (macOS or Linux with Homebrew)

Install the LocalStack CLI using our [official LocalStack Brew Tap](https://github.com/localstack/homebrew-tap):

```bash
brew install localstack/tap/localstack-cli
```

### Binary download (macOS, Linux, Windows)

If you don't have Brew, download the pre-built LocalStack CLI binary:

-   Visit [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the latest release for your platform.
-   Extract the archive to a directory in your `PATH` variable.
    -   For macOS/Linux, use: `sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin`

### PyPI (macOS, Linux, Windows)

Install the LocalStack CLI using `pip`:

```bash
python3 -m pip install localstack
```

After installing the `localstack-cli`, install the `awslocal` CLI to interact with the local AWS services, following the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

> **Important**: Avoid using `sudo` or running as `root`. LocalStack must be installed and started by a local, non-root user. If you encounter issues in macOS High Sierra, try `pip install --user localstack`.

## Quickstart

Start LocalStack in a Docker container:

```bash
 % localstack start -d

     __                     _______ __             __
    / /   ____  _________ _/ / ___// /_____ ______/ /__
   / /   / __ \/ ___/ __ `/ /\__ \/ __/ __ `/ ___/ //_/
  / /___/ /_/ / /__/ /_/ / /___/ / /_/ /_/ / /__/ ,<
 /_____/\____/\___/\__,_/_//____/\__/\__,_/\___/_/|_|

- LocalStack CLI: 4.7.0
- Profile: default
- App: https://app.localstack.cloud

[17:00:15] starting LocalStack in Docker mode 🐳               localstack.py:512
           preparing environment                               bootstrap.py:1322
           configuring container                               bootstrap.py:1330
           starting container                                  bootstrap.py:1340
[17:00:16] detaching                                           bootstrap.py:1344
```

Check service statuses:

```bash
% localstack status services
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Service                  ┃ Status      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ acm                      │ ✔ available │
│ apigateway               │ ✔ available │
│ cloudformation           │ ✔ available │
│ cloudwatch               │ ✔ available │
│ config                   │ ✔ available │
│ dynamodb                 │ ✔ available │
...
```

Use SQS, for example:

```shell
% awslocal sqs create-queue --queue-name sample-queue
{
    "QueueUrl": "http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/sample-queue"
}
```

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and the `awslocal` CLI.

## Running

Run LocalStack using these options:

-   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
-   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
-   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
-   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Explore our [documentation](https://docs.localstack.cloud) to start using LocalStack.

-   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
-   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
-   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
-   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
-   [Understanding LocalStack](https://docs.localstack.cloud/references/)
-   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

Use these UI clients for a graphical interface:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

See the [GitHub releases](https://github.com/localstack/localstack/releases) for complete release details and the [changelog](https://docs.localstack.cloud/references/changelog/) for extended release notes.

## Contributing

Contribute to LocalStack:

-   Read our [contributing guide](docs/CONTRIBUTING.md).
-   Set up your [development environment](docs/development-environment-setup/README.md).
-   Explore the codebase and [open issues](https://github.com/localstack/localstack/issues).

We appreciate all contributions and feedback.

## Get in Touch

Contact the LocalStack Team to:

-   Report 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose)
-   Upvote 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
-   Ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/)
-   Discuss local cloud development:

-   [LocalStack Slack Community](https://localstack.cloud/contact/)
-   [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

### Contributors

We thank all project contributors.

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

Support the project by becoming a backer on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

Become a sponsor on [Open Collective](https://opencollective.com/localstack#sponsor). Your logo will appear here.

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