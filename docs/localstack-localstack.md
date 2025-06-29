# LocalStack: Develop and Test AWS Applications Locally

**LocalStack enables you to develop and test your AWS applications locally, eliminating the need for a remote cloud provider.** ([View on GitHub](https://github.com/localstack/localstack))

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

**Key Features:**

*   **Local AWS Cloud Emulation:** Run and test AWS services locally, including Lambda, S3, DynamoDB, Kinesis, SQS, and more.
*   **Rapid Development & Testing:** Accelerate your development workflow by avoiding deployment delays and costs associated with remote cloud providers.
*   **Comprehensive Coverage:** Supports a wide range of AWS services, with more added frequently.  Check our [feature coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) for a full list.
*   **Flexible Installation:** Install via CLI, Docker, Docker Compose, Helm, or PyPI, making it easy to integrate into your preferred environment.
*   **CI/CD Integration:** Ideal for integrating into Continuous Integration and Continuous Delivery (CI/CD) pipelines for automated testing.
*   **Pro Version:** Offers advanced features and extended API support.
*   **GUI Clients:** Web Application, LocalStack Desktop and Docker Extension
*   **Active Community & Support:** Benefit from a vibrant community and extensive documentation.

**Table of Contents:**

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

[LocalStack](https://localstack.cloud) is a cloud service emulator that allows developers to run and test their AWS applications locally, in a single container. This is ideal for testing complex CDK applications, Terraform configurations, or simply learning about AWS services without incurring cloud provider costs.

LocalStack supports a wide array of AWS services, including AWS Lambda, S3, DynamoDB, Kinesis, SQS, SNS, and more, making it a versatile tool for local development and testing. The [Pro version of LocalStack](https://localstack.cloud/pricing) unlocks even more APIs and advanced features. For a complete list of supported APIs, see our [☑️ Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) page.

Find more information in LocalStack's [User Guides](https://docs.localstack.cloud/user-guide/).

## Install

Choose your preferred installation method:

### Brew (macOS or Linux with Homebrew)

Install the LocalStack CLI using our [official LocalStack Brew Tap](https://github.com/localstack/homebrew-tap):

```bash
brew install localstack/tap/localstack-cli
```

### Binary download (macOS, Linux, Windows)

Download the pre-built LocalStack CLI binary from [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and extract it to a directory in your `PATH`:

-   For macOS/Linux, use the command: `sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin`

### PyPI (macOS, Linux, Windows)

Install the LocalStack CLI with `pip`:

```bash
python3 -m pip install localstack
```

Install the `awslocal` CLI separately for interacting with local AWS services. See the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal) for details.

>   **Important:** Avoid using `sudo` or running as `root`. LocalStack should be installed and run under a local non-root user. For macOS High Sierra permission issues, try `pip install --user localstack`.

## Quickstart

Start LocalStack inside a Docker container:

```bash
% localstack start -d

     __                     _______ __             __
    / /   ____  _________ _/ / ___// /_____ ______/ /__
   / /   / __ \/ ___/ __ `/ /\__ \/ __/ __ `/ ___/ //_/
  / /___/ /_/ / /__/ /_/ / /___/ / /_/ /_/ / /__/ ,<
 /_____/\____/\___/\__,_/_//____/\__/\__,_/\___/_/|_|

- LocalStack CLI: 4.5.0
- Profile: default
- App: https://app.localstack.cloud

[17:00:15] starting LocalStack in Docker mode 🐳               localstack.py:512
           preparing environment                               bootstrap.py:1322
           configuring container                               bootstrap.py:1330
           starting container                                  bootstrap.py:1340
[17:00:16] detaching                                           bootstrap.py:1344
```

Check the status of services:

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

Create an SQS queue:

```shell
% awslocal sqs create-queue --queue-name sample-queue
{
    "QueueUrl": "http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/sample-queue"
}
```

Explore [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and the `awslocal` CLI.

## Running

Choose your preferred method for running LocalStack:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Find detailed usage instructions in our [documentation](https://docs.localstack.cloud):

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

GUI Clients:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

See the [GitHub releases](https://github.com/localstack/localstack/releases) for release details and the [changelog](https://docs.localstack.cloud/references/changelog/) for extended release notes.

## Contributing

To contribute:

*   Read our [contributing guide](docs/CONTRIBUTING.md).
*   Consult our [development environment setup guide](docs/development-environment-setup/README.md).
*   Explore the codebase and [open issues](https://github.com/localstack/localstack/issues).

## Get in touch

*   Report 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose)
*   Upvote 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   Ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/)
*   Discuss local cloud development:
    *   [LocalStack Slack Community](https://localstack.cloud/contact/)
    *   [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

### Contributors

Thanks to everyone who has contributed!

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

Support the project by becoming a backer on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

Become a sponsor on [Open Collective](https://opencollective.com/localstack#sponsor) and have your logo featured here.

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