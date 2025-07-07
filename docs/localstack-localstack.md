<p align="center">
  <img src="https://raw.githubusercontent.com/localstack/localstack/master/docs/localstack-readme-banner.svg" alt="LocalStack - A fully functional local cloud stack">
</p>

<p align="center">
  <a href="https://github.com/localstack/localstack/actions/workflows/aws-main.yml?query=branch%3Amaster"><img alt="GitHub Actions" src="https://github.com/localstack/localstack/actions/workflows/aws-main.yml/badge.svg?branch=master"></a>
  <a href="https://coveralls.io/github/localstack/localstack?branch=master"><img alt="Coverage Status" src="https://coveralls.io/repos/github/localstack/localstack/badge.svg?branch=master"></a>
  <a href="https://pypi.org/project/localstack/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/localstack?color=blue"></a>
  <a href="https://hub.docker.com/r/localstack/localstack"><img alt="Docker Pulls" src="https://img.shields.io/docker/pulls/localstack/localstack"></a>
  <a href="https://pypi.org/project/localstack"><img alt="PyPi downloads" src="https://static.pepy.tech/badge/localstack"></a>
  <a href="#backers"><img alt="Backers on Open Collective" src="https://opencollective.com/localstack/backers/badge.svg"></a>
  <a href="#sponsors"><img alt="Sponsors on Open Collective" src="https://opencollective.com/localstack/sponsors/badge.svg"></a>
  <a href="https://img.shields.io/pypi/l/localstack.svg"><img alt="PyPI License" src="https://img.shields.io/pypi/l/localstack.svg"></a>
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
  <a href="https://twitter.com/localstack"><img alt="Twitter" src="https://img.shields.io/twitter/url/http/shields.io.svg?style=social"></a>
</p>

# LocalStack: Develop and Test AWS Applications Locally

**LocalStack empowers developers to build and test AWS applications locally, significantly speeding up development and ensuring cloud-readiness.**

## Key Features

*   **Local Cloud Environment:** Run a fully functional AWS cloud environment on your local machine.
*   **Comprehensive AWS Service Support:** Supports a wide range of AWS services including:
    *   AWS Lambda
    *   S3
    *   DynamoDB
    *   Kinesis
    *   SQS
    *   SNS
    *   And many more!
*   **Fast Development & Testing:** Accelerate your development workflow by eliminating the need to deploy to the cloud for testing.
*   **Integration with CI/CD:** Seamlessly integrate LocalStack into your Continuous Integration and Continuous Deployment pipelines.
*   **Easy Setup & Management:** Simple installation via CLI, Docker, Docker Compose, and Helm.
*   **User-Friendly Interfaces:** Provides a Web Application, Desktop app, and Docker extension for easy interaction and monitoring.
*   **Pro Version:** Offers extended APIs and advanced features for professional use ([pricing](https://localstack.cloud/pricing)).

**[Visit the LocalStack Repository on GitHub](https://github.com/localstack/localstack)**

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

---

## Overview

[LocalStack](https://localstack.cloud) is a powerful cloud service emulator that allows you to run your AWS applications and Lambdas entirely on your local machine. This means you can test complex applications, including CDK applications and Terraform configurations, without connecting to a remote cloud provider.  LocalStack streamlines your development and testing process whether you're a beginner or an experienced cloud developer.  Explore the [Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) page for a detailed list of supported APIs. Explore the [User Guides](https://docs.localstack.cloud/user-guide/) for more information.

## Install

Get started quickly by installing the LocalStack CLI. Ensure you have a working [`docker` environment](https://docs.docker.com/get-docker/) before you start.

### Brew (macOS or Linux with Homebrew)

Install the LocalStack CLI using the [official LocalStack Brew Tap](https://github.com/localstack/homebrew-tap):

```bash
brew install localstack/tap/localstack-cli
```

### Binary download (macOS, Linux, Windows)

If you do not have Brew, you can download the pre-built LocalStack CLI directly:

*   Download the latest release from [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) for your platform.
*   Extract the downloaded archive to a directory in your `PATH` variable:
    *   For macOS/Linux: `sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin`

### PyPI (macOS, Linux, Windows)

LocalStack is built with Python. Install the LocalStack CLI using `pip`:

```bash
python3 -m pip install localstack
```

After installing `localstack-cli`, install the `awslocal` CLI separately to interact with local AWS services. Refer to the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal) for installation instructions.

> **Important:** Avoid using `sudo` or running as `root`. Install and start LocalStack under a local, non-root user. For macOS High Sierra users, try `pip install --user localstack` if you face permission issues.

## Quickstart

Start LocalStack in a Docker container:

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

Use SQS (fully managed message queuing service) with LocalStack:

```shell
% awslocal sqs create-queue --queue-name sample-queue
{
    "QueueUrl": "http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/sample-queue"
}
```

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and the `awslocal` CLI.

## Running

Run LocalStack using:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Explore our [documentation](https://docs.localstack.cloud) to start using LocalStack:

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

Use these UI clients to interact with LocalStack:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

View release details in the [GitHub releases](https://github.com/localstack/localstack/releases) section and the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

Contribute to LocalStack:

*   Read the [contributing guide](docs/CONTRIBUTING.md).
*   Review the [development environment setup guide](docs/development-environment-setup/README.md).
*   Explore the codebase and [open issues](https://github.com/localstack/localstack/issues).

We appreciate all contributions.

## Get in touch

Contact the LocalStack Team for:

*   Reporting 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose)
*   Upvoting 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   Asking [support questions](https://docs.localstack.cloud/getting-started/help-and-support/)
*   Discussing local cloud development:
    *   [LocalStack Slack Community](https://localstack.cloud/contact/)
    *   [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

### Contributors

We thank all contributors to this project.

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

We are thankful to our backers on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

Support this project by becoming a sponsor on [Open Collective](https://opencollective.com/localstack#sponsor). Your logo will appear here!

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
```
Key improvements and SEO considerations:

*   **Concise Hook:** A clear and compelling one-sentence introduction to immediately grab the reader's attention.
*   **Keyword Optimization:** Incorporated relevant keywords like "AWS applications," "local development," "cloud testing," and specific AWS service names throughout the document.
*   **Structured Headings:** Uses clear and descriptive headings and subheadings for better readability and SEO.
*   **Bulleted Key Features:** Highlights key benefits in an easily scannable bulleted list.
*   **Internal Links:** Linking to sections within the README (e.g., [Install](#install)) improves user navigation and potentially SEO.
*   **External Links:** All external links are preserved, and where applicable given descriptive anchor text to aid SEO.
*   **Concise Language:** Avoided unnecessary wordiness to improve readability.
*   **Call to Action:** Included clear calls to action, such as "Visit the LocalStack Repository on GitHub".
*   **Clean Code:** Improved formatting for better presentation and maintainability.