# LocalStack: Your Local Cloud for AWS Development and Testing

**LocalStack** is your all-in-one solution for developing and testing AWS applications locally.

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

*   **Local AWS Cloud:** Run your AWS applications locally, eliminating the need for a remote cloud connection during development and testing.
*   **Comprehensive Service Coverage:** Supports a wide range of AWS services, including Lambda, S3, DynamoDB, SQS, SNS, and more.
*   **Faster Development Cycles:**  Significantly speeds up testing and development workflows by enabling rapid iteration without cloud deployment delays.
*   **Cost-Effective:** Reduce cloud costs by developing and testing locally.
*   **CI/CD Integration:** Easily integrates with your CI/CD pipelines for automated testing.
*   **Multiple Installation Options:** Install via CLI, Docker, Docker Compose, Helm, or PyPI.

## Table of Contents

*   [Overview](#overview)
*   [Key Features](#key-features)
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

[LocalStack](https://localstack.cloud) is a powerful cloud service emulator designed to run in a single container on your local machine or within your CI environment. This enables you to build, test, and deploy your AWS applications and Lambdas entirely on your local system, without interacting with a remote cloud provider.  Whether you're working on complex CDK applications, Terraform configurations, or simply learning the ropes of AWS services, LocalStack provides a streamlined and efficient testing and development workflow.

LocalStack offers robust support for a growing number of AWS services, like AWS Lambda, S3, DynamoDB, Kinesis, SQS, SNS, and many more. The [Pro version of LocalStack](https://localstack.cloud/pricing) offers additional APIs and advanced features. A comprehensive list of supported APIs is available on our [☑️ Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) page.

Explore LocalStack's [User Guides](https://docs.localstack.cloud/user-guide/) for further insights.

## Install

Choose your preferred installation method:

### Brew (macOS or Linux with Homebrew)

```bash
brew install localstack/tap/localstack-cli
```

### Binary download (macOS, Linux, Windows)

1.  Visit [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the binary for your platform.
2.  Extract the downloaded archive to a directory included in your `PATH` variable, e.g.,:
    ```bash
    sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin
    ```

### PyPI (macOS, Linux, Windows)

```bash
python3 -m pip install localstack
```

Remember,  install and run LocalStack under a local non-root user. For help with permissions, see the provided documentation.  Install `awslocal` separately for interacting with local services. See the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

## Quickstart

Start LocalStack with the command line:

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

Check service status with:

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

Create a sample SQS queue:

```shell
% awslocal sqs create-queue --queue-name sample-queue
{
    "QueueUrl": "http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/sample-queue"
}
```

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and using them with LocalStack's `awslocal` CLI.

## Running

Choose from several methods to run LocalStack:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Explore LocalStack's [documentation](https://docs.localstack.cloud) for help.

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

UI clients are available:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

See the complete list of changes in each release on [GitHub releases](https://github.com/localstack/localstack/releases). For more information, see the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

To contribute to LocalStack:

*   Review the [contributing guide](docs/CONTRIBUTING.md).
*   Set up your [development environment](docs/development-environment-setup/README.md).
*   Browse the codebase and [open issues](https://github.com/localstack/localstack/issues).

## Get in touch

Contact the LocalStack team to:

*   Report 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose)
*   Upvote 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   Ask 🙋🏽 [support questions](https://docs.localstack.cloud/getting-started/help-and-support/)
*   Discuss local cloud development in the [LocalStack Slack Community](https://localstack.cloud/contact/)
*   Use the [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

### Contributors

Thank you to all contributors:

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

Support the project by becoming a backer on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

Sponsor LocalStack on [Open Collective](https://opencollective.com/localstack#sponsor).

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

[Back to Top](#localstack-your-local-cloud-for-aws-development-and-testing)
```
Key improvements:

*   **SEO Optimization:**  Includes relevant keywords like "AWS," "local cloud," "development," "testing," "emulator," and service names (Lambda, S3, etc.) throughout the content and in headings.
*   **Clear Structure:** Uses headings, subheadings, and bullet points to organize information and make it easier to scan.
*   **Concise Summaries:**  Provides brief, informative descriptions for each section.
*   **Actionable Installation Instructions:**  Presents clear, step-by-step installation guides for multiple methods.
*   **Quickstart Example:** Includes a short example showing how to use a service (SQS).
*   **Call to Action:** Encourages exploration of the project.
*   **Back to Top Link:** Added a link at the bottom of the page for easy navigation.
*   **One-Sentence Hook:**  Provides a compelling introduction to capture the reader's attention.
*   **Clear and Concise Language:** Rewrites original sentences for readability.
*   **Focus on Benefits:**  Highlights the key benefits of using LocalStack.
*   **Link Back to Repo:**  Added a link to the original repository.
*   **Updated Badges:** Preserves all badges from the original.
*   **Added Table of Contents.** Added a table of contents at the top to improve navigation.