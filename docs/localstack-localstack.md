<p align="center">
  <img src="https://raw.githubusercontent.com/localstack/localstack/main/docs/localstack-readme-banner.svg" alt="LocalStack - A fully functional local cloud stack">
</p>

<p align="center">
  <a href="https://github.com/localstack/localstack/actions/workflows/aws-main.yml?query=branch%3Amain"><img alt="GitHub Actions" src="https://github.com/localstack/localstack/actions/workflows/aws-main.yml/badge.svg?branch=main"></a>
  <a href="https://coveralls.io/github/localstack/localstack?branch=main"><img alt="Coverage Status" src="https://coveralls.io/repos/github/localstack/localstack/badge.svg?branch=main"></a>
  <a href="https://pypi.org/project/localstack/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/localstack?color=blue"></a>
  <a href="https://hub.docker.com/r/localstack/localstack"><img alt="Docker Pulls" src="https://img.shields.io/docker/pulls/localstack/localstack"></a>
  <a href="https://pypi.org/project/localstack"><img alt="PyPi downloads" src="https://static.pepy.tech/badge/localstack"></a>
  <a href="#backers"><img alt="Backers on Open Collective" src="https://opencollective.com/localstack/backers/badge.svg"></a>
  <a href="#sponsors"><img alt="Sponsors on Open Collective" src="https://opencollective.com/localstack/sponsors/badge.svg"></a>
  <a href="https://img.shields.io/pypi/l/localstack.svg"><img alt="PyPI License" src="https://img.shields.io/pypi/l/localstack.svg"></a>
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
  <a href="https://bsky.app/profile/localstack.cloud"><img alt="Bluesky" src="https://img.shields.io/badge/bluesky-Follow-blue?logo=bluesky"></a>
</p>

# LocalStack: Your Local AWS Cloud for Development and Testing

**LocalStack empowers developers to build, test, and deploy AWS applications locally, accelerating development and reducing cloud costs.**  Learn more on the [original repository](https://github.com/localstack/localstack).

**Key Features:**

*   **Local AWS Environment:** Run a comprehensive set of AWS services on your local machine.
*   **Accelerated Development:** Develop and test your AWS applications locally, without the need for a remote cloud connection.
*   **Cost Savings:** Avoid incurring costs associated with using cloud resources during development and testing.
*   **Comprehensive Service Support:** Supports a growing number of AWS services like Lambda, S3, DynamoDB, SQS, SNS, and more.
*   **Pro Version:** Access additional APIs and advanced features through the [Pro version](https://localstack.cloud/pricing).

## Table of Contents

*   [Overview](#overview)
*   [Install](#install)
    *   [Brew (macOS or Linux with Homebrew)](#brew-macos-or-linux-with-homebrew)
    *   [Binary Download (macOS, Linux, Windows)](#binary-download-macos-linux-windows)
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

## Overview

[LocalStack](https://localstack.cloud) is a powerful cloud service emulator that runs in a single container, enabling you to develop and test AWS applications on your local machine.  This allows you to run your AWS applications or Lambdas entirely locally, eliminating the need to connect to a remote cloud provider during development and testing. Whether you're working on complex CDK applications, Terraform configurations, or simply learning about AWS services, LocalStack simplifies your development workflow.

LocalStack supports a wide array of AWS services, including AWS Lambda, S3, DynamoDB, Kinesis, SQS, SNS, and many more.  For expanded functionality and access to additional APIs, consider the [Pro version of LocalStack](https://localstack.cloud/pricing).  A complete list of supported APIs can be found on the [☑️ Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) page.

Enhance your cloud development experience with LocalStack's helpful features. Explore the [User Guides](https://docs.localstack.cloud/user-guide/) for more information.

## Install

The easiest way to get started is using the LocalStack CLI. Ensure a working [`docker` environment](https://docs.docker.com/get-docker/) is installed before proceeding.

### Brew (macOS or Linux with Homebrew)

Install the LocalStack CLI through our [official LocalStack Brew Tap](https://github.com/localstack/homebrew-tap):

```bash
brew install localstack/tap/localstack-cli
```

### Binary Download (macOS, Linux, Windows)

If Brew is not installed, download the pre-built LocalStack CLI binary:

*   Visit [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the latest release for your platform.
*   Extract the archive to a directory included in your `PATH` variable:
    *   For macOS/Linux, use: `sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin`

### PyPI (macOS, Linux, Windows)

Since LocalStack is developed using Python, you can use `pip`:

```bash
python3 -m pip install localstack
```

Install the `awslocal` CLI separately to interact with the local AWS services. Refer to the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

>   **Important**: Install and start LocalStack with a local, non-root user; avoid `sudo` or running as `root`. For permission issues in macOS High Sierra, install with `pip install --user localstack`.

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

Check service status:

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

Use SQS on LocalStack:

```shell
% awslocal sqs create-queue --queue-name sample-queue
{
    "QueueUrl": "http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/sample-queue"
}
```

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and using them with the `awslocal` CLI.

## Running

Run LocalStack using these options:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Explore our documentation to begin using LocalStack:

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

Find complete release changes on [GitHub releases](https://github.com/localstack/localstack/releases) and extended release notes in the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

To contribute to LocalStack:

*   Review our [contributing guide](docs/CONTRIBUTING.md).
*   Set up your [development environment](docs/development-environment-setup/README.md).
*   Browse the codebase and [open issues](https://github.com/localstack/localstack/issues).

We welcome all contributions and feedback.

## Get in touch

Connect with the LocalStack Team:

*   Report 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose).
*   Upvote 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+).
*   Ask 🙋🏽 [support questions](https://docs.localstack.cloud/getting-started/help-and-support/).
*   Discuss local cloud development on the [LocalStack Slack Community](https://localstack.cloud/contact/) or [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues).

### Contributors

We are thankful to all the people who have contributed to this project.

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

We are also grateful to all our backers who have donated to the project. You can become a backer on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

You can also support this project by becoming a sponsor on [Open Collective](https://opencollective.com/localstack#sponsor). Your logo will show up here along with a link to your website.

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

*   **Strong Title and Hook:** The updated title "LocalStack: Your Local AWS Cloud for Development and Testing" is more descriptive and includes relevant keywords.  The one-sentence hook immediately introduces the core value proposition.
*   **Keyword Optimization:** Keywords like "AWS," "local," "development," "testing," "cloud," are integrated naturally throughout the document.
*   **Clear Headings and Structure:** The use of headings, subheadings, and a Table of Contents enhances readability and SEO.  Google favors well-structured content.
*   **Concise Bullet Points:** Key features are listed in concise bullet points, making it easy for users to understand the benefits quickly.
*   **Relevant Links:** Links to important resources like the original repository, documentation, feature coverage, and the Pro version are included.
*   **Emphasis on Benefits:** The text emphasizes the benefits of using LocalStack, such as cost savings, accelerated development, and comprehensive service support.
*   **Call to Action:** Encourages users to explore documentation and get involved with the project.
*   **Complete Information:** Installation instructions, quickstart examples, and links to helpful resources have been retained and optimized.
*   **Backers and Sponsors Sections:** Improved to be more SEO friendly, incorporating relevant keywords and providing context for each section.
*   **License Information:** The license information is clearly presented at the end.
*   **Meta Description (Not Directly in Markdown):** A meta description could be added to the HTML `<head>` tag of a webpage that contains this Markdown, which further enhances SEO. Example:  `<meta name="description" content="Develop and test AWS applications locally with LocalStack. Save costs and accelerate your development workflow.">`