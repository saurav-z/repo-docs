<!--
  _   _   _       _   _   _       _       _   _
 | \ | | | |     | | | | | |     | |     | | | |
 |  \| | | |     | | | | | |     | |     | | | |
 | . ` | | |     | | | | | |     | |     | | | |
 |_|\__| |_|     |_| |_| |_|     |_|     |_| |_|
-->

<h1 align="center">LocalStack: Develop and Test AWS Applications Locally</h1>

<p align="center">
  <a href="https://github.com/localstack/localstack">
    <img src="https://raw.githubusercontent.com/localstack/localstack/master/docs/localstack-readme-banner.svg" alt="LocalStack Banner" width="800">
  </a>
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
  <a href="https://bsky.app/profile/localstack.cloud"><img alt="Bluesky" src="https://img.shields.io/badge/bluesky-Follow-blue?logo=bluesky"></a>
</p>

LocalStack enables you to develop and test your AWS applications locally, without the need for a real cloud environment.

**Key Features:**

*   **Local AWS Cloud Emulation:** Run a wide range of AWS services (Lambda, S3, DynamoDB, etc.) on your local machine.
*   **Faster Development & Testing:**  Accelerate your development workflow by testing locally, reducing deployment times and costs.
*   **Supports Many AWS Services:**  Includes support for popular AWS services like Lambda, S3, DynamoDB, Kinesis, SQS, and SNS.
*   **Integration with Tools:** Seamlessly integrates with popular tools and frameworks for CI/CD, testing, and local development.
*   **Pro Version for Extended Capabilities:** The [Pro version](https://localstack.cloud/pricing) offers additional features, APIs, and advanced capabilities.

**Links:**

*   [LocalStack Documentation](https://docs.localstack.cloud)
*   [LocalStack Pro Version](https://app.localstack.cloud)
*   [Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/)
*   [GitHub Repository](https://github.com/localstack/localstack)

---

## Table of Contents

*   [Overview](#overview)
*   [Install](#install)
    *   [Brew](#brew-macos-or-linux-with-homebrew)
    *   [Binary download](#binary-download-macos-linux-windows)
    *   [PyPI](#pypi-macos-linux-windows)
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

[LocalStack](https://localstack.cloud) is a powerful cloud service emulator designed to run within a single container, whether on your laptop or in a CI environment. It provides a convenient way to develop and test your AWS applications and Lambdas locally, eliminating the need to connect to a remote cloud provider during development and testing. Perfect for testing CDK applications, Terraform configurations, or simply learning about AWS services, LocalStack significantly speeds up and streamlines your testing and development workflow.

LocalStack offers a wide range of supported AWS services, including AWS Lambda, S3, DynamoDB, Kinesis, SQS, and SNS, with additional APIs and advanced features available in the [Pro version of LocalStack](https://localstack.cloud/pricing). For a comprehensive list of supported APIs, please consult our [Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) page.

LocalStack also includes a variety of helpful features designed to simplify the cloud development experience, find detailed instructions and guides in LocalStack's [User Guides](https://docs.localstack.cloud/user-guide/).

## Install

The most efficient method for getting started with LocalStack is by leveraging the LocalStack CLI, enabling you to directly manage and launch the LocalStack Docker container through your command line. Prior to starting, ensure you have a properly functioning [`docker` environment](https://docs.docker.com/get-docker/) installed on your machine.

### Brew (macOS or Linux with Homebrew)

Install the LocalStack CLI using our [official LocalStack Brew Tap](https://github.com/localstack/homebrew-tap):

```bash
brew install localstack/tap/localstack-cli
```

### Binary download (macOS, Linux, Windows)

If you are not using Brew, you can download the pre-built LocalStack CLI binary directly:

-   Go to [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the newest release for your specific platform.
-   Unpack the downloaded archive into a directory that's included in your `PATH` variable. For instance, for macOS/Linux: `sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin`

### PyPI (macOS, Linux, Windows)

LocalStack is developed using Python. Install the LocalStack CLI via `pip` by executing:

```bash
python3 -m pip install localstack
```

The `localstack-cli` package facilitates the execution of the Docker image that contains the LocalStack runtime. To interact with the local AWS services, the `awslocal` CLI must be installed separately. Refer to the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal) for installation guides.

>   **Important**: Avoid using `sudo` or running as the `root` user. LocalStack must be installed and executed under a local, non-root user account. If you encounter permission problems on macOS High Sierra, use `pip install --user localstack` for installation.

## Quickstart

Launch LocalStack within a Docker container using:

```bash
 % localstack start -d

     __                     _______ __             __
    / /   ____  _________ _/ / ___// /_____ ______/ /__
   / /   / __ \/ ___/ __ `/ /\__ \/ __/ __ `/ ___/ //_/
  / /___/ /_/ / /__/ /_/ / /___/ / /_/ /_/ / /__/ ,<
 /_____/\____/\___/\__,_/_//____/\__/\__,_/\___/_/|_|

- LocalStack CLI: 4.6.0
- Profile: default
- App: https://app.localstack.cloud

[17:00:15] starting LocalStack in Docker mode 🐳               localstack.py:512
           preparing environment                               bootstrap.py:1322
           configuring container                               bootstrap.py:1330
           starting container                                  bootstrap.py:1340
[17:00:16] detaching                                           bootstrap.py:1344
```

Check the status of various LocalStack services by running:

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

To use SQS, a fully managed distributed message queuing service, on LocalStack, run:

```shell
% awslocal sqs create-queue --queue-name sample-queue
{
    "QueueUrl": "http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/sample-queue"
}
```

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and how to use them with LocalStack's `awslocal` CLI.

## Running

You have several options to run LocalStack:

-   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
-   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
-   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
-   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

To start using LocalStack, please refer to our [documentation](https://docs.localstack.cloud).

-   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
-   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
-   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
-   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
-   [Understanding LocalStack](https://docs.localstack.cloud/references/)
-   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

You can also utilize the following UI clients to interact with LocalStack:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

For the full list of changes in each release, please consult the [GitHub releases](https://github.com/localstack/localstack/releases). Extended release notes are available in the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

If you want to contribute to LocalStack:

-   Begin by reading our [contributing guide](docs/CONTRIBUTING.md).
-   Review our [development environment setup guide](docs/development-environment-setup/README.md).
-   Explore our codebase and [open issues](https://github.com/localstack/localstack/issues).

Your contributions and feedback are greatly appreciated.

## Get in touch

Reach out to the LocalStack Team to
report 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose),
upvote 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+),
🙋🏽 ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/),
or 🗣️ discuss local cloud development:

-   [LocalStack Slack Community](https://localstack.cloud/contact/)
-   [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

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

*   **Clear Headline & Summary:**  Uses a strong, SEO-friendly headline and a concise, keyword-rich introductory sentence.
*   **Keyword Optimization:**  Incorporates important keywords like "AWS," "local," "development," "testing," and specific AWS service names.
*   **Structured Content:**  Uses clear headings, subheadings, and bullet points for readability and SEO benefits.
*   **Internal Linking:**  Includes links to key sections and pages within the documentation.
*   **External Linking:**  Maintains the original links to the project's resources.
*   **Conciseness:** Streamlined installation instructions.
*   **Call to Action:**  Directly links to the GitHub repository for user interaction.
*   **Accessibility:** Clean formatting and appropriate use of headings for screen readers and better SEO.
*   **Focus on Value Proposition:** Highlights the core benefits of using LocalStack.