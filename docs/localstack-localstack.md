# LocalStack: Develop and Test AWS Applications Locally

**LocalStack empowers developers to build, test, and run AWS applications locally without the need for a remote cloud provider.**

[![GitHub Actions](https://github.com/localstack/localstack/actions/workflows/aws-main.yml/badge.svg?branch=master)](https://github.com/localstack/localstack/actions/workflows/aws-main.yml?query=branch%3Amaster)
[![Coverage Status](https://coveralls.io/repos/github/localstack/localstack/badge.svg?branch=master)](https://coveralls.io/github/localstack/localstack?branch=master)
[![PyPI Version](https://img.shields.io/pypi/v/localstack?color=blue)](https://pypi.org/project/localstack/)
[![Docker Pulls](https://img.shields.io/docker/pulls/localstack/localstack)](https://hub.docker.com/r/localstack/localstack)
[![PyPi downloads](https://static.pepy.tech/badge/localstack)](https://pypi.org/project/localstack)
[![Backers on Open Collective](https://opencollective.com/localstack/backers/badge.svg)](https://opencollective.com/localstack#backers)
[![Sponsors on Open Collective](https://opencollective.com/localstack/sponsors/badge.svg)](https://opencollective.com/localstack#sponsors)
[![PyPI License](https://img.shields.io/pypi/l/localstack.svg)](https://img.shields.io/pypi/l/localstack.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/localstack)

---

## Key Features

*   **Local AWS Cloud Emulation:** Replicates a wide range of AWS services for local development and testing.
*   **Faster Development Cycles:** Test your applications locally, eliminating the need to deploy to the cloud for every iteration.
*   **Cost Savings:** Avoid incurring cloud costs during development and testing.
*   **CI/CD Integration:** Easily integrate LocalStack into your CI/CD pipelines for automated testing.
*   **Comprehensive Service Support:** Supports popular AWS services like Lambda, S3, DynamoDB, SQS, SNS, and more.
*   **Pro Version:** Access additional APIs and advanced features with LocalStack Pro ([https://localstack.cloud/pricing](https://localstack.cloud/pricing)).

## Overview

[LocalStack](https://localstack.cloud) is a powerful cloud service emulator that allows developers to develop and test AWS applications locally. It runs within a single container, enabling you to simulate a complete AWS environment on your laptop or in your CI/CD environment. This allows you to test everything from complex CDK applications and Terraform configurations to getting started with AWS services - all locally.  This improves developer productivity and accelerates your development workflow.

LocalStack supports a growing list of AWS services, enabling you to test your applications without a live internet connection.  Check out the [feature coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) to see supported APIs.

## Installation

Get started by installing the LocalStack CLI. Make sure you have a functional Docker environment.

### Brew (macOS or Linux with Homebrew)

```bash
brew install localstack/tap/localstack-cli
```

### Binary download (macOS, Linux, Windows)

1.  Visit [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the latest release.
2.  Extract the archive to a directory in your `PATH`.

### PyPI (macOS, Linux, Windows)

```bash
python3 -m pip install localstack
```

After installing the LocalStack CLI, install the `awslocal` CLI separately.  See the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal) for installation instructions.

**Important:** Do not use `sudo` or run as `root`.

## Quickstart

Run LocalStack in a Docker container:

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

## Running

Run LocalStack using these methods:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Explore the [documentation](https://docs.localstack.cloud) to learn how to use LocalStack:

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

View all changes in our [GitHub releases](https://github.com/localstack/localstack/releases). For extended release notes, see the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

Contribute to LocalStack:

*   Read our [contributing guide](docs/CONTRIBUTING.md).
*   See the [development environment setup guide](docs/development-environment-setup/README.md).
*   Review our [open issues](https://github.com/localstack/localstack/issues).

## Get in Touch

*   Report 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose)
*   Upvote 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   Ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/)
*   Discuss local cloud development on the [LocalStack Slack Community](https://localstack.cloud/contact/) or the [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues).

### Contributors

We appreciate all contributors.

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

Become a backer on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

Support LocalStack by becoming a sponsor on [Open Collective](https://opencollective.com/localstack#sponsor).

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

Released under the Apache License, Version 2.0 ([LICENSE](LICENSE.txt)).  By downloading and using this software you agree to the [End-User License Agreement (EULA)](docs/end_user_license_agreement).

---
[Back to top](#localstack-develop-and-test-aws-applications-locally) ([https://github.com/localstack/localstack](https://github.com/localstack/localstack))
```
Key changes and improvements:

*   **SEO Optimization:** Added a descriptive one-sentence hook and improved the title for better search engine visibility.  Used keywords like "AWS", "local development", "testing", and "cloud emulator".
*   **Clear Structure:**  Organized the content with clear headings and subheadings for better readability.
*   **Bulleted Key Features:**  Used bullet points to highlight the main features of LocalStack, making it easier for users to grasp the core benefits.
*   **Concise Summaries:** Provided concise summaries of each section to keep the information focused and easy to scan.
*   **Improved Installation Instructions:**  Clarified installation steps and included important notes like not running as root.
*   **Simplified Quickstart:**  Streamlined the Quickstart section with essential commands.
*   **Direct Links:**  Provided direct links to the original repository, documentation, and other important resources.
*   **Back to Top Link:**  Added a "Back to Top" link and a link to the original repo.
*   **Removed unnecessary HTML:** Replaced `<p>` tags with Markdown-friendly formatting.