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

## LocalStack: Develop and Test AWS Applications Locally

**LocalStack is a powerful cloud service emulator that lets you develop and test your AWS applications locally, dramatically speeding up your development cycle.** ([Back to the top](#overview))

### Key Features:

*   **Local Cloud Development:** Run your AWS applications, including Lambdas, S3, DynamoDB, and more, entirely on your local machine.
*   **Accelerated Testing:** Speed up your testing and development workflow by eliminating the need to connect to a remote cloud provider.
*   **Comprehensive AWS Service Support:** Supports a growing number of AWS services such as Lambda, S3, DynamoDB, Kinesis, SQS, and SNS.  [See Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) for details.
*   **Easy Integration:** Works seamlessly with various tools and frameworks, including CDK and Terraform.
*   **CLI and GUI Support:** Manage and interact with LocalStack using a command-line interface, web UI, and desktop applications.

---

## Overview

[LocalStack](https://localstack.cloud) enables you to emulate a cloud environment on your local machine. This allows developers to build, test, and debug AWS applications without needing to connect to the actual cloud. It runs in a single container, simplifying the development and CI/CD processes. Whether you're building CDK applications, Terraform configurations, or just starting with AWS, LocalStack streamlines your workflow.

LocalStack supports a wide array of AWS services, with the [Pro version](https://localstack.cloud/pricing) offering even more features.  A comprehensive list of supported APIs is available on the [Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) page. Explore [User Guides](https://docs.localstack.cloud/user-guide/) for additional features and use cases.

## Install

The LocalStack CLI is the easiest way to get started. Make sure Docker is installed.

### Brew (macOS or Linux with Homebrew)

```bash
brew install localstack/tap/localstack-cli
```

### Binary Download (macOS, Linux, Windows)

-   Download the latest release from [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest).
-   Extract the archive to a directory in your `PATH`.

### PyPI (macOS, Linux, Windows)

```bash
python3 -m pip install localstack
```

After installing `localstack-cli`, install the `awslocal` CLI separately to interact with the emulated AWS services, following the instructions in the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

> **Important:**  Install LocalStack as a local non-root user to avoid permission issues.

## Quickstart

Get up and running quickly with LocalStack.

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

Choose your preferred method to run LocalStack:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

For detailed instructions, see the [documentation](https://docs.localstack.cloud).

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

Use these UI clients for a graphical interface:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

Find a complete list of changes in each release on [GitHub releases](https://github.com/localstack/localstack/releases) and extended release notes in the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

Contribute to LocalStack:

*   Read the [contributing guide](docs/CONTRIBUTING.md).
*   Set up your [development environment](docs/development-environment-setup/README.md).
*   Explore the codebase and [open issues](https://github.com/localstack/localstack/issues).

Contributions and feedback are welcome!

## Get in Touch

Connect with the LocalStack Team:

*   Report 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose)
*   Upvote 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   Ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/)
*   Discuss local cloud development:
    *   [LocalStack Slack Community](https://localstack.cloud/contact/)
    *   [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

### Contributors

Thank you to all contributors!

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

Support the project by becoming a backer on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

Become a sponsor on [Open Collective](https://opencollective.com/localstack#sponsor) and get your logo displayed here!

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