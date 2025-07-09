# LocalStack: Your Local AWS Cloud Development & Testing Solution

LocalStack empowers developers to build, test, and deploy AWS applications entirely on their local machine, accelerating development cycles.  [Explore the LocalStack Repository](https://github.com/localstack/localstack)

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

---

## Key Features:

*   **Local AWS Cloud Emulation:** Run a wide range of AWS services locally.
*   **Faster Development:** Accelerate your development and testing workflows.
*   **Cost-Effective Testing:** Eliminate the need for expensive cloud resources during development.
*   **Comprehensive AWS Support:** Supports services like Lambda, S3, DynamoDB, SQS, SNS, and more.
*   **CI/CD Integration:** Seamlessly integrate LocalStack into your CI/CD pipelines for automated testing.
*   **Multiple Installation Options:** Install via CLI, Docker, Docker Compose, or Helm.
*   **Pro Version:** Offers advanced features and extended API support.
*   **Active Community:** Benefit from a vibrant community and extensive documentation.

## Overview

LocalStack is a powerful cloud service emulator designed to run in a single container, allowing you to develop and test your AWS applications locally.  This means you can develop, test, and debug your applications without connecting to the real AWS cloud, saving time and money.  Whether you're testing CDK applications, Terraform configurations, or just starting your AWS journey, LocalStack streamlines your workflow.

LocalStack supports a growing number of popular AWS services including AWS Lambda, S3, DynamoDB, Kinesis, SQS, SNS, and many more!  The [Pro version of LocalStack](https://localstack.cloud/pricing) provides even more APIs and advanced features. Check out the [Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) page for a detailed list of supported APIs.

## Install

The easiest way to get started is with the LocalStack CLI:

### Brew (macOS or Linux)

```bash
brew install localstack/tap/localstack-cli
```

### Binary Download (macOS, Linux, Windows)

1.  Visit [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the correct binary.
2.  Extract the archive to a directory in your `PATH` (e.g., `/usr/local/bin`).

### PyPI (macOS, Linux, Windows)

```bash
python3 -m pip install localstack
```

**Important:** Avoid using `sudo` or running as `root`.  Install and run LocalStack with a local user.  If you experience permission issues on macOS High Sierra, use `pip install --user localstack`.

## Quickstart

Start LocalStack with:

```bash
% localstack start -d
```

Check service status:

```bash
% localstack status services
```

Use SQS (example):

```shell
% awslocal sqs create-queue --queue-name sample-queue
```

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and `awslocal` CLI.

## Running

LocalStack can be run using the following methods:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Explore these resources to effectively utilize LocalStack:

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

**Graphical User Interface Options:**

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

Review the complete list of changes in each release via [GitHub releases](https://github.com/localstack/localstack/releases) and the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

Contribute to LocalStack by:

*   Reading the [contributing guide](docs/CONTRIBUTING.md).
*   Setting up your [development environment](docs/development-environment-setup/README.md).
*   Exploring the codebase and opening [issues](https://github.com/localstack/localstack/issues).

## Get in Touch

Connect with the LocalStack team and community:

*   Report 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose)
*   Upvote 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   Ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/)
*   Join the conversation on [LocalStack Slack Community](https://localstack.cloud/contact/) or the [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues).

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

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)).

By downloading and using this software you agree to the [End-User License Agreement (EULA)](docs/end_user_license_agreement).