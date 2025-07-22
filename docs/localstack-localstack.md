# LocalStack: Develop and Test AWS Applications Locally (with LocalStack 4.6!)

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

LocalStack empowers developers to build, test, and experiment with AWS applications locally, without the need for a live cloud connection. **[Check out the latest release of LocalStack 4.6!](https://blog.localstack.cloud/localstack-for-aws-release-v-4-6-0/)**

**Key Features:**

*   **Local AWS Cloud Emulation:** Run a wide range of AWS services (Lambda, S3, DynamoDB, etc.) on your local machine.
*   **Accelerate Development:** Test and debug your applications faster and more efficiently.
*   **Cost Savings:** Eliminate costs associated with using real cloud resources during development and testing.
*   **Offline Development:** Develop and test AWS applications even without an internet connection.
*   **CI/CD Integration:** Seamlessly integrate LocalStack into your CI/CD pipelines for automated testing.
*   **Comprehensive Service Support:** Supports a growing list of AWS services, with more available in [LocalStack Pro](https://localstack.cloud/pricing).

**Get Started:**

*   [**Overview**](#overview)
*   [**Install**](#install)
*   [**Quickstart**](#quickstart)
*   [**Running**](#running)
*   [**Usage**](#usage)
*   [**Releases**](#releases)
*   [**Contributing**](#contributing)

---

## Overview

[LocalStack](https://localstack.cloud) is a powerful cloud service emulator that runs in a single container, allowing you to develop and test your AWS applications and infrastructure locally.  Whether you're building serverless applications, managing infrastructure with tools like Terraform, or simply learning about AWS, LocalStack simplifies your development workflow.

LocalStack supports a broad spectrum of AWS services, including popular options like AWS Lambda, S3, DynamoDB, Kinesis, SQS, and SNS.  For a comprehensive list of supported APIs and features, visit the [Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) page.  The [Pro version of LocalStack](https://localstack.cloud/pricing) unlocks additional APIs and advanced features.

Explore LocalStack's [User Guides](https://docs.localstack.cloud/user-guide/) for more in-depth information.

## Install

The simplest way to get started with LocalStack is using the LocalStack CLI. Before you begin, make sure you have a working [`docker` environment](https://docs.docker.com/get-docker/) installed.

### Brew (macOS or Linux with Homebrew)

Install the LocalStack CLI through our [official LocalStack Brew Tap](https://github.com/localstack/homebrew-tap):

```bash
brew install localstack/tap/localstack-cli
```

### Binary download (macOS, Linux, Windows)

If you don't have Brew, download the pre-built LocalStack CLI binary:

-   Visit [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the release for your OS.
-   Extract the archive to a directory that is in your `PATH`:
    -   For macOS/Linux, use: `sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin`

### PyPI (macOS, Linux, Windows)

LocalStack is developed with Python, so install the CLI with `pip`:

```bash
python3 -m pip install localstack
```

After installing the `localstack-cli`, you'll run the Docker image containing the LocalStack runtime. To interact with local AWS services, separately install the `awslocal` CLI. See the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal) for install instructions.

> **Important**: Avoid `sudo` or running as `root`. LocalStack needs to be installed and run as a local, non-root user. For macOS High Sierra, use `pip install --user localstack`.

## Quickstart

Start LocalStack in a Docker container:

```bash
% localstack start -d
```

Check the status of services with:

```bash
% localstack status services
```

Use SQS with:

```shell
% awslocal sqs create-queue --queue-name sample-queue
```

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and the `awslocal` CLI.

## Running

You can run LocalStack using these methods:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

See our [documentation](https://docs.localstack.cloud) for details.

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

Use these UIs with LocalStack:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

See the complete release list on [GitHub releases](https://github.com/localstack/localstack/releases) and the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

*   Read our [contributing guide](docs/CONTRIBUTING.md).
*   Set up your [development environment](docs/development-environment-setup/README.md).
*   View and [open issues](https://github.com/localstack/localstack/issues).

## Get in touch

*   Report 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose)
*   Upvote 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   Ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/)
*   Discuss local cloud development in our [Slack Community](https://localstack.cloud/contact/) or the [GitHub Issue tracker](https://github.com/localstack/localstack/issues).

### Contributors

We are thankful to all contributors.

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

Thank you to our backers on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

Support this project by becoming a sponsor on [Open Collective](https://opencollective.com/localstack#sponsor).

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

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)).  See the [End-User License Agreement (EULA)](docs/end_user_license_agreement) for details.