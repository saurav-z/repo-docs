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

# LocalStack: Develop and Test AWS Applications Locally

LocalStack empowers developers to build and test AWS applications locally, providing a fully functional cloud environment on your machine.

**[View the LocalStack GitHub Repository](https://github.com/localstack/localstack)**

**Key Features:**

*   **Local AWS Environment:** Run and test your AWS applications locally without needing a remote cloud provider.
*   **Comprehensive AWS Service Support:** Supports a wide range of AWS services including Lambda, S3, DynamoDB, Kinesis, SQS, and SNS, with more being added constantly.
*   **Simplified Testing & Development:** Speeds up your workflow by allowing you to test complex applications, CDK projects, and Terraform configurations locally.
*   **CLI & Docker Integration:** Offers an easy-to-use CLI for managing the LocalStack Docker container.
*   **Pro Version:** Provides advanced features and additional API support through the [Pro version of LocalStack](https://localstack.cloud/pricing).

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

LocalStack is a powerful cloud service emulator that runs in a single container, enabling you to develop and test your AWS applications locally.  It supports a growing number of AWS services, allowing you to test your applications locally.

## Install

Get started with LocalStack using the CLI. Ensure you have a Docker environment installed.

### Brew (macOS or Linux with Homebrew)

```bash
brew install localstack/tap/localstack-cli
```

### Binary Download (macOS, Linux, Windows)

1.  Download the latest release from the [localstack/localstack-cli releases](https://github.com/localstack/localstack-cli/releases/latest) page for your platform.
2.  Extract the archive to a directory in your `PATH`. For example:

```bash
sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin  # macOS/Linux
```

### PyPI (macOS, Linux, Windows)

```bash
python3 -m pip install localstack
```

Install `awslocal` separately for interacting with local AWS services; refer to the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

> **Important**: Avoid using `sudo` or running as `root`. Install and start LocalStack under a local, non-root user. If experiencing permission issues on macOS High Sierra, install with `pip install --user localstack`.

## Quickstart

Start LocalStack in a Docker container:

```bash
% localstack start -d
```

Check service status:

```bash
% localstack status services
```

Use SQS:

```bash
% awslocal sqs create-queue --queue-name sample-queue
```

Explore [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and the `awslocal` CLI.

## Running

Run LocalStack using:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Refer to the [LocalStack documentation](https://docs.localstack.cloud) for detailed usage information:

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

See the [GitHub releases](https://github.com/localstack/localstack/releases) for release details and the [changelog](https://docs.localstack.cloud/references/changelog/) for extended notes.

## Contributing

Contribute to LocalStack:

*   Read the [contributing guide](docs/CONTRIBUTING.md).
*   Review the [development environment setup guide](docs/development-environment-setup/README.md).
*   Explore the codebase and open [issues](https://github.com/localstack/localstack/issues).

## Get in touch

Contact the LocalStack Team:

*   Report [issues](https://github.com/localstack/localstack/issues/new/choose).
*   Upvote [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+).
*   Ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/).
*   Discuss local cloud development.

*   [LocalStack Slack Community](https://localstack.cloud/contact/)
*   [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

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

Copyright (c) 2017-2025 LocalStack maintainers and contributors.

Copyright (c) 2016 Atlassian and others.

This version of LocalStack is released under the Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)). By downloading and using this software you agree to the [End-User License Agreement (EULA)](docs/end_user_license_agreement).