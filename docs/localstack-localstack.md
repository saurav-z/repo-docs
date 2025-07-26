# LocalStack: Your Local Cloud Development Toolkit

LocalStack empowers developers to build, test, and deploy AWS applications locally, accelerating development cycles. [Explore the LocalStack Repo](https://github.com/localstack/localstack).

<p align="center">
  <img src="https://raw.githubusercontent.com/localstack/localstack/main/docs/localstack-readme-banner.svg" alt="LocalStack - A fully functional local cloud stack">
</p>

[![GitHub Actions](https://github.com/localstack/localstack/actions/workflows/aws-main.yml/badge.svg?branch=main)](https://github.com/localstack/localstack/actions/workflows/aws-main.yml?query=branch%3Amain)
[![Coverage Status](https://coveralls.io/repos/github/localstack/localstack/badge.svg?branch=main)](https://coveralls.io/github/localstack/localstack?branch=main)
[![PyPI Version](https://img.shields.io/pypi/v/localstack?color=blue)](https://pypi.org/project/localstack/)
[![Docker Pulls](https://img.shields.io/docker/pulls/localstack/localstack)](https://hub.docker.com/r/localstack/localstack)
[![PyPi downloads](https://static.pepy.tech/badge/localstack)](https://pypi.org/project/localstack)
[![Backers on Open Collective](https://opencollective.com/localstack/backers/badge.svg)](https://opencollective.com/localstack/backers)
[![Sponsors on Open Collective](https://opencollective.com/localstack/sponsors/badge.svg)](https://opencollective.com/localstack/sponsors)
[![PyPI License](https://img.shields.io/pypi/l/localstack.svg)](https://img.shields.io/pypi/l/localstack.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Bluesky](https://img.shields.io/badge/bluesky-Follow-blue?logo=bluesky)](https://bsky.app/profile/localstack.cloud)

---

## Key Features

*   **Local AWS Emulation:** Run and test your AWS applications locally without a cloud connection.
*   **Comprehensive Service Support:** Supports a wide array of AWS services like Lambda, S3, DynamoDB, and more.
*   **Accelerated Development:** Speeds up your development and testing workflow.
*   **Integration with Existing Tools:** Works seamlessly with your existing AWS CLI and SDKs.
*   **Docker-Based:** Easy to set up and run using Docker.
*   **Pro Version:** Explore advanced features and additional APIs with LocalStack Pro.

## Installation

Get started quickly with the LocalStack CLI:

### Brew (macOS or Linux with Homebrew)

```bash
brew install localstack/tap/localstack-cli
```

### Binary download (macOS, Linux, Windows)

1.  Visit [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the latest release for your platform.
2.  Extract the downloaded archive to a directory included in your `PATH` variable. For macOS/Linux, use: `sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin`

### PyPI (macOS, Linux, Windows)

```bash
python3 -m pip install localstack
```

Remember to install the `awslocal` CLI for interacting with local AWS services.  See the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal) for details.

> **Important**:  Install and run LocalStack under a local non-root user.  Avoid `sudo`.

## Quickstart

Start LocalStack in a Docker container:

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

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/).

## Running LocalStack

You can run LocalStack using:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage and Resources

*   [Documentation](https://docs.localstack.cloud)
    *   [Configuration](https://docs.localstack.cloud/references/configuration/)
    *   [CI](https://docs.localstack.cloud/user-guide/ci/)
    *   [Integrations](https://docs.localstack.cloud/user-guide/integrations/)
    *   [Tools](https://docs.localstack.cloud/user-guide/tools/)
    *   [References](https://docs.localstack.cloud/references/)
    *   [FAQ](https://docs.localstack.cloud/getting-started/faq/)
*   **UI Clients:**
    *   [LocalStack Web Application](https://app.localstack.cloud)
    *   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
    *   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

See the [GitHub releases](https://github.com/localstack/localstack/releases) for detailed release notes and the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

Contribute to the project:

*   [Contributing Guide](docs/CONTRIBUTING.md)
*   [Development Environment Setup](docs/development-environment-setup/README.md)
*   [Open Issues](https://github.com/localstack/localstack/issues)

## Get in Touch

*   Report [issues](https://github.com/localstack/localstack/issues/new/choose).
*   Upvote [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+).
*   Ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/).
*   Discuss local cloud development in the [Slack Community](https://localstack.cloud/contact/) and the [GitHub Issue tracker](https://github.com/localstack/localstack/issues).

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

Apache License, Version 2.0 ([LICENSE](LICENSE.txt)) - see the [End-User License Agreement (EULA)](docs/end_user_license_agreement).