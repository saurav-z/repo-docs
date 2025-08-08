<!-- Improved README with SEO Optimization -->

<p align="center">
  <img src="https://raw.githubusercontent.com/localstack/localstack/main/docs/localstack-readme-banner.svg" alt="LocalStack - A fully functional local cloud stack">
</p>

<h1 align="center">LocalStack: Develop and Test AWS Applications Locally</h1>

<p align="center"><b>LocalStack empowers developers to build, test, and deploy AWS applications offline, accelerating development workflows.</b></p>

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

[LocalStack](https://github.com/localstack/localstack) is a cloud service emulator that runs in a single container on your laptop or in your CI environment.

---

## Key Features

*   **Local AWS Development:** Develop and test your AWS applications locally without the need for a remote cloud provider.
*   **Comprehensive AWS Service Support:** Supports a growing number of AWS services including Lambda, S3, DynamoDB, Kinesis, SQS, and SNS.
*   **Faster Development Cycles:** Speed up testing and development workflows, allowing you to iterate quickly.
*   **CI/CD Integration:** Seamlessly integrate LocalStack into your CI/CD pipelines for automated testing.
*   **Cost-Effective:** Reduce cloud costs by developing and testing locally.
*   **CLI and UI Tools:** Offers a CLI and UI for easy management and interaction with local AWS services.
*   **Pro Version:** Access additional APIs and features with the [Pro version](https://localstack.cloud/pricing).

## Installation

Get started with LocalStack using the following methods:

### LocalStack CLI

The quickest way to get started. Ensure you have a functional [`docker` environment](https://docs.docker.com/get-docker/) installed.

#### Brew (macOS or Linux with Homebrew)

```bash
brew install localstack/tap/localstack-cli
```

#### Binary Download (macOS, Linux, Windows)

- Visit [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the latest release for your platform.
- Extract the downloaded archive to a directory included in your `PATH` variable.

#### PyPI (macOS, Linux, Windows)

```bash
python3 -m pip install localstack
```

*For `awslocal` CLI installation guidelines, refer to the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).*

> **Important:** Install and start LocalStack as a local non-root user. If you have permissions issues in macOS High Sierra, install with `pip install --user localstack`.

## Quickstart

1.  **Start LocalStack:**

```bash
% localstack start -d
```

2.  **Check Service Status:**

```bash
% localstack status services
```

3.  **Use SQS Example:**

```shell
% awslocal sqs create-queue --queue-name sample-queue
```

Explore more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and `awslocal` CLI usage.

## Running Options

Choose how you want to run LocalStack:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage and Resources

Explore the following resources to get the most out of LocalStack:

*   [Documentation](https://docs.localstack.cloud)
*   [Configuration](https://docs.localstack.cloud/references/configuration/)
*   [CI Integration](https://docs.localstack.cloud/user-guide/ci/)
*   [Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [FAQ](https://docs.localstack.cloud/getting-started/faq/)

### UI Clients

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

See the complete list of changes in each release on [GitHub releases](https://github.com/localstack/localstack/releases).  For extended release notes, refer to the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

Interested in contributing?

*   Read our [contributing guide](docs/CONTRIBUTING.md).
*   Check out our [development environment setup guide](docs/development-environment-setup/README.md).
*   Browse the [open issues](https://github.com/localstack/localstack/issues).

## Get in Touch

*   Report 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose)
*   Upvote 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   Ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/)
*   Discuss local cloud development in the [LocalStack Slack Community](https://localstack.cloud/contact/) or [GitHub Issue tracker](https://github.com/localstack/localstack/issues)

### Contributors

We are thankful to all the people who have contributed to this project.

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

Support the project on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

Become a sponsor on [Open Collective](https://opencollective.com/localstack#sponsor).

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

Released under the Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)). By using this software, you agree to the [End-User License Agreement (EULA)](docs/end_user_license_agreement).