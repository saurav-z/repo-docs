# LocalStack: Your Local AWS Cloud - Develop & Test AWS Applications Offline

LocalStack ([GitHub Repo](https://github.com/localstack/localstack)) empowers developers to build and test AWS applications locally, without the need for a remote cloud connection.

**Key Features:**

*   **Local AWS Development:** Run and test your AWS applications and Lambdas entirely on your local machine.
*   **Comprehensive AWS Service Support:**  Supports a growing list of AWS services, including Lambda, S3, DynamoDB, Kinesis, SQS, and SNS.
*   **Simplified Testing and Development:** Speeds up and simplifies your testing and development workflows.
*   **Multiple Installation Options:**  Install via LocalStack CLI, Docker, Docker Compose, Helm, or PyPI.
*   **Pro Version:** Offers additional APIs and advanced features for enhanced functionality.
*   **User-Friendly GUI Options:**  Interact with LocalStack via the Web Application, Desktop App, or Docker Extension.

---

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
*   [Get in Touch](#get-in-touch)
    *   [Contributors](#contributors)
    *   [Backers](#backers)
    *   [Sponsors](#sponsors)
*   [License](#license)

---

## Overview

LocalStack ([https://localstack.cloud](https://localstack.cloud)) is a cloud service emulator that runs in a single container, enabling local development and testing of AWS applications. This allows you to test complex CDK applications, Terraform configurations, and learn AWS services without incurring costs or relying on internet connectivity.  The [Pro version of LocalStack](https://localstack.cloud/pricing) offers more features. See the [Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) page.

## Install

Get started quickly using the LocalStack CLI, ensuring you have a functional [`docker` environment](https://docs.docker.com/get-docker/) installed.

### Brew (macOS or Linux with Homebrew)

Install the LocalStack CLI:

```bash
brew install localstack/tap/localstack-cli
```

### Binary Download (macOS, Linux, Windows)

Download the latest release from the [localstack/localstack-cli releases](https://github.com/localstack/localstack-cli/releases/latest) page. Extract the archive and move the binary to a directory in your `PATH`.  For example:

```bash
sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin  # macOS/Linux
```

### PyPI (macOS, Linux, Windows)

Install using `pip`:

```bash
python3 -m pip install localstack
```

Install the `awslocal` CLI separately for interacting with local AWS services. Refer to the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

> **Important**: Avoid using `sudo` or running as `root`. Install and start LocalStack under a local, non-root user. If you have issues on macOS High Sierra, use `pip install --user localstack`.

## Quickstart

Start LocalStack in a Docker container:

```bash
% localstack start -d
```

Check the status of services:

```bash
% localstack status services
```

Create an SQS queue:

```shell
% awslocal sqs create-queue --queue-name sample-queue
```

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and using them with `awslocal`.

## Running

Run LocalStack using:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Explore the [documentation](https://docs.localstack.cloud) for detailed guidance:

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

Use GUI clients for interaction:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

Review the [GitHub releases](https://github.com/localstack/localstack/releases) and [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

Contribute by following the [contributing guide](docs/CONTRIBUTING.md) and the [development environment setup guide](docs/development-environment-setup/README.md).  Explore the codebase and [open issues](https://github.com/localstack/localstack/issues).

## Get in Touch

Contact the LocalStack team to report issues, request features, ask support questions, and discuss development:

*   [LocalStack Slack Community](https://localstack.cloud/contact/)
*   [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

### Contributors

Thanks to all contributors:

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

Released under the Apache License, Version 2.0 ([LICENSE](LICENSE.txt)), and the [End-User License Agreement (EULA)](docs/end_user_license_agreement).