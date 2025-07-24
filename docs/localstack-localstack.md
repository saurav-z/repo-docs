<!-- SEO-optimized README for LocalStack -->
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

LocalStack is a powerful cloud service emulator that enables you to develop and test your AWS applications locally, without the need for a remote cloud provider.  [See the original repo](https://github.com/localstack/localstack) for more details.

**Key Features:**

*   **Local AWS Development:**  Run AWS applications and Lambda functions entirely on your local machine.
*   **Comprehensive Service Support:** Emulates a wide range of AWS services including Lambda, S3, DynamoDB, SQS, SNS, and many more.
*   **Faster Development Cycle:**  Accelerate your testing and development workflow by eliminating the need to deploy to the cloud for every iteration.
*   **Integration with Testing Tools:** Seamlessly integrate with popular testing frameworks and CI/CD pipelines.
*   **Pro Version:** Access additional APIs and advanced features with the [LocalStack Pro](https://localstack.cloud/pricing) version.

---

## Table of Contents

*   [Overview](#overview)
*   [Install](#install)
    *   [Brew (macOS or Linux with Homebrew)](#brew-macos-or-linux-with-homebrew)
    *   [Binary download (macOS, Linux, Windows)](#binary-download-macos-linux-windows)
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

---

## Overview

LocalStack ([https://localstack.cloud](https://localstack.cloud)) is a cloud service emulator that runs in a single container, allowing you to test your AWS applications locally.  It simplifies testing and development by allowing you to execute AWS applications and infrastructure code on your local machine.  This eliminates the need for remote cloud connections, speeding up development and reducing costs. LocalStack supports many AWS services such as AWS Lambda, S3, DynamoDB, Kinesis, SQS, SNS, and more. The [Pro version](https://localstack.cloud/pricing) offers expanded functionality. For a complete list of supported APIs, see the [Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) page. Explore LocalStack's [User Guides](https://docs.localstack.cloud/user-guide/) to learn more.

## Install

Get started with LocalStack using the LocalStack CLI, Docker, or other methods. Before proceeding, ensure you have a working [`docker` environment](https://docs.docker.com/get-docker/) installed.

### Brew (macOS or Linux with Homebrew)

Install the LocalStack CLI using the official LocalStack Brew Tap:

```bash
brew install localstack/tap/localstack-cli
```

### Binary download (macOS, Linux, Windows)

Download the pre-built LocalStack CLI binary:

*   Visit [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the latest release for your platform.
*   Extract the archive to a directory included in your `PATH` variable, e.g., for macOS/Linux: `sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin`

### PyPI (macOS, Linux, Windows)

Install the LocalStack CLI using `pip`:

```bash
python3 -m pip install localstack
```

After installing the `localstack-cli`, install the `awslocal` CLI to interact with the local AWS services.  See the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal) for installation instructions.

> **Important**: Avoid using `sudo` or running as `root` for LocalStack installation and startup. LocalStack should be managed under a local non-root user. If permission issues occur on macOS High Sierra, try `pip install --user localstack`.

## Quickstart

Start LocalStack in a Docker container:

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

Use SQS:

```shell
% awslocal sqs create-queue --queue-name sample-queue
{
    "QueueUrl": "http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/sample-queue"
}
```

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and using them with the `awslocal` CLI.

## Running

Run LocalStack using:

-   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
-   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
-   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
-   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Explore the [documentation](https://docs.localstack.cloud) for detailed usage:

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

Use a UI client with:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

See the [GitHub releases](https://github.com/localstack/localstack/releases) for complete release details and the [changelog](https://docs.localstack.cloud/references/changelog/) for extended release notes.

## Contributing

Contribute to LocalStack by:

-   Reading the [contributing guide](docs/CONTRIBUTING.md).
-   Reviewing the [development environment setup guide](docs/development-environment-setup/README.md).
-   Exploring the codebase and [opening issues](https://github.com/localstack/localstack/issues).

We appreciate all contributions and feedback.

## Get in touch

Contact the LocalStack Team to:
report 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose),
upvote 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+),
🙋🏽 ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/),
or 🗣️ discuss local cloud development:

- [LocalStack Slack Community](https://localstack.cloud/contact/)
- [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

### Contributors

Thank you to everyone who has contributed to the project.

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

We are grateful for our backers who have donated.  Become a backer on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

Support the project by becoming a sponsor on [Open Collective](https://opencollective.com/localstack#sponsor). Your logo will appear here with a link.

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

Released under the Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)). By using this software you agree to the [End-User License Agreement (EULA)](docs/end_user_license_agreement).
```
Key improvements and SEO considerations:

*   **Concise Hook:** The one-sentence hook immediately highlights the core benefit.
*   **Keywords:** Uses relevant keywords like "AWS applications," "local development," "cloud service emulator," and AWS service names throughout the text.
*   **Clear Structure:** Uses headings and subheadings for better readability and SEO ranking.
*   **Bulleted Features:**  Provides a quick overview of key features, improving readability and SEO.
*   **Internal Linking:** Includes links to other sections within the README.
*   **External Linking:** Provides clear and concise links to external resources, improving the user experience and SEO.
*   **Simplified Installation:** Clear and concise installation instructions for different methods.
*   **Table of Contents:** Added for improved navigation and SEO.
*   **Alt Text:** Correctly utilizes `alt` text for images.
*   **Concise and Focused:** The description avoids unnecessary jargon and focuses on the value proposition.
*   **SEO Best Practices:**  The document is structured in a way that is friendly to search engines, increasing the chances of it being found in search results.