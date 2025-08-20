<!-- Improved README with SEO and Key Features -->
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

LocalStack provides a fully functional cloud service emulator, enabling developers to build, test, and debug AWS applications locally, accelerating development cycles.  [Visit the GitHub Repository](https://github.com/localstack/localstack).

## Key Features

*   **Local AWS Cloud Environment:** Run your AWS applications and Lambda functions entirely on your local machine.
*   **Comprehensive AWS Service Support:** Supports a wide array of AWS services, including Lambda, S3, DynamoDB, Kinesis, SQS, SNS, and more.
*   **Faster Development and Testing:** Speeds up testing of complex applications, CDK and Terraform configurations.
*   **Docker Integration:** Easy to use with Docker, Docker Compose, and Helm.
*   **CLI and UI Tools:** Supports CLI commands for service interactions and web UI clients for easier management and monitoring.
*   **CI/CD Integration:** Seamlessly integrate with your CI/CD pipelines for automated testing.
*   **Pro Version:** Offers additional APIs and advanced features for extended functionality.

## Overview

[LocalStack](https://localstack.cloud) emulates cloud services within a single container, letting you develop and test AWS applications locally. This allows you to build, test, and debug your applications without needing to connect to a remote cloud provider. Ideal for testing CDK applications, Terraform configurations, or anyone learning AWS services, LocalStack simplifies testing and development workflows.

LocalStack supports various AWS services like AWS Lambda, S3, DynamoDB, Kinesis, SQS, SNS, and more. The [Pro version](https://localstack.cloud/pricing) offers additional APIs and features.

## Installation

Get started quickly with the LocalStack CLI. Ensure you have Docker installed.

### Brew (macOS or Linux with Homebrew)

```bash
brew install localstack/tap/localstack-cli
```

### Binary Download (macOS, Linux, Windows)

1.  Download the latest release from [LocalStack CLI Releases](https://github.com/localstack/localstack-cli/releases/latest).
2.  Extract the archive to a directory in your `PATH`.

### PyPI (macOS, Linux, Windows)

```bash
python3 -m pip install localstack
```

Install the `awslocal` CLI separately. See [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

> **Important:** Install and run LocalStack under a local, non-root user.  Avoid `sudo` or running as root.  If you have macOS High Sierra permissions issues, use `pip install --user localstack`.

## Quickstart

Run LocalStack in a Docker container:

```bash
% localstack start -d

     __                     _______ __             __
    / /   ____  _________ _/ / ___// /_____ ______/ /__
   / /   / __ \/ ___/ __ `/ /\__ \/ __/ __ `/ ___/ //_/
  / /___/ /_/ / /__/ /_/ / /___/ / /_/ /_/ / /__/ ,<
 /_____/\____/\___/\__,_/_//____/\__/\__,_/\___/_/|_|

- LocalStack CLI: 4.7.0
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

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and using them with LocalStack's `awslocal` CLI.

## Running Options

Choose how to run LocalStack:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Explore the [documentation](https://docs.localstack.cloud) to get started.

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

Use GUI clients:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

See the [GitHub releases](https://github.com/localstack/localstack/releases) and the [changelog](https://docs.localstack.cloud/references/changelog/) for release details.

## Contributing

Contribute to LocalStack:

*   Read the [contributing guide](docs/CONTRIBUTING.md).
*   Set up your [development environment](docs/development-environment-setup/README.md).
*   Explore the codebase and check [open issues](https://github.com/localstack/localstack/issues).

## Get in Touch

Report issues, upvote feature requests, ask for support, or discuss cloud development:

*   [LocalStack Slack Community](https://localstack.cloud/contact/)
*   [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

### Contributors

Thanks to all contributors.

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

Support the project by becoming a backer on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

Sponsor the project on [Open Collective](https://opencollective.com/localstack#sponsor).

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

Released under the Apache License, Version 2.0 ([LICENSE](LICENSE.txt)). Agree to the [End-User License Agreement (EULA)](docs/end_user_license_agreement) by using this software.
```
Key improvements and SEO optimization:

*   **Clear, Concise Hook:** The one-sentence hook at the beginning clearly states the value proposition.
*   **Targeted Keywords:** Used relevant keywords throughout (e.g., "AWS applications", "local development", "testing", "Docker").
*   **Structured Headings:**  Using clear and descriptive headings (Overview, Installation, Quickstart, etc.) improves readability and SEO.
*   **Bulleted Key Features:**  Highlights the most important features in an easy-to-scan format, improving user understanding.
*   **Installation Steps:**  Provides clear, step-by-step installation instructions, important for user onboarding.
*   **Quickstart Example:**  Includes a useful quickstart guide, encouraging immediate use.
*   **Detailed Documentation Links:** Includes direct links to important documentation sections.
*   **Contact and Contribution Sections:**  Makes it easy for users to contribute and get support.
*   **SEO-Friendly Formatting:** Uses Markdown for proper heading structure, and emphasis.
*   **Clear and Concise Language:** Simplifies the language.
*   **Focus on Benefits:** The text emphasizes the benefits of using LocalStack (e.g., faster development, easier testing).
*   **Updated for Recent Release:** Mentions "Develop and Test AWS Applications Locally", which is more descriptive.
*   **Complete and Informative:** The README covers all essential information a user would need.