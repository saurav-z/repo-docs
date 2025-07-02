<!-- SEO-optimized README for LocalStack -->

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


# LocalStack: Your Local AWS Cloud for Development and Testing

**LocalStack is a cloud service emulator that allows you to develop and test your AWS applications locally, without requiring a connection to the real AWS cloud.**

[See the original repo](https://github.com/localstack/localstack)

## Key Features

*   **Local AWS Development:** Develop and test AWS applications locally using a wide range of AWS services.
*   **Comprehensive AWS Service Support:** Supports services like Lambda, S3, DynamoDB, Kinesis, SQS, SNS, and many more.
*   **Faster Development Cycles:** Speed up testing and development workflows by eliminating the need to deploy to the cloud for every change.
*   **Cost-Effective Testing:** Avoid incurring costs associated with using real cloud resources during development and testing.
*   **CI/CD Integration:** Easily integrate LocalStack into your CI/CD pipelines for automated testing.
*   **Multiple Installation Options:** Supports installation via CLI, Docker, Docker Compose, and Helm.

## Getting Started

### Install

Choose your preferred installation method:

*   **LocalStack CLI (Recommended):**
    *   **Brew (macOS/Linux):** `brew install localstack/tap/localstack-cli`
    *   **Binary Download (macOS, Linux, Windows):** Download the latest CLI binary from [localstack/localstack-cli releases](https://github.com/localstack/localstack-cli/releases/latest).
    *   **PyPI (macOS, Linux, Windows):** `python3 -m pip install localstack`

*   **Docker:**  See the [Docker documentation](https://docs.localstack.cloud/getting-started/installation/#docker) for detailed instructions.

*   **Docker Compose:**  See the [Docker Compose documentation](https://docs.localstack.cloud/getting-started/installation/#docker-compose) for detailed instructions.

*   **Helm:**  See the [Helm documentation](https://docs.localstack.cloud/getting-started/installation/#helm) for detailed instructions.

Remember to install `awslocal` CLI separately to interact with local AWS services as outlined in the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

### Quickstart

1.  **Start LocalStack:**
    ```bash
    localstack start -d
    ```

2.  **Check Service Status:**
    ```bash
    localstack status services
    ```

3.  **Use an AWS Service (SQS Example):**
    ```shell
    awslocal sqs create-queue --queue-name sample-queue
    ```

For further guidance, refer to the [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) documentation.

## Run

Run LocalStack using your preferred method:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage & Documentation

Explore the full power of LocalStack with these resources:

*   **Documentation:**  [https://docs.localstack.cloud](https://docs.localstack.cloud)
*   **Configuration:** [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   **CI/CD Integration:** [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   **Integrations:** [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   **Tools:** [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   **Understanding LocalStack:** [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   **FAQ:** [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

### UI Clients
Explore LocalStack through user interfaces:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

View all the changes for each release in the [GitHub releases](https://github.com/localstack/localstack/releases). Refer to the [changelog](https://docs.localstack.cloud/references/changelog/) for more details.

## Contribute

We welcome contributions!

*   Read our [contributing guide](docs/CONTRIBUTING.md).
*   Check out our [development environment setup guide](docs/development-environment-setup/README.md).
*   Browse our [open issues](https://github.com/localstack/localstack/issues).

## Get in Touch

Report issues, request features, ask support questions, or discuss local cloud development:

*   [LocalStack Slack Community](https://localstack.cloud/contact/)
*   [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

## Support

*   **[Docs](https://docs.localstack.cloud):** Comprehensive documentation.
*   **[LocalStack Pro](https://app.localstack.cloud):** Unlock extra features and functionality.
*   **[LocalStack Coverage](https://docs.localstack.cloud/references/coverage/):** Details on feature and API support.

## Contributors

Thank you to all our contributors!

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

## Backers

Support the project by becoming a backer on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

## Sponsors

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

Released under the Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)).  By using this software, you agree to the [End-User License Agreement (EULA)](docs/end_user_license_agreement).