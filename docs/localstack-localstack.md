<!-- Improved README.md -->

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

---

# LocalStack: Your Local Cloud for AWS Development and Testing

**LocalStack empowers developers to build, test, and deploy AWS applications locally, without the need for a live cloud connection.**

## Key Features

*   **Comprehensive AWS Service Support:** Emulates a wide range of AWS services, including Lambda, S3, DynamoDB, SQS, SNS, and many more.
*   **Local Development and Testing:** Develop and test your AWS applications and infrastructure code entirely on your local machine.
*   **Faster Development Cycles:** Speed up your development process by eliminating the need to deploy to the cloud for every test.
*   **Cost Savings:** Reduce cloud costs by testing and debugging locally.
*   **CI/CD Integration:** Seamlessly integrate LocalStack into your CI/CD pipelines for automated testing.
*   **Pro Version:** Access additional APIs and advanced features with the [Pro version](https://localstack.cloud/pricing).

## Getting Started

### Installation

Choose your preferred installation method:

*   **LocalStack CLI (Recommended):**
    *   **Brew (macOS/Linux):**
        ```bash
        brew install localstack/tap/localstack-cli
        ```
    *   **Binary Download (macOS, Linux, Windows):** Download the CLI binary from the [releases page](https://github.com/localstack/localstack-cli/releases/latest) and add it to your PATH.
    *   **PyPI (macOS, Linux, Windows):**
        ```bash
        python3 -m pip install localstack
        ```
*   **Docker:** See [Docker Installation](https://docs.localstack.cloud/getting-started/installation/#docker) for details.
*   **Docker Compose:** See [Docker Compose Installation](https://docs.localstack.cloud/getting-started/installation/#docker-compose) for details.
*   **Helm:** See [Helm Installation](https://docs.localstack.cloud/getting-started/installation/#helm) for details.

**Note:** Install the `awslocal` CLI separately for interacting with local AWS services. See [awslocal documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

### Quickstart with LocalStack CLI

1.  **Start LocalStack:**
    ```bash
    localstack start -d
    ```
2.  **Check Service Status:**
    ```bash
    localstack status services
    ```
3.  **Use an AWS Service (SQS Example):**
    ```bash
    awslocal sqs create-queue --queue-name sample-queue
    ```

## Usage and Documentation

*   **Comprehensive Documentation:** Explore our detailed [documentation](https://docs.localstack.cloud) to learn how to configure, integrate, and use LocalStack effectively.
*   **Configuration:** [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   **CI/CD:** [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   **Integrations:** [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   **Tools:** [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   **Understanding LocalStack:** [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   **FAQ:** [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

## GUI Clients

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

Stay up-to-date with the latest changes and features.  See [GitHub releases](https://github.com/localstack/localstack/releases) and the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contribute

We welcome contributions from the community!

*   Read our [contributing guide](docs/CONTRIBUTING.md).
*   Set up your [development environment](docs/development-environment-setup/README.md).
*   Browse [open issues](https://github.com/localstack/localstack/issues) to find areas to contribute.

## Get in Touch

*   Report issues: [GitHub Issues](https://github.com/localstack/localstack/issues/new/choose)
*   Upvote feature requests: [GitHub Feature Requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   Ask support questions: [Help and Support](https://docs.localstack.cloud/getting-started/help-and-support/)
*   Discuss cloud development: [LocalStack Slack Community](https://localstack.cloud/contact/)
*   LocalStack GitHub Issue tracker: [GitHub Issues](https://github.com/localstack/localstack/issues)

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

[Back to top](#localstack-your-local-cloud-for-aws-development-and-testing)