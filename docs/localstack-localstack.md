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

# LocalStack: Develop and Test AWS Applications Locally

**LocalStack provides a fully functional local cloud stack, enabling developers to build, test, and deploy AWS applications without the need for a remote cloud provider.** ([View on GitHub](https://github.com/localstack/localstack))

**Key Features:**

*   **Local AWS Environment:** Run your AWS applications and Lambdas entirely on your local machine.
*   **Comprehensive AWS Services Support:** Supports a wide range of AWS services including Lambda, S3, DynamoDB, Kinesis, SQS, and SNS.
*   **Faster Development:** Speeds up testing and development workflows by eliminating the need to connect to a remote cloud provider.
*   **Pro Version:** Offers additional APIs and advanced features.
*   **Multiple Installation Options:** Supports CLI, Docker, Docker Compose, and Helm.
*   **Integrations:** Works seamlessly with various tools and CI/CD pipelines.

## Getting Started

### Installation

Choose your preferred method to install LocalStack:

*   **LocalStack CLI:** The recommended way.
    *   **Brew (macOS or Linux with Homebrew):** `brew install localstack/tap/localstack-cli`
    *   **Binary Download (macOS, Linux, Windows):** Download the latest release from [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest).
    *   **PyPI (macOS, Linux, Windows):** `python3 -m pip install localstack`
*   **Docker:** Refer to the [Docker documentation](https://docs.localstack.cloud/getting-started/installation/#docker).
*   **Docker Compose:** Refer to the [Docker Compose documentation](https://docs.localstack.cloud/getting-started/installation/#docker-compose).
*   **Helm:** Refer to the [Helm documentation](https://docs.localstack.cloud/getting-started/installation/#helm).

### Quickstart

1.  **Start LocalStack:**

```bash
localstack start -d
```

2.  **Check Service Status:**

```bash
localstack status services
```

3.  **Use SQS:**

```shell
awslocal sqs create-queue --queue-name sample-queue
```

For further details on `awslocal` CLI, please refer to the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

## Running

Explore various methods to run LocalStack:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

**UI Clients:**

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

See the complete list of changes in [GitHub releases](https://github.com/localstack/localstack/releases).

## Contributing

*   Read our [contributing guide](docs/CONTRIBUTING.md).
*   Set up your [development environment](docs/development-environment-setup/README.md).
*   [Open issues](https://github.com/localstack/localstack/issues).

## Get in Touch

*   [Issues](https://github.com/localstack/localstack/issues/new/choose)
*   [Feature Requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   [Support Questions](https://docs.localstack.cloud/getting-started/help-and-support/)
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

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)).