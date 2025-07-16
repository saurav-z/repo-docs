# LocalStack: Your Local Cloud for AWS Development and Testing

**Develop and test your AWS applications locally with LocalStack, a fully functional cloud service emulator.**

[![GitHub Actions](https://github.com/localstack/localstack/actions/workflows/aws-main.yml/badge.svg?branch=master)](https://github.com/localstack/localstack/actions/workflows/aws-main.yml?query=branch%3Amaster)
[![Coverage Status](https://coveralls.io/repos/github/localstack/localstack/badge.svg?branch=master)](https://coveralls.io/github/localstack/localstack?branch=master)
[![PyPI Version](https://img.shields.io/pypi/v/localstack?color=blue)](https://pypi.org/project/localstack/)
[![Docker Pulls](https://img.shields.io/docker/pulls/localstack/localstack)](https://hub.docker.com/r/localstack/localstack)
[![PyPi downloads](https://static.pepy.tech/badge/localstack)](https://pypi.org/project/localstack)
[![Backers on Open Collective](https://opencollective.com/localstack/backers/badge.svg)](https://opencollective.com/localstack)
[![Sponsors on Open Collective](https://opencollective.com/localstack/sponsors/badge.svg)](https://opencollective.com/localstack)
[![PyPI License](https://img.shields.io/pypi/l/localstack.svg)](https://img.shields.io/pypi/l/localstack.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Bluesky](https://img.shields.io/badge/bluesky-Follow-blue?logo=bluesky)](https://bsky.app/profile/localstack.cloud)

---

## Key Features

*   **Local AWS Cloud:** Run a comprehensive set of AWS services locally, including Lambda, S3, DynamoDB, Kinesis, SQS, and SNS, without needing a remote cloud provider.
*   **Faster Development:**  Accelerate your development and testing cycles by eliminating the need to deploy to the cloud for every iteration.
*   **Simplified Testing:**  Test complex applications, CDK applications, and Terraform configurations quickly and efficiently on your local machine.
*   **Wide Service Coverage:** Supports a growing list of AWS services, with new APIs and features added regularly (see [Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/)).
*   **Easy Setup:**  Get started quickly with the LocalStack CLI, Docker, Docker Compose, or Helm.
*   **Pro Version:** Access additional APIs and advanced features with the [Pro version](https://localstack.cloud/pricing) of LocalStack.
*   **Multiple Integration Options:** Integrate with various tools and frameworks through the LocalStack CLI, `awslocal` CLI, and UI clients.

## Installation

Choose your preferred method to install the LocalStack CLI. Ensure that your machine has a functional [`docker` environment](https://docs.docker.com/get-docker/) installed before proceeding.

### Brew (macOS or Linux with Homebrew)

```bash
brew install localstack/tap/localstack-cli
```

### Binary Download (macOS, Linux, Windows)

1.  Visit [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the latest release.
2.  Extract the archive to a directory included in your `PATH` variable.

### PyPI (macOS, Linux, Windows)

```bash
python3 -m pip install localstack
```

**Note:**  Install `awslocal` CLI separately to interact with the local AWS services, see the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

## Quickstart

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

    Find details on the [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) to start using them.

## Run Options

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage and Resources

Explore the following resources to start using LocalStack:

*   [Documentation](https://docs.localstack.cloud)
    *   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
    *   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
    *   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
    *   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
    *   [Understanding LocalStack](https://docs.localstack.cloud/references/)
    *   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

*   **UI Clients:**
    *   [LocalStack Web Application](https://app.localstack.cloud)
    *   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
    *   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

View all the changes for each release by visiting [GitHub releases](https://github.com/localstack/localstack/releases), or go to the [changelog](https://docs.localstack.cloud/references/changelog/) for extended release notes.

## Contributing

Contribute to LocalStack:

1.  Read the [contributing guide](docs/CONTRIBUTING.md).
2.  Set up your [development environment](docs/development-environment-setup/README.md).
3.  Explore the codebase and [open issues](https://github.com/localstack/localstack/issues).

## Get in Touch

*   [LocalStack Slack Community](https://localstack.cloud/contact/)
*   [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

### Contributors

We are thankful to all the people who have contributed to this project.

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

Become a backer on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

Support the project on [Open Collective](https://opencollective.com/localstack#sponsor).

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

[Apache License, Version 2.0](LICENSE.txt).  See the [End-User License Agreement (EULA)](docs/end_user_license_agreement).

[Back to Top](#localstack-your-local-cloud-for-aws-development-and-testing)