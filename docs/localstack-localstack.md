<!-- Improved README for SEO Optimization -->

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

**LocalStack is the go-to cloud service emulator for developing, testing, and debugging your AWS applications without the need for a real cloud provider.**

*   [View the original repository](https://github.com/localstack/localstack)

## Key Features

*   **Local AWS Cloud Emulation:** Run a wide range of AWS services locally, including Lambda, S3, DynamoDB, Kinesis, SQS, and SNS, allowing you to test your applications in a realistic environment without incurring cloud costs.
*   **Accelerated Development:** Speed up your development and testing cycles by eliminating the need to deploy to the cloud for every change, enabling rapid iteration and debugging on your local machine.
*   **Comprehensive Service Coverage:** Supports a growing list of AWS services, including core services and advanced features, with the Pro version offering even more functionality.
*   **Flexible Installation:** Offers multiple installation options, including CLI, Docker, Docker Compose, and Helm, providing flexibility to fit your preferred workflow.
*   **Integration with AWS Tools:** Compatible with the AWS CLI and SDKs, allowing you to interact with LocalStack services using familiar tools and workflows.
*   **CI/CD Integration:** Integrate LocalStack into your CI/CD pipelines to automate testing and ensure the reliability of your cloud applications.
*   **GUI Access:** Utilize a web UI to work with LocalStack services.

## Installation

Get started with LocalStack using the following methods:

### LocalStack CLI

The LocalStack CLI allows you to start and manage the LocalStack Docker container directly from your command line.

**Prerequisites:** A functional [`docker` environment](https://docs.docker.com/get-docker/) installed.

#### Brew (macOS or Linux with Homebrew)

```bash
brew install localstack/tap/localstack-cli
```

#### Binary download (macOS, Linux, Windows)

1.  Visit [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the latest release for your platform.
2.  Extract the downloaded archive to a directory included in your `PATH` variable:
    -   For macOS/Linux, use the command: `sudo tar xvzf ~/Downloads/localstack-cli-*-darwin-*-onefile.tar.gz -C /usr/local/bin`

#### PyPI (macOS, Linux, Windows)

```bash
python3 -m pip install localstack
```

The `localstack-cli` installation enables you to run the Docker image containing the LocalStack runtime. To interact with the local AWS services, you need to install the `awslocal` CLI separately. For installation guidelines, refer to the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal).

> **Important**: Do not use `sudo` or run as `root` user. LocalStack must be installed and started entirely under a local non-root user. If you have problems with permissions in macOS High Sierra, install with `pip install --user localstack`

### Docker

See the [Docker documentation](https://docs.localstack.cloud/getting-started/installation/#docker) for details.

### Docker Compose

See the [Docker Compose documentation](https://docs.localstack.cloud/getting-started/installation/#docker-compose) for details.

### Helm

See the [Helm documentation](https://docs.localstack.cloud/getting-started/installation/#helm) for details.

## Quickstart

Start LocalStack in Docker mode:

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

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and using them with LocalStack's `awslocal` CLI.

## Running

You can run LocalStack through the following options:

-   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
-   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
-   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
-   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

## Usage

Explore LocalStack's capabilities through our comprehensive [documentation](https://docs.localstack.cloud).

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

## GUI Clients

Interact with LocalStack using these UI clients:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

## Releases

For a complete list of changes in each release, refer to [GitHub releases](https://github.com/localstack/localstack/releases) and the [changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

Contribute to LocalStack:

*   Read our [contributing guide](docs/CONTRIBUTING.md).
*   Review our [development environment setup guide](docs/development-environment-setup/README.md).
*   Explore the codebase and [open issues](https://github.com/localstack/localstack/issues).

## Get in Touch

*   Report 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose)
*   Upvote 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   Ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/)
*   Discuss local cloud development in the [LocalStack Slack Community](https://localstack.cloud/contact/) and [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

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
```
Key improvements and SEO optimizations:

*   **Clear and Concise Title:**  Uses "Develop and Test AWS Applications Locally" which is a strong keyword phrase.
*   **Keyword-Rich Hook:**  The one-sentence hook immediately grabs attention and clearly states the core function.
*   **Detailed Key Features Section:**  Uses bullet points to highlight benefits and services.  This is very important for SEO.
*   **Organized Headings:**  Uses clear, descriptive headings (Installation, Quickstart, etc.) to improve readability and SEO.
*   **Structured Content:** Improves the flow of information.
*   **Links to Documentation:**  Includes links to key documentation and resources.
*   **Call to Actions:** Include actions to encourage usage, contributions, and getting in touch.
*   **Clear and Concise Language:**  Avoids jargon and focuses on the benefits of using LocalStack.
*   **Optimized for Search:** Uses keywords throughout the document to improve search ranking.
*   **Complete Sections:** All sections are complete.
*   **Clean and Organized:** Formatting is consistent and easy to read.
*   **Included links to relevant docs.**
*   **Highlights benefits.**
*   **Maintains all original content.**
*   **Includes links back to the original repo.**