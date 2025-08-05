# LocalStack: Develop and Test Your AWS Applications Locally

**LocalStack enables you to develop and test your AWS applications locally, without the need for a remote cloud provider.**

[View the original repository on GitHub](https://github.com/localstack/localstack)

---

## Key Features

*   **Local AWS Cloud Emulation:** Run your AWS applications and Lambdas entirely on your local machine.
*   **Fast Development & Testing:** Speed up your development and testing workflow by eliminating the need for cloud deployments.
*   **Comprehensive AWS Service Support:** Supports a wide range of AWS services like Lambda, S3, DynamoDB, Kinesis, SQS, SNS, and more.
*   **Pro Version for Advanced Features:** Access additional APIs and advanced features with the [Pro version](https://localstack.cloud/pricing).
*   **Easy Installation:** Install via CLI (Brew, Binary, PyPI), Docker, Docker Compose, or Helm.
*   **Integration with Popular Tools:** Works seamlessly with your existing AWS CLI and other tools.
*   **Extensive Documentation & Support:** Comprehensive [documentation](https://docs.localstack.cloud) and a vibrant community for support.

---

## Overview

LocalStack is a cloud service emulator that runs in a single container on your laptop or in your CI environment.  It allows you to run your AWS applications entirely on your local machine, speeding up testing and simplifying development workflows, whether you're testing complex CDK applications, Terraform configurations, or just starting with AWS.  LocalStack supports a wide range of AWS services, with the [Pro version](https://localstack.cloud/pricing) offering even more.  Explore the [Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) for a complete list of supported APIs.

---

## Install

The quickest way to get started is by using the LocalStack CLI.  Ensure you have a functional `docker` environment.

### Brew (macOS or Linux with Homebrew)

```bash
brew install localstack/tap/localstack-cli
```

### Binary download (macOS, Linux, Windows)

1.  Visit [localstack/localstack-cli](https://github.com/localstack/localstack-cli/releases/latest) and download the latest release for your platform.
2.  Extract the archive to a directory in your `PATH`.

### PyPI (macOS, Linux, Windows)

```bash
python3 -m pip install localstack
```

After installing the CLI, install the `awslocal` CLI separately to interact with local AWS services. Refer to the [`awslocal` documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal) for installation guidance.

> **Important**: Do not use `sudo` or run as `root` user. Install and start LocalStack under a local non-root user.  For macOS High Sierra permission issues, install with `pip install --user localstack`.

---

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

Learn more about [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) and how to use them with the `awslocal` CLI.

---

## Running

Choose your preferred method to run LocalStack:

*   [LocalStack CLI](https://docs.localstack.cloud/getting-started/installation/#localstack-cli)
*   [Docker](https://docs.localstack.cloud/getting-started/installation/#docker)
*   [Docker Compose](https://docs.localstack.cloud/getting-started/installation/#docker-compose)
*   [Helm](https://docs.localstack.cloud/getting-started/installation/#helm)

---

## Usage

Explore the comprehensive [documentation](https://docs.localstack.cloud) for in-depth guides and references.

*   [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

Utilize these UI clients:

*   [LocalStack Web Application](https://app.localstack.cloud)
*   [LocalStack Desktop](https://docs.localstack.cloud/user-guide/tools/localstack-desktop/)
*   [LocalStack Docker Extension](https://docs.localstack.cloud/user-guide/tools/localstack-docker-extension/)

---

## Releases

Review the [GitHub releases](https://github.com/localstack/localstack/releases) for complete release information. For extended release notes, see the [changelog](https://docs.localstack.cloud/references/changelog/).

---

## Contributing

Contribute to LocalStack!

*   Read the [contributing guide](docs/CONTRIBUTING.md).
*   Set up your [development environment](docs/development-environment-setup/README.md).
*   Find and address [open issues](https://github.com/localstack/localstack/issues).

---

## Get in touch

Connect with the LocalStack team:

*   Report 🐞 [issues](https://github.com/localstack/localstack/issues/new/choose)
*   Upvote 👍 [feature requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   Ask [support questions](https://docs.localstack.cloud/getting-started/help-and-support/)
*   Discuss local cloud development in the [LocalStack Slack Community](https://localstack.cloud/contact/)
*   Use the [LocalStack GitHub Issue tracker](https://github.com/localstack/localstack/issues)

### Contributors

Thank you to everyone who contributes to LocalStack!

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

Support LocalStack by becoming a backer on [Open Collective](https://opencollective.com/localstack#backer).

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890"></a>

### Sponsors

Sponsor LocalStack on [Open Collective](https://opencollective.com/localstack#sponsor).

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

---

## License

Copyright (c) 2017-2025 LocalStack maintainers and contributors.

Copyright (c) 2016 Atlassian and others.

This version of LocalStack is released under the Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)). By downloading and using this software you agree to the [End-User License Agreement (EULA)](docs/end_user_license_agreement).