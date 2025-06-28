<!-- Improved README.md - SEO Optimized -->

<div align="center">
  <img src="https://raw.githubusercontent.com/localstack/localstack/master/docs/localstack-readme-banner.svg" alt="LocalStack - A fully functional local cloud stack">
  <br/>
  <a href="https://github.com/localstack/localstack">
    <img src="https://img.shields.io/github/stars/localstack/localstack?style=social" alt="GitHub stars">
  </a>
</div>

# LocalStack: Develop and Test AWS Applications Locally

**LocalStack empowers developers to build, test, and debug AWS applications locally, without the need for a live cloud connection.**

**[View the original repository](https://github.com/localstack/localstack)**

## Key Features

*   **Local Cloud Emulation:** Runs a fully functional AWS cloud environment on your local machine.
*   **Comprehensive AWS Service Support:** Supports a wide range of AWS services including:
    *   Lambda
    *   S3
    *   DynamoDB
    *   Kinesis
    *   SQS
    *   SNS
    *   And many more, with growing support!
*   **Fast Development and Testing:** Accelerates development cycles by eliminating the need to deploy to the cloud for testing.
*   **Cost-Effective:** Reduces cloud costs by enabling local testing and development.
*   **Integration with Popular Tools:** Seamlessly integrates with popular tools like the AWS CLI, Terraform, CDK, and others.
*   **Multiple Installation Options:** Available via CLI, Docker, Docker Compose, and more.
*   **Pro Version:** Offers advanced features and extended API support (see [pricing](https://localstack.cloud/pricing)).

## What is LocalStack?

LocalStack is a cloud service emulator, allowing you to replicate AWS cloud services locally, for development and testing purposes. It allows you to run AWS applications, test complex CDK applications and Terraform configurations, all on your local machine. This simplifies and speeds up your testing and development workflow, by providing a local environment for cloud application development.

## Getting Started

### Installation

Choose your preferred method:

*   **LocalStack CLI:** The easiest way to manage LocalStack via command line.
    *   **Homebrew (macOS/Linux):** `brew install localstack/tap/localstack-cli`
    *   **Binary Download (macOS, Linux, Windows):** Download the latest release from [localstack/localstack-cli/releases](https://github.com/localstack/localstack-cli/releases/latest)
    *   **PyPI (macOS, Linux, Windows):** `python3 -m pip install localstack`
*   **Docker:**  (Recommended for most users) Ensure you have Docker installed.  Refer to the [Docker documentation](https://docs.docker.com/get-docker/) for setup.
*   **Docker Compose:** Leverage Docker Compose for more complex setups.
*   **Helm:** Deploy LocalStack within Kubernetes.

### Quickstart Example

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

    *(Requires the `awslocal` CLI.  See the [awslocal documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal) for installation)*

## Additional Resources

*   **[Documentation](https://docs.localstack.cloud):** Comprehensive documentation for detailed usage.
*   **[Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/):** See what AWS services are supported.
*   **[Pro Version](https://app.localstack.cloud):** Explore the advanced features available with the LocalStack Pro version.
*   **[Releases](https://github.com/localstack/localstack/releases):** Get the latest changes.
*   **[Changelog](https://docs.localstack.cloud/references/changelog/):** Extended release notes.
*   **[Contributing Guide](docs/CONTRIBUTING.md):** Learn how to contribute to the project.
*   **[Development Environment Setup Guide](docs/development-environment-setup/README.md):** Setup your development environment.
*   **[FAQ](https://docs.localstack.cloud/getting-started/faq/):** Find answers to frequently asked questions.

## Community and Support

*   **[LocalStack Slack Community](https://localstack.cloud/contact/)**
*   **[GitHub Issues](https://github.com/localstack/localstack/issues)**

### Get Involved

We welcome contributions! Please review the [contributing guide](docs/CONTRIBUTING.md) to get started.

### Contributors

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" /></a>

### Backers

Support LocalStack by becoming a backer on [Open Collective](https://opencollective.com/localstack#backer).

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

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)).
By using this software, you agree to the [End-User License Agreement (EULA)](docs/end_user_license_agreement).