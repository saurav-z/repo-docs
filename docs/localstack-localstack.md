# LocalStack: Develop and Test AWS Applications Locally ☁️

**LocalStack is a fully functional cloud service emulator that enables you to develop and test your AWS applications locally, saving time and resources.**

[![GitHub Actions](https://github.com/localstack/localstack/actions/workflows/aws-main.yml/badge.svg?branch=main)](https://github.com/localstack/localstack/actions/workflows/aws-main.yml?query=branch%3Amain)
[![Coverage Status](https://coveralls.io/repos/github/localstack/localstack/badge.svg?branch=main)](https://coveralls.io/github/localstack/localstack?branch=main)
[![PyPI Version](https://img.shields.io/pypi/v/localstack?color=blue)](https://pypi.org/project/localstack/)
[![Docker Pulls](https://img.shields.io/docker/pulls/localstack/localstack)](https://hub.docker.com/r/localstack/localstack)
[![PyPi downloads](https://static.pepy.tech/badge/localstack)](https://pypi.org/project/localstack)
[![Backers on Open Collective](https://opencollective.com/localstack/backers/badge.svg)](https://opencollective.com/localstack#backers)
[![Sponsors on Open Collective](https://opencollective.com/localstack/sponsors/badge.svg)](https://opencollective.com/localstack#sponsors)
[![PyPI License](https://img.shields.io/pypi/l/localstack.svg)](https://img.shields.io/pypi/l/localstack.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Bluesky](https://img.shields.io/badge/bluesky-Follow-blue?logo=bluesky)](https://bsky.app/profile/localstack.cloud)

[<img src="https://raw.githubusercontent.com/localstack/localstack/main/docs/localstack-readme-banner.svg" alt="LocalStack - A fully functional local cloud stack" width="100%">](https://github.com/localstack/localstack)

---

## Key Features

*   **Local AWS Development**: Run and test AWS applications and Lambdas locally.
*   **Comprehensive AWS Service Support**: Emulates a wide range of AWS services like Lambda, S3, DynamoDB, SQS, SNS, and more.
*   **Faster Development Cycles**: Speed up your testing and development workflows without connecting to the remote cloud.
*   **Cost-Effective**: Develop and test applications without incurring cloud provider charges.
*   **CI/CD Integration**: Seamlessly integrate LocalStack into your CI/CD pipelines.
*   **Cross-Platform Compatibility**: Works on macOS, Linux, and Windows.
*   **Easy to Use**: Supports CLI, Docker, Docker Compose, and Helm for different use cases.

## Getting Started

### Installation

Choose your preferred installation method:

*   **LocalStack CLI (Recommended)**:

    *   **Homebrew (macOS/Linux)**:
        ```bash
        brew install localstack/tap/localstack-cli
        ```
    *   **Binary Download (macOS, Linux, Windows)**: Download the latest release from the [LocalStack CLI releases](https://github.com/localstack/localstack-cli/releases/latest) and add it to your PATH.
    *   **PyPI (macOS, Linux, Windows)**:
        ```bash
        python3 -m pip install localstack
        ```
    *   Install the `awslocal` CLI separately to interact with the local AWS services; see the [awslocal documentation](https://docs.localstack.cloud/user-guide/integrations/aws-cli/#localstack-aws-cli-awslocal) for details.
*   **Docker**:  Refer to the [Docker Installation Guide](https://docs.localstack.cloud/getting-started/installation/#docker).
*   **Docker Compose**: Check out the [Docker Compose Installation Guide](https://docs.localstack.cloud/getting-started/installation/#docker-compose).
*   **Helm**:  Learn how to install via [Helm](https://docs.localstack.cloud/getting-started/installation/#helm).

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
    ```bash
    awslocal sqs create-queue --queue-name sample-queue
    ```

    You can find information about the available AWS services at the [LocalStack AWS services](https://docs.localstack.cloud/references/coverage/) page.

### Additional Resources

*   **Documentation**: [LocalStack Documentation](https://docs.localstack.cloud)
*   **Pro Version**: [LocalStack Pro](https://app.localstack.cloud) - Explore advanced features.
*   **Feature Coverage**: [Feature Coverage](https://docs.localstack.cloud/user-guide/aws/feature-coverage/) - See supported AWS APIs.
*   **User Guides**: [User Guides](https://docs.localstack.cloud/user-guide/) - Comprehensive guides and tutorials.

## Usage

*   **Configuration**: [LocalStack Configuration](https://docs.localstack.cloud/references/configuration/)
*   **CI/CD Integration**: [LocalStack in CI](https://docs.localstack.cloud/user-guide/ci/)
*   **Integrations**: [LocalStack Integrations](https://docs.localstack.cloud/user-guide/integrations/)
*   **Tools**: [LocalStack Tools](https://docs.localstack.cloud/user-guide/tools/)
*   **Understanding LocalStack**: [Understanding LocalStack](https://docs.localstack.cloud/references/)
*   **FAQ**: [Frequently Asked Questions](https://docs.localstack.cloud/getting-started/faq/)

## Releases

View the complete list of changes in each release at [GitHub Releases](https://github.com/localstack/localstack/releases). For detailed release notes, see the [Changelog](https://docs.localstack.cloud/references/changelog/).

## Contributing

We welcome contributions!

*   Read our [Contributing Guide](docs/CONTRIBUTING.md).
*   Set up your [Development Environment](docs/development-environment-setup/README.md).
*   Browse [Open Issues](https://github.com/localstack/localstack/issues).

## Get in Touch

*   Report [Issues](https://github.com/localstack/localstack/issues/new/choose)
*   Upvote [Feature Requests](https://github.com/localstack/localstack/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+)
*   Ask [Support Questions](https://docs.localstack.cloud/getting-started/help-and-support/)
*   Join the [LocalStack Slack Community](https://localstack.cloud/contact/)
*   Discuss on [GitHub Issues](https://github.com/localstack/localstack/issues)

### Contributors

<a href="https://github.com/localstack/localstack/graphs/contributors"><img src="https://opencollective.com/localstack/contributors.svg?width=890" alt="Contributors"></a>

### Backers

<a href="https://opencollective.com/localstack#backers" target="_blank"><img src="https://opencollective.com/localstack/backers.svg?width=890" alt="Backers"></a>

### Sponsors

<a href="https://opencollective.com/localstack/sponsor/0/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/0/avatar.svg" alt="Sponsor"></a>
<a href="https://opencollective.com/localstack/sponsor/1/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/1/avatar.svg" alt="Sponsor"></a>
<a href="https://opencollective.com/localstack/sponsor/2/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/2/avatar.svg" alt="Sponsor"></a>
<a href="https://opencollective.com/localstack/sponsor/3/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/3/avatar.svg" alt="Sponsor"></a>
<a href="https://opencollective.com/localstack/sponsor/4/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/4/avatar.svg" alt="Sponsor"></a>
<a href="https://opencollective.com/localstack/sponsor/5/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/5/avatar.svg" alt="Sponsor"></a>
<a href="https://opencollective.com/localstack/sponsor/6/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/6/avatar.svg" alt="Sponsor"></a>
<a href="https://opencollective.com/localstack/sponsor/7/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/7/avatar.svg" alt="Sponsor"></a>
<a href="https://opencollective.com/localstack/sponsor/8/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/8/avatar.svg" alt="Sponsor"></a>
<a href="https://opencollective.com/localstack/sponsor/9/website" target="_blank"><img src="https://opencollective.com/localstack/sponsor/9/avatar.svg" alt="Sponsor"></a>

## License

Copyright (c) 2017-2025 LocalStack maintainers and contributors.

Copyright (c) 2016 Atlassian and others.

Released under the Apache License, Version 2.0 (see [LICENSE](LICENSE.txt)). By using this software, you agree to the [End-User License Agreement (EULA)](docs/end_user_license_agreement).

[Back to Top](#localstack-develop-and-test-aws-applications-locally-️) - View the [original repo](https://github.com/localstack/localstack).
```
Key improvements and SEO optimizations:

*   **Strong Headline:**  Clear and concise headline with keywords.
*   **One-Sentence Hook:**  Provides an immediate benefit.
*   **Keyword-Rich Subheadings:**  Uses keywords relevant to AWS, local development, and testing.
*   **Bulleted Key Features:** Highlights key benefits.
*   **Clear Installation Instructions:**  Improved and consolidated installation steps.
*   **Actionable Quickstart:**  Provides a clear, working example.
*   **Internal Linking:** Uses anchor links to jump to specific sections.
*   **External Linking:** Includes links to documentation, related projects, and helpful resources.
*   **Alt Text for Images:** Added alt text to all images to enhance accessibility and SEO.
*   **Call to Action:** Encourages the user to take action (install, contribute, get in touch).
*   **Clear Structure:** Uses headings, subheadings, and bullet points for readability.
*   **SEO-Friendly Language:**  Uses search-friendly terms and phrases.
*   **Concise and Focused:** Removed unnecessary information.
*   **Back to Top Anchor:** Added an anchor link at the end for better navigation.
*   **Original Repo Link**: Added a direct link back to the GitHub repo.