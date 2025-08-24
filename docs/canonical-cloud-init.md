# Cloud-init: Automate Your Cloud Instance Initialization

**Cloud-init is the industry-leading solution for cross-platform cloud instance initialization, simplifying system setup across various environments.**  [See the original repository](https://github.com/canonical/cloud-init) for more details.

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

## Key Features of Cloud-init

Cloud-init streamlines the initial setup of your cloud instances, offering:

*   **Cross-Platform Compatibility:** Works seamlessly across major public cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated Initialization:** Automatically configures your instance during boot, reading metadata and user data to set up network devices, storage, SSH keys, and more.
*   **Cloud Metadata Processing:** Identifies the cloud environment and initializes the system based on provided cloud metadata.
*   **User and Vendor Data Support:** Parses and processes optional user and vendor data for customized configurations.
*   **Broad Distribution and Cloud Support:** Supports a wide range of Linux/Unix operating systems and cloud providers.

## Getting Started and Support

*   **User Documentation:** Explore the comprehensive [user documentation](https://docs.cloud-init.io/en/latest/) for detailed information.
*   **Community Support:** Get help and connect with other users through:
    *   [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
    *   [GitHub Discussions](https://github.com/canonical/cloud-init/discussions)
    *   [GitHub Issues](https://github.com/canonical/cloud-init/issues) for bug reports

## Development and Contribution

*   **Contribution Guidelines:** Learn how to contribute by reviewing the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) document for development, testing, and code submission guidelines.

## Daily Builds

*   **Ubuntu Daily Builds:** Access the latest upstream code through the [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily).
*   **CentOS Daily Builds:** Find daily builds in the [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/).

## Build and Packaging Resources

*   Refer to the [packages](packages) directory for reference build and packaging implementations.