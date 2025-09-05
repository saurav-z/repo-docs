# Cloud-init: Automate Your Cloud Instance Initialization

**Cloud-init is the leading cross-platform cloud instance initialization tool, simplifying the deployment of Linux and Unix systems across various cloud environments.** [Original Repository](https://github.com/canonical/cloud-init)

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

## Key Features of Cloud-init

*   **Cross-Platform Compatibility:** Works seamlessly across major public cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated System Configuration:**  Automatically configures network settings, storage devices, SSH access keys, and more during the boot process.
*   **Metadata-Driven Initialization:** Reads cloud metadata, user data, and vendor data to customize instances based on the environment.
*   **Broad Cloud Provider Support:** Supports a wide range of cloud providers and Linux/Unix distributions.
*   **Customization Options:** Processes optional user and vendor data for tailored instance configurations.

## How Cloud-init Works

Cloud-init initializes cloud instances from a disk image and instance data, including:

*   Cloud metadata
*   User data (optional)
*   Vendor data (optional)

During boot, Cloud-init identifies the cloud environment, reads the provided metadata, and configures the system accordingly. It then processes any user or vendor data passed to the instance.

## Getting Support

For assistance, consult the following resources:

*   **User Documentation:** [https://docs.cloud-init.io/en/latest/](https://docs.cloud-init.io/en/latest/)
*   **Matrix Channel:**  Ask questions in the  [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
*   **GitHub Discussions:** Follow announcements or ask questions on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions)
*   **Report Bugs:** Report issues on [GitHub Issues](https://github.com/canonical/cloud-init/issues)

## Supported Distributions and Clouds

Cloud-init supports most [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux / Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html). If your distribution or cloud is not supported, please reach out to the distribution maintainers.

## Contributing to Cloud-init

Explore the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) document to learn about developing, testing, and submitting code.

## Daily Builds

Stay up-to-date with the latest features and bug fixes using daily builds:

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build / Packaging

Find reference build/packaging implementations in the [packages](packages) directory.