# Cloud-init: Automate Cloud Instance Initialization (Industry Standard)

**Cloud-init is the go-to solution for automating cloud instance initialization across all major platforms, streamlining your deployment process.**

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

Cloud-init is the industry-standard, cross-platform tool for initializing cloud instances. It supports a wide range of cloud providers, provisioning systems, and bare-metal installations, offering a consistent and automated approach to system setup.

## Key Features of Cloud-init:

*   **Cross-Platform Compatibility:** Works seamlessly across all major public cloud providers (AWS, Azure, GCP, etc.), private cloud infrastructure, and bare-metal environments.
*   **Automated Instance Initialization:** Automatically identifies the cloud environment and configures the system based on cloud metadata.
*   **Metadata and Data Handling:** Reads and processes cloud metadata, user data, and vendor data to configure network settings, storage, SSH keys, and other system aspects.
*   **Supports Diverse OS and Cloud Environments:** Wide support for various Linux/Unix distributions and cloud platforms.
*   **Customizable:** Flexible to accommodate user-specific configurations through user and vendor data.

## How Cloud-init Works

Cloud instances are initialized from a disk image and data provided by the cloud provider, which can include:

*   **Cloud Metadata:** Information about the instance, such as its name, network configuration, and storage.
*   **User Data (Optional):** Instructions and configurations specified by the user, such as scripts or configuration files.
*   **Vendor Data (Optional):** Data provided by the cloud provider or vendor to configure specific services or features.

Cloud-init reads this data during boot and configures the system accordingly, automating tasks like network setup, SSH key configuration, and more.

## Getting Started & Resources

*   **User Documentation:** [Comprehensive documentation](https://docs.cloud-init.io/en/latest/) to guide you through the usage and configuration.
*   **Community Support:**
    *   **Matrix Channel:** Get help and connect with other users in the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com).
    *   **GitHub Discussions:** Stay informed on announcements and engage in discussions through [GitHub Discussions](https://github.com/canonical/cloud-init/discussions).
    *   **Bug Reporting:** [Report bugs on GitHub Issues](https://github.com/canonical/cloud-init/issues) to contribute to the project's improvement.
*   **Cloud and OS Support:** Cloud-init supports the majority of [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux/Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html). If your environment is not supported, please contact your distribution provider.

## Development and Contribution

*   **Contribution Guide:** Learn how to contribute to the project, including development, testing, and code submission, in the [contributing document](https://docs.cloud-init.io/en/latest/development/index.html).
*   **Daily Builds:** Stay up-to-date with the latest features and bug fixes through daily builds:
    *   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
    *   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)
*   **Build/Packaging Information:** Refer to the [packages](packages) directory for reference build and packaging implementations.

**Explore the original repository for Cloud-init at:** [https://github.com/canonical/cloud-init](https://github.com/canonical/cloud-init)