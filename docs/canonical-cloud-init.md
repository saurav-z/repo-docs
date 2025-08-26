# Cloud-init: Automate Cloud Instance Initialization

**Cloud-init is the leading open-source solution for automating cloud instance initialization across all major platforms, streamlining your cloud deployments.** For more information, visit the original repository: [https://github.com/canonical/cloud-init](https://github.com/canonical/cloud-init).

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

## Key Features of Cloud-init

Cloud-init simplifies the process of configuring cloud instances by:

*   **Cross-Platform Compatibility:** Works seamlessly across all major public cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated System Configuration:** Automatically detects the cloud environment during boot and configures network settings, storage, SSH keys, and more.
*   **Metadata and Data Processing:** Reads and processes cloud metadata, optional user data, and vendor data for tailored instance initialization.
*   **Wide Distribution Support:**  Supports a vast majority of Linux distributions and cloud platforms.

## Getting Started

*   **User Documentation:** Comprehensive [user documentation](https://docs.cloud-init.io/en/latest/) to get you started.

## Need Help?

Get support and connect with the community:

*   **Matrix Channel:** Ask questions in the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com).
*   **GitHub Discussions:** Follow announcements and engage in discussions on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions).
*   **Report Bugs:**  Report any bugs on [GitHub Issues](https://github.com/canonical/cloud-init/issues).

## Supported Distributions and Clouds

Cloud-init is designed to work with a wide range of [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux / Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html).  If your specific environment is not yet supported, please contact your distribution provider and encourage them to integrate with cloud-init.

## Developing Cloud-init

*   **Contributing Guide:** Learn how to contribute and submit code by reviewing the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) document.

## Daily Builds

*   **Ubuntu Daily PPAs:** Test the latest features and bug fixes with [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily).
*   **CentOS COPR Build Repos:** Access daily builds for CentOS via the [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/).

## Build / Packaging Information

*   **Packages:** See reference build/packaging implementations in the [packages](packages) directory.