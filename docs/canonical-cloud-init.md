# Cloud-init: Automate Cloud Instance Initialization (The Industry Standard)

**Cloud-init** is the industry-leading solution for cross-platform cloud instance initialization, streamlining the deployment and configuration of your cloud infrastructure.

[View the original repository on GitHub](https://github.com/canonical/cloud-init)

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

## Key Features of Cloud-init

Cloud-init simplifies the process of launching and configuring cloud instances. Here's how:

*   **Cross-Platform Support:** Works seamlessly across major public cloud providers like AWS, Azure, and GCP.
*   **Private Cloud and Bare-Metal Compatibility:** Supports private cloud infrastructure and bare-metal installations.
*   **Automated Instance Initialization:** Initializes instances using metadata and user/vendor data provided by the cloud.
*   **Flexible Configuration:** Sets up network devices, storage, SSH access, and more.
*   **Metadata-Driven Configuration:** Reads cloud metadata to automatically configure the system during boot.
*   **User and Vendor Data Processing:** Parses and processes optional user and vendor data for advanced customization.

## How Cloud-init Works

Cloud instances are initialized from a disk image and data provided by the cloud provider, including:

*   Cloud Metadata
*   User Data (Optional)
*   Vendor Data (Optional)

During the boot process, cloud-init identifies the cloud environment, reads the provided metadata, and configures the system accordingly, automating a wide range of initialization tasks.

## Getting Started with Cloud-init

*   **User Documentation:** Start with the comprehensive [user documentation](https://docs.cloud-init.io/en/latest/) for detailed guidance.

## Get Help and Support

Need assistance? Here's how to get support:

*   **Matrix Channel:** Ask questions in the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com).
*   **GitHub Discussions:** Follow announcements or ask questions on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions).
*   **Report Bugs:** Report bugs on [GitHub Issues](https://github.com/canonical/cloud-init/issues).

## Cloud and Distribution Support

Cloud-init supports a wide range of [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux / Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html). Contact your distribution if support is missing.

## Contribute to Cloud-init

To contribute, refer to the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) document.

## Daily Builds

Access the latest upstream code and bug fixes via the daily builds:

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build and Packaging

Refer to the [packages](packages) directory for build and packaging implementations.