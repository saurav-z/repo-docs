# Cloud-init: Automate Cloud Instance Initialization

Cloud-init is the leading open-source tool for automating the initialization of cloud instances across all major platforms. (See the original repo here: [https://github.com/canonical/cloud-init](https://github.com/canonical/cloud-init))

![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)
![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)
![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)

## Key Features of Cloud-init

*   **Cross-Platform Compatibility:** Works seamlessly across all major public cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated Configuration:** Automatically configures network settings, storage devices, SSH access keys, and other system aspects.
*   **Metadata Driven:** Reads and processes cloud metadata to initialize instances based on the specific cloud environment.
*   **User and Vendor Data Processing:** Parses and executes user-provided and vendor-provided data for advanced customization.
*   **Industry Standard:** The *industry standard* multi-distribution method for cross-platform cloud instance initialization.

## How Cloud-init Works

Cloud-init initializes cloud instances from a disk image and the following data:

*   Cloud metadata
*   User data (optional)
*   Vendor data (optional)

During boot, cloud-init identifies the cloud environment, reads the provided metadata, and initializes the system accordingly. It then processes any optional user or vendor data.

## Getting Started and Support

*   **Documentation:** Comprehensive [user documentation](https://docs.cloud-init.io/en/latest/).
*   **Community Support:**
    *   Join the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com) for real-time discussions.
    *   Follow announcements and ask questions on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions).
    *   Report bugs and issues on [GitHub Issues](https://github.com/canonical/cloud-init/issues).

## Cloud and Distribution Support

Cloud-init supports a wide range of [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux / Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html). If your distribution or cloud is not supported, please contact your distribution to request support.

## Contributing to Cloud-init

Learn how to contribute by reviewing the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) document, which outlines the steps necessary to develop, test, and submit code.

## Daily Builds

Stay up-to-date with the latest features and bug fixes through daily builds:

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build and Packaging Information

For reference build/packaging implementations, refer to [packages](packages).