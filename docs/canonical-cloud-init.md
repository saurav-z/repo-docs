# cloud-init: Automate Your Cloud Instance Initialization

**Cloud-init is the industry-standard tool for initializing cloud instances across all major platforms.**  For the original project, visit the [cloud-init GitHub repository](https://github.com/canonical/cloud-init).

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

## Key Features of cloud-init:

*   **Cross-Platform Support:** Works seamlessly across all major public cloud providers (AWS, Azure, GCP, etc.), private cloud infrastructure, and bare-metal installations.
*   **Automated Instance Initialization:**  Cloud-init automatically configures your cloud instances during the boot process.
*   **Metadata-Driven Configuration:**  Reads cloud metadata, user data, and vendor data to initialize network settings, storage devices, SSH keys, and more.
*   **Distribution & OS Compatibility:**  Supports a wide range of Linux/Unix operating systems.
*   **Flexible and Extensible:**  Allows for custom configurations through user data and vendor data.

## How cloud-init Works:

Cloud instances are initialized using a disk image and instance-specific data:

*   Cloud Metadata
*   User Data (Optional)
*   Vendor Data (Optional)

During boot, cloud-init identifies the cloud environment, reads the provided metadata, and configures the system accordingly. This includes setting up networking, storage, and other essential aspects of your cloud instance. Cloud-init then processes any user or vendor data provided to the instance for further customization.

## Getting Help and Support:

*   **User Documentation:**  Find detailed information in the [user documentation](https://docs.cloud-init.io/en/latest/).
*   **Community Support:**
    *   Ask questions in the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
    *   Follow announcements and ask questions on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions)
    *   Report bugs on [GitHub Issues](https://github.com/canonical/cloud-init/issues)

## Cloud and Distribution Support:

Cloud-init supports the majority of [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux/Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html).

## Contributing to cloud-init:

Interested in contributing?  Check out the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) document to learn how to develop, test, and submit code.

## Daily Builds:

Stay up-to-date with the latest features and bug fixes by using daily builds:

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build and Packaging Information:

Refer to [packages](packages) for reference build/packaging implementations.