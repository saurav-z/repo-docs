# Cloud-init: Automate Cloud Instance Initialization

Cloud-init is the **industry-leading** solution for automatically initializing cloud instances across diverse platforms, simplifying deployment and management. You can find the original source code and more details on the [Canonical cloud-init GitHub repository](https://github.com/canonical/cloud-init).

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

## Key Features of Cloud-init:

*   **Cross-Platform Compatibility:** Works seamlessly with all major public cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated Initialization:** Automatically configures cloud instances based on cloud metadata, user data, and vendor data.
*   **Flexible Configuration:** Sets up network and storage devices, configures SSH access, and handles many other system aspects.
*   **Multi-Distribution Support:** Supports a wide range of Linux/Unix operating systems.
*   **Data Source Integration:**  Identifies and integrates with various cloud providers via supported data sources.

## How Cloud-init Works:

Cloud-init initializes cloud instances using data provided during instance creation. This includes:

*   **Cloud Metadata:**  Information about the cloud environment.
*   **User Data (Optional):** Custom user-defined configuration scripts or data.
*   **Vendor Data (Optional):**  Vendor-specific configuration data.

During boot, cloud-init identifies the cloud environment, reads the provided metadata, and initializes the system accordingly.  It then processes any user or vendor data.

## Getting Help and Support:

*   **User Documentation:**  [Comprehensive documentation](https://docs.cloud-init.io/en/latest/)
*   **Matrix Channel:** Join the `#cloud-init` channel on [Matrix](https://matrix.to/#/#cloud-init:ubuntu.com) for community support.
*   **GitHub Discussions:**  Follow announcements or ask questions on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions).
*   **Report Bugs:**  [Report bugs on GitHub Issues](https://github.com/canonical/cloud-init/issues).

## Supported Distributions and Clouds:

Cloud-init supports a wide variety of [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux/Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html).  Contact your distribution provider to add support for Cloud-init.

## Contributing to Cloud-init Development:

Interested in contributing? Refer to the [contributing guide](https://docs.cloud-init.io/en/latest/development/index.html) for details on how to develop, test, and submit code.

## Daily Builds:

Try the latest features and bug fixes with the daily builds:

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build and Packaging Information:

See the [packages](packages) directory for reference build and packaging implementations.