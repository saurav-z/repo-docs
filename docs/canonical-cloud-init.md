# Cloud-init: Automate Cloud Instance Initialization

**Cloud-init is the go-to open-source solution for automating cloud instance configuration across all major cloud providers and operating systems.**

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

Cloud-init is an **industry-standard** tool for automating the initialization of cloud instances across various platforms. It streamlines the process of configuring cloud instances, making them ready to use upon boot.

## Key Features of Cloud-init:

*   **Cross-Platform Compatibility:** Supports all major public cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated Configuration:** Automatically identifies the cloud environment and configures the system accordingly.
*   **Metadata Processing:** Reads and processes cloud metadata, user data, and vendor data to initialize the instance.
*   **Networking & Storage Setup:** Configures network interfaces, storage devices, and other essential system components.
*   **SSH Key Configuration:** Simplifies SSH access key setup.
*   **Open Source & Widely Supported:** Supported by a large community and integrated into most Linux distributions.

## How Cloud-init Works:

Cloud-init initializes cloud instances using three primary data sources:

*   **Cloud Metadata:** Information about the instance provided by the cloud provider.
*   **User Data (Optional):** Custom configuration scripts and instructions provided by the user.
*   **Vendor Data (Optional):** Information specific to the cloud provider or vendor.

Cloud-init reads these data sources during the boot process to automatically configure the system.

## Get Help and Support

For detailed information and assistance, consult the following resources:

*   **User Documentation:** [https://docs.cloud-init.io/en/latest/](https://docs.cloud-init.io/en/latest/)
*   **Community Chat:** Join the ``#cloud-init`` channel on Matrix: [https://matrix.to/#/#cloud-init:ubuntu.com](https://matrix.to/#/#cloud-init:ubuntu.com)
*   **GitHub Discussions:** [https://github.com/canonical/cloud-init/discussions](https://github.com/canonical/cloud-init/discussions)
*   **Report Bugs:** [https://github.com/canonical/cloud-init/issues](https://github.com/canonical/cloud-init/issues)

## Supported Distributions and Clouds

Cloud-init supports a wide range of [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux/Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html). If your distribution or cloud is not supported, please contact the distribution maintainers and encourage them to integrate cloud-init.

## Contributing

Interested in contributing to cloud-init? Review the [contributing guidelines](https://docs.cloud-init.io/en/latest/development/index.html) to learn how to develop, test, and submit code.

## Daily Builds

Stay up-to-date with the latest features and bug fixes by using daily builds:

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build / Packaging

Refer to the [packages](packages) directory for build and packaging implementations.

For more information, visit the original repository: [https://github.com/canonical/cloud-init](https://github.com/canonical/cloud-init)