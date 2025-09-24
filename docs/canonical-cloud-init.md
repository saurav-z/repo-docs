# Cloud-init: Automate Cloud Instance Initialization

**Cloud-init is the industry-leading, cross-platform tool that simplifies and automates the initialization of cloud instances across diverse environments.**

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

Cloud-init empowers you to seamlessly configure and customize your cloud instances from the moment they boot. This powerful tool is supported by all major public cloud providers, private cloud infrastructure systems, and bare-metal installations.

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly across a wide range of cloud providers and Linux distributions.
*   **Automated Configuration:** Automatically configures network settings, storage devices, SSH access keys, and more.
*   **Metadata Driven:** Leverages cloud metadata, user data, and vendor data to initialize instances.
*   **Industry Standard:** The go-to solution for cloud instance initialization.
*   **Broad Support:** Compatible with a vast majority of cloud platforms and operating systems.

## How Cloud-init Works

Cloud instances are initialized from:

*   **Cloud Metadata:** Information about the instance provided by the cloud provider.
*   **User Data (Optional):** Custom configurations and scripts specified by the user.
*   **Vendor Data (Optional):** Vendor-specific data for instance customization.

During boot, cloud-init identifies the cloud environment, reads the provided metadata, and initializes the system accordingly. It then processes any user or vendor data, tailoring the instance to your specifications.

## Get Started

*   **User Documentation:** [https://docs.cloud-init.io/en/latest/](https://docs.cloud-init.io/en/latest/)
*   **Community Support:**
    *   [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
    *   [GitHub Discussions](https://github.com/canonical/cloud-init/discussions)
    *   [Report Bugs on GitHub Issues](https://github.com/canonical/cloud-init/issues)

## Supported Platforms

Cloud-init supports the majority of clouds and Linux/Unix operating systems. See the following documentation for details:

*   [Supported Clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported)
*   [Supported Distributions](https://docs.cloud-init.io/en/latest/reference/distros.html)

## Contributing

Learn how to develop and contribute to Cloud-init: [Contributing Guide](https://docs.cloud-init.io/en/latest/development/index.html)

## Daily Builds

Access the latest upstream code for testing and bug fixes:

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR Build Repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build and Packaging

Refer to the [packages](packages) directory for reference build and packaging implementations.

[View the Cloud-init Repository on GitHub](https://github.com/canonical/cloud-init)