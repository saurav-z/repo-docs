# cloud-init: Automate Cloud Instance Initialization

**Cloud-init is the leading cross-platform tool for automating cloud instance initialization across all major cloud providers and environments.**

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

## Key Features of cloud-init

*   **Cross-Platform Compatibility:** Works seamlessly across diverse cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated Instance Initialization:**  Automatically configures cloud instances during boot based on cloud metadata.
*   **Metadata & Data Handling:** Reads and processes cloud metadata, along with optional user and vendor data for comprehensive system setup.
*   **Network & Storage Configuration:** Automates the setup of network interfaces, storage devices, and other essential system components.
*   **SSH Key Management:**  Simplifies SSH access by configuring SSH keys for secure access.
*   **Broad OS Support:** Supports the majority of Linux and Unix distributions.

## How cloud-init Works

cloud-init initializes cloud instances by:

1.  **Booting from Disk Image:** Starts with a base disk image.
2.  **Reading Instance Data:** Accesses cloud metadata, optional user data, and vendor data.
3.  **Cloud Identification:** Detects the cloud environment.
4.  **System Configuration:** Configures the system based on metadata, including network settings, storage setup, SSH keys, and more.
5.  **User and Vendor Data Processing:** Parses and executes any provided user or vendor data for custom configuration.

## Getting Started

*   **User Documentation:** [User Documentation](https://docs.cloud-init.io/en/latest/)
*   **Join the Community:**
    *   [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
    *   [GitHub Discussions](https://github.com/canonical/cloud-init/discussions)
    *   [Report Bugs on GitHub Issues](https://github.com/canonical/cloud-init/issues)

## Supported Clouds & Distributions

Cloud-init supports a wide range of [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux/Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html). If your distribution or cloud is not supported, contact your distribution maintainers.

## Contributing to cloud-init

Interested in contributing? Check out the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) document.

## Daily Builds

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build/Packaging Information

Refer to the [packages](packages) directory for build and packaging implementations.

---

**[View the original cloud-init repository on GitHub](https://github.com/canonical/cloud-init)**