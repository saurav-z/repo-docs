# Cloud-init: Automate Cloud Instance Initialization

**Cloud-init is the leading cross-platform tool for automating cloud instance initialization across all major cloud providers and infrastructure platforms.**

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

Cloud-init simplifies the deployment and configuration of cloud instances by automating crucial setup tasks. This ensures consistent and reliable deployments across different cloud environments.

## Key Features of Cloud-init:

*   **Cross-Platform Compatibility:** Works seamlessly with all major public cloud providers (AWS, Azure, GCP, etc.) and private cloud infrastructure.
*   **Automated Instance Initialization:**  Reads cloud metadata and initializes the system during boot, configuring network, storage, and more.
*   **User Data Processing:** Parses and processes user-supplied data for custom configurations, including SSH keys and application setup.
*   **Vendor Data Support:** Integrates with vendor-specific data for optimized configurations.
*   **Multi-Distribution Support:** Widely supported across various Linux/Unix distributions.

## How Cloud-init Works

Cloud instances are initialized from a disk image and instance data, including:

*   **Cloud Metadata:** Information about the cloud environment.
*   **User Data (Optional):** User-supplied configuration data.
*   **Vendor Data (Optional):**  Vendor-specific configuration data.

During the boot process, cloud-init identifies the cloud environment, reads the provided metadata, and initializes the system accordingly. This process sets up the network, storage devices, SSH access, and other essential configurations.  It then processes any user or vendor-supplied data.

## Getting Started and Support

*   **User Documentation:**  Comprehensive information is available in the [user documentation](https://docs.cloud-init.io/en/latest/).
*   **Community Support:**
    *   Ask questions in the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
    *   Follow announcements or ask questions on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions)
    *   Report bugs on [GitHub Issues](https://github.com/canonical/cloud-init/issues)

## Cloud and Distribution Support

Cloud-init offers broad support for various [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux / Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html).

## Contributing to Cloud-init

Learn how to contribute to the project by reviewing the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) documentation.

## Daily Builds

*   **Ubuntu:**  [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:**  [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build and Packaging

Refer to the [packages](packages) directory for reference build/packaging implementations.

---

**Original Repository:** Find the source code and more details on the official [cloud-init GitHub repository](https://github.com/canonical/cloud-init).