# Cloud-init: Automate Cloud Instance Initialization

Cloud-init is the leading cross-platform solution for automating cloud instance initialization, streamlining your cloud deployments.

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

**Cloud-init** is the industry-standard tool for initializing cloud instances across various platforms, providers, and environments. As a cross-platform tool, it simplifies the provisioning process.  It supports a wide range of cloud providers and Linux/Unix operating systems.

## Key Features of Cloud-init:

*   **Cross-Platform Compatibility:** Works seamlessly across major public cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated Initialization:**  Automatically configures cloud instances based on metadata and user data provided during boot.
*   **Metadata Processing:** Reads and utilizes cloud metadata to configure network settings, storage devices, and other system aspects.
*   **User Data Processing:** Parses and executes user data, enabling the configuration of SSH keys, application installation, and custom scripts.
*   **Vendor Data Support:** Processes vendor data for specialized configurations and integrations.

## How Cloud-init Works:

Cloud-init initializes cloud instances using data from three sources:

*   **Cloud Metadata:**  Information about the instance, such as its cloud provider and region.
*   **User Data (Optional):**  Custom configuration data, typically in the form of a cloud-config file or shell script.
*   **Vendor Data (Optional):**  Vendor-specific configuration data.

During boot, cloud-init identifies the cloud environment, reads the provided data, and configures the system accordingly.

## Getting Started and Resources:

*   **User Documentation:** Access detailed information and guides on [cloud-init documentation](https://docs.cloud-init.io/en/latest/).
*   **Community Support:**
    *   Ask questions in the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com).
    *   Follow announcements or ask questions on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions).
    *   Report bugs on [GitHub Issues](https://github.com/canonical/cloud-init/issues).
*   **Supported Clouds and Distributions:** Cloud-init supports most [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux / Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html).

## Development and Contributing:

*   **Contributing Guide:** Learn how to contribute to cloud-init by reviewing the [contributing documentation](https://docs.cloud-init.io/en/latest/development/index.html).

## Daily Builds:

*   **Ubuntu:** Access daily builds via the [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily).
*   **CentOS:**  Find daily builds in the [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/).

## Build and Packaging

*   Refer to [packages](packages) to see reference build/packaging implementations.

**Explore the source code and learn more about cloud-init on GitHub:** [https://github.com/canonical/cloud-init](https://github.com/canonical/cloud-init)