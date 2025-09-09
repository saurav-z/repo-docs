# Cloud-init: Automate Cloud Instance Initialization

Cloud-init is the **industry-leading** solution for automating the initialization of cloud instances across diverse platforms. Find the original repo [here](https://github.com/canonical/cloud-init).

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

## Key Features of Cloud-init:

*   **Cross-Platform Support:** Works seamlessly across major public cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated Initialization:** Automatically configures cloud instances during boot based on cloud metadata.
*   **Metadata Driven:** Reads and processes instance data including:
    *   Cloud metadata
    *   User data (optional)
    *   Vendor data (optional)
*   **Configuration Flexibility:** Sets up network and storage devices, configures SSH access, and handles numerous other system configurations.
*   **Wide Distribution & Cloud Support:**  Supports the majority of [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux / Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html).

## Getting Started with Cloud-init

### Resources for Support

*   **User Documentation:** Start with the comprehensive [user documentation](https://docs.cloud-init.io/en/latest/).
*   **Community Support:**
    *   Ask questions in the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
    *   Follow announcements or ask questions on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions)
    *   Report bugs on [GitHub Issues](https://github.com/canonical/cloud-init/issues)

## Development & Contribution

*   **Contributing:** Learn how to contribute by checking out the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) document.

## Daily Builds

*   **Ubuntu:** Access the latest upstream code via the [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily).
*   **CentOS:** Use the [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/) for the latest updates.

## Build / Packaging

*   Refer to [packages](packages) for build/packaging implementations.