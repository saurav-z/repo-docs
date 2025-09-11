# cloud-init: Automate Cloud Instance Initialization Across Platforms

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

Cloud-init is the **industry-leading solution** for initializing cloud instances, simplifying deployment across various platforms.

## Key Features & Benefits

*   **Cross-Platform Compatibility:** Works seamlessly with all major public cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated Configuration:** Automatically detects the cloud environment and configures the system based on metadata provided.
*   **Flexible Initialization:** Manages networking, storage, SSH access keys, and more.
*   **User & Vendor Data Processing:** Parses and processes optional user and vendor data for custom configurations.
*   **Wide Distribution & Cloud Support:** Supports a broad range of Linux distributions and cloud providers.

## How Cloud-init Works

Cloud-init initializes cloud instances from a disk image and instance data, including:

*   Cloud Metadata
*   User Data (Optional)
*   Vendor Data (Optional)

During boot, cloud-init identifies the cloud environment, reads metadata, and configures the system accordingly.

## Getting Help and Support

*   **User Documentation:** [https://docs.cloud-init.io/en/latest/](https://docs.cloud-init.io/en/latest/)
*   **Matrix Channel:** [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
*   **GitHub Discussions:** [https://github.com/canonical/cloud-init/discussions](https://github.com/canonical/cloud-init/discussions)
*   **Report Bugs:** [GitHub Issues](https://github.com/canonical/cloud-init/issues)

## Contributing to Cloud-init

Learn how to contribute to the project by reviewing the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) documentation.

## Daily Builds

Test the latest upstream code with daily builds:

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build and Packaging

See reference build/packaging implementations in the [packages](packages) directory.

---

**Original Repository:**  For more information and the complete source code, visit the original repository: [https://github.com/canonical/cloud-init](https://github.com/canonical/cloud-init)