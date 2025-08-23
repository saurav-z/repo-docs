# Cloud-init: Automate Cloud Instance Initialization

Cloud-init is the go-to, open-source solution for automating cloud instance initialization across diverse platforms and infrastructures. (See the original repository on [GitHub](https://github.com/canonical/cloud-init).)

![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)
![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)
![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)

## Key Features of Cloud-init

Cloud-init simplifies and streamlines the process of provisioning cloud instances with these core features:

*   **Cross-Platform Compatibility:** Works seamlessly across all major public cloud providers, private cloud provisioning systems, and bare-metal installations.
*   **Automated Initialization:** Automatically identifies the cloud environment during boot and configures the system based on cloud metadata.
*   **Metadata-Driven Configuration:** Reads cloud metadata, user data, and vendor data to set up essential configurations, including network settings, storage devices, and SSH access.
*   **Multi-Distribution Support:** Supported by and shipped with a wide range of Linux/Unix distributions.

## How Cloud-init Works

Cloud instances are initialized using a disk image and instance-specific data:

*   **Cloud Metadata:** Provides information about the cloud environment.
*   **User Data (Optional):** Allows users to specify custom configurations.
*   **Vendor Data (Optional):** Enables vendor-specific configurations.

## Get Support and Contribute

*   **User Documentation:** Start with the comprehensive [user documentation](https://docs.cloud-init.io/en/latest/).
*   **Community Support:**
    *   Ask questions in the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com).
    *   Follow announcements and engage in discussions on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions).
    *   Report bugs on [GitHub Issues](https://github.com/canonical/cloud-init/issues).
*   **Contributing:** Learn how to develop, test, and submit code by reviewing the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) document.

## Daily Builds

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build and Packaging

*   Refer to the [packages](packages) directory for reference build/packaging implementations.