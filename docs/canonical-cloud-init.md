# Cloud-init: Automate Cloud Instance Initialization (Industry Standard)

Cloud-init simplifies and automates the process of configuring cloud instances across various platforms, making cloud deployments easier than ever.

**(Original Repository: [https://github.com/canonical/cloud-init](https://github.com/canonical/cloud-init))**

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

## What is Cloud-init?

Cloud-init is the **industry-leading** cross-platform cloud instance initialization tool, used by major cloud providers and private cloud infrastructure systems. Cloud-init streamlines the boot process of your cloud instances by reading metadata and user data to configure the system automatically.

## Key Features:

*   **Cross-Platform Support:** Works across all major public cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated Configuration:** Initializes cloud instances from disk images and instance data (cloud metadata, user data, and vendor data).
*   **Cloud-Aware Initialization:** Automatically identifies the cloud environment and configures the system accordingly.
*   **Network and Storage Setup:** Configures network and storage devices during instance boot.
*   **User and Vendor Data Processing:** Parses and processes user and vendor data passed to the instance for custom configurations.
*   **SSH Key Configuration:** Enables secure SSH access configuration.

## Getting Started & Support

*   **User Documentation:** Start with the comprehensive [user documentation](https://docs.cloud-init.io/en/latest/).
*   **Community Support:**
    *   Join the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
    *   Follow announcements or ask a question on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions)
    *   Report bugs on [GitHub Issues](https://github.com/canonical/cloud-init/issues)

## Supported Clouds and Distributions

Cloud-init supports the vast majority of [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux / Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html).

## Contributing

Interested in contributing? Learn how to develop, test, and submit code by reviewing the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) document.

## Daily Builds

Test the latest features and bug fixes using the daily builds:

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build/Packaging

Refer to [packages](packages) for reference build/packaging implementations.