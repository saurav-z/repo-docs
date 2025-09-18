# Cloud-init: Automate Cloud Instance Initialization

**Cloud-init is the industry-leading solution for automatically initializing cloud instances across various platforms, simplifying your cloud deployment workflow.** For the most up-to-date information, visit the [original cloud-init repository](https://github.com/canonical/cloud-init).

![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)
![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)
![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)

## Key Features of Cloud-init

Cloud-init simplifies and automates the process of configuring and initializing cloud instances. Here's what it can do:

*   **Cross-Platform Support:** Works seamlessly across major public cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated Instance Configuration:** Automatically identifies the cloud environment and initializes the system based on metadata provided.
*   **Metadata-Driven Initialization:** Leverages cloud metadata, user data, and vendor data to configure network settings, storage devices, SSH keys, and more.
*   **Flexible Data Handling:** Processes optional user data and vendor data for customized configurations.
*   **Broad OS and Cloud Compatibility:** Supports a wide range of Linux/Unix distributions and cloud platforms.

## How Cloud-init Works

Cloud instances are initialized from a disk image and instance data, including:

*   Cloud metadata
*   User data (optional)
*   Vendor data (optional)

Cloud-init reads this information during the boot process and configures the system accordingly. This allows for automated server setups with minimal manual configuration.

## Getting Support and Contributing

Need help or want to contribute? Here's how:

*   **User Documentation:** Start with the comprehensive [user documentation](https://docs.cloud-init.io/en/latest/).
*   **Community Support:**
    *   Ask questions in the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com).
    *   Join the conversation on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions).
    *   Report bugs on [GitHub Issues](https://github.com/canonical/cloud-init/issues).
*   **Contributing:** Learn how to develop and submit code by visiting the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) documentation.

## Distribution and Cloud Support

Cloud-init offers robust support for a wide variety of [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux / Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html). If your specific distribution or cloud platform isn't supported, please reach out to the respective distribution for support.

## Daily Builds

Stay up-to-date with the latest features and bug fixes by using daily builds:

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build/Packaging Information

Refer to the [packages](packages) directory for information on build and packaging implementations.