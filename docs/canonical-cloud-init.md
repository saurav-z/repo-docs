# cloud-init: Automate Cloud Instance Initialization (Industry Standard)

**cloud-init simplifies cloud instance initialization across all major platforms, ensuring consistent and automated setup.** Learn more and contribute on the [original GitHub repository](https://github.com/canonical/cloud-init).

![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)
![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)
![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)

## What is cloud-init?

cloud-init is the *industry-standard* multi-distribution tool for cross-platform cloud instance initialization. It automates the initial configuration of your cloud instances, saving you time and effort. It's supported across:

*   All major public cloud providers
*   Private cloud infrastructure provisioning systems
*   Bare-metal installations

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly across various cloud providers and Linux distributions.
*   **Automated Configuration:** Handles network setup, storage configuration, SSH key setup, and more.
*   **Metadata-Driven Initialization:** Reads cloud metadata to configure the instance accordingly.
*   **User and Vendor Data Processing:** Processes optional user and vendor data for advanced customization.

## How it Works

Cloud instances are initialized from a disk image and instance data, including:

*   Cloud metadata
*   User data (optional)
*   Vendor data (optional)

Cloud-init identifies the cloud environment during boot and utilizes the provided metadata to initialize the system, automatically configuring it according to the cloud provider's specifications and any user-supplied data.

## Getting Help and Support

For comprehensive assistance, explore the following resources:

*   **User Documentation:** [https://docs.cloud-init.io/en/latest/](https://docs.cloud-init.io/en/latest/)
*   **Matrix Channel:** [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
*   **GitHub Discussions:** [https://github.com/canonical/cloud-init/discussions](https://github.com/canonical/cloud-init/discussions)
*   **Bug Reports:** [https://github.com/canonical/cloud-init/issues](https://github.com/canonical/cloud-init/issues)

## Supported Distributions and Clouds

Cloud-init supports a wide array of [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux / Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html). If your specific distribution or cloud is not supported, contact them and direct them to the cloud-init project.

## Contributing to cloud-init

If you'd like to contribute, refer to the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) document for development, testing, and code submission guidelines.

## Daily Builds

Test the latest upstream code and bug fixes with daily builds:

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build and Packaging Information

For build and packaging implementations, refer to the [packages](packages) directory.