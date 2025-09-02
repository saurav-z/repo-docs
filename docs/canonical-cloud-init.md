# Cloud-init: Automate Cloud Instance Initialization

**Cloud-init is the leading open-source tool for cross-platform cloud instance initialization, simplifying and automating the setup of your cloud environments.**

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

Cloud-init is the **industry-standard** solution for automating the initialization of cloud instances across various platforms. It seamlessly integrates with all major public cloud providers, private cloud infrastructure, and bare-metal installations. Cloud-init leverages instance data, including cloud metadata, user data, and vendor data, to configure your system during the boot process.

## Key Features of Cloud-init:

*   **Cross-Platform Compatibility:** Works with all major cloud providers and a wide range of Linux/Unix distributions.
*   **Automated Configuration:** Automates network and storage device setup, SSH key configuration, and other system settings.
*   **Metadata-Driven:** Reads metadata from the cloud to tailor the instance configuration.
*   **User and Vendor Data Processing:** Parses and processes user and vendor data provided to the instance.
*   **Open Source:** Cloud-init is free and open-source software.

## Getting Started & Support

*   **User Documentation:** Explore the [official documentation](https://docs.cloud-init.io/en/latest/) for detailed information.
*   **Community Support:**
    *   Join the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com) for real-time discussions.
    *   Engage with the community on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions) for announcements and questions.
    *   Report any bugs or issues on [GitHub Issues](https://github.com/canonical/cloud-init/issues).

## Supported Clouds and Distributions

Cloud-init boasts extensive support for various [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux / Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html). Contact your distribution if support for your specific environment is missing.

## Contributing

Interested in contributing to cloud-init? Review the [contributing guide](https://docs.cloud-init.io/en/latest/development/index.html) for development, testing, and code submission guidelines.

## Daily Builds

Access the latest upstream code, features, and bug fixes with daily builds:

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build and Packaging Information

Refer to the [packages](packages) directory for build and packaging implementations.

**[View the original repository on GitHub](https://github.com/canonical/cloud-init)**