# Cloud-init: Automate Cloud Instance Initialization

**Cloud-init is the leading cross-platform tool that automates cloud instance initialization across major public and private cloud platforms.**

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

Cloud-init is the *industry standard* for automating the initialization of cloud instances across various platforms. It streamlines the process of provisioning and configuring cloud instances by reading instance metadata, user data, and vendor data. This process simplifies tasks like setting up networking, storage, SSH keys, and other system configurations.

**Key Features:**

*   **Cross-Platform Support:** Works across all major public cloud providers, private cloud infrastructure systems, and bare-metal installations.
*   **Automated Instance Configuration:** Automatically identifies the cloud environment during boot and configures the system accordingly.
*   **Metadata Driven:** Utilizes cloud metadata, user data, and vendor data to customize instance settings.
*   **Network and Storage Setup:** Configures network and storage devices.
*   **Security Configuration:** Configures SSH access keys.
*   **Wide Distribution and Cloud Support:** Broadly supported by various [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux/Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html).

## Getting Help and Support

*   **User Documentation:** Get started with the [user documentation](https://docs.cloud-init.io/en/latest/).
*   **Community Support:**
    *   Join the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com) to ask questions.
    *   Follow announcements or ask questions on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions).
    *   Report bugs on [GitHub Issues](https://github.com/canonical/cloud-init/issues).

## Contributing

Learn how to contribute to the project by reviewing the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) document which outlines the steps necessary to develop, test, and submit code.

## Daily Builds

Get the latest features and bug fixes by trying the daily builds:

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build and Packaging

See reference build/packaging implementations in the [packages](packages) directory.

**[Learn more and explore the source code on GitHub](https://github.com/canonical/cloud-init)**