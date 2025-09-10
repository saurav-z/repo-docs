# Cloud-init: Automate Cloud Instance Initialization (Industry Standard)

**Cloud-init simplifies and automates the process of initializing cloud instances across various platforms.**

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

Cloud-init is the **industry-leading** multi-distribution tool for cross-platform cloud instance initialization, streamlining the setup of your cloud environments. It's designed to work with all major cloud providers, private cloud infrastructure solutions, and even bare-metal installations.

## Key Features of Cloud-init

*   **Cross-Platform Compatibility:** Supports a wide range of Linux distributions and cloud providers.
*   **Automated Initialization:** Automatically identifies the cloud environment and initializes systems accordingly.
*   **Metadata Driven:** Reads cloud metadata, user data, and vendor data to configure systems.
*   **Network Configuration:** Sets up network devices.
*   **Storage Configuration:** Configures storage devices.
*   **SSH Key Setup:** Configures SSH access keys.
*   **Customization:** Parses and processes optional user and vendor data.

## How Cloud-init Works

Cloud instances are initialized using a disk image and instance data. Cloud-init utilizes cloud metadata, optional user data, and optional vendor data to configure cloud instances. Cloud-init identifies the cloud it is running on during boot and initializes the system appropriately, for example, setting up network and storage devices, configuring SSH access, and other aspects of a system.

## Get Started with Cloud-init

*   **User Documentation:** Dive into the [user documentation](https://docs.cloud-init.io/en/latest/) for detailed guides and information.

## Get Help and Support

If you need assistance, explore these options:

*   **Matrix Channel:** Ask questions in the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
*   **GitHub Discussions:** Stay updated and ask questions on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions)
*   **Report Issues:** Found a bug? [Report bugs on GitHub Issues](https://github.com/canonical/cloud-init/issues)

## Cloud and Distribution Support

Cloud-init has broad support for a wide range of [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported)
and [Linux / Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html). Contact the distribution if it is not supported.

## Contributing to Cloud-init

Learn how to contribute by reviewing the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) documentation for development, testing, and submitting code.

## Daily Builds

Utilize daily builds to try the latest upstream code for the latest features or to verify bug fixes.

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build / Packaging

Refer to the [packages](packages) directory for example build/packaging implementations.

---

**[Original Repository](https://github.com/canonical/cloud-init)**