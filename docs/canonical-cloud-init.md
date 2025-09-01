# Cloud-init: Automate Cloud Instance Initialization

**Cloud-init simplifies and streamlines the initialization of your cloud instances, making deployment across various platforms effortless.** (Original repo: [https://github.com/canonical/cloud-init](https://github.com/canonical/cloud-init))

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

## Key Features of Cloud-init

Cloud-init is the industry-standard tool for automating cloud instance initialization, offering:

*   **Cross-Platform Compatibility:** Supports major public cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated Configuration:** Automatically configures network settings, storage devices, SSH access keys, and more.
*   **Metadata & Data Processing:** Reads cloud metadata, user data, and vendor data to initialize systems effectively.
*   **Broad Distribution & Cloud Support:** Compatible with a wide range of Linux distributions and cloud platforms.
*   **Easy Integration:** Integrates seamlessly with existing cloud workflows.

## How Cloud-init Works

Cloud instances are initialized from a disk image and instance data, including:

*   Cloud Metadata
*   User Data (Optional)
*   Vendor Data (Optional)

Cloud-init identifies the cloud environment during boot, reads the provided metadata, and initializes the system accordingly, automating various aspects of the setup process.

## Getting Support and Contributing

### Need Help?

*   **User Documentation:** Start with the comprehensive [user documentation](https://docs.cloud-init.io/en/latest/).
*   **Matrix Channel:** Ask questions in the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com).
*   **GitHub Discussions:** Follow announcements or ask questions on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions).
*   **Report Bugs:** Report bugs on [GitHub Issues](https://github.com/canonical/cloud-init/issues).

### Want to Contribute?

*   Check out the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) document for steps on developing, testing, and submitting code.

## Daily Builds

Get the latest features and bug fixes with daily builds:

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build / Packaging Information

Refer to the [packages](packages) directory for reference build and packaging implementations.