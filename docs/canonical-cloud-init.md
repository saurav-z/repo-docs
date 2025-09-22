# cloud-init: Automate Cloud Instance Initialization for Seamless Deployment

[View the original repository on GitHub](https://github.com/canonical/cloud-init)

![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)
![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)
![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)

**cloud-init is the leading cross-platform cloud instance initialization tool, simplifying the provisioning of your cloud infrastructure across diverse environments.**

## Key Features:

*   **Cross-Platform Compatibility:** Works seamlessly across major public cloud providers (AWS, Azure, GCP, etc.), private cloud infrastructure, and bare-metal installations.
*   **Automated Instance Configuration:** Automatically configures network settings, storage devices, SSH access, and more based on cloud metadata.
*   **Flexible Data Processing:** Parses and processes user and vendor data, enabling custom configurations and application deployments.
*   **Industry Standard:** Cloud-init is the *industry standard* for cloud instance initialization, ensuring consistent behavior across different cloud platforms.
*   **Wide OS Support:** Supports a broad range of Linux/Unix distributions.

## How Cloud-init Works

Cloud-init initializes cloud instances from a disk image and instance data:

*   **Cloud Metadata:** Provides information about the cloud environment.
*   **User Data (Optional):** Allows for custom configurations and scripts.
*   **Vendor Data (Optional):** Enables platform-specific configurations.

During boot, cloud-init identifies the cloud environment, reads the provided metadata, and configures the system accordingly.

## Getting Help and Support

Access comprehensive documentation and community resources for cloud-init:

*   **User Documentation:** [https://docs.cloud-init.io/en/latest/](https://docs.cloud-init.io/en/latest/)
*   **Matrix Channel:** [``#cloud-init`` on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
*   **GitHub Discussions:** [https://github.com/canonical/cloud-init/discussions](https://github.com/canonical/cloud-init/discussions)
*   **Report Bugs:** [GitHub Issues](https://github.com/canonical/cloud-init/issues)

## Distribution and Cloud Support

Cloud-init supports a wide range of clouds and Linux/Unix operating systems:

*   **Supported Clouds:** [https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported)
*   **Supported OSes:** [https://docs.cloud-init.io/en/latest/reference/distros.html](https://docs.cloud-init.io/en/latest/reference/distros.html)

## Contribute to cloud-init

Learn how to develop, test, and submit code:

*   **Contributing Guide:** [https://docs.cloud-init.io/en/latest/development/index.html](https://docs.cloud-init.io/en/latest/development/index.html)

## Daily Builds

Stay up-to-date with the latest features and bug fixes with daily builds:

*   **Ubuntu Daily PPAs:** [https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS COPR Build Repos:** [https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build/Packaging

*   **Packaging Information:** [packages](packages)