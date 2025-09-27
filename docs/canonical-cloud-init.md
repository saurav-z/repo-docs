# Cloud-init: Automate Cloud Instance Initialization (Industry Standard)

**Cloud-init is the leading open-source tool for cross-platform cloud instance initialization, streamlining the setup of your instances across various cloud providers.**

[View the original Cloud-init repository on GitHub](https://github.com/canonical/cloud-init)

<br>

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

<br>

## Key Features of Cloud-init:

*   **Cross-Platform Compatibility:** Works seamlessly across all major public cloud providers (AWS, Azure, Google Cloud, etc.), private cloud infrastructure, and bare-metal installations.
*   **Automated System Configuration:** Automates critical system initialization tasks during instance boot.
*   **Metadata Driven:** Reads cloud metadata, user data, and vendor data to configure network settings, storage devices, SSH keys, and other crucial system aspects.
*   **Industry Standard:** Cloud-init is the *de facto* standard for cloud instance initialization, widely adopted and supported.
*   **Open Source:**  Benefit from a robust and actively developed open-source project with community support.

<br>

## How Cloud-init Works

Cloud-init initializes cloud instances using:

*   **Cloud Metadata:** Provides information about the instance, such as instance ID and region.
*   **User Data (Optional):** Allows you to provide custom scripts and configuration data.
*   **Vendor Data (Optional):**  Provides information from the cloud vendor.

During boot, cloud-init identifies the cloud environment, reads the provided data, and configures the system accordingly.

<br>

## Getting Help & Support

*   **User Documentation:**  [Comprehensive documentation](https://docs.cloud-init.io/en/latest/)
*   **Community Support:**
    *   [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
    *   [GitHub Discussions](https://github.com/canonical/cloud-init/discussions)
    *   [Report Bugs on GitHub Issues](https://github.com/canonical/cloud-init/issues)

<br>

## Supported Distributions and Clouds

Cloud-init supports a wide range of [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux / Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html).  If your environment is not listed, please contact your distribution maintainers and encourage them to integrate cloud-init support.

<br>

## Contributing to Cloud-init

Learn how to develop, test, and submit code: [Contributing Guide](https://docs.cloud-init.io/en/latest/development/index.html).

<br>

## Daily Builds

Stay up-to-date with the latest features and bug fixes by using daily builds.

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

<br>

## Build/Packaging Information

Refer to the [packages](packages) directory for reference build and packaging implementations.