# Cloud-init: Automate Cloud Instance Initialization ☁️

**Cloud-init is the leading open-source tool for automating the initialization of cloud instances across all major platforms, streamlining deployment and management.** For more detailed information, please visit the original repository: [https://github.com/canonical/cloud-init](https://github.com/canonical/cloud-init).

[![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/unit.yml)
[![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/integration.yml)
[![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml)

## Key Features of Cloud-init

Cloud-init simplifies and automates the process of configuring cloud instances by:

*   **Cross-Platform Support:** Works seamlessly across all major public cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated Instance Initialization:** Reads cloud metadata, user data, and vendor data to configure networking, storage, SSH keys, and more.
*   **Flexible Configuration:** Processes user and vendor data to customize instance settings.
*   **Broad Cloud & OS Compatibility:** Supports a wide range of [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux/Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html).

## How Cloud-init Works

Cloud instances are initialized using a disk image and instance-specific data:

1.  **Cloud Metadata:** Cloud-init retrieves information about the instance from the cloud provider.
2.  **User Data (Optional):**  Allows users to provide custom configuration scripts or data.
3.  **Vendor Data (Optional):** Enables vendors to provision instance-specific configurations.

Cloud-init identifies the cloud environment, reads the provided metadata, and configures the system accordingly. This can include network and storage setup, SSH key configuration, and more. It then processes any user or vendor data to further customize the instance.

## Get Help and Support

*   **User Documentation:** [https://docs.cloud-init.io/en/latest/](https://docs.cloud-init.io/en/latest/)
*   **Matrix Channel:** [``#cloud-init`` on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
*   **GitHub Discussions:** [https://github.com/canonical/cloud-init/discussions](https://github.com/canonical/cloud-init/discussions)
*   **Report Bugs:** [GitHub Issues](https://github.com/canonical/cloud-init/issues)

## Development

Contribute to the project by reviewing the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) document which outlines development, testing, and submission of code.

## Daily Builds

Stay up-to-date with the latest features and bug fixes with daily builds:

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build / Packaging Information

Reference build/packaging implementations are available in the [packages](packages) directory.