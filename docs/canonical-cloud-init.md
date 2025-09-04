# Cloud-init: Automate Your Cloud Instance Initialization

**Cloud-init is the industry-standard solution for automating the initialization of your cloud instances across all major platforms.** (See the original repository [here](https://github.com/canonical/cloud-init).)

![Unit Tests](https://github.com/canonical/cloud-init/actions/workflows/unit.yml/badge.svg?branch=main)
![Integration Tests](https://github.com/canonical/cloud-init/actions/workflows/integration.yml/badge.svg?branch=main)
![Documentation](https://github.com/canonical/cloud-init/actions/workflows/check_format.yml/badge.svg?branch=main)

## Key Features of Cloud-init:

*   **Cross-Platform Support:** Works seamlessly across major public cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated Initialization:** Initializes cloud instances from a disk image and instance data, including cloud metadata, user data, and vendor data.
*   **Dynamic Configuration:** Identifies the cloud environment during boot and configures the system accordingly, setting up network, storage, SSH keys, and more.
*   **User & Vendor Data Processing:** Processes optional user and vendor data passed to the instance, enabling further customization.
*   **Wide Distribution and Cloud Support:** Cloud-init is supported by the majority of clouds and Linux/Unix operating systems.

## Getting Started & Support

*   **User Documentation:** [https://docs.cloud-init.io/en/latest/](https://docs.cloud-init.io/en/latest/)
*   **Community Support:**
    *   Matrix Channel: [``#cloud-init`` on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
    *   GitHub Discussions: [https://github.com/canonical/cloud-init/discussions](https://github.com/canonical/cloud-init/discussions)
    *   Bug Reports: [GitHub Issues](https://github.com/canonical/cloud-init/issues)

## Development & Contribution

*   **Contributing Guide:** [https://docs.cloud-init.io/en/latest/development/index.html](https://docs.cloud-init.io/en/latest/development/index.html) outlines steps to develop, test, and submit code.

## Daily Builds

*   **Ubuntu Daily PPAs:** [https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS COPR Build Repos:** [https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build/Packaging Reference

*   **Packages:** [packages](packages)