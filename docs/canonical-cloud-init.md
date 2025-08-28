# Cloud-init: Automate Cloud Instance Initialization

**Cloud-init simplifies and standardizes the process of configuring cloud instances across various platforms, making cloud deployments easier than ever.**

[View the original repository on GitHub](https://github.com/canonical/cloud-init)

Cloud-init is the industry-leading, cross-platform solution for initializing cloud instances, supporting a wide range of public and private cloud providers, as well as bare-metal installations. This powerful tool automates the initial setup of your cloud instances, allowing you to quickly deploy and configure your infrastructure.

## Key Features of Cloud-init:

*   **Cross-Platform Compatibility:** Supports all major public cloud providers, private cloud provisioning systems, and bare-metal installations.
*   **Automated Initialization:** Reads instance metadata, user data, and vendor data to configure network settings, storage devices, SSH keys, and more.
*   **Multi-Distribution Support:** Compatible with most Linux and Unix distributions.
*   **Flexible Configuration:** Processes cloud metadata, optional user data, and vendor data to customize your instance.
*   **Industry Standard:** The most widely used and trusted method for cloud instance initialization.

## Getting Started and Support:

*   **Comprehensive Documentation:** Explore the [user documentation](https://docs.cloud-init.io/en/latest/) for detailed guidance.
*   **Community Support:** Get help and connect with other users through:
    *   The [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
    *   [GitHub Discussions](https://github.com/canonical/cloud-init/discussions)
    *   [Report bugs on GitHub Issues](https://github.com/canonical/cloud-init/issues)

## Cloud and Distribution Support:

Cloud-init offers broad support for various [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux / Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html). If you need support for a specific cloud or distribution, contact the respective distribution and suggest they use cloud-init.

## Contributing:

If you are interested in contributing to cloud-init, please read the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) document that outlines the steps necessary to develop, test, and submit code.

## Daily Builds:

Try the latest upstream code by using the daily builds:

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build/Packaging:

Refer to [packages](packages) for reference build/packaging implementations.