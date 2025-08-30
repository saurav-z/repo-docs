# Cloud-init: Automate Cloud Instance Initialization - The Industry Standard

Cloud-init is the leading solution for cross-platform cloud instance initialization, streamlining the process across major cloud providers and infrastructure platforms.

[Check out the original repository here](https://github.com/canonical/cloud-init).

## Key Features of Cloud-init:

*   **Cross-Platform Compatibility:** Works seamlessly across all major public cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated Initialization:** Initializes cloud instances from disk images and instance data, including cloud metadata, user data, and vendor data.
*   **Dynamic Configuration:**  Identifies the cloud environment during boot and configures the system accordingly, including network setup, storage configuration, and SSH key access.
*   **User & Vendor Data Processing:** Parses and processes optional user and vendor data provided to the instance for customized configurations.

## Getting Help and Support

Find answers and assistance through these resources:

*   **User Documentation:** Explore comprehensive [user documentation](https://docs.cloud-init.io/en/latest/) for detailed guidance.
*   **Community Support:**
    *   Connect with the community in the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com).
    *   Join the conversation and ask questions on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions).
*   **Bug Reporting:**  Report bugs and issues directly on [GitHub Issues](https://github.com/canonical/cloud-init/issues).

## Cloud and Distribution Support

Cloud-init offers broad support for a wide range of cloud providers and Linux/Unix operating systems. Find a complete list of supported:

*   **Clouds:** Check the supported [cloud providers](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported).
*   **Distributions:** Explore the supported [Linux/Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html).

## Contribute to Cloud-init Development

Interested in contributing?  Follow the steps outlined in the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) documentation for development, testing, and code submission.

## Daily Builds

Access the latest features and bug fixes with daily builds:

*   **Ubuntu:** Use the [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily).
*   **CentOS:** Explore the [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/).

## Build / Packaging Reference

Refer to [packages](packages) for build and packaging implementation examples.