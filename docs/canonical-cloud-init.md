# Cloud-init: Automate Cloud Instance Initialization (Industry Standard)

**Cloud-init simplifies and automates the initialization of cloud instances across various platforms.** This open-source tool streamlines the setup of your cloud environments.

[View the original repository on GitHub](https://github.com/canonical/cloud-init)

Cloud-init is the **industry-leading** multi-distribution method for cross-platform cloud instance initialization. It's supported by all major public cloud providers, private cloud provisioning systems, and bare-metal installations. Cloud instances are initialized using disk images and data provided by the cloud, including:

*   Cloud metadata
*   User data (optional)
*   Vendor data (optional)

Cloud-init automatically identifies the cloud environment during boot, reads the provided metadata, and configures the system. This may involve tasks such as configuring network and storage devices, setting up SSH access, and more. Cloud-init then processes any user or vendor data provided to the instance.

## Key Features and Benefits:

*   **Cross-Platform Compatibility:** Works seamlessly across various cloud providers and Linux distributions.
*   **Automated Configuration:** Automates system setup tasks, saving time and reducing manual configuration.
*   **Metadata-Driven Initialization:** Leverages cloud metadata to customize the instance based on the specific environment.
*   **User and Vendor Data Processing:** Handles user data and vendor data to further personalize the instance.
*   **Industry Standard:** Widely adopted and supported, ensuring compatibility and ongoing development.

## Getting Help and Support

*   **User Documentation:** Start with the comprehensive [user documentation](https://docs.cloud-init.io/en/latest/).
*   **Matrix Channel:** Ask questions and engage with the community in the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com).
*   **GitHub Discussions:** Follow announcements and participate in discussions on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions).
*   **Bug Reporting:** Report any bugs you find on [GitHub Issues](https://github.com/canonical/cloud-init/issues).

## Supported Distributions and Clouds

Cloud-init supports a wide range of [clouds](https://docs.cloud-init.io/en/latest/reference/datasources.html#datasources_supported) and [Linux/Unix OSes](https://docs.cloud-init.io/en/latest/reference/distros.html). If your distribution or cloud is not supported, please contact the distribution and suggest they reach out!

## Contributing to Cloud-init

*   Refer to the [contributing](https://docs.cloud-init.io/en/latest/development/index.html) document to learn how to develop, test, and submit code.

## Daily Builds

*   **Ubuntu:** [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
*   **CentOS:** [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)

## Build and Packaging

*   Refer to the [packages](packages) directory for build and packaging implementations.