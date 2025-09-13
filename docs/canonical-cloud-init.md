# cloud-init: Automate Cloud Instance Initialization

**Cloud-init simplifies and automates the process of initializing cloud instances across various platforms, making it the industry standard.**

[View the original repository on GitHub](https://github.com/canonical/cloud-init)

**Key Features of cloud-init:**

*   **Cross-Platform Compatibility:** Supports major public cloud providers, private cloud infrastructure, and bare-metal installations.
*   **Automated Initialization:** Initializes cloud instances from a disk image and instance data, automating system setup.
*   **Metadata-Driven Configuration:** Reads cloud metadata to identify the cloud environment and initialize the system accordingly.
*   **User and Vendor Data Processing:** Processes optional user and vendor data for further customization.
*   **Networking and Storage Configuration:** Sets up network and storage devices during initialization.
*   **SSH Key Configuration:** Configures SSH access keys for secure access.
*   **Wide Distribution and Cloud Support:** Compatible with a vast majority of Linux/Unix OSes and cloud providers.

**How cloud-init Works:**

Cloud instances are initialized using a disk image along with instance data, including:

*   Cloud metadata
*   User data (optional)
*   Vendor data (optional)

Cloud-init identifies the cloud environment during boot, reads the metadata, and configures the system, including network setup, storage configuration, and SSH access key setup. Subsequently, it processes optional user or vendor data.

**Get Support:**

*   **Documentation:** Start with the comprehensive [user documentation](https://docs.cloud-init.io/en/latest/).
*   **Community:**
    *   Ask questions in the [``#cloud-init`` channel on Matrix](https://matrix.to/#/#cloud-init:ubuntu.com)
    *   Follow announcements or ask questions on [GitHub Discussions](https://github.com/canonical/cloud-init/discussions)
*   **Report Issues:** [Report bugs on GitHub Issues](https://github.com/canonical/cloud-init/issues)

**Development and Contributions:**

*   Learn how to [contribute](https://docs.cloud-init.io/en/latest/development/index.html) to cloud-init.
*   **Daily Builds:** Access the latest upstream code for features and bug fixes:
    *   Ubuntu: [Daily PPAs](https://code.launchpad.net/~cloud-init-dev/+archive/ubuntu/daily)
    *   CentOS: [COPR build repos](https://copr.fedorainfracloud.org/coprs/g/cloud-init/cloud-init-dev/)
*   **Build/Packaging:** Refer to [packages](packages) for reference build/packaging implementations.