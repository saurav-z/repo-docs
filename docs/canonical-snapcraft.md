<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Build, Package, and Distribute Your Software Across Linux & IoT Devices

**Snapcraft** is a powerful command-line tool that simplifies the process of packaging and distributing your software as snaps, ensuring compatibility and ease of installation across various Linux distributions and IoT devices. [Learn more and contribute on GitHub!](https://github.com/canonical/snapcraft)

## Key Features

*   **Cross-Distribution Compatibility:** Package your software once and deploy it on all major Linux distributions.
*   **Dependency Management:** Bundle all dependencies within the snap, eliminating dependency conflicts.
*   **Simple Configuration:**  Use a `snapcraft.yaml` file to define your project's build and runtime details.
*   **Seamless Distribution:** Publish your snaps to public and private app stores, including the Snap Store.
*   **Easy Installation:** Simple installation with `snap install snapcraft --classic` on snap-enabled systems.
*   **IoT Support:**  Package and deploy your applications on IoT devices.

## Getting Started

Get up and running with Snapcraft using these basic commands:

1.  **Initialize your project:**
    ```bash
    snapcraft init
    ```
2.  **Package your software into a snap:**
    ```bash
    snapcraft pack
    ```
3.  **Register your project on the Snap Store:**
    ```bash
    snapcraft register
    ```
4.  **Upload and release your snap:**
    ```bash
    snapcraft upload
    ```

For a more in-depth guide, explore the [Crafting Your First Snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) tutorial.

## Installation

Snapcraft is readily available across various platforms:

*   **Snap-enabled Linux systems:** Install using `sudo snap install snapcraft --classic`.
*   **Traditional package management:**  Available through various Linux repositories.
*   **Windows & macOS:** Supported as well.

Refer to the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft) for complete setup instructions.

## Resources

*   **Documentation:** [Snapcraft Documentation](https://documentation.ubuntu.com/snapcraft/stable) - Comprehensive guidance on building, debugging, and distributing snaps.
*   **Community Forum:** [Snapcraft Forum](https://forum.snapcraft.io) - Ask questions and connect with other Snapcraft users.
*   **Matrix Channel:** [Snapcraft Matrix Channel](https://matrix.to/#/#snapcraft:ubuntu.com) - Join the real-time discussion.
*   **Issue Tracker:** [GitHub Repository](https://github.com/canonical/snapcraft/issues) - Report bugs and issues.
*   **Contribution Guide:** [CONTRIBUTING.md](CONTRIBUTING.md) - Learn how to contribute to the project.
*   **Documentation Academy:** [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) - Contribute to the Snapcraft documentation.

## License and Copyright

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.