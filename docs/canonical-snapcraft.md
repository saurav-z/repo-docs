<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software for Any Linux Distribution

**Snapcraft** empowers developers to easily package and distribute their software across all major Linux distributions and IoT devices in a secure, dependency-managed container format called snaps. [See the Snapcraft project on GitHub](https://github.com/canonical/snapcraft).

## Key Features of Snapcraft

*   **Simplified Packaging:** Easily package any application, program, toolkit, or library for all major Linux distributions and IoT devices.
*   **Dependency Management:** Bundle all dependencies within the snap, ensuring consistent execution across different environments.
*   **Architecture Support:** Build once, run everywhere - snaps support multiple architectures.
*   **Cross-Distribution Compatibility:** Publish your software for all major Linux distros.
*   **Easy to Use:** Create a `snapcraft.yaml` configuration file to define your project and build process.
*   **Store Integration:** Seamlessly publish and manage your snaps on public and private app stores, including the Snap Store.

## Getting Started with Snapcraft

### Basic Usage:

1.  **Initialize:** Create a basic `snapcraft.yaml` file with:

    ```bash
    snapcraft init
    ```

2.  **Configure:** Add your project's build and runtime details to the `snapcraft.yaml` file.
3.  **Build:** Package your project into a snap with:

    ```bash
    snapcraft pack
    ```

4.  **Publish:** Register and upload your snap to the Snap Store:

    ```bash
    snapcraft register
    snapcraft upload
    ```

### Installation

Snapcraft is readily available for most major Linux distributions, Windows, and macOS.

*   **Recommended Installation (Linux):** Install Snapcraft as a snap:

    ```bash
    sudo snap install snapcraft --classic
    ```

*   **Alternative Installation:**  Install as a traditional package (consult your distribution's documentation for instructions).
*   **Setup Documentation**: [Set up Snapcraft](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft)

## Resources and Support

*   **Documentation:**  Comprehensive [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) covers all aspects of building snaps.
*   **Community:**  Engage with the Snapcraft community on the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Issue Tracking:** Report bugs and issues on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).
*   **Contribution:** The [contribution guide](CONTRIBUTING.md) and [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) welcome your contributions.

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.