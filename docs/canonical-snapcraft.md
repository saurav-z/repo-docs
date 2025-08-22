<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software Across Linux, IoT, and Beyond

Snapcraft empowers developers to easily package and distribute their software applications, libraries, and toolkits across all major Linux distributions and IoT devices, simplifying dependency management and architecture support.  [See the original repo on GitHub](https://github.com/canonical/snapcraft).

## Key Features of Snapcraft

*   **Simplified Packaging:** Create application packages in the snap container format using a simple `snapcraft.yaml` file.
*   **Cross-Distribution Compatibility:** Build once and run on all major Linux distributions, eliminating compatibility issues.
*   **Dependency Management:**  Bundles all necessary libraries and dependencies within the snap container for seamless operation.
*   **App Store Integration:** Easily register and publish your snaps to public and private app stores, including the Snap Store.
*   **Command-Line Interface:**  Utilize intuitive commands for initialization, building, packing, registering, and uploading your snaps.
*   **Wide Platform Support:** Supports building and running snaps on various platforms, including Linux, Windows, and macOS.

## Getting Started with Snapcraft

### Installation

Install Snapcraft on your system using the following command (requires a snap-enabled system):

```bash
sudo snap install snapcraft --classic
```

Complete installation may require additional setup, as detailed in the [Snapcraft Documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

### Basic Usage

1.  **Initialize:** Create a basic `snapcraft.yaml` file in your project directory:
    ```bash
    snapcraft init
    ```
2.  **Configure:** Add your project's build and runtime details to the `snapcraft.yaml` file.
3.  **Build:** Package your project into a snap:
    ```bash
    snapcraft pack
    ```
4.  **Publish:**  Register and upload your snap to app stores like the Snap Store:
    ```bash
    snapcraft register
    snapcraft upload
    ```

## Resources

*   **Documentation:** Comprehensive documentation is available at [Snapcraft Documentation](https://documentation.ubuntu.com/snapcraft/stable).
*   **Community:** Engage with the Snapcraft community in the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com) for support and discussions.
*   **Report Issues:** Report bugs and issues on the [GitHub repository](https://github.com/canonical/snapcraft/issues).
*   **Contribute:** Contribute to the project by following the [contribution guide](CONTRIBUTING.md) or the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy).

## License and Copyright

Snapcraft is licensed under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.