<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software Across Linux and IoT Devices

**Snapcraft** is a powerful command-line tool for packaging and distributing software as snaps, making it easy to deploy your applications across various Linux distributions and IoT devices. [Explore the original repository on GitHub](https://github.com/canonical/snapcraft).

## Key Features

*   **Cross-Distribution Compatibility:** Build once, run on all major Linux distributions, simplifying software deployment.
*   **Dependency Management:**  Snapcraft bundles all necessary dependencies within the snap, ensuring consistent runtime environments.
*   **Easy Packaging:** Uses a simple `snapcraft.yaml` configuration file for straightforward project setup.
*   **App Store Integration:** Publish your snaps to public and private app stores, including the Snap Store.
*   **Simple Commands:**  Initiate, pack, register, and upload your snaps with intuitive commands.
*   **IoT Device Support:**  Package software for IoT devices, expanding your software's reach.

## Getting Started with Snapcraft

### Installation

Snapcraft is easy to install on multiple platforms, including Linux, Windows, and macOS.

To install Snapcraft on snap-ready systems:

```bash
sudo snap install snapcraft --classic
```

Detailed installation instructions and setup guides are available in the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

### Basic Usage

1.  **Initialize your project:**
    ```bash
    snapcraft init
    ```
    This generates a `snapcraft.yaml` file for your project configuration.

2.  **Define Build and Runtime Details:** Customize your `snapcraft.yaml` file with your project's build and runtime requirements.

3.  **Package your software:**
    ```bash
    snapcraft pack
    ```
    This command bundles your project into a snap package.

4.  **Register and Publish:** Register your project on the Snap Store and upload releases:
    ```bash
    snapcraft register
    snapcraft upload
    ```

## Resources

*   **Documentation:** Comprehensive guidance on building, debugging, and publishing snaps can be found in the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable).
*   **Community:** Engage with other Snapcraft users and developers in the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Contributing:** Contribute to Snapcraft's development through the [GitHub repository](https://github.com/canonical/snapcraft/issues) and the [contribution guide](CONTRIBUTING.md).

## License and Copyright

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.