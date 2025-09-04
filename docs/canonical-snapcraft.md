<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software Across Linux and IoT Devices

**Snapcraft** empowers developers to easily package, distribute, and manage their software across various Linux distributions and IoT devices.  [View the source code on GitHub](https://github.com/canonical/snapcraft).

## Key Features:

*   **Simplified Packaging:** Create software packages in the snap container format with a simple `snapcraft.yaml` configuration file.
*   **Cross-Distribution Compatibility:** Build once and run on major Linux distributions and IoT devices.
*   **Dependency Management:** Bundle all dependencies within the snap container for seamless execution.
*   **Easy Publishing:** Register and publish your snaps to public and private app stores, including the Snap Store.
*   **Command-Line Tool:** Utilize a powerful command-line interface for building, packaging, and publishing snaps.
*   **Supports Multiple Platforms:** Works on major Linux distributions, Windows, and macOS.

## Getting Started

### Initialization
Create a `snapcraft.yaml` file with:
```bash
snapcraft init
```
### Build & Package
Build a snap from your project directory:
```bash
snapcraft pack
```
### Publish
Register & Upload your snaps:
```bash
snapcraft register
snapcraft upload
```

Learn more about crafting your first snap at [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap).

## Installation

Install Snapcraft on your system using the following command for snap-enabled systems:

```bash
sudo snap install snapcraft --classic
```

Installation guides for other methods can be found in the official documentation.

## Documentation and Support

*   **Comprehensive Documentation:** Access detailed [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) for in-depth guidance.
*   **Active Community:** Engage with the Snapcraft community via the [Snapcraft Forum](https://forum.snapcraft.io) and [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Issue Tracking:** Report bugs and issues on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contribute

Snapcraft is an open-source project under the [GPL-3.0 license](LICENSE), and contributions are welcome! Review the [contribution guide](CONTRIBUTING.md) for more information.
The [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) is the hub for doc development, including Snapcraft docs.

© 2015-2025 Canonical Ltd.