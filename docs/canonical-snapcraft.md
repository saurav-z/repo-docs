<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software with Ease

**Snapcraft empowers developers to effortlessly package, distribute, and manage software across all major Linux distributions and IoT devices.**

[Original Repository](https://github.com/canonical/snapcraft)

## Key Features

*   **Universal Packaging:** Create snaps, a container format that bundles all dependencies, ensuring consistent performance across different Linux distributions.
*   **Simplified Dependency Management:**  Eliminate dependency conflicts by including all required libraries within the snap.
*   **Cross-Platform Compatibility:**  Build once and run on multiple Linux distributions and IoT devices.
*   **Easy-to-Use Project Files:**  Define your build configuration using the `snapcraft.yaml` file, making it simple to integrate into your existing workflow.
*   **Seamless Distribution:**  Publish your snaps to public or private app stores, including the Snap Store.
*   **Command-Line Interface:** Provides a suite of CLI commands to initialize, build, pack, register, and upload your snaps.

## Getting Started

1.  **Initialize:** Create a basic `snapcraft.yaml` file in your project's root directory with `snapcraft init`.
2.  **Configure:**  Add your project's build and runtime details to the `snapcraft.yaml` file.
3.  **Build:** Package your project into a snap with `snapcraft pack`.
4.  **Distribute:** Register and upload your snap to the Snap Store or other app stores using `snapcraft register` and `snapcraft upload`.

For a more detailed tutorial, check out [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap).

## Installation

Snapcraft is easily installed on various platforms:

*   **Snap-Enabled Systems:**  Install Snapcraft via the command line: `sudo snap install snapcraft --classic`.
*   **Traditional Packages:** Available through many popular Linux repositories.

Refer to the [setup documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft) for complete installation instructions.

## Resources

*   **Documentation:** Comprehensive [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) for building, debugging, resolving interfaces, and command references.
*   **Community:**  Engage with the community on the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com) for support and discussions.
*   **Issues:** Report bugs and issues on the [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contribute

Snapcraft is an open-source project under the [GPL-3.0 license](LICENSE).  We welcome contributions! Check out the [contribution guide](CONTRIBUTING.md) to get started.  The [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) is the hub for doc development.

© 2015-2025 Canonical Ltd.