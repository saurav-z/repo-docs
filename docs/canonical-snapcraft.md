<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: The Ultimate Tool for Packaging and Distributing Linux Applications

**Snapcraft** simplifies software packaging and distribution across all major Linux distributions and IoT devices, making application deployment a breeze.  [Learn more on GitHub](https://github.com/canonical/snapcraft).

## Key Features of Snapcraft

*   **Cross-Platform Compatibility:** Package your applications for all major Linux distributions and IoT devices.
*   **Simplified Dependency Management:** Bundles all dependencies within the snap, ensuring consistent behavior across different systems.
*   **Easy Packaging with `snapcraft.yaml`:** Define your application's build and runtime configuration in a simple, easy-to-understand file.
*   **Seamless Distribution via the Snap Store:** Publish your applications to public and private app stores, including the Snap Store.
*   **Version Control & Parallel Releases:** Manage and publish different versions of your snaps with ease.

## Getting Started with Snapcraft

### Initialize Your Project

Create a basic `snapcraft.yaml` file in your project's root directory:

```bash
snapcraft init
```

### Build Your Snap

Package your application into a snap:

```bash
snapcraft pack
```

### Register and Upload

Publish your snap to the Snap Store:

```bash
snapcraft register
snapcraft upload
```

For detailed instructions, refer to the [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) tutorial.

## Installation

Snapcraft can be installed on various platforms, including all major Linux distributions, Windows, and macOS.

### Install via Snap (Recommended)

If you have a snap-ready system, the easiest way to install Snapcraft is using:

```bash
sudo snap install snapcraft --classic
```

### Alternative Installation Methods

Snapcraft is also available as a traditional package on many Linux repositories. For detailed installation instructions, see the [Snapcraft setup documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Documentation and Resources

*   **Comprehensive Documentation:**  [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) provides in-depth guidance on building snaps, debugging, command references, and more.
*   **Community Support:** Engage with the Snapcraft community on the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Report Issues:** Submit bug reports or issues on the [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contribute to Snapcraft

Snapcraft is an open-source project.  Contributions are welcome!

*   Explore the [contribution guide](CONTRIBUTING.md) to get started.
*   Join the community on the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) to assist with documentation.

## License

Snapcraft is licensed under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.