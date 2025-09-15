<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software Seamlessly

**Snapcraft** empowers developers to package and distribute their software across major Linux distributions and IoT devices, simplifying dependency management and streamlining the build process.  For more details, visit the [Snapcraft GitHub repository](https://github.com/canonical/snapcraft).

## Key Features & Benefits

*   **Universal Packaging:** Create packages that run on all major Linux distributions and IoT devices.
*   **Simplified Dependency Management:**  Bundles all necessary libraries and dependencies within the snap container.
*   **Easy to Use:** Define your build configuration with a simple `snapcraft.yaml` project file.
*   **Cross-Platform Compatibility:** Supported on Linux, Windows, and macOS.
*   **Seamless Distribution:** Publish your snaps to public and private app stores, including the Snap Store.

## Getting Started

### 1. Initialization

Start a new project by creating a minimal `snapcraft.yaml` configuration file:

```bash
snapcraft init
```

### 2. Build Your Snap

Once you've configured your project details in `snapcraft.yaml`, build your snap with:

```bash
snapcraft pack
```

### 3. Register and Publish

Register your project and upload your snap to app stores:

```bash
snapcraft register
snapcraft upload
```

For a detailed tutorial on crafting your first snap, refer to the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap).

## Installation

Snapcraft is available on various platforms.

*   **Snap-enabled systems:**

    ```bash
    sudo snap install snapcraft --classic
    ```
*   **Traditional package:** Install through a Linux repository.

For detailed setup instructions, see the [Snapcraft setup documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Resources

*   **Documentation:**  Comprehensive [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) provides in-depth guidance, tutorials, and command references.
*   **Community Support:** Connect with the Snapcraft community through the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Issue Reporting:** Report any issues or bugs on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).
*   **Contribution:** Contribute to Snapcraft by following the [contribution guide](CONTRIBUTING.md) and participating in the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy).

## License

Snapcraft is licensed under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.