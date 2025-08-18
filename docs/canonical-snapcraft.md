<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software with Ease

**Snapcraft** simplifies software packaging and distribution for all major Linux distributions and IoT devices, making your software accessible to a wider audience.  For more details, visit the [Snapcraft GitHub Repository](https://github.com/canonical/snapcraft).

## Key Features

*   **Cross-Platform Compatibility:** Package your applications for all major Linux distributions, Windows, and macOS.
*   **Dependency Management:** Bundle all necessary libraries and dependencies within a single, self-contained snap.
*   **Simplified Packaging:** Create a `snapcraft.yaml` file to define your project's build and runtime details.
*   **Easy Distribution:**  Publish your snaps to public and private app stores, including the Snap Store.
*   **Command-Line Interface:**  Utilize straightforward commands for initialization, packaging, and uploading your software.

## Getting Started

### Installation

Snapcraft is readily available across a wide range of operating systems.

**Installation via Snap (recommended):**

```bash
sudo snap install snapcraft --classic
```

**Alternative Installation Methods:**

Snapcraft can also be installed via traditional package managers on various Linux distributions.  Refer to the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft) for detailed setup instructions.

### Basic Usage

1.  **Initialize:** Create a basic `snapcraft.yaml` file in your project's root directory:

    ```bash
    snapcraft init
    ```

2.  **Configure:**  Edit `snapcraft.yaml` to define your project's build and runtime configurations.

3.  **Package:** Build your snap:

    ```bash
    snapcraft pack
    ```

4.  **Register & Upload:**  Publish your snap to app stores, including the Snap Store:

    ```bash
    snapcraft register
    snapcraft upload
    ```

### Further Exploration

Discover how to create your first snap with the tutorial: [Crafting Your First Snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap).

## Resources

*   **Documentation:**  Comprehensive guidance is available in the [Snapcraft Documentation](https://documentation.ubuntu.com/snapcraft/stable).
*   **Community:**  Engage with the Snapcraft community on the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Issue Tracking:** Report bugs and issues on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contribute

Contribute to Snapcraft through the [contribution guide](CONTRIBUTING.md). The [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) welcomes suggestions and help with documentation.

## License

Snapcraft is licensed under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.