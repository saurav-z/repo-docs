<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software with Ease

**Snapcraft** is a powerful command-line tool that simplifies the packaging and distribution of software across various Linux distributions and IoT devices, providing a unified solution for developers.  [Learn more on GitHub](https://github.com/canonical/snapcraft).

## Key Features

*   **Cross-Platform Compatibility:** Package your applications for all major Linux distributions, Windows, and macOS.
*   **Simplified Dependency Management:**  Snapcraft bundles all required libraries and dependencies within a single, self-contained snap package.
*   **Easy Packaging:**  Uses a simple `snapcraft.yaml` project file for easy integration into existing codebases.
*   **Streamlined Distribution:**  Supports publishing to public and private app stores, including the Snap Store.
*   **Version Control & Releases:** Enables easy management of snap versions and revisions, including parallel releases.

## Getting Started

1.  **Initialize a Project:** Start by creating a basic `snapcraft.yaml` file:

    ```bash
    snapcraft init
    ```

2.  **Customize Your Build:** Add your project's build and runtime details to the `snapcraft.yaml` file.
3.  **Package Your Application:** Bundle your project into a snap with:

    ```bash
    snapcraft pack
    ```

4.  **Publish Your Snap:** Register and upload your snap to app stores:

    ```bash
    snapcraft register
    snapcraft upload
    ```

For a more detailed guide on creating your first snap, see the [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) tutorial.

## Installation

Snapcraft can be installed on various systems.

To install Snapcraft on a snap-ready system using the command line:

```bash
sudo snap install snapcraft --classic
```

Complete installation requires a Linux container tool.  Consult the [setup documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft) for detailed instructions.

## Documentation and Support

*   **Comprehensive Documentation:**  The [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) provides in-depth guidance, tutorials, and command references.
*   **Community Forum:** Engage with other Snapcraft users and get support on the [Snapcraft Forum](https://forum.snapcraft.io).
*   **Matrix Channel:** Join the conversation on the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Report Issues:**  Report bugs or issues on the [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contribute

Snapcraft is an open-source project, and contributions are welcome!  See the [contribution guide](CONTRIBUTING.md) for details.

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.