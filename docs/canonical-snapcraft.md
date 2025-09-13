<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: The Ultimate Tool for Packaging and Distributing Linux Software

**Snapcraft** is a powerful command-line tool designed to simplify the packaging and distribution of software applications across various Linux distributions and IoT devices. [Learn more on GitHub](https://github.com/canonical/snapcraft).

## Key Features

*   **Cross-Distribution Compatibility:** Package your software once and deploy it on all major Linux distributions.
*   **Dependency Management:**  Bundles all dependencies within the snap, ensuring consistent software behavior regardless of the host system.
*   **Easy to Use:**  Build configuration is stored in a simple `snapcraft.yaml` file.
*   **Simplified Build Process:** Streamlined commands for initializing, building, packaging, and uploading your snaps.
*   **Integration with Snap Store:** Seamlessly register and upload your snaps to public and private app stores, including the Snap Store.
*   **Supports Parallel Releases:** Manage and publish different versions and revisions of your software easily.

## Getting Started with Snapcraft

Creating a snap is straightforward:

1.  **Initialize:** Create a basic `snapcraft.yaml` file:
    ```bash
    snapcraft init
    ```
2.  **Configure:** Add your project's build and runtime details to `snapcraft.yaml`.
3.  **Build:** Package your project into a snap:
    ```bash
    snapcraft pack
    ```
4.  **Register and Upload:** Publish your snap to app stores:
    ```bash
    snapcraft register
    snapcraft upload
    ```

For a more in-depth guide, explore the [Crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) tutorial.

## Installation

Snapcraft is available on most Linux distributions, Windows, and macOS. Installation is easy with:

```bash
sudo snap install snapcraft --classic
```

Comprehensive setup instructions and alternative installation methods are available in the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Resources

*   **Documentation:** The comprehensive [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) offers detailed guidance on all aspects of snap creation, including building project files, debugging, and command references.
*   **Community:** Engage with other snap developers in the [Snapcraft Forum](https://forum.snapcraft.io) and on the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Report Issues:**  Report bugs and issues directly on the [GitHub repository](https://github.com/canonical/snapcraft/issues).
*   **Contribute:**  Contribute to the project via the [contribution guide](CONTRIBUTING.md).

## License and Copyright

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.