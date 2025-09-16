<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Software with Ease

**Snapcraft** is the command-line tool that simplifies software packaging and distribution across major Linux distributions and IoT devices, making your software accessible everywhere.  [Learn more at the original repository](https://github.com/canonical/snapcraft).

## Key Features

*   **Cross-Platform Compatibility:** Package your applications for all major Linux distributions, Windows, and macOS.
*   **Simplified Dependency Management:**  Snapcraft bundles all dependencies within the container, ensuring consistent execution.
*   **Easy Build Process:**  Use `snapcraft.yaml` project files to define your application's build and runtime configurations.
*   **Seamless Distribution:**  Publish your snaps to public and private app stores, including the Snap Store.
*   **Parallel Releases:**  Support for snap versions and revisions, including parallel releases.

## Getting Started

1.  **Initialize Your Project:**  Create a basic `snapcraft.yaml` file with:

    ```bash
    snapcraft init
    ```

2.  **Define Build & Runtime Details:** Add your project's specifications to the `snapcraft.yaml` file.
3.  **Package Your Application:** Create a snap package with:

    ```bash
    snapcraft pack
    ```

4.  **Register and Publish:** Register your project and publish snaps to app stores:

    ```bash
    snapcraft register
    snapcraft upload
    ```

## Installation

Snapcraft is primarily available as a snap.

To install Snapcraft:

```bash
sudo snap install snapcraft --classic
```

Additional Linux container tools may also be required.  Refer to the official documentation for detailed setup instructions.

## Resources

*   **Documentation:**  Comprehensive [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) provides detailed guidance.
*   **Community:**  Engage with the Snapcraft community on the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Issue Tracking:** Report issues and contribute to the project via the [GitHub repository](https://github.com/canonical/snapcraft/issues).
*   **Contribution:** Contribute to Snapcraft development by following the [contribution guide](CONTRIBUTING.md).
*   **Documentation Academy:**  Contribute to the docs with the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy).

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.