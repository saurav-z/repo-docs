<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software with Ease

**Snapcraft** is a powerful command-line tool that simplifies packaging and distribution of your software across major Linux distributions and IoT devices, making deployment a breeze. [Explore the original repository](https://github.com/canonical/snapcraft).

## Key Features of Snapcraft:

*   **Cross-Platform Compatibility:** Package your applications for all major Linux distributions, Windows, and macOS.
*   **Simplified Dependency Management:** Bundles all software dependencies into a single container, ensuring consistent execution.
*   **Easy to Use:** Create and manage software packages with a simple `snapcraft.yaml` configuration file.
*   **Seamless Distribution:** Publish your snaps to public and private app stores, including the Snap Store.
*   **Comprehensive Documentation:** Access extensive guides and learning resources to master Snapcraft.

## Getting Started

### Installation

Install Snapcraft on snap-ready systems with:

```bash
sudo snap install snapcraft --classic
```

Detailed setup instructions are available in the [documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

### Basic Usage

1.  **Initialize your project:** Create a `snapcraft.yaml` file.
    ```bash
    snapcraft init
    ```
2.  **Bundle your project:** Package your application into a snap.
    ```bash
    snapcraft pack
    ```
3.  **Register and Upload:** Publish your snap to app stores.
    ```bash
    snapcraft register
    snapcraft upload
    ```

## Documentation and Support

*   **Extensive Documentation:** Dive into the comprehensive [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) for detailed guidance.
*   **Active Community:** Engage with the Snapcraft community on the [Snapcraft Forum](https://forum.snapcraft.io) and [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com) for support and discussions.
*   **Report Issues:** Report bugs and issues on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contribute

Snapcraft is an open-source project. Learn how to contribute by reviewing the [contribution guide](CONTRIBUTING.md).
The [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) is the hub for doc development, including Snapcraft docs.

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.