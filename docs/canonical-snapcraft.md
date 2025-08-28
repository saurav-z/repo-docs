<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software with Ease

**Snapcraft** is a powerful command-line tool that simplifies software packaging and distribution across major Linux distributions and IoT devices.  [View the original repository](https://github.com/canonical/snapcraft)

## Key Features of Snapcraft:

*   **Simplified Packaging:** Bundle your software and all its dependencies into a single, portable snap package.
*   **Cross-Distribution Compatibility:** Package your applications once and deploy them on various Linux distributions and IoT devices.
*   **Dependency Management:**  Snapcraft handles dependencies, ensuring your software runs smoothly on different systems.
*   **Easy Build Process:** Uses a simple `snapcraft.yaml` file for configuration, making it easy to integrate into your existing workflows.
*   **App Store Integration:** Seamlessly publish and distribute your snaps to public and private app stores, including the Snap Store.
*   **Version Control and Releases:**  Manage snap versions and releases, including parallel releases.

## Getting Started

1.  **Initialize your project:**  Create a basic `snapcraft.yaml` file with:

    ```bash
    snapcraft init
    ```

2.  **Configure your project:**  Edit `snapcraft.yaml` to include build and runtime details.
3.  **Build your snap:** Package your project into a snap:

    ```bash
    snapcraft pack
    ```

4.  **Publish your snap:** Register and upload your snap to app stores, such as the Snap Store:

    ```bash
    snapcraft register
    snapcraft upload
    ```
    Learn more by [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap).

## Installation

Snapcraft is readily available on major Linux distributions, Windows, and macOS.

To install Snapcraft on snap-ready systems:

```bash
sudo snap install snapcraft --classic
```

For complete installation, you'll also need a Linux container tool.  Find detailed setup instructions in the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Resources

*   **Documentation:** Comprehensive [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) provides guidance on building projects, debugging, and resolving issues.
*   **Community Support:**  Engage with the community on the [Snapcraft Forum](https://forum.snapcraft.io) and [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com) to get support.
*   **Issue Reporting:** Report bugs and issues on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).
*   **Contribution:** Contribute to Snapcraft by following the [contribution guide](CONTRIBUTING.md) and the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy).

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.