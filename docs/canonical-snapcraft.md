<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software with Ease

**Snapcraft** is the command-line tool that simplifies software packaging and distribution across all major Linux distributions and IoT devices, allowing developers to create universal, self-contained packages called "snaps."  Get started with [Snapcraft on GitHub](https://github.com/canonical/snapcraft).

## Key Features

*   **Universal Packaging:** Package your app, program, toolkit, or library once and deploy it across various Linux distributions.
*   **Dependency Management:** Bundles all required libraries within the snap, solving dependency issues.
*   **Simple Configuration:** Uses a `snapcraft.yaml` file for easy project setup and management.
*   **Cross-Platform Support:**  Available on Linux, Windows, and macOS.
*   **App Store Integration:** Seamlessly register and upload your snaps to public and private app stores, including the Snap Store.
*   **Version Control & Parallel Releases:** Supports snap versioning and parallel releases for easy updates and rollbacks.

## Getting Started

### Initialize a Snap Project

Create a basic `snapcraft.yaml` file in your project directory:

```bash
snapcraft init
```

### Build Your Snap

Package your project into a snap:

```bash
snapcraft pack
```

### Upload Your Snap

Publish your snap to the Snap Store:

```bash
snapcraft upload
```

For a more detailed guide, refer to the [crafting your first snap tutorial](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap).

## Installation

Snapcraft can be installed in multiple ways:

*   **As a Snap (Recommended):**  If you're on a snap-ready system:

    ```bash
    sudo snap install snapcraft --classic
    ```
*   **As a Traditional Package:** Available in many popular Linux repositories.

Detailed installation instructions can be found in the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Documentation

Comprehensive documentation, including tutorials and a command reference, is available at the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable).

## Community and Support

Join the vibrant Snapcraft community and get help:

*   **Snapcraft Forum:** [https://forum.snapcraft.io](https://forum.snapcraft.io)
*   **Snapcraft Matrix channel:** [https://matrix.to/#/#snapcraft:ubuntu.com](https://matrix.to/#/#snapcraft:ubuntu.com)
*   **Report Issues:** [GitHub repository](https://github.com/canonical/snapcraft/issues)

## Contribute

Contribute to Snapcraft and help improve the project:

*   **Contribution Guide:** [CONTRIBUTING.md](CONTRIBUTING.md)
*   **Documentation:**  Help improve the documentation through the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy).

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.