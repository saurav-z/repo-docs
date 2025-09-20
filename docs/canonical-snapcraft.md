<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software for Linux and IoT

**Snapcraft** is a powerful command-line tool that simplifies the process of packaging and distributing software as snaps for seamless deployment across various Linux distributions and IoT devices.  [Learn more and contribute on the original repo](https://github.com/canonical/snapcraft).

## Key Features of Snapcraft:

*   **Simplified Packaging:** Create application packages with the snap container format for easy installation and management.
*   **Cross-Distribution Compatibility:** Build once and deploy your software across major Linux distributions.
*   **Dependency Management:**  Snapcraft bundles all required dependencies within the snap, eliminating compatibility issues.
*   **Architecture Support:**  Package your software to support a wide range of hardware architectures.
*   **Easy to Use:** Utilizes a simple `snapcraft.yaml` project file for streamlined configuration.
*   **Integration with Snap Store:** Easily publish and manage your snaps on public and private app stores, including the Snap Store.

## Getting Started with Snapcraft

### Initialize Your Project:

Start by creating a `snapcraft.yaml` file with:

```bash
snapcraft init
```

### Build Your Snap:

Package your project into a snap:

```bash
snapcraft pack
```

### Publish Your Snap:

Register and upload your snaps to the Snap Store:

```bash
snapcraft register
snapcraft upload
```

## Installation

Snapcraft can be easily installed on multiple platforms.

**Install Snapcraft via Snap (Recommended):**

```bash
sudo snap install snapcraft --classic
```

**Other Installation Methods:**

*   Snapcraft is available on most major Linux distributions and can be installed via their package managers.
*   See the official documentation for detailed installation instructions.

## Documentation and Resources

*   **Comprehensive Documentation:**  Explore the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) for detailed guides, tutorials, and command references.
*   **Community Support:** Engage with the Snapcraft community on the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com) to ask questions and collaborate.
*   **Issue Tracking:** Report bugs and issues on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contribute

Snapcraft is an open-source project and welcomes contributions from the community.  Review the [contribution guide](CONTRIBUTING.md) to get started.

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.