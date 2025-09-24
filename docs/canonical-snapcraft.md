<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Build, Package, and Distribute Your Software with Ease

**Snapcraft** is your go-to command-line tool for creating and distributing software packages across major Linux distributions and IoT devices, streamlining the development and deployment process. ([See the original repository](https://github.com/canonical/snapcraft))

## Key Features

*   **Universal Packaging:** Package any app, program, toolkit, or library for all major Linux distributions and IoT devices.
*   **Dependency Management:**  Snapcraft handles dependencies by bundling everything your software needs into a single container.
*   **Simplified Build Process:** Uses a simple `snapcraft.yaml` project file for easy integration with your existing code base.
*   **Cross-Platform Support:** Available on all major Linux distributions, Windows, and macOS.
*   **Seamless Distribution:** Integrate with public and private app stores, including the Snap Store, for easy software publishing.
*   **Version Control & Updates:** Manage snap versions, revisions, and parallel releases effectively.

## Getting Started

### 1. Initialize Your Project

Create a basic `snapcraft.yaml` file:

```bash
snapcraft init
```

### 2. Package Your Software

Bundle your project into a snap package:

```bash
snapcraft pack
```

### 3. Register and Upload

Register your project and upload it to stores:

```bash
snapcraft register
snapcraft upload
```

For a detailed guide, see the [Crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) tutorial.

## Installation

Snapcraft is easy to install on most systems. On snap-ready systems:

```bash
sudo snap install snapcraft --classic
```

For more detailed installation instructions, including setting up the necessary Linux container tools, consult the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Documentation

Comprehensive documentation is available, covering project file creation, debugging, interface resolution, and command references.  Explore the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable).

## Community and Support

Join the growing Snapcraft community for support and collaboration:

*   **Forum:** [Snapcraft Forum](https://forum.snapcraft.io)
*   **Matrix Channel:** [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com)
*   **Issue Tracking:** Report bugs or issues on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contribute

Snapcraft is open source and welcomes community contributions.  Start with the [contribution guide](CONTRIBUTING.md). The [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) provides a hub for doc development, even without coding experience.

## License

Snapcraft is licensed under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.