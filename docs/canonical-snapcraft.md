<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software for Any Linux Distribution

**Snapcraft** is the command-line tool that simplifies software packaging and distribution across all major Linux distributions and IoT devices.  Get started packaging your software and see all the available features via the [Snapcraft GitHub repository](https://github.com/canonical/snapcraft).

## Key Features of Snapcraft:

*   **Universal Packaging:** Package your applications once and deploy them across various Linux distributions.
*   **Dependency Management:** Bundle all required libraries and dependencies within a single snap package for consistent execution.
*   **Architecture Support:** Build your software for various architectures, ensuring compatibility across different devices.
*   **Simplified Build Process:** Define your snap's configuration using a simple `snapcraft.yaml` file.
*   **App Store Integration:** Easily register and publish your snaps to public and private app stores, including the Snap Store.

## Getting Started with Snapcraft

### 1. Initialize Your Project

Create a basic `snapcraft.yaml` file to start packaging your software.

```bash
snapcraft init
```

### 2. Package Your Software

Build your project into a snap container.

```bash
snapcraft pack
```

### 3. Publish Your Snap

Register and upload your snap to the Snap Store or other app stores.

```bash
snapcraft register
snapcraft upload
```

For a more in-depth guide, check out [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap).

## Installation

Snapcraft is available on most major Linux distributions, Windows, and macOS.

### Install on Snap-Ready Systems

```bash
sudo snap install snapcraft --classic
```

Refer to the [installation documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft) for comprehensive installation instructions.

## Resources

*   **Documentation:** Comprehensive [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) for building project files, debugging, command references, and more.
*   **Community:** Engage with the Snapcraft community through the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Issue Tracking:** Report issues and bugs on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).
*   **Contribution:** Contribute to Snapcraft by following the [contribution guide](CONTRIBUTING.md).

## License and Copyright

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.