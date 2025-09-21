<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: The Ultimate Tool for Packaging and Distributing Software

**Snapcraft** simplifies software packaging and distribution for all major Linux distributions and IoT devices, allowing developers to effortlessly create and deploy applications in the secure and versatile snap format.  [See the original repository](https://github.com/canonical/snapcraft).

## Key Features

*   **Effortless Packaging:**  Create snaps from any app, program, toolkit, or library.
*   **Cross-Distribution Compatibility:**  Package software for all major Linux distributions.
*   **Dependency Management:**  Bundles all necessary libraries within the snap, eliminating dependency conflicts.
*   **Architecture Support:** Compatible with various architectures for broad device support.
*   **Simplified Workflow:** Build configurations stored in a user-friendly `snapcraft.yaml` file.
*   **App Store Integration:** Easily register and upload your snaps to public and private app stores, including the Snap Store.

## Getting Started

### Initialize Your Project

Start by creating a basic `snapcraft.yaml` file:

```bash
snapcraft init
```

### Build Your Snap

Once you've configured your project, build your snap:

```bash
snapcraft pack
```

### Publish Your App

Register your project and upload it to app stores:

```bash
snapcraft register
snapcraft upload
```

For a more in-depth tutorial, check out the official [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) guide.

## Installation

Snapcraft is readily available on all major Linux distributions, Windows, and macOS.

### Install via Snap

For systems with snap support, install Snapcraft with a single command:

```bash
sudo snap install snapcraft --classic
```

### Other Installation Options

Snapcraft can also be installed as a traditional package on many popular Linux repositories. Comprehensive setup instructions are available in the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Documentation & Support

Comprehensive guidance and learning materials are available in the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable), covering project file creation, debugging, interface resolution, and command references.

Join the community and get support:

*   **Forum:** [Snapcraft Forum](https://forum.snapcraft.io)
*   **Matrix Channel:** [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com)
*   **Issue Tracker:** Report issues or bugs on the [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contribute

Snapcraft is an open-source project. We welcome contributions from the community!

*   **Contribution Guide:** [CONTRIBUTING.md](CONTRIBUTING.md)
*   **Documentation:** Help with the docs through the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy).

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.