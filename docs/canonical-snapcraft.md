<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software Across Linux with Ease

**Snapcraft** is the command-line tool that simplifies packaging and distributing software as snaps, making it easy to deploy your applications on all major Linux distributions and IoT devices. ([See the original repo](https://github.com/canonical/snapcraft))

## Key Features

*   **Universal Packaging:** Package your software once and distribute it across various Linux distributions.
*   **Dependency Management:** Bundles all dependencies, libraries, and architectures within a container, eliminating dependency conflicts.
*   **Simplified Build Process:** Uses a simple `snapcraft.yaml` file for easy project configuration.
*   **App Store Integration:** Seamlessly publish your snaps to public and private app stores, including the Snap Store.
*   **Cross-Platform Support:** Available on all major Linux distributions, Windows, and macOS.

## Getting Started

### Initialization

Create a basic `snapcraft.yaml` file:

```bash
snapcraft init
```

### Build and Package

Build your snap:

```bash
snapcraft pack
```

### Publish

Register and upload your snap:

```bash
snapcraft register
snapcraft upload
```

## Installation

Snapcraft is easily installable via snap:

```bash
sudo snap install snapcraft --classic
```

Alternatively, you can install it through traditional package managers on many Linux distributions. See the [installation documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft) for further guidance.

## Documentation and Support

Comprehensive documentation is available, including guidance on project file creation, debugging, interfaces, and command references.

*   [Snapcraft Documentation](https://documentation.ubuntu.com/snapcraft/stable)
*   [Snapcraft Forum](https://forum.snapcraft.io)
*   [Snapcraft Matrix Channel](https://matrix.to/#/#snapcraft:ubuntu.com)
*   [GitHub Repository](https://github.com/canonical/snapcraft/issues)

## Contribute

Snapcraft is open source and welcomes contributions!  Check out the [contribution guide](CONTRIBUTING.md) and the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) for documentation contributions.

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.