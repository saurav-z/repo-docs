<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software for Linux and IoT

**Snapcraft** is the essential command-line tool for effortlessly packaging and distributing your software as snaps, reaching all major Linux distributions and IoT devices. **[Learn more at the official repository](https://github.com/canonical/snapcraft)**

[![Snapcraft][snapcraft-badge]][snapcraft-site]
[![Documentation Status][rtd-badge]][rtd-latest]
[![Spread tests][gha-spread-badge]][gha-spread]
[![Codecov Status][codecov-badge]][codecov-status]
[![Ruff status][ruff-badge]][ruff-site]

## Key Features of Snapcraft

*   **Simplified Packaging:** Package any application, program, toolkit, or library into a single, portable snap package.
*   **Universal Compatibility:** Build snaps that run seamlessly across all major Linux distributions and IoT devices.
*   **Dependency Management:** Snapcraft handles dependency management by bundling all necessary libraries within the snap.
*   **Easy Build Process:** Uses a simple `snapcraft.yaml` configuration file for easy project setup and updates.
*   **Store Integration:** Publish your snaps to public and private app stores, including the Snap Store.
*   **Efficient Publishing:** Easily version, revise, and release parallel versions of your snaps.

## Getting Started with Snapcraft

### Initialization

Create a basic `snapcraft.yaml` file in your project's root directory:

```bash
snapcraft init
```

### Build and Package

Create your snap:

```bash
snapcraft pack
```

### Publishing

Register and upload your snap to the Snap Store:

```bash
snapcraft register
snapcraft upload
```

For a more in-depth guide, consult the [Crafting Your First Snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) tutorial.

## Installation

Snapcraft is readily available on all major Linux distributions, Windows, and macOS.

### Snap Installation (Recommended)

On snap-ready systems:

```bash
sudo snap install snapcraft --classic
```

### Traditional Package Installation

Alternatively, install Snapcraft through your distribution's package manager. Detailed setup instructions can be found in the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Resources and Support

*   **Documentation:** Comprehensive [Snapcraft Documentation](https://documentation.ubuntu.com/snapcraft/stable) for in-depth guidance.
*   **Community Forum:** Engage with other Snapcraft users and experts on the [Snapcraft Forum](https://forum.snapcraft.io).
*   **Matrix Channel:** Join the conversation on the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Issue Reporting:** Report bugs and issues on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contributing

Snapcraft is open source and welcomes contributions! Refer to the [contribution guide](CONTRIBUTING.md) and the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) for details on contributing to the project.

## License and Copyright

Snapcraft is licensed under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.

[snapcraft-badge]: https://snapcraft.io/snapcraft/badge.svg
[snapcraft-site]: https://snapcraft.io/snapcraft
[rtd-badge]: https://readthedocs.com/projects/canonical-snapcraft/badge/?version=latest
[rtd-latest]: https://documentation.ubuntu.com/snapcraft/latest/?badge=latest
[gha-spread-badge]: https://github.com/canonical/snapcraft/actions/workflows/spread-scheduled.yaml/badge.svg?branch=main
[gha-spread]: https://github.com/canonical/snapcraft/actions/workflows/spread-scheduled.yaml
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff-site]: https://github.com/astral-sh/ruff
[codecov-badge]: https://codecov.io/github/canonical/snapcraft/coverage.svg?branch=master
[codecov-status]: https://codecov.io/github/canonical/snapcraft?branch=master