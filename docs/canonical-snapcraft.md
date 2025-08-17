<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Software Easily

**Snapcraft** is a powerful command-line tool that simplifies packaging and distributing software as self-contained, cross-platform snaps.  ([View on GitHub](https://github.com/canonical/snapcraft))

[![Snapcraft][snapcraft-badge]][snapcraft-site]
[![Documentation Status][rtd-badge]][rtd-latest]
[![Spread tests][gha-spread-badge]][gha-spread]
[![Codecov Status][codecov-badge]][codecov-status]
[![Ruff status][ruff-badge]][ruff-site]

## Key Features of Snapcraft

*   **Simplified Packaging:** Easily package applications, libraries, and tools into the snap format.
*   **Cross-Platform Compatibility:** Build once, run on all major Linux distributions and IoT devices.
*   **Dependency Management:** Bundles all dependencies within the snap, ensuring consistent behavior across different environments.
*   **Easy Build Configuration:**  Uses a straightforward `snapcraft.yaml` file for defining build and runtime settings.
*   **Integration with App Stores:**  Publish and distribute your snaps through public and private app stores, including the Snap Store.

## Getting Started with Snapcraft

### Initialize a Project

Create a basic `snapcraft.yaml` file with:

```bash
snapcraft init
```

### Build Your Snap

Package your project into a snap:

```bash
snapcraft pack
```

### Publish Your Snap

Register and upload your snap to app stores:

```bash
snapcraft register
snapcraft upload
```

For more detailed instructions, explore the [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) tutorial.

## Installation

Snapcraft is readily available across various platforms.

**Install on snap-enabled systems (recommended):**

```bash
sudo snap install snapcraft --classic
```

**Alternative installation:**

Snapcraft can also be installed as a traditional package on various Linux distributions. Refer to the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft) for detailed setup instructions.

## Resources

*   **Documentation:** Comprehensive [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) provides guidance and learning materials.
*   **Community:** Engage with the Snapcraft community through the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Issue Tracking:** Report issues and bugs on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).
*   **Contribution:** Contribute to Snapcraft! Check out the [contribution guide](CONTRIBUTING.md) and the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) for doc development.

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

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