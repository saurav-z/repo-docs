<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: The Ultimate Tool for Packaging and Distributing Linux Software

**Simplify software distribution and reach all major Linux distributions and IoT devices with Snapcraft.** ([See the original repository](https://github.com/canonical/snapcraft))

[![Snapcraft][snapcraft-badge]][snapcraft-site]
[![Documentation Status][rtd-badge]][rtd-latest]
[![Spread tests][gha-spread-badge]][gha-spread]
[![Codecov Status][codecov-badge]][codecov-status]
[![Ruff status][ruff-badge]][ruff-site]

Snapcraft is a powerful command-line tool designed for packaging and distributing software and applications in the universal snap container format. It streamlines dependency management and ensures architecture compatibility, making it easy to package your applications for a wide range of Linux distributions and IoT devices.

## Key Features

*   **Simplified Packaging:** Creates a snap package, bundling all dependencies for easy distribution.
*   **Cross-Distribution Compatibility:** Package your software once and run it on all major Linux distributions.
*   **Dependency Management:** Handles dependencies, ensuring your software runs correctly on various systems.
*   **Easy-to-Use Configuration:** Uses `snapcraft.yaml` for clear and concise project configuration.
*   **Broad Platform Support:**  Supports Linux, Windows, and macOS.
*   **Integration with App Stores:** Seamless integration with the Snap Store and other public/private app stores for distribution.

## Getting Started

### Initialize your project:

```bash
snapcraft init
```

### Build your snap:

```bash
snapcraft pack
```

### Publish your snap:

```bash
snapcraft upload
```

For a detailed guide, see the [Crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) tutorial.

## Installation

Snapcraft is available on most major Linux distributions, Windows, and macOS.

**Installation via Snap (recommended):**

```bash
sudo snap install snapcraft --classic
```

Refer to the [setup documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft) for complete setup instructions, including container tool requirements.

## Resources

*   **Documentation:** [Snapcraft Documentation](https://documentation.ubuntu.com/snapcraft/stable) - Comprehensive guides, tutorials, and command references.
*   **Community:** [Snapcraft Forum](https://forum.snapcraft.io) and [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com) - Connect with the community, ask questions, and stay up-to-date.
*   **Issue Tracking:** [GitHub Repository](https://github.com/canonical/snapcraft/issues) - Report bugs and issues.

## Contributing

Snapcraft is an open-source project, and contributions are welcome. Review the [contribution guide](CONTRIBUTING.md) to get started. The [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) welcomes contributions to the documentation.

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