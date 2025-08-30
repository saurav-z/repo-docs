<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Build, Package, and Distribute Your Software with Ease

**Snapcraft** is the powerful command-line tool that simplifies packaging and distributing software across various Linux distributions and IoT devices.  [Learn more on GitHub](https://github.com/canonical/snapcraft).

[![Snapcraft][snapcraft-badge]][snapcraft-site]
[![Documentation Status][rtd-badge]][rtd-latest]
[![Spread tests][gha-spread-badge]][gha-spread]
[![Codecov Status][codecov-badge]][codecov-status]
[![Ruff status][ruff-badge]][ruff-site]

## Key Features

*   **Cross-Distribution Compatibility:** Package your software once and deploy it on all major Linux distributions.
*   **Dependency Management:**  Snapcraft bundles all necessary libraries and dependencies within the snap, eliminating compatibility issues.
*   **Easy Project File:** Uses a simple `snapcraft.yaml` file for defining your build and runtime configurations.
*   **Simplified Packaging:** Streamline your build process with intuitive commands like `snapcraft init`, `snapcraft pack`, `snapcraft register`, and `snapcraft upload`.
*   **Supports IoT Devices:** Package applications specifically for IoT platforms.
*   **Integration with Snap Store:** Easily publish and manage your snaps on public and private app stores, including the Snap Store.

## Getting Started

### Installation

Snapcraft is available on all major Linux distributions, Windows, and macOS.

Install on snap-ready systems using the command line:

```bash
sudo snap install snapcraft --classic
```

Or, install as a traditional package on many popular Linux repositories.

For complete installation, you need an additional Linux container tool.

### Basic Usage

1.  **Initialize:** Create a `snapcraft.yaml` file in your project root:

    ```bash
    snapcraft init
    ```

2.  **Configure:** Add your project's build and runtime details to the `snapcraft.yaml` file.
3.  **Build:** Package your project into a snap:

    ```bash
    snapcraft pack
    ```

4.  **Publish:** Register and upload your snap to the Snap Store or other app stores:

    ```bash
    snapcraft register
    snapcraft upload
    ```

## Resources

*   **Documentation:**  [Snapcraft Documentation](https://documentation.ubuntu.com/snapcraft/stable) provides comprehensive guides and tutorials.
*   **Tutorial:** [Crafting Your First Snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap)
*   **Community Forum:** [Snapcraft Forum](https://forum.snapcraft.io) for discussions and support.
*   **Matrix Channel:** [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com) for real-time chat.
*   **Issue Tracking:** Report issues on the [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contribute

Snapcraft is an open-source project, and contributions are welcome!  Review the [contribution guide](CONTRIBUTING.md) and join the community.

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