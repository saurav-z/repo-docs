<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package Your Software for Any Linux Distribution

**Snapcraft is the ultimate command-line tool for effortlessly packaging and distributing your software applications across all major Linux distributions and IoT devices.**  [Visit the original repository](https://github.com/canonical/snapcraft)

[![Snapcraft][snapcraft-badge]][snapcraft-site]
[![Documentation Status][rtd-badge]][rtd-latest]
[![Spread tests][gha-spread-badge]][gha-spread]
[![Codecov Status][codecov-badge]][codecov-status]
[![Ruff status][ruff-badge]][ruff-site]

## Key Features of Snapcraft

*   **Cross-Distribution Compatibility:** Package your software once and run it on various Linux distributions without modification.
*   **Dependency Management:**  Snapcraft bundles all required libraries and dependencies, ensuring your software runs smoothly.
*   **Simple YAML Configuration:** Define your build and runtime configurations easily using the `snapcraft.yaml` project file.
*   **Simplified Build Process:**  Use intuitive commands like `snapcraft init`, `snapcraft pack`, `snapcraft register`, and `snapcraft upload` to build, package, and publish your snaps.
*   **Support for Public and Private App Stores:**  Easily distribute your software through the Snap Store and other private stores.
*   **IoT Device Compatibility:** Package your applications for IoT devices, expanding your software's reach.

## Getting Started with Snapcraft

### Installation

Snapcraft is easily installed on all major Linux distributions, Windows, and macOS.  On snap-ready systems, use:

```bash
sudo snap install snapcraft --classic
```

For detailed installation instructions and setup guidance, see the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

### Basic Usage

1.  **Initialize:** Create a basic `snapcraft.yaml` file in your project's root directory:
    ```bash
    snapcraft init
    ```
2.  **Configure:**  Add your project's build and runtime details to the `snapcraft.yaml` file.
3.  **Package:** Build your application into a snap:
    ```bash
    snapcraft pack
    ```
4.  **Publish:**  Register and upload your snap to app stores:
    ```bash
    snapcraft register
    snapcraft upload
    ```

## Resources and Support

*   **Documentation:** Comprehensive [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) with tutorials, command references, and debugging guides.
*   **Community Forum:** Get help and connect with other Snapcraft users on the [Snapcraft Forum](https://forum.snapcraft.io).
*   **Matrix Channel:** Discuss Snapcraft and stay up-to-date on the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Issue Tracking:** Report bugs and issues on the [GitHub repository](https://github.com/canonical/snapcraft/issues).
*   **Contribution:** Contribute to the project! See the [contribution guide](CONTRIBUTING.md). and join the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy).

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