<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Software Easily with Snaps

**Snapcraft empowers developers to build, package, and distribute their applications across major Linux distributions and IoT devices.**

[![Snapcraft][snapcraft-badge]][snapcraft-site]
[![Documentation Status][rtd-badge]][rtd-latest]
[![Spread tests][gha-spread-badge]][gha-spread]
[![Codecov Status][codecov-badge]][codecov-status]
[![Ruff status][ruff-badge]][ruff-site]

## Key Features of Snapcraft

*   **Simplified Packaging:** Create self-contained application packages (snaps) that include all dependencies.
*   **Cross-Distribution Compatibility:** Package your software once and run it on various Linux distributions.
*   **Easy Build Process:** Utilize the `snapcraft.yaml` configuration file for streamlined project setup and packaging.
*   **Snap Store Integration:** Seamlessly publish and distribute your snaps through public and private app stores, including the Snap Store.
*   **Dependency Management:**  Snapcraft handles dependency management, ensuring your application runs consistently across different systems.
*   **Supports IoT Devices:** Package your application for IoT devices running Linux.

## Getting Started with Snapcraft

Snapcraft utilizes a simple `snapcraft.yaml` file to define your project's build and runtime details.

1.  **Initialize your project:**
    ```bash
    snapcraft init
    ```
2.  **Package your project:**
    ```bash
    snapcraft pack
    ```
3.  **Register and upload your snap:**
    ```bash
    snapcraft register
    snapcraft upload
    ```

For a detailed guide on crafting your first snap, see [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap).

## Installation

Snapcraft is available on major Linux distributions, Windows, and macOS. The easiest way to install Snapcraft is via snap itself:

```bash
sudo snap install snapcraft --classic
```

For more installation options and setup assistance, refer to the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Documentation and Support

*   **Comprehensive Documentation:**  The [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) provides in-depth guidance on all aspects of snap creation, including project file composition, debugging, and command references.
*   **Community Forum:** Get help and discuss Snapcraft on the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Report Issues:** Report bugs and issues on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contributing to Snapcraft

Snapcraft is an open-source project, and contributions are welcome!  Check out the [contribution guide](CONTRIBUTING.md) to get started.

The [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) is a resource for documentation development, including Snapcraft docs.

## License and Copyright

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

**Original Repo:** [https://github.com/canonical/snapcraft](https://github.com/canonical/snapcraft)