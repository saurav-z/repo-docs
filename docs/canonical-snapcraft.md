<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software Across Linux with Ease

**Snapcraft** empowers developers to package and distribute software effortlessly across all major Linux distributions and IoT devices. ([See the original repository](https://github.com/canonical/snapcraft))

[![Snapcraft][snapcraft-badge]][snapcraft-site]
[![Documentation Status][rtd-badge]][rtd-latest]
[![Spread tests][gha-spread-badge]][gha-spread]
[![Codecov Status][codecov-badge]][codecov-status]
[![Ruff status][ruff-badge]][ruff-site]

## Key Features of Snapcraft:

*   **Simplified Packaging:** Easily package any application, program, toolkit, or library into a snap container format.
*   **Dependency Management:** Resolves dependency issues by bundling all necessary libraries within the snap.
*   **Cross-Distribution Support:** Ensures your software runs consistently across various Linux distributions.
*   **Easy to Use:** Build configuration uses a simple `snapcraft.yaml` project file.
*   **Publishing and Distribution:**  Seamlessly integrate with public and private app stores, including the Snap Store.

## Getting Started with Snapcraft

1.  **Initialize:** Create a basic `snapcraft.yaml` file with `snapcraft init`.
2.  **Configure:** Add build and runtime details to the `snapcraft.yaml` file.
3.  **Package:** Create your snap package with `snapcraft pack`.
4.  **Publish:** Register and upload your snap to the Snap Store or other app stores with `snapcraft register` and `snapcraft upload`.

For detailed instructions on creating your first snap, explore the [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) tutorial.

## Installation

Snapcraft is readily available on major Linux distributions, Windows, and macOS. Install Snapcraft using the following command:

```bash
sudo snap install snapcraft --classic
```

For comprehensive installation guidance, refer to the [Snapcraft setup documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Documentation and Support

*   **Comprehensive Documentation:** Explore the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) for in-depth guides on project file creation, debugging, command references, and more.
*   **Community Forum:** Engage with the Snapcraft community on the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com) to ask questions and stay informed.
*   **Issue Reporting:** Report issues and bugs on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contribute to Snapcraft

Snapcraft is an open-source project under the Canonical umbrella, welcoming contributions from the community.

*   **Contribution Guide:** Begin by reviewing the [contribution guide](CONTRIBUTING.md).
*   **Documentation:** Assist with documentation updates through the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy).

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