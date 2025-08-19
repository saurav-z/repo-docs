<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Software Seamlessly Across Linux and IoT Devices

**Snapcraft** empowers developers to easily package, build, and distribute software in the universal snap container format, ensuring compatibility across major Linux distributions and IoT devices.  Learn more and contribute on [GitHub](https://github.com/canonical/snapcraft).

## Key Features

*   **Universal Packaging:** Create a single package that runs across various Linux distributions and IoT devices.
*   **Dependency Management:**  Automatically handles software dependencies, ensuring everything your application needs is included.
*   **Simplified Build Process:** Utilize a simple `snapcraft.yaml` configuration file for easy setup and integration.
*   **One-liner Installation:** Install Snapcraft on most systems using a single command: `sudo snap install snapcraft --classic`.
*   **Store Integration:**  Seamlessly publish your snaps to public and private app stores, including the Snap Store.
*   **Flexible Publishing:** Manage snap versions, revisions, and parallel releases easily through publishing workflows.

## Getting Started

1.  **Initialize Your Project:** Start by creating a `snapcraft.yaml` file:
    ```bash
    snapcraft init
    ```
2.  **Configure Your Project:** Add your project's build and runtime details to the `snapcraft.yaml` file.
3.  **Build Your Snap:** Package your project into a snap:
    ```bash
    snapcraft pack
    ```
4.  **Publish Your Snap:** Register and upload your snap to app stores:
    ```bash
    snapcraft register
    snapcraft upload
    ```

For a detailed tutorial on creating your first snap, visit the [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) tutorial.

## Installation

Snapcraft is readily available for:

*   **Linux:** Major distributions including Ubuntu, Fedora, Debian, and more.
*   **Windows:** Available via WSL2.
*   **macOS:**  Supports MacOS development.

The recommended installation method is via Snap: `sudo snap install snapcraft --classic`.  Alternative installation methods and setup instructions are available in the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Documentation

Comprehensive documentation is available to guide you through every stage:

*   [Snapcraft Documentation](https://documentation.ubuntu.com/snapcraft/stable): Covers project file creation, debugging, interface resolution, and command references.

## Community and Support

Join the Snapcraft community and get support:

*   **Snapcraft Forum:** [Snapcraft Forum](https://forum.snapcraft.io)
*   **Matrix Channel:** [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com)
*   **Issue Tracking:** Report issues and bugs on the [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contribute

Snapcraft is an open-source project, and contributions are welcome!

*   **Contribution Guide:** Start with the [contribution guide](CONTRIBUTING.md) for details.
*   **Documentation:** Help improve the documentation through the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy).

## License

Snapcraft is licensed under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.