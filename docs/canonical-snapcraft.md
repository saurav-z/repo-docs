<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software Across Linux and IoT Devices

**Snapcraft** is a powerful command-line tool that simplifies the process of packaging and distributing software applications in the snap container format, making it easy to deploy your software across various Linux distributions and IoT devices. ([See the original repository](https://github.com/canonical/snapcraft))

## Key Features

*   **Simplified Packaging:** Easily package your applications with a simple `snapcraft.yaml` configuration file.
*   **Cross-Distribution Compatibility:** Build once and deploy to all major Linux distributions and IoT devices.
*   **Dependency Management:**  Handles dependencies automatically, ensuring your software runs consistently across different environments.
*   **App Store Integration:** Publish your snaps to public and private app stores, including the Snap Store.
*   **Version Control and Updates:** Manage snap versions and revisions with ease, including parallel releases.

## Getting Started

### Basic Usage

1.  **Initialize a project:** Create a basic `snapcraft.yaml` file with `snapcraft init`.
2.  **Define your project:** Add build and runtime details to your `snapcraft.yaml` file.
3.  **Build your snap:** Bundle your project into a snap with `snapcraft pack`.
4.  **Register and Upload:** Register your project on a store and publish with `snapcraft register` and `snapcraft upload`.

For a more detailed guide, start with [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap).

### Installation

Snapcraft is readily available on all major Linux distributions, Windows, and macOS.

*   **Snap Installation (Recommended):** On snap-enabled systems, install with `sudo snap install snapcraft --classic`.
*   **Traditional Package Installation:** Available through many popular Linux repositories.

Refer to the [documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft) for detailed installation instructions.

## Documentation and Support

*   **Comprehensive Documentation:** Explore the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) for in-depth guides, tutorials, and command references.
*   **Community Support:** Engage with the Snapcraft community on the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Issue Reporting:** Report bugs and issues on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contribute

Snapcraft is an open-source project, and contributions are welcome!  Consult the [contribution guide](CONTRIBUTING.md) for details. You can also contribute to documentation through the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy).

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.