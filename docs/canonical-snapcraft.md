<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software Across Linux with Ease

**Snapcraft** is a powerful command-line tool that simplifies software packaging and distribution across a wide range of Linux distributions and IoT devices. Solve dependency issues and simplify app deployment. Visit the [Snapcraft GitHub repository](https://github.com/canonical/snapcraft) to learn more.

## Key Features:

*   **Cross-Distribution Compatibility:** Build once, run everywhere on major Linux distributions.
*   **Simplified Dependency Management:**  Bundles all dependencies within the snap package for consistent execution.
*   **Easy Packaging:** Create snap packages using a simple `snapcraft.yaml` configuration file.
*   **Supports Multiple Architectures:** Package your application for various architectures.
*   **Integration with App Stores:** Publish your snaps to public and private app stores, including the Snap Store.
*   **Command-Line Interface:** Manage your snaps with intuitive commands like `snapcraft init`, `snapcraft pack`, `snapcraft register`, and `snapcraft upload`.

## Getting Started

### Installation

Snapcraft is easily installed on all major Linux distributions, macOS, and Windows.

On snap-enabled systems:

```bash
sudo snap install snapcraft --classic
```

For detailed setup instructions and installation as a traditional package, refer to the [Snapcraft Documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

### Basic Usage

1.  **Initialize a project:**  `snapcraft init` - create a basic `snapcraft.yaml` file.
2.  **Configure your project:** Edit the `snapcraft.yaml` file to define build and runtime settings.
3.  **Build your snap:** `snapcraft pack` - bundles your project into a snap.
4.  **Register and Upload:**  `snapcraft register` and `snapcraft upload` - publish to app stores.

For a more in-depth tutorial, check out the [Crafting Your First Snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) guide.

## Resources & Community

*   **Documentation:** Comprehensive [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) provides in-depth guidance.
*   **Community Forum:** Engage with the Snapcraft community on the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Issue Tracking:** Report issues and bugs on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contribute

Snapcraft is open-source and welcomes contributions!  Start with the [contribution guide](CONTRIBUTING.md) and help improve the project. The [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) welcomes contributions to the documentation.

## License and Copyright

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.