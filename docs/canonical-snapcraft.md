<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software Easily

**Snapcraft** is the command-line tool that empowers developers to package and distribute software across all major Linux distributions and IoT devices.  Learn more on the [Snapcraft GitHub repository](https://github.com/canonical/snapcraft).

## Key Features of Snapcraft:

*   **Simplified Packaging:** Easily bundle your application and its dependencies into a single snap package.
*   **Cross-Distribution Compatibility:**  Build once and run on any Linux distribution that supports snaps.
*   **Dependency Management:** Automatically handles dependencies, ensuring your software runs correctly.
*   **Architecture Support:** Supports various architectures, making your software widely accessible.
*   **Seamless Distribution:**  Publish your snaps to the Snap Store or your own private app stores.
*   **Easy to Use:**  Uses a simple `snapcraft.yaml` configuration file, making it easy to integrate into your existing workflow.

## Getting Started with Snapcraft

Snapcraft simplifies the process of packaging your software:

1.  **Initialize:** Create a `snapcraft.yaml` file with `snapcraft init`.
2.  **Configure:** Add your project's build and runtime details to the `snapcraft.yaml` file.
3.  **Package:** Build your snap with `snapcraft pack`.
4.  **Distribute:** Register and upload your snap to app stores using `snapcraft register` and `snapcraft upload`.

For a guided tutorial on creating your first snap, see [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap).

## Installation

Snapcraft is available on various platforms: Linux, Windows, and macOS.

To install Snapcraft on snap-ready systems:

```bash
sudo snap install snapcraft --classic
```

Consult the documentation for [setting up Snapcraft](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft) for detailed installation instructions, including traditional package installations.

## Resources

*   **Documentation:**  Explore the comprehensive [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) for in-depth guides, tutorials, and command references.
*   **Community:**
    *   Engage with the community on the [Snapcraft Forum](https://forum.snapcraft.io).
    *   Connect with other crafters on the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Bug Reporting:**  Report issues and contribute to the project via the [GitHub repository](https://github.com/canonical/snapcraft/issues).
*   **Contribution:**  Learn how to contribute to Snapcraft by reviewing the [contribution guide](CONTRIBUTING.md) and the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy).

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.