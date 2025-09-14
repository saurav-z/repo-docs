<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software Seamlessly

**Snapcraft** is a powerful command-line tool that simplifies software packaging and distribution for all major Linux distributions and IoT devices.  Learn more about Snapcraft on its [GitHub repository](https://github.com/canonical/snapcraft).

## Key Features

*   **Cross-Platform Compatibility:** Package your applications for all major Linux distributions, Windows, and macOS.
*   **Dependency Management:**  Snapcraft bundles all dependencies into a container, eliminating dependency conflicts.
*   **Simplified Packaging:** Build your snap packages using a straightforward `snapcraft.yaml` project file.
*   **Easy Distribution:** Publish your software to public and private app stores, including the Snap Store.
*   **Flexible Releases:**  Manage snap versions, revisions, and parallel releases.
*   **Open Source:**  Snapcraft is open-source and welcomes community contributions.

## Getting Started

### Initialization

Create a basic `snapcraft.yaml` project file with:

```bash
snapcraft init
```

### Building a Snap

Bundle your project into a snap with:

```bash
snapcraft pack
```

### Publishing Your App

Publish your project with:

```bash
snapcraft register
snapcraft upload
```

For detailed guidance, check out the [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) tutorial.

## Installation

Snapcraft is available on various platforms. The recommended installation method is via snap:

```bash
sudo snap install snapcraft --classic
```

You may need to set up a Linux container tool to enable full functionality. Refer to the [installation documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft) for comprehensive setup instructions.

## Documentation and Support

*   **Comprehensive Documentation:** The [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) provides in-depth guides on building, debugging, and managing snaps.
*   **Community Forums:** Engage with the Snapcraft community on the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Issue Tracking:** Report bugs and issues on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contributing

Snapcraft is a community-driven project, and contributions are welcome! Review the [contribution guide](CONTRIBUTING.md) to get started. You can also help improve the documentation through the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy).

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.