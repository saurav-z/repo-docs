<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software Across Linux with Ease

**Snapcraft** is a powerful command-line tool that simplifies software packaging and distribution for Linux, enabling developers to create and manage snaps for various distributions and IoT devices.  [View the original repository](https://github.com/canonical/snapcraft).

## Key Features

*   **Universal Packaging:** Package any application, program, toolkit, or library for all major Linux distributions and IoT devices.
*   **Dependency Management:** Bundles all necessary libraries and dependencies within the snap container.
*   **Simplified Build Process:** Uses `snapcraft.yaml` for easy project configuration and integration with existing codebases.
*   **Cross-Platform Compatibility:** Supports building snaps for all major Linux distributions, Windows, and macOS.
*   **Seamless Distribution:** Enables publishing your snaps to public and private app stores, including the Snap Store.
*   **Version Control & Releases**: Easily publish new versions and revisions, including parallel releases.

## Getting Started

### Initialize Your Project

Create a basic `snapcraft.yaml` file with:

```bash
snapcraft init
```

### Build Your Snap

Package your project into a snap:

```bash
snapcraft pack
```

### Publish Your Snap

Register and upload your snap to the Snap Store:

```bash
snapcraft register
snapcraft upload
```

For detailed command options and project file configuration, explore the [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) tutorial.

## Installation

Snapcraft can be easily installed on most systems:

```bash
sudo snap install snapcraft --classic
```

Or via traditional package managers on many Linux distributions.  Consult the [setup documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft) for complete setup instructions.

## Resources

*   **Documentation:** [Snapcraft Documentation](https://documentation.ubuntu.com/snapcraft/stable) - Comprehensive guides, tutorials, and command references.
*   **Community:**
    *   [Snapcraft Forum](https://forum.snapcraft.io) - Discuss Snapcraft and connect with other developers.
    *   [Snapcraft Matrix Channel](https://matrix.to/#/#snapcraft:ubuntu.com) - Real-time chat and collaboration.
    *   [GitHub Issues](https://github.com/canonical/snapcraft/issues) - Report bugs and issues.
*   **Contribution:** [Contribution Guide](CONTRIBUTING.md) - Learn how to contribute to the Snapcraft project.
*   **Docs:** [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) - Hub for documentation development.

## License

Snapcraft is licensed under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.