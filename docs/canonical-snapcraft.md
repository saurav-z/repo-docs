<!-- Snapcraft Logo -->
<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software Across Linux with Ease

Snapcraft is the command-line tool that empowers developers to package and distribute software seamlessly across all major Linux distributions and IoT devices.

## Key Features of Snapcraft

*   **Simplified Packaging:** Create and manage application packages in the snap format with ease.
*   **Dependency Management:** Bundles all software dependencies within a single container, ensuring consistent performance.
*   **Cross-Platform Compatibility:** Package your applications to run on various Linux distributions.
*   **App Store Integration:** Register and publish your snaps to public and private app stores, including the Snap Store.
*   **Easy to Use:**  A simple `snapcraft.yaml` project file allows for easy configuration and integration with existing codebases.

## Getting Started with Snapcraft

**1. Initialize your project:**
   ```bash
   snapcraft init
   ```

**2. Package your project into a snap:**
   ```bash
   snapcraft pack
   ```

**3. Register and upload your project to stores:**
   ```bash
   snapcraft register
   snapcraft upload
   ```

For a more in-depth guide on how to get started, see the [crafting your first snap tutorial](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap).

## Installation

Snapcraft is readily available on popular operating systems.

**Installation via Snap (Recommended):**

```bash
sudo snap install snapcraft --classic
```

For detailed installation instructions, including installation on other operating systems or using traditional package managers, see the [setup documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Documentation and Resources

*   **Comprehensive Documentation:** Explore the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) for detailed guidance on building project files, debugging, command references, and more.
*   **Community Forum:** Engage with the Snapcraft community on the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com)
*   **GitHub Repository:** Report issues and contribute to the project on the [GitHub repository](https://github.com/canonical/snapcraft).

## Contribute

Snapcraft is an open-source project, and contributions are welcome! Check out the [contribution guide](CONTRIBUTING.md) for more information. Join the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) for help with documentation.

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.

**[Visit the original Snapcraft repository on GitHub](https://github.com/canonical/snapcraft)**