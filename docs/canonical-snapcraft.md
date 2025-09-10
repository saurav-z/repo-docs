<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Software Easily Across Linux and IoT Devices

**Snapcraft** is a powerful command-line tool that simplifies packaging and distributing software as self-contained snap packages, ensuring compatibility and ease of deployment for all major Linux distributions and IoT devices.

[View the original repository on GitHub](https://github.com/canonical/snapcraft)

## Key Features

*   **Universal Packaging:** Create snap packages that work across various Linux distributions and IoT devices.
*   **Dependency Management:** Bundle all necessary libraries and dependencies within the snap, eliminating compatibility issues.
*   **Simplified Build Process:**  Utilize `snapcraft.yaml` for easy configuration and integration with existing codebases.
*   **Command-Line Interface:**  Use intuitive commands like `snapcraft init`, `snapcraft pack`, `snapcraft register`, and `snapcraft upload` for efficient packaging, registration, and publishing.
*   **Integration with Snap Store:** Seamlessly publish your snaps to public and private app stores, including the Snap Store.

## Getting Started

1.  **Initialize your project:**
    ```bash
    snapcraft init
    ```
2.  **Define your build and runtime details:**
    *   Add your project's build and runtime details to the `snapcraft.yaml` file.
3.  **Package your project:**
    ```bash
    snapcraft pack
    ```
4.  **Register and upload:**
    ```bash
    snapcraft register
    snapcraft upload
    ```

To learn more, explore the [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) tutorial.

## Installation

Snapcraft is readily available for major Linux distributions, Windows, and macOS.

**Install as a Snap (Recommended):**

```bash
sudo snap install snapcraft --classic
```

For complete setup, a Linux container tool is also necessary.  Find detailed instructions on [setting up Snapcraft](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Resources

*   **Documentation:** Comprehensive [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) covering all aspects of snap creation, debugging, and more.
*   **Community:** Connect with other Snapcraft users on the [Snapcraft Forum](https://forum.snapcraft.io) and [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Contribute:** Help improve Snapcraft! Review the [contribution guide](CONTRIBUTING.md) and the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy).
*   **Report Issues:** Report bugs and issues on the [Snapcraft GitHub repository](https://github.com/canonical/snapcraft/issues).

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.