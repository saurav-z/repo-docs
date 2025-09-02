<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software Easily

**Snapcraft** is a powerful command-line tool that simplifies packaging and distributing your software as snaps, making it easy to reach a wide audience across various Linux distributions and IoT devices. ([Original Repo](https://github.com/canonical/snapcraft))

## Key Features

*   **Simplified Packaging:** Create software packages (snaps) using a straightforward `snapcraft.yaml` configuration file.
*   **Dependency Management:** Bundle all necessary libraries and dependencies within the snap, ensuring compatibility across different systems.
*   **Cross-Platform Compatibility:**  Build snaps that run on major Linux distributions and IoT devices.
*   **Easy Distribution:** Publish your software to public and private app stores, including the Snap Store.
*   **Version Control & Releases:** Manage snap versions, revisions, and parallel releases efficiently.
*   **Free and Open Source:**  Committed to open source principles with open-source contributions welcome.

## Getting Started

1.  **Initialize your project:** Create a basic `snapcraft.yaml` file:

    ```bash
    snapcraft init
    ```

2.  **Configure your project:** Add build and runtime details to the `snapcraft.yaml` file.

3.  **Build your snap:** Package your project into a snap:

    ```bash
    snapcraft pack
    ```

4.  **Register & Upload:**  Register and publish your snap to app stores:

    ```bash
    snapcraft register
    snapcraft upload
    ```

## Installation

Install Snapcraft on your system:

```bash
sudo snap install snapcraft --classic
```

Detailed installation instructions can be found in the [Snapcraft Documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Resources

*   **Documentation:** Comprehensive [Snapcraft Documentation](https://documentation.ubuntu.com/snapcraft/stable) for in-depth guidance.
*   **Community:** Engage with the Snapcraft community via the [Snapcraft Forum](https://forum.snapcraft.io) and the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Report Issues:**  Submit bug reports and issues on the [Snapcraft GitHub Repository](https://github.com/canonical/snapcraft/issues).
*   **Contribute:** Get involved with the project by following the [contribution guide](CONTRIBUTING.md).

## License

Snapcraft is licensed under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.