<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software Across Linux with Ease

**Snapcraft** simplifies the process of packaging and distributing your software, allowing you to reach a broad audience across all major Linux distributions and IoT devices.  [Learn more on the original repository](https://github.com/canonical/snapcraft).

*   **Cross-Distribution Compatibility:** Build once, run everywhere. Snapcraft packages your application in a container format that works seamlessly across various Linux distributions, eliminating dependency headaches.
*   **Dependency Management:** Snapcraft bundles all your software's dependencies directly into the container, ensuring consistent behavior and reducing compatibility issues.
*   **Simple Configuration:** Easily create a `snapcraft.yaml` project file to define your build process and dependencies.
*   **Publishing to Stores:**  Effortlessly register and upload your snaps to public and private app stores, including the Snap Store.
*   **Comprehensive Documentation:**  Access extensive documentation with tutorials, command references, and debugging guides to help you master Snapcraft.

## Key Features of Snapcraft

Snapcraft enables developers to package any app, program, toolkit, or library for all major Linux distributions and IoT devices. Here's what you can do:

*   **Package Any Application:** Package any application, program, toolkit, or library.
*   **Dependency Management:** Manages all the dependencies.
*   **Cross-Distribution Support:** Package your applications for multiple Linux distributions.
*   **Simplified Build Process:** Utilizes a user-friendly `snapcraft.yaml` configuration file.
*   **Integration with App Stores:** Supports publishing to the Snap Store and other app stores.

## Getting Started with Snapcraft

Follow these simple steps to start using Snapcraft:

1.  **Initialize Your Project:**  From your project's root directory, run `snapcraft init` to create a basic `snapcraft.yaml` file.
2.  **Configure Your Build:**  Edit `snapcraft.yaml` to include your project's build and runtime details.
3.  **Build Your Snap:** Package your project into a snap with the command: `snapcraft pack`
4.  **Publish Your Snap:** Register and upload your snap to app stores.
5.  **Continuous Delivery:** Upload new versions via `snapcraft upload` to push releases and parallel releases to stores.

To delve deeper, explore the [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) tutorial.

## Installation

Snapcraft is easily available on all major Linux distributions, Windows, and macOS. Install it with the following command on snap-ready systems:

```bash
sudo snap install snapcraft --classic
```

## Resources

*   **Documentation:**  Find detailed guidance and learning materials on the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable).
*   **Community Forum:**  Connect with the Snapcraft community on the [Snapcraft Forum](https://forum.snapcraft.io).
*   **Matrix Channel:**  Join the conversation on the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Issue Tracker:**  Report issues and contribute to the project on the [Snapcraft GitHub repository](https://github.com/canonical/snapcraft/issues).
*   **Contribution Guide:**  Learn how to contribute to the project with the [contribution guide](CONTRIBUTING.md).

## License and Copyright

Snapcraft is licensed under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.