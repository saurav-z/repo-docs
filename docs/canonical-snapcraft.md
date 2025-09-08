<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Effortlessly Package and Distribute Your Software Across Linux with Snaps

**Snapcraft** empowers developers to easily package, distribute, and manage their applications across a wide range of Linux distributions and IoT devices. [Check out the original repository on GitHub](https://github.com/canonical/snapcraft).

## Key Features

*   **Cross-Distribution Compatibility:** Build once, run everywhere.  Snapcraft packages your software into a single snap, ensuring it runs seamlessly on all major Linux distributions.
*   **Simplified Dependency Management:** Snapcraft bundles all necessary libraries and dependencies within the snap, eliminating compatibility issues and ensuring a consistent user experience.
*   **Easy Packaging with `snapcraft.yaml`:** Define your build configuration in a simple and intuitive `snapcraft.yaml` file, making it easy to integrate into your existing build process.
*   **Built-in Build and Release Commands:** Use straightforward commands like `snapcraft init`, `snapcraft pack`, and `snapcraft upload` to build, package, and publish your snaps to app stores.
*   **Support for Multiple Platforms:** Develop applications for all major Linux distributions and IoT devices.
*   **Secure and Isolated Environment:** Snapcraft uses container technology to provide a secure and isolated environment for your applications, reducing the risk of conflicts and vulnerabilities.

## Getting Started

1.  **Initialize your project:**  Create a basic `snapcraft.yaml` file with:

    ```bash
    snapcraft init
    ```

2.  **Define your build and runtime details:** Configure the `snapcraft.yaml` file to specify your project's dependencies, build steps, and other settings.
3.  **Build your snap:** Package your project into a snap using:

    ```bash
    snapcraft pack
    ```

4.  **Register & Upload:** Register your project and publish snaps to public or private app stores:

    ```bash
    snapcraft register
    snapcraft upload
    ```

## Installation

Snapcraft is easy to install on various platforms.

*   **Snap-Ready Systems:**

    ```bash
    sudo snap install snapcraft --classic
    ```

*   **Traditional Package Managers:**  Installation instructions for traditional package managers are available in the [documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Resources

*   **Documentation:**  Comprehensive [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) provides detailed guides, tutorials, and command references.
*   **Community Forum:** Engage with other developers and ask questions on the [Snapcraft Forum](https://forum.snapcraft.io).
*   **Community Matrix Channel:**  Join the discussion on the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **GitHub Repository:**  Report issues, contribute code, and stay updated on the project's progress on the [GitHub repository](https://github.com/canonical/snapcraft/issues).
*   **Contribution Guide:** Learn how to contribute to Snapcraft development with the [contribution guide](CONTRIBUTING.md).
*   **Open Documentation Academy:**  Help improve the documentation on the [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy).

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.