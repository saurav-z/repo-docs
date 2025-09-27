<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package & Distribute Your Software Across Linux, IoT, & More

**Snapcraft** empowers developers to effortlessly package and distribute software as **snaps**, enabling cross-platform compatibility and simplified dependency management.  [Learn more at the original repo](https://github.com/canonical/snapcraft).

## Key Features of Snapcraft

*   **Universal Packaging:** Package any application, program, toolkit, or library for all major Linux distributions and IoT devices.
*   **Simplified Dependency Management:**  Bundles all software dependencies within a single, self-contained snap package, eliminating compatibility issues.
*   **Easy to Use:**  Build configuration stored in `snapcraft.yaml`, making it easy to add as a new package format to your existing code.
*   **Cross-Platform Support:**  Available on all major Linux distributions, Windows, and macOS.
*   **Integration with App Stores:**  Seamlessly register and upload your snaps to public and private app stores, including the Snap Store.
*   **Parallel Releases:** Supports snap versions and revisions, including parallel releases.

## Getting Started with Snapcraft

Snapcraft simplifies the software packaging process with these basic steps:

1.  **Initialize:** Create a `snapcraft.yaml` file in your project's root directory:
    ```bash
    snapcraft init
    ```
2.  **Configure:** Add your project's build and runtime details to the `snapcraft.yaml` file.
3.  **Package:** Build your snap:
    ```bash
    snapcraft pack
    ```
4.  **Register & Upload:** Register and upload your snap to app stores like the Snap Store:
    ```bash
    snapcraft register
    snapcraft upload
    ```

For a comprehensive guide, explore the [crafting your first snap tutorial](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap).

## Installation

Install Snapcraft on any snap-ready system via the command line:

```bash
sudo snap install snapcraft --classic
```

Complete installation may require a Linux container tool, or install as a traditional package.  Refer to the documentation for [setup instructions](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Resources

*   **Documentation:** Access comprehensive guidance and learning materials at the [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable).
*   **Community Forum:** Engage with fellow crafters and ask questions in the [Snapcraft Forum](https://forum.snapcraft.io).
*   **Matrix Channel:** Join the conversation on the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Issue Tracker:** Report bugs and issues on the project's [GitHub repository](https://github.com/canonical/snapcraft/issues).
*   **Contribution:**  Contribute to the project by following the [contribution guide](CONTRIBUTING.md).  The [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) is also available for doc development.

## License & Copyright

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.