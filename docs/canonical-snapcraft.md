<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software with Ease

**Snapcraft** simplifies software packaging and distribution, enabling developers to create secure, reliable, and easily installable applications across various Linux distributions and IoT devices.  [Visit the Snapcraft GitHub Repository](https://github.com/canonical/snapcraft)

## Key Features of Snapcraft

*   **Cross-Distribution Compatibility:** Package your applications once and deploy them across major Linux distributions, simplifying development and reach.
*   **Dependency Management:** Snapcraft bundles all required libraries and dependencies within a single container, ensuring consistent application behavior across different environments.
*   **Easy Packaging with `snapcraft.yaml`:** Define your application's build configuration using a simple, intuitive `snapcraft.yaml` file, making the packaging process straightforward.
*   **Seamless Deployment:**  Publish your snaps to public and private app stores, including the Snap Store, for easy distribution and updates.
*   **Supports IoT Devices:** Snapcraft is designed for IoT, allowing you to build and deploy applications on edge devices.
*   **Simple Commands:**  Quickly initialize projects, build, package, and register/upload your applications with intuitive command-line tools.

## Getting Started with Snapcraft

1.  **Initialize your project:**

    ```bash
    snapcraft init
    ```
2.  **Define your build configuration:**  Edit the created `snapcraft.yaml` to specify your project's details.
3.  **Build your snap:**

    ```bash
    snapcraft pack
    ```
4.  **Register and upload your snap:**

    ```bash
    snapcraft register
    snapcraft upload
    ```

    For a more detailed guide, follow the [crafting your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap).

## Installation

Snapcraft is available on all major Linux distributions, Windows, and macOS. You can install it using:

```bash
sudo snap install snapcraft --classic
```
For complete installation, you need an additional Linux container tool.  Refer to the [setup documentation](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft) for more details.

## Resources & Support

*   **Documentation:** Explore comprehensive [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) for in-depth guidance and tutorials.
*   **Community Forum:**  Engage with other developers and ask questions in the [Snapcraft Forum](https://forum.snapcraft.io).
*   **Community Chat:**  Join the conversation on the [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com).
*   **Issue Tracking:** Report bugs and issues on the [GitHub repository](https://github.com/canonical/snapcraft/issues).
*   **Contribution:**  Get involved!  Refer to the [contribution guide](CONTRIBUTING.md) to help improve Snapcraft.  The [Canonical Open Documentation Academy](https://github.com/canonical/open-documentation-academy) welcomes contributions to the documentation.

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.