<img src="https://dashboard.snapcraft.io/site_media/appmedia/2018/04/Snapcraft-logo-bird.png" alt="Snapcraft logo" style="height: 128px; display: block">

# Snapcraft: Package and Distribute Your Software for Any Linux Distribution

**Snapcraft** is the powerful command-line tool that simplifies software packaging and distribution across all major Linux distributions and IoT devices. Visit the [Snapcraft GitHub Repository](https://github.com/canonical/snapcraft) to learn more.

## Key Features

*   **Universal Packaging:** Package your applications once and deploy them on any Linux distribution.
*   **Dependency Management:** Automatically bundles all necessary libraries and dependencies within the snap.
*   **Architecture Support:** Supports a wide range of architectures.
*   **Easy to Use:** Simple `snapcraft.yaml` configuration file for easy integration into your existing projects.
*   **App Store Integration:** Seamlessly publish your snaps to public and private app stores, including the Snap Store.

## Get Started Quickly

Creating your first snap is easy. 

1.  **Initialize your project:**
    ```bash
    snapcraft init
    ```
2.  **Define your build and runtime details in `snapcraft.yaml`.**
3.  **Build your snap:**
    ```bash
    snapcraft pack
    ```
4.  **Publish your app to the Snap Store (optional):**
    ```bash
    snapcraft register
    snapcraft upload
    ```

[Craft your first snap](https://documentation.ubuntu.com/snapcraft/stable/tutorials/craft-a-snap) to get a hands-on introduction.

## Installation

Snapcraft is available on Linux, Windows, and macOS.

*   **Snap-ready Systems:**
    ```bash
    sudo snap install snapcraft --classic
    ```
*   **Traditional Package:**
    Snapcraft can also be installed as a traditional package on many popular Linux repositories.
    The documentation covers how to [set up Snapcraft](https://documentation.ubuntu.com/snapcraft/stable/how-to/setup/set-up-snapcraft).

## Documentation and Support

*   Comprehensive [Snapcraft documentation](https://documentation.ubuntu.com/snapcraft/stable) provides detailed guidance.
*   Engage with the [Snapcraft Forum](https://forum.snapcraft.io) and [Snapcraft Matrix channel](https://matrix.to/#/#snapcraft:ubuntu.com) for community support.
*   Report issues on the [GitHub repository](https://github.com/canonical/snapcraft/issues).

## Contribute

Snapcraft is an open-source project and welcomes contributions.
Get started with the [contribution guide](CONTRIBUTING.md).

## License

Snapcraft is released under the [GPL-3.0 license](LICENSE).

© 2015-2025 Canonical Ltd.