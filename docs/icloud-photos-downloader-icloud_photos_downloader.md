# iCloud Photos Downloader: Effortlessly Download Your iCloud Photos

**Tired of being locked into iCloud?** iCloud Photos Downloader is a powerful command-line tool that allows you to download all your precious photos and videos from iCloud, giving you complete control over your memories.

[View the original repository on GitHub](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting laptops, desktops, and NAS devices.
*   **Multiple Download Modes:** Choose between copy, sync, and move modes to fit your needs.
*   **Live Photo & RAW Support:** Download both the image and video components of Live Photos, as well as RAW images (including RAW+JPEG).
*   **Automated Features:** Includes automatic de-duplication, incremental downloads, and EXIF metadata updates.
*   **Flexible Installation:** Available as an executable, or via package managers like Docker, PyPI, AUR, and npm.
*   **Continuous Monitoring:** Optionally watch for iCloud changes and automatically download new content.
*   **Full Command-Line Control:** Offers extensive options for customization (use `--help` to see a full list).

## iCloud Prerequisites

To ensure the iCloud Photos Downloader functions correctly, please configure your iCloud account as follows:

*   **Enable "Access iCloud Data on the Web":** Within your iPhone/iPad settings: `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable "Advanced Data Protection":** Within your iPhone/iPad settings: `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation

You can run `icloudpd` in several ways:

1.  **Download Executable:** Download the executable for your platform from the [GitHub Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2).
2.  **Package Managers:** Install via Docker, PyPI, AUR, or npm (see the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for details).
3.  **Build From Source:** Build and run the tool from the source code.

## Usage

**Example: Synchronize Your iCloud Photos**

To synchronize your iCloud photo collection to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Important:**  Use `icloudpd` (not `icloud`) to run the tool.  Use `icloudpd --help` for a full list of options.

## Experimental Mode

Explore new features before they graduate into the main package in [EXPERIMENTAL.md](EXPERIMENTAL.md).

## Contributing

We welcome contributions!  Review the [contributing guidelines](CONTRIBUTING.md) to get started.