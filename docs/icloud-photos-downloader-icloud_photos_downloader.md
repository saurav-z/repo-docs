# iCloud Photos Downloader: Download Your iCloud Photos Easily

Tired of being locked into iCloud? **iCloud Photos Downloader** is the perfect command-line tool to effortlessly download all your photos and videos from iCloud, offering you complete control of your memories.  [See the original repo](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml) [![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting desktops, laptops, and NAS devices.
*   **Multiple Download Modes:** Choose between Copy, Sync (with auto-delete), and Move (with iCloud deletion) modes.
*   **Live Photo & RAW Support:** Downloads Live Photos (image and video) and RAW images (including RAW+JPEG).
*   **Smart De-duplication:** Automatically prevents downloading duplicate photos with the same name.
*   **Incremental & Continuous Download:**  Download once or continuously monitor for changes in iCloud.
*   **Metadata Preservation:**  Option to update photo metadata (EXIF data).
*   **Flexible Installation:**  Available as an executable, Docker image, and through package managers like PyPI, AUR, and npm.

## iCloud Prerequisites

Before you start downloading, ensure your iCloud account has the following settings enabled to avoid "ACCESS_DENIED" errors:

*   **Enable Access iCloud Data on the Web:** In your iPhone/iPad settings: `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable Advanced Data Protection:** In your iPhone/iPad settings: `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation & Usage

### Installation

You can install iCloud Photos Downloader in several ways:

1.  **Download Executable:** Download the executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.30.0) page.
2.  **Package Managers:** Use Docker, PyPI, AUR, or npm for installation.  See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for details.
3.  **Build from Source:** Build and run the application from the source code.

### Basic Usage Example

To keep your iCloud photos synchronized to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> **Important:**  Use `icloudpd`, not `icloud` when running the command.

For more advanced options and detailed instructions, refer to the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) and use the `--help` option to explore the full command-line interface.

## Experimental Mode

Explore cutting-edge features in experimental mode.  Read more about it in [EXPERIMENTAL.md](EXPERIMENTAL.md).

## Contributing

We welcome contributions!  If you'd like to help improve iCloud Photos Downloader, please check out the [contributing guidelines](CONTRIBUTING.md).