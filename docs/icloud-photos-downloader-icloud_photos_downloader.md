# iCloud Photos Downloader: Easily Back Up Your iCloud Photos

Tired of relying on iCloud's storage and ready to take control of your photo library? **iCloud Photos Downloader is the command-line tool you need to securely download and back up all your iCloud photos to your computer or NAS.**

[View the Original Repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, perfect for desktops, laptops, and NAS devices.
*   **Multiple Download Modes:** Choose between Copy, Sync (with auto-delete), and Move to manage your photos.
*   **Comprehensive File Support:** Downloads Live Photos (images and videos) and RAW images (including RAW+JPEG).
*   **Metadata Preservation:** Preserves and updates photo metadata (EXIF) during download.
*   **Automatic De-duplication:** Handles duplicate file names to avoid conflicts.
*   **Flexible Synchronization:** Offers both one-time downloads and continuous monitoring for iCloud changes with configurable intervals.
*   **Incremental Downloads:** Optimized for efficient incremental downloads using options like `--until-found` and `--recent`.
*   **Easy Installation:** Available as a direct executable and through package managers such as [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), and [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
*   **Open Source & Community Driven:** Developed and maintained by volunteers; contributions are always welcome!

## iCloud Prerequisites

To ensure smooth operation, please configure your iCloud account as follows:

*   **Enable "Access iCloud Data on the Web":**  Navigate to `Settings > Apple ID > iCloud > Access iCloud Data on the Web` on your iPhone/iPad.
*   **Disable "Advanced Data Protection":**  Navigate to `Settings > Apple ID > iCloud > Advanced Data Protection` on your iPhone/iPad.

## Getting Started

### Installation

You can install `icloudpd` in several ways:

1.  **Download Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.28.2) page.
2.  **Package Managers:** Install, update, and run the tool using package managers like Docker, PyPI, AUR, and npm (see [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for details).
3.  **Build from Source:** Build and run from the source code.

See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed installation instructions.

### Usage Examples

To keep your iCloud photo collection synchronized:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> [!IMPORTANT]
> Use the executable `icloudpd` instead of `icloud`.

To authorize a session:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

> [!TIP]
> This can also verify a session's authentication status.

## Experimental Mode

Check out the [EXPERIMENTAL.md](EXPERIMENTAL.md) file for features that are under development.

## Contributing

Interested in contributing? Check out the [contributing guidelines](CONTRIBUTING.md).