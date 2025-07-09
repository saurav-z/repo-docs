# iCloud Photos Downloader: Download & Back Up Your iCloud Photos

**Easily and securely download all your photos and videos from iCloud with the open-source iCloud Photos Downloader.**  [View the original repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, across laptops, desktops, and NAS devices.
*   **Multiple Installation Options:** Available as an executable, and through package managers like Docker, PyPI, AUR, and npm, offering flexibility in installation.
*   **Three Download Modes:**
    *   **Copy (Default):** Downloads new photos from iCloud.
    *   **Sync:** Downloads new photos and deletes local files removed from iCloud (`--auto-delete`).
    *   **Move:** Downloads new photos and deletes them from iCloud (`--keep-icloud-recent-days`).
*   **Comprehensive Media Support:** Handles Live Photos (image and video), RAW images (including RAW+JPEG).
*   **Intelligent Features:**  Includes automatic de-duplication, incremental download options, and metadata (EXIF) updates.
*   **Continuous Monitoring:** Optionally monitors iCloud for changes and downloads new photos automatically (`--watch-with-interval`).

## iCloud Prerequisites

To ensure successful downloads, configure your iCloud account as follows:

*   **Enable Web Access:** In your iCloud settings on your iPhone/iPad, enable `Settings > Apple ID > iCloud > Access iCloud Data on the Web`.
*   **Disable Advanced Data Protection:** On your iPhone/iPad, disable `Settings > Apple ID > iCloud > Advanced Data Protection`.

## Installation and Usage

You have several options to install and run iCloud Photos Downloader:

1.  **Download Executable:** Get the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.28.2) page.
2.  **Use Package Manager:** Install and update via [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), or [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
3.  **Build from Source:** Compile and run the tool from the source code.

Detailed installation instructions are available in the [documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

### Basic Usage Example

To continuously synchronize your iCloud photo library to your local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> **Important:**  Use `icloudpd`, not `icloud`, in your commands.

### Authentication

You can also independently create and authorize a session (and complete 2SA/2FA validation if needed) on your local system:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Experimental Mode

Explore new features and changes in the experimental mode before they become part of the main package.  See [EXPERIMENTAL.md](EXPERIMENTAL.md) for details.

## Contribute

We welcome contributions!  Review the [contributing guidelines](CONTRIBUTING.md) to learn how you can get involved and help improve iCloud Photos Downloader.