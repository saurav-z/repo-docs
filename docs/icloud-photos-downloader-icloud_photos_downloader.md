# iCloud Photos Downloader: Securely Download & Back Up Your iCloud Photos 📸

Easily back up your precious memories with **iCloud Photos Downloader**, a versatile command-line tool for downloading all your iCloud photos and videos, with robust features and cross-platform compatibility.  ([Original Repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS across various devices (laptop, desktop, NAS).
*   **Multiple Operation Modes:**
    *   **Copy:** Download new photos from iCloud (default).
    *   **Sync:** Download new photos and delete local files removed from iCloud (`--auto-delete`).
    *   **Move:** Download new photos and delete photos from iCloud (`--keep-icloud-recent-days`).
*   **Comprehensive Media Support:**  Downloads Live Photos (image and video), RAW images (including RAW+JPEG), and various video formats.
*   **Intelligent File Handling:** Automatic de-duplication of photos with the same name.
*   **Flexible Download Options:**  One-time download and continuous monitoring for iCloud changes (`--watch-with-interval`).
*   **Efficient Incremental Downloads:** Optimized for incremental runs with `--until-found` and `--recent` options.
*   **Metadata Preservation:** Option to update photo metadata (EXIF) (`--set-exif-datetime`).
*   **Installation Variety:** Available via executable downloads, Docker, PyPI, AUR, and npm.

## iCloud Prerequisites

Before using iCloud Photos Downloader, please ensure these settings are configured in your iCloud account to avoid "ACCESS_DENIED" errors:

*   **Enable Access iCloud Data on the Web:**  `Settings > Apple ID > iCloud > Access iCloud Data on the Web` (iPhone/iPad)
*   **Disable Advanced Data Protection:** `Settings > Apple ID > iCloud > Advanced Data Protection` (iPhone/iPad)

## Installation & Running

You can use iCloud Photos Downloader in several ways:

1.  **Executable Download:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.4) page.
2.  **Package Managers:** Install via package managers like Docker, PyPI, AUR, or npm. See the [Installation Guide](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for details.
3.  **Build from Source:** Build and run from the source code.

## Usage Examples

To synchronize your iCloud photo library to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

To create and authorize a session (including 2FA validation):

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

> [!IMPORTANT]
> Use `icloudpd`, *not* `icloud` for the executable name.

> [!TIP]
> Adjust synchronization behavior using command-line parameters. Run `icloudpd --help` for a comprehensive list.

## Experimental Mode

Explore cutting-edge features in the experimental mode before they are added to the main package.  See [EXPERIMENTAL.md](EXPERIMENTAL.md) for more details.

## Contributing

We welcome contributions!  Check out the [contributing guidelines](CONTRIBUTING.md) to learn how you can get involved and help improve iCloud Photos Downloader.