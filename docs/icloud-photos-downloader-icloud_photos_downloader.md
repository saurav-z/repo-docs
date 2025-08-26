# iCloud Photos Downloader: Effortlessly Download Your iCloud Photos 

**Easily back up your precious memories by downloading all your photos from iCloud with the command-line tool iCloud Photos Downloader!**  [View the original repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, on laptops, desktops, and NAS devices.
*   **Multiple Installation Options:** Available as an executable, and through package managers such as Docker, PyPI, AUR, and npm, giving you flexibility in how you install it.
*   **Versatile Download Modes:**
    *   **Copy:** Download new photos from iCloud (default).
    *   **Sync:** Download new photos and delete local files removed from iCloud (`--auto-delete`).
    *   **Move:** Download new photos and delete photos in iCloud (`--keep-icloud-recent-days`).
*   **Advanced File Support:** Supports Live Photos (image and video), RAW images (including RAW+JPEG), and automatic de-duplication.
*   **Continuous Monitoring:** Choose one-time downloads or continuously monitor for iCloud changes with the `--watch-with-interval` option.
*   **Optimized Downloads:** Uses options like `--until-found` and `--recent` for efficient incremental downloads.
*   **Metadata Preservation:** Updates photo metadata (EXIF) using the `--set-exif-datetime` option.
*   **Easy to Use:** Includes example commands for simple photo synchronization.

## iCloud Prerequisites

To ensure a successful download, please configure your iCloud account as follows:

*   **Enable Access iCloud Data on the Web:**  Enable this setting in your iCloud settings on your iPhone/iPad.
*   **Disable Advanced Data Protection:**  Disable this setting in your iCloud settings on your iPhone/iPad.

## Installation and Usage

You can get started with the iCloud Photos Downloader in a few ways:

1.  **Download Executable:** Download the pre-built executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.31.0) page.
2.  **Package Managers:** Install using Docker, PyPI, AUR, or npm. See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for installation instructions specific to each package manager.
3.  **Build from Source:** Compile and run the tool directly from the source code.

### Example Usage

To automatically synchronize your iCloud photo collection to a local directory every hour:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

To independently create and authorize a session:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```
> [!IMPORTANT]
> Make sure you use `icloudpd` and not `icloud` in your command.

## Experimental Features

Explore cutting-edge updates in the experimental mode.  Find details in the [EXPERIMENTAL.md](EXPERIMENTAL.md) file.

## Contributing

We welcome contributions!  Review the [contributing guidelines](CONTRIBUTING.md) to learn how you can help improve iCloud Photos Downloader.