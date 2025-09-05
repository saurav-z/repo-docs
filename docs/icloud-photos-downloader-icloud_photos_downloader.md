# iCloud Photos Downloader: Effortlessly Download Your iCloud Photos (and More!)

Tired of your iCloud photos being locked away? **iCloud Photos Downloader** is a powerful command-line tool that lets you download, sync, and manage your entire iCloud photo library, all from your computer. [Get started today!](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, across laptops, desktops, and NAS devices.
*   **Multiple Operation Modes:** Choose how you want to manage your photos:
    *   **Copy:** Download new photos from iCloud (default).
    *   **Sync:** Download new photos and delete local files removed from iCloud (using the `--auto-delete` option).
    *   **Move:** Download new photos and delete them from iCloud (using the `--keep-icloud-recent-days` option).
*   **Comprehensive Media Support:** Handles Live Photos (image and video), RAW images (including RAW+JPEG), and more.
*   **Intelligent De-duplication:** Automatically avoids downloading duplicate photos.
*   **Continuous Synchronization:** Monitor iCloud for changes and download updates automatically (using the `--watch-with-interval` option).
*   **Incremental Download Optimizations:** Utilize options like `--until-found` and `--recent` for faster, more efficient downloads.
*   **Metadata Preservation:** Option to update photo metadata (EXIF) to preserve important information (using the `--set-exif-datetime` option).
*   **Flexible Installation:** Available as an executable, via package managers (Docker, PyPI, AUR, npm), and from source code.
*   **Much More:** Explore additional options with `--help` to customize your experience.

## iCloud Prerequisites

To ensure successful downloads, configure your iCloud account as follows:

*   **Enable "Access iCloud Data on the Web":** In your iPhone/iPad settings: `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable "Advanced Data Protection":** In your iPhone/iPad settings: `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation & Usage

### Installation Methods

Choose your preferred installation method:

1.  **Executable:** Download pre-built executables for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.32.2) page.
2.  **Package Managers:** Install using [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), or [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
3.  **From Source:** Build and run from the source code.

### Example Usage: Synchronizing Your Photos

To automatically synchronize your iCloud photo collection to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> [!IMPORTANT]
>  Remember to use the `icloudpd` executable, not `icloud`.

> [!TIP]
> Customize synchronization with command-line parameters. Run `icloudpd --help` for a full list.

### Authentication Only

To independently create and authorize a session (and complete 2SA/2FA validation if needed) on your local system:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

> [!TIP]
> This feature can also be used to check and verify that the session is still authenticated.

## Experimental Mode

Stay informed about upcoming features in the experimental mode.  [Details](EXPERIMENTAL.md)

## Contribute

We welcome contributions!  Check out the [contributing guidelines](CONTRIBUTING.md) to get involved.