# iCloud Photos Downloader: Download and Back Up Your iCloud Photos (and More!)

Tired of being locked into iCloud? **iCloud Photos Downloader** is a versatile command-line tool that empowers you to securely download and manage your entire iCloud photo library on your own terms. [See the original repo](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Multiple Download Modes:** Choose from Copy, Sync (with automatic deletion of local files), and Move (download and delete from iCloud).
*   **Comprehensive Media Support:** Downloads Live Photos (image and video), RAW images (including RAW+JPEG), and more.
*   **Intelligent Handling:** Includes automatic de-duplication of photos with the same names and metadata updates.
*   **Continuous Monitoring:** Option to monitor iCloud for changes continuously, with customizable intervals.
*   **Flexible Operation:** Supports one-time downloads and incremental updates for efficient backups.
*   **Multiple Installation Options:** Available as an executable, or via package managers like Docker, PyPI, AUR, and npm.
*   **Metadata Preservation:** Preserves and updates photo metadata (EXIF).

## iCloud Prerequisites

Before using iCloud Photos Downloader, ensure your iCloud account is configured:

*   **Enable Web Access:** On your iPhone/iPad, enable `Settings > Apple ID > iCloud > Access iCloud Data on the Web`.
*   **Disable Advanced Data Protection:** On your iPhone/iPad disable `Settings > Apple ID > iCloud > Advanced Data Protection`.

## Installation

You can install and run `icloudpd` in several ways:

1.  **Download Executable:** Download the pre-built executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.32.2) page.
2.  **Package Managers:** Utilize package managers like [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), or [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm) for easy installation and updates.
3.  **Build from Source:** Build and run the tool directly from the source code.

For detailed installation instructions, please refer to the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

## Usage

To automatically back up your iCloud photos:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> **Important:**  The executable name is `icloudpd`, not `icloud`.
>
> **Tip:** Customize the synchronization behavior using command-line parameters. Run `icloudpd --help` to see the full list.

To authorize your session:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

> **Tip:** This can also check the session status.

## Experimental Mode

Test out new features before they are fully implemented. Learn more [here](EXPERIMENTAL.md).

## Contribute

Your contributions are welcome! Review the [contributing guidelines](CONTRIBUTING.md) to get involved.