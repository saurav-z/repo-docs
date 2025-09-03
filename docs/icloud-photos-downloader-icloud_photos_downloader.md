# iCloud Photos Downloader: Securely Download Your iCloud Photos

**Easily and reliably back up your precious memories by downloading your iCloud photos to your computer with iCloud Photos Downloader.**

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[View the project on GitHub](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Multiple Installation Options:** Available as an executable, and through package managers (Docker, PyPI, AUR, npm) for easy setup and updates.
*   **Flexible Download Modes:** Choose from Copy, Sync, or Move to manage your photo downloads.
*   **Live Photo & RAW Support:** Downloads both image and video components of Live Photos, and supports RAW images (including RAW+JPEG).
*   **Automatic Deduplication:** Prevents downloading duplicate photos with the same name.
*   **Continuous Monitoring:**  Option to watch for iCloud changes and automatically download new photos.
*   **Metadata Preservation:**  Option to update photo EXIF data.
*   **Optimized for Incremental Downloads:** Features options for efficient downloads (`--until-found` and `--recent`).

## iCloud Prerequisites

Before using iCloud Photos Downloader, ensure the following settings are configured in your iCloud account:

*   **Enable Access iCloud Data on the Web:** Enable this setting on your iPhone/iPad: `Settings > Apple ID > iCloud > Access iCloud Data on the Web`.
*   **Disable Advanced Data Protection:**  Disable this setting on your iPhone/iPad: `Settings > Apple ID > iCloud > Advanced Data Protection`.

## Installation & Usage

You can install and run `icloudpd` in several ways:

1.  **Download Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.32.2) page and run it.
2.  **Package Managers:** Use package managers like Docker, PyPI, AUR, or npm. See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for instructions.
3.  **Build from Source:** Build and run the tool from the source code.

### Basic Usage

To automatically download and synchronize your iCloud photo collection to a local directory:

```bash
icloudpd --directory /path/to/your/photos --username your@email.com --watch-with-interval 3600
```

### Authentication

You can also independently create and authorize a session for secure access.

```bash
icloudpd --username your@email.com --password your_password --auth-only
```

**Important Note:**  The executable is called `icloudpd`, not `icloud`.

## Experimental Mode

Test out new features before they graduate into the main package!  See [EXPERIMENTAL.md](EXPERIMENTAL.md) for details.

## Contribute

We welcome contributions!  Check out the [contributing guidelines](CONTRIBUTING.md) to get involved.