# iCloud Photos Downloader: Effortlessly Download Your iCloud Photos 📸

Easily back up your precious memories with **iCloud Photos Downloader**, a powerful command-line tool designed to download all your photos and videos from iCloud to your local device.  [Check out the original repository here](https://github.com/icloud-photos-downloader/icloud_photos_downloader).

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting laptops, desktops, and NAS devices.
*   **Multiple Installation Options:** Available as an executable, or installable via package managers and ecosystems like Docker, PyPI, AUR, and npm.
*   **Flexible Download Modes:** Choose between "Copy", "Sync", and "Move" modes to manage your photo library effectively.
*   **Comprehensive Media Support:** Downloads Live Photos (image and video), RAW images (including RAW+JPEG), and handles various media types.
*   **Smart Features:** Includes automatic de-duplication, continuous monitoring for iCloud changes, and optimizations for incremental downloads.
*   **Metadata Preservation:** Updates photo metadata (EXIF) for enhanced organization and searchability.
*   **Efficient Operations:** `--until-found` and `--recent` options for optimized incremental runs.
*   **Advanced Options:** Offers `--watch-with-interval`, `--set-exif-datetime` and many more (use `--help` option for full list)

## iCloud Prerequisites

Before you begin, configure your iCloud account with these settings to ensure successful downloads:

*   **Enable Access iCloud Data on the Web:**  On your iPhone / iPad, enable `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable Advanced Data Protection:** On your iPhone /iPad disable `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation and Usage

### Installation

Install `icloudpd` with these methods:

1.  **Executable Download:** Get the latest executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.32.2) page.
2.  **Package Managers:** Utilize package managers such as Docker, PyPI, AUR, or npm.
3.  **Build from Source:** Build and run the tool from the source code.

Find more detailed installation instructions in the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

### Example Usage

To synchronize your iCloud photo collection to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> [!IMPORTANT]
> Remember to use the `icloudpd` executable, not `icloud`.

### Authentication

Independently create and authorize a session on your local system:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

> [!TIP]
> This feature can also be used to check and verify that the session is still authenticated.

## Experimental Mode

Explore cutting-edge features in the experimental mode. Learn more in the [EXPERIMENTAL.md](EXPERIMENTAL.md) file.

## Contribute

Want to help improve iCloud Photos Downloader?  We welcome your contributions!  Read the [contributing guidelines](CONTRIBUTING.md) to get started.