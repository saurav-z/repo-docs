# iCloud Photos Downloader: Easily Download Your iCloud Photos (and Keep Them Safe!)

Tired of your precious memories being locked in iCloud? **iCloud Photos Downloader** is a powerful command-line tool that lets you effortlessly download and back up your entire iCloud photo library, ensuring you always have access to your photos.  Find the original repo here: [https://github.com/icloud-photos-downloader/icloud_photos_downloader](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features:

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Multiple Installation Options:** Download as an executable, or install via Docker, PyPI, AUR, or npm.
*   **Flexible Download Modes:** Choose between copying, syncing, or moving your photos.
*   **Live Photo and RAW Image Support:** Downloads both the image and video components of Live Photos, and supports RAW and RAW+JPEG images.
*   **Automatic De-duplication:** Avoids downloading duplicate photos.
*   **Continuous Monitoring:** Option to monitor for iCloud changes and automatically download new photos.
*   **Metadata Preservation:**  Option to update photo metadata (EXIF).

## iCloud Prerequisites

To ensure the tool works correctly, configure your iCloud account with these settings:

*   **Enable Web Access:** In your iPhone/iPad settings, enable `Settings > Apple ID > iCloud > Access iCloud Data on the Web`.
*   **Disable Advanced Data Protection:** On your iPhone/iPad disable `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation and Usage

You can install and run `icloudpd` in several ways:

1.  **Download Executable:** Download the pre-built executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.28.2) page.
2.  **Package Manager:** Install using Docker, PyPI, AUR, or npm (see the [Installation Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for details).
3.  **Build from Source:** Build and run the tool from the source code.

### Example Usage:

To keep your iCloud photos synchronized to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```
> [!IMPORTANT]
> Use `icloudpd`, not `icloud` to run the tool.