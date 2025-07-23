# iCloud Photos Downloader: Effortlessly Back Up Your iCloud Photos

**Back up and archive your precious iCloud photos with ease using the powerful and versatile iCloud Photos Downloader.** ([View the project on GitHub](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting various devices (laptop, desktop, and NAS).
*   **Multiple Download Modes:** Choose the right approach for your needs:
    *   **Copy:** Download new photos from iCloud.
    *   **Sync:** Download new photos and automatically delete local files removed from iCloud.
    *   **Move:** Download new photos and delete them from iCloud (use with caution!).
*   **Comprehensive Media Support:** Downloads Live Photos (images and videos) and RAW images (including RAW+JPEG).
*   **Smart Features:**
    *   Automatic de-duplication of photos.
    *   Option to continuously monitor iCloud for changes.
    *   Optimized for incremental downloads.
*   **Metadata Preservation:** Option to update photo metadata (EXIF) to maintain your photos' original information.
*   **Flexible Installation:** Available as an executable, through package managers (Docker, PyPI, AUR, npm), or by building from source.

## Prerequisites for iCloud

To ensure smooth operation, configure your iCloud account as follows:

*   **Enable Web Access:** On your iPhone/iPad, go to `Settings > [Your Name] > iCloud > Access iCloud Data on the Web`.
*   **Disable Advanced Data Protection:** On your iPhone/iPad, go to `Settings > [Your Name] > iCloud > Advanced Data Protection`

## Installation and Running

You can install and run `icloudpd` in several ways:

1.  **Download Executable:** Download the pre-built executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2) page.
2.  **Package Managers:** Use your preferred package manager (Docker, PyPI, AUR, npm) for easy installation, updates, and, in some cases, running.  See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for details.
3.  **Build from Source:** Build and run the tool directly from the source code.  See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for build instructions.

## Usage Example

To synchronize your iCloud photos to a local directory every hour:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Important:**  Remember to use the `icloudpd` command.

## Experimental Mode

Explore cutting-edge features in the experimental mode.  See [EXPERIMENTAL.md](EXPERIMENTAL.md) for details.

## Contributing

We welcome contributions!  Review the [contributing guidelines](CONTRIBUTING.md) to learn how to get involved.