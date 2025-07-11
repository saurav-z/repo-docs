# iCloud Photos Downloader: Easily Download and Back Up Your iCloud Photos

**Tired of being locked into Apple's ecosystem?** iCloud Photos Downloader is a powerful command-line tool that lets you effortlessly download and back up all your photos and videos from iCloud to your computer or NAS.  [View the project on GitHub](https://github.com/icloud-photos-downloader/icloud_photos_downloader).

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features:

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Flexible Installation:** Available as an executable, via package managers (Docker, PyPI, AUR, npm), or from source.
*   **Multiple Download Modes:** Choose from "Copy," "Sync," and "Move" modes for various backup and management strategies.
*   **Comprehensive Media Support:** Handles Live Photos (images and videos), RAW images (including RAW+JPEG), and more.
*   **Intelligent Features:** Includes automatic de-duplication, continuous monitoring, and EXIF data preservation.
*   **Easy to Use:** Simple command-line interface for effortless downloads and synchronization.

## iCloud Prerequisites

To ensure a smooth experience, please configure your iCloud account as follows:

*   **Enable Web Access:** On your iPhone/iPad, go to `Settings > Apple ID > iCloud > Access iCloud Data on the Web`.
*   **Disable Advanced Data Protection:** On your iPhone/iPad, go to `Settings > Apple ID > iCloud > Advanced Data Protection` and disable it.

## Installation and Running

You have multiple options to install and run `icloudpd`:

1.  **Executable:** Download the pre-built executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.28.2).
2.  **Package Managers:** Install using your preferred package manager: [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
3.  **Build from Source:** Build and run from the source code.

Refer to the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed instructions.

## Example Usage:

To download your photos and continuously synchronize them to a directory:

```bash
icloudpd --directory /path/to/your/photos --username your@email.com --watch-with-interval 3600
```

Use `--help` for a full list of options and features.

## Contributing

Help improve iCloud Photos Downloader!  Check out the [contributing guidelines](CONTRIBUTING.md) to get involved.