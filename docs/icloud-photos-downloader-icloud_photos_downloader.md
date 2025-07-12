# iCloud Photos Downloader: Download and Back Up Your iCloud Photos Easily

Easily back up and manage your iCloud photo library with the **iCloud Photos Downloader**, a powerful command-line tool.  [Visit the original repository on GitHub](https://github.com/icloud-photos-downloader/icloud_photos_downloader).

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Multiple Installation Options:** Available via executable downloads, Docker, PyPI, AUR, and npm for flexible installation.
*   **Sync Modes:** Copy, Sync (with auto-delete), and Move (delete from iCloud) modes for versatile photo management.
*   **Live Photo and RAW Support:** Downloads both Live Photos (image & video) and RAW images (including RAW+JPEG).
*   **Automatic De-duplication:** Avoids downloading duplicate photos.
*   **Continuous Monitoring:** Supports one-time downloads and continuous monitoring for iCloud changes.
*   **Metadata Preservation:**  Updates photo metadata (EXIF) to maintain details.
*   **Incremental Downloads:** Optimized options for efficient incremental downloads.

## iCloud Prerequisites

To use iCloud Photos Downloader, ensure your iCloud account has the following settings enabled:

*   **Enable Access iCloud Data on the Web:**  `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable Advanced Data Protection:** `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation and Running

You can run `icloudpd` using these methods:

1.  **Download Executable:**  Download the executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.28.2) page.
2.  **Package Managers:** Install via Docker, PyPI, AUR, or npm (see [Installation Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for details).
3.  **Build from Source:** Build and run from the source code.

## Usage Examples

**Sync your iCloud photos to a local directory:**

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Authenticate your session separately:**

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

**Note:**  The executable is named `icloudpd`, not `icloud`.

## Contributing

We welcome contributions! Please review our [contributing guidelines](CONTRIBUTING.md) to learn how you can help.