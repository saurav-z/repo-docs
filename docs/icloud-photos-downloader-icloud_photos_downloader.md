# iCloud Photos Downloader: Download Your iCloud Photos with Ease

Easily back up and manage your precious memories by downloading all your iCloud photos to your local device with **iCloud Photos Downloader** - a versatile and cross-platform command-line tool. [(Back to Original Repo)](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Multiple Download Modes:** Copy, Sync (with auto-delete), and Move options for flexible management.
*   **Live Photo & RAW Image Support:** Download Live Photos (image and video) and RAW images, including RAW+JPEG.
*   **Automatic De-duplication:** Prevents downloading duplicate photos with the same names.
*   **Continuous Monitoring:** Option to watch for iCloud changes continuously.
*   **Incremental Downloads:** Optimized with `--until-found` and `--recent` options for efficient incremental updates.
*   **Metadata Preservation:**  Option to update photo metadata (EXIF).
*   **Multiple Installation Options:** Available as an executable, via package managers (Docker, PyPI, AUR, npm), and from source.

## iCloud Prerequisites

To ensure the iCloud Photos Downloader works correctly, please configure your iCloud account as follows:

*   **Enable Access iCloud Data on the Web:**  `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable Advanced Data Protection:** `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation

You can install and run `icloudpd` in several ways:

1.  **Download Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.4) page.
2.  **Package Managers:** Install via Docker, PyPI, AUR, or npm (See [Installation Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for details).
3.  **Build from Source:** Build and run the tool from the source code.

## Usage

**Basic Synchronization:**

To keep your iCloud photo collection synchronized to a local directory, use a command like this:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> [!IMPORTANT]
> Remember to use `icloudpd`, not `icloud`, when running the executable.

**Authentication:**

To create and authorize a session independently (and complete 2FA validation if needed):

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Experimental Mode

Explore cutting-edge features in the [Experimental Mode](EXPERIMENTAL.md) before they graduate into the main package.

## Contributing

We welcome contributions!  Please review our [contributing guidelines](CONTRIBUTING.md) to get involved.