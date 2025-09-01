# Download Your iCloud Photos Easily with iCloud Photos Downloader

Need a simple way to back up your iCloud photos?  **iCloud Photos Downloader** is a command-line tool that makes downloading your entire iCloud photo library straightforward and reliable, available for a variety of platforms!  [View the original repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Runs on Linux, Windows, and macOS, supporting desktops, laptops, and NAS devices.
*   **Multiple Download Modes:** Choose between Copy, Sync, and Move modes for flexible management of your photos.
*   **Live Photo and RAW Image Support:** Handles Live Photos (both image and video) and RAW images, including RAW+JPEG pairs.
*   **Automatic De-duplication:** Avoids duplicate downloads by automatically detecting and skipping photos with the same name.
*   **Continuous Monitoring:** Includes options for one-time downloads or continuous monitoring for iCloud changes.
*   **Metadata Preservation:** Preserves and updates photo metadata (EXIF) during the download process.
*   **Incremental Downloads:** Optimizations like `--until-found` and `--recent` for efficient incremental backups.
*   **Flexible Installation:** Available via executables, Docker, PyPI, AUR, and npm.

## Installation

You can install and run `icloudpd` in several ways:

*   **Download Executable:** Get the pre-built executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.32.1) page.
*   **Package Managers:** Install via [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), or [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
*   **Build from Source:** Build and run the application from the source code.

Detailed installation instructions are available in the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

## iCloud Prerequisites

To ensure successful downloads, please configure your iCloud account as follows:

*   **Enable Access iCloud Data on the Web:** On your iPhone / iPad, enable `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable Advanced Data Protection:** On your iPhone /iPad disable `Settings > Apple ID > iCloud > Advanced Data Protection`

## Usage Examples

Keep your iCloud photos synchronized to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

Independently create and authorize a session:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

> [!IMPORTANT]
> Remember to use `icloudpd`, not `icloud` when running commands.

> [!TIP]
> Customize the synchronization process using command-line parameters. Run `icloudpd --help` to see the full list of options.

## Contributing

We welcome contributions!  If you'd like to help improve iCloud Photos Downloader, please review the [contributing guidelines](CONTRIBUTING.md).

## Experimental Mode

Check the [EXPERIMENTAL.md](EXPERIMENTAL.md) file for details of changes added in experimental mode before they graduate into the main package.