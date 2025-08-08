# iCloud Photos Downloader: Download and Back Up Your iCloud Photos with Ease

**Effortlessly download and back up your precious iCloud photos and videos with the reliable and versatile iCloud Photos Downloader.** (See the original repo [here](https://github.com/icloud-photos-downloader/icloud_photos_downloader).)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml) [![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting laptops, desktops, and NAS devices.
*   **Multiple Installation Options:** Download as an executable or install via package managers like Docker, PyPI, AUR, and npm.
*   **Flexible Operation Modes:**
    *   **Copy:** Downloads new photos from iCloud (default).
    *   **Sync:** Downloads new photos and deletes local files removed from iCloud (`--auto-delete`).
    *   **Move:** Downloads new photos from iCloud and deletes photos from iCloud (`--keep-icloud-recent-days`).
*   **Comprehensive Support:** Handles Live Photos (separate image and video files) and RAW images (including RAW+JPEG).
*   **Intelligent Features:** Automatic de-duplication of photos with the same name.
*   **Continuous Monitoring:** Option to monitor iCloud for changes continuously (`--watch-with-interval`).
*   **Optimized Downloads:** Incremental run optimizations with `--until-found` and `--recent` options.
*   **Metadata Preservation:** Updates photo metadata (EXIF) with the `--set-exif-datetime` option.
*   **And much more!** Explore the full list of features by using the `--help` command-line option.

## iCloud Prerequisites

To ensure the tool works correctly, configure your iCloud account as follows:

*   **Enable Access iCloud Data on the Web:** On your iPhone/iPad, enable `Settings > Apple ID > iCloud > Access iCloud Data on the Web`.
*   **Disable Advanced Data Protection:** On your iPhone/iPad disable `Settings > Apple ID > iCloud > Advanced Data Protection`.

## Installation and Usage

### Installation

Choose your preferred installation method:

1.  **Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2) page.
2.  **Package Managers:** Install using Docker, PyPI, AUR, or npm (see links in the original README).
3.  **Build from Source:** Build and run from the source code.

See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed instructions.

### Basic Usage

To synchronize your iCloud photo collection to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Important:** Remember to use `icloudpd`, not `icloud` when running the command.

To authorize a session independently:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Experimental Mode

Explore cutting-edge features in the experimental mode before they are integrated into the main package. [Details](EXPERIMENTAL.md)

## Contributing

Your contributions are welcome! Review the [contributing guidelines](CONTRIBUTING.md) to learn how to get involved and help improve iCloud Photos Downloader.