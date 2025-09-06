# iCloud Photos Downloader: Download Your iCloud Photos Effortlessly

Tired of being locked into the iCloud ecosystem? **iCloud Photos Downloader** is the perfect command-line tool to securely download all your precious photos and videos from iCloud to your computer or NAS.

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[**View the project on GitHub**](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, suitable for laptops, desktops, and NAS devices.
*   **Multiple Download Modes:** Choose between Copy (default), Sync (with auto-delete), and Move (delete from iCloud).
*   **Live Photo & RAW Support:** Downloads both the image and video components of Live Photos and supports RAW images (including RAW+JPEG).
*   **Automatic De-Duplication:** Prevents duplicate downloads.
*   **Continuous Monitoring:**  Options to download once or continuously monitor iCloud for new changes using the `--watch-with-interval` option.
*   **Incremental Downloads:** Optimizations with `--until-found` and `--recent` to download only new or specific files.
*   **Metadata Preservation:** Option to update EXIF data (`--set-exif-datetime` option).
*   **Multiple Install Options:**  Available as an executable, through package managers (Docker, PyPI, AUR, npm), or by building from source.

## iCloud Prerequisites

Before using iCloud Photos Downloader, please configure your iCloud account:

*   **Enable Access iCloud Data on the Web:**  In your iPhone/iPad settings: `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable Advanced Data Protection:** In your iPhone/iPad settings: `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation and Running

You can install and run `icloudpd` in several ways:

1.  **Download Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.32.2) page.
2.  **Package Managers:** Install via package managers:
    *   [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker)
    *   [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi)
    *   [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur)
    *   [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm)
3.  **Build from Source:** Build and run the tool from the source code.

## Usage Examples

To continuously sync your iCloud photos to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

To create and authorize a session:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Contributing

We welcome contributions!  Review the [contributing guidelines](CONTRIBUTING.md) to get involved in the project.

## Further Information

*   [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/)
*   [Issues](https://github.com/icloud-photos-downloader/icloud_photos_downloader/issues)