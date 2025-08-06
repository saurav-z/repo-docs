# Download Your iCloud Photos Easily with iCloud Photos Downloader

**Tired of being locked into iCloud?** iCloud Photos Downloader is a powerful command-line tool that lets you download your entire iCloud photo library to your computer, giving you complete control over your memories. You can find the original repo [here](https://github.com/icloud-photos-downloader/icloud_photos_downloader).

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Multiple Installation Options:** Downloadable executables, Docker, PyPI, AUR, and npm packages.
*   **Flexible Download Modes:**
    *   **Copy:** Download new photos (default).
    *   **Sync:** Download new photos and delete locally removed ones (with `--auto-delete`).
    *   **Move:** Download and delete photos from iCloud (with `--keep-icloud-recent-days`).
*   **Advanced Support:** Handles Live Photos (separate image and video), RAW images (including RAW+JPEG).
*   **Smart Features:** Automatic de-duplication, incremental downloads, photo metadata (EXIF) updates, and continuous monitoring for iCloud changes.

## iCloud Prerequisites

Before using iCloud Photos Downloader, please ensure your iCloud account is configured as follows:

*   **Enable Access iCloud Data on the Web:**  `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable Advanced Data Protection:**  `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation and Usage

Choose your preferred installation method:

1.  **Download Executable:** Get the latest release from the [GitHub Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2).
2.  **Package Managers:** Install via Docker, PyPI, AUR, or npm.
3.  **Build from Source:** Compile and run the program yourself.

Detailed installation instructions are available in the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

### Example Usage

To automatically synchronize your iCloud photos to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

Remember to use `icloudpd`, not `icloud`. For a comprehensive list of commands and options, run `icloudpd --help`.

## Experimental Mode

Explore cutting-edge features in [EXPERIMENTAL.md](EXPERIMENTAL.md) before they are integrated into the main release.

## Contributing

We welcome contributions! Check out the [contributing guidelines](CONTRIBUTING.md) to get started.