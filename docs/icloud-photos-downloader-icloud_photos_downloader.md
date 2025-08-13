# iCloud Photos Downloader: Download and Back Up Your iCloud Photos Easily

Tired of being locked into iCloud? **iCloud Photos Downloader** is a powerful command-line tool that lets you download and back up your iCloud photos and videos to your computer. ([See the original repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, making it easy to back up your photos from any device.
*   **Multiple Installation Options:**  Download and run directly as an executable, or install via package managers such as Docker, PyPI, AUR, and npm.
*   **Flexible Download Modes:** Choose between "Copy," "Sync," and "Move" modes to manage your photo library effectively.
*   **Supports Various Media Types:** Download Live Photos (image and video), RAW images (including RAW+JPEG), and other media types.
*   **Automatic De-duplication:** Prevents redundant downloads by automatically identifying and skipping photos with the same names.
*   **Continuous Monitoring:**  Keep your local backup up-to-date with the `--watch-with-interval` option, ensuring your photos are always backed up.
*   **Metadata Preservation:**  Preserves photo metadata (EXIF data) with the `--set-exif-datetime` option.
*   **Optimized Incremental Downloads:**  Utilize `--until-found` and `--recent` options for faster incremental downloads.

## iCloud Prerequisites

To ensure a smooth download experience, configure your iCloud account with the following:

*   **Enable Access iCloud Data on the Web:** In your iPhone/iPad settings: `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable Advanced Data Protection:** In your iPhone/iPad settings: `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation and Running

You can run `icloudpd` in three ways:

1.  **Executable:** Download the executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.4) section and run it.
2.  **Package Manager:**  Install, update, and, in some cases, run using package managers like [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), or [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
3.  **From Source:** Build and run from the source code.

For detailed installation instructions, refer to the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

## Example Usage

To synchronize your iCloud photo collection with your local system:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Note:** The executable name is `icloudpd`, not `icloud`.  Explore the command-line options further with `icloudpd --help`.

To independently create and authorize a session:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Experimental Mode

Explore new features and changes in the [Experimental Mode](EXPERIMENTAL.md) before they are integrated into the main package.

## Contributing

We welcome contributions!  Review our [contributing guidelines](CONTRIBUTING.md) to get started.