# iCloud Photos Downloader: Securely Download Your iCloud Photos (Command-Line Tool)

Effortlessly back up and manage your precious iCloud photos with the **iCloud Photos Downloader**, a powerful command-line tool.  ([Original Repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml) [![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features of iCloud Photos Downloader:

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, suitable for various devices (laptops, desktops, NAS).
*   **Multiple Download Modes:**  Choose from Copy, Sync, or Move to tailor your download strategy.
*   **Live Photo and RAW Support:**  Downloads both image and video components of Live Photos and handles RAW image formats (including RAW+JPEG).
*   **Automatic De-duplication:**  Intelligently avoids downloading duplicate photos.
*   **Continuous Monitoring:**  Option to watch for and automatically download new iCloud changes.
*   **Incremental Download Options:** Optimizations like `--until-found` and `--recent` speed up incremental runs.
*   **Metadata Preservation:** Option to update photo EXIF data (`--set-exif-datetime`).
*   **Multiple Installation Options:** Install using Docker, PyPI, AUR, or npm, or download executables directly.

## iCloud Prerequisites

To ensure the iCloud Photos Downloader functions correctly, please configure your iCloud account as follows:

*   **Enable Web Access:** On your iPhone/iPad, enable `Settings > Apple ID > iCloud > Access iCloud Data on the Web`.
*   **Disable Advanced Data Protection:**  On your iPhone/iPad, disable `Settings > Apple ID > iCloud > Advanced Data Protection`.

## Installation and Usage

You can install and run iCloud Photos Downloader in several ways:

1.  **Download Executable:** Download the pre-built executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2) section.
2.  **Package Managers:** Install and manage via package managers like Docker, PyPI, AUR, or npm. See [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed instructions.
3.  **Build from Source:**  Build and run the tool from the source code.  See [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

**Example Usage:**

To synchronize your iCloud photos to a local directory and automatically monitor for changes:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> **Important:**  Ensure you use `icloudpd`, not `icloud`, when executing commands.

> **Tip:**  Customize the synchronization behavior with various command-line parameters. Run `icloudpd --help` to explore all options.

## Experimental Mode

Explore new features in the experimental mode before they become part of the main package. [Details](EXPERIMENTAL.md)

## Contributing

We welcome contributions!  Review the [contributing guidelines](CONTRIBUTING.md) to learn how you can help improve iCloud Photos Downloader.