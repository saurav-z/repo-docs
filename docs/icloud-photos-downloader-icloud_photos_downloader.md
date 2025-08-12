# iCloud Photos Downloader: Download Your iCloud Photos with Ease

Tired of being locked into Apple's ecosystem? **iCloud Photos Downloader** is a powerful command-line tool that allows you to easily download all your iCloud photos and videos to your local storage, giving you complete control over your memories. ([View on GitHub](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, on laptops, desktops, and NAS devices.
*   **Multiple Download Modes:** Choose from "Copy" (default), "Sync" (with auto-delete), and "Move" (delete from iCloud).
*   **Comprehensive Media Support:** Downloads Live Photos (image and video), RAW images (including RAW+JPEG), and handles photo de-duplication.
*   **Continuous Monitoring:**  Option to monitor for iCloud changes automatically using the `--watch-with-interval` option.
*   **Optimized for Incremental Downloads:** Features like `--until-found` and `--recent` for efficient incremental runs.
*   **Metadata Preservation:**  Option to preserve or update photo metadata (EXIF) with the `--set-exif-datetime` option.
*   **Flexible Installation:** Available as an executable, through package managers like Docker, PyPI, AUR, npm, and from source.

## iCloud Prerequisites

Before you start, ensure your iCloud account settings are configured correctly to avoid access errors:

*   **Enable Access iCloud Data on the Web:**  On your iPhone/iPad, enable `Settings > Apple ID > iCloud > Access iCloud Data on the Web`.
*   **Disable Advanced Data Protection:** On your iPhone/iPad, disable `Settings > Apple ID > iCloud > Advanced Data Protection`.

## Installation and Running

You can run `icloudpd` using the following methods:

1.  **Executable Download:** Download the executable for your platform from the [GitHub Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.3) page.
2.  **Package Managers:** Install via Docker, PyPI, AUR, or npm (see [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html)).
3.  **Build from Source:** Build and run from the source code.

See the [Installation Guide](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed instructions.

## Usage Examples

**Synchronize your iCloud photo collection to a local directory:**

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Authorize a session (and complete 2SA/2FA validation):**

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

> **Important:** Remember to use `icloudpd` (not `icloud`) as the executable.

## Experimental Mode

Explore new features and changes in the [Experimental Mode](EXPERIMENTAL.md) before they are integrated into the main package.

## Contributing

We welcome contributions! Review our [contributing guidelines](CONTRIBUTING.md) to get involved in making iCloud Photos Downloader even better.