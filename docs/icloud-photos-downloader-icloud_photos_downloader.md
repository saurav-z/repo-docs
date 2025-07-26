# iCloud Photos Downloader: Easily Backup and Manage Your iCloud Photos

**Securely download and manage your iCloud photos with ease using iCloud Photos Downloader, a versatile command-line tool.**  ([Back to original repo](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, on various devices (laptop, desktop, and NAS).
*   **Multiple Operation Modes:**
    *   **Copy:** Download new photos from iCloud (default).
    *   **Sync:** Download new photos and delete local files removed from iCloud (`--auto-delete`).
    *   **Move:** Download new photos and delete photos in iCloud (`--keep-icloud-recent-days`).
*   **Comprehensive Media Support:** Downloads Live Photos (image and video), RAW images (including RAW+JPEG).
*   **Intelligent Handling:**
    *   Automatic de-duplication of photos with the same name.
    *   One-time download and continuous monitoring for iCloud changes (`--watch-with-interval`).
    *   Optimizations for incremental downloads (`--until-found` and `--recent`).
*   **Metadata Preservation:** Maintains photo metadata (EXIF) with the `--set-exif-datetime` option.
*   **Flexible Installation:** Available as an executable, and through package managers like Docker, PyPI, AUR, and npm.

## Installation

You can install `icloudpd` in several ways:

1.  **Executable Download:** Download the pre-built executable for your platform from the [GitHub Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2).
2.  **Package Managers:** Install using package managers like [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), or [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
3.  **Build from Source:** Build and run directly from the source code.

For detailed installation instructions, refer to the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

## iCloud Prerequisites

Before using iCloud Photos Downloader, ensure your iCloud account settings are configured correctly to avoid "ACCESS_DENIED" errors:

*   **Enable "Access iCloud Data on the Web":**  Enable this setting on your iPhone/iPad: `Settings > Apple ID > iCloud > Access iCloud Data on the Web`.
*   **Disable "Advanced Data Protection":** Disable "Advanced Data Protection" on your iPhone/iPad: `Settings > Apple ID > iCloud > Advanced Data Protection`.

## Usage

**To synchronize your iCloud photo collection to a local directory:**

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> [!IMPORTANT]
> Use `icloudpd` and not `icloud` to run the tool.

> [!TIP]
> Customize synchronization behavior with command-line options. Run `icloudpd --help` for a complete list.

**To authenticate and authorize a session:**

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

> [!TIP]
>  Use this feature to verify your session is still authenticated.

## Experimental Mode

Explore and test new features in the experimental mode. See [EXPERIMENTAL.md](EXPERIMENTAL.md) for details.

## Contributing

We welcome contributions!  Check out the [contributing guidelines](CONTRIBUTING.md) to learn how you can help improve iCloud Photos Downloader.