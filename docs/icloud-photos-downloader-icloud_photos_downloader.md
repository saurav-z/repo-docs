# iCloud Photos Downloader: Easily Download Your iCloud Photos 

**Effortlessly back up your precious memories with iCloud Photos Downloader, a powerful command-line tool that allows you to download all your photos and videos from iCloud.** Find the original repository [here](https://github.com/icloud-photos-downloader/icloud_photos_downloader).

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting laptops, desktops, and NAS devices.
*   **Multiple Installation Options:** Available as an executable for direct use and through package managers like Docker, PyPI, AUR, and npm.
*   **Flexible Download Modes:**
    *   **Copy:** Downloads new photos from iCloud (default).
    *   **Sync:** Downloads new photos and deletes local files removed from iCloud (`--auto-delete`).
    *   **Move:** Downloads new photos and deletes photos in iCloud (`--keep-icloud-recent-days`).
*   **Comprehensive Media Support:** Handles Live Photos (image and video files), RAW images (including RAW+JPEG).
*   **Intelligent Deduplication:** Automatically avoids downloading duplicate photos with the same filenames.
*   **Continuous Monitoring:** Option to continuously watch for iCloud changes (`--watch-with-interval`).
*   **Optimized Incremental Downloads:** Includes options for efficient incremental downloads (`--until-found`, `--recent`).
*   **Metadata Preservation:** Updates photo metadata (EXIF) (`--set-exif-datetime`).
*   **And much more!** Explore all features using the `--help` option.

## iCloud Prerequisites

Before you begin, ensure your iCloud account is configured with these settings to avoid "ACCESS_DENIED" errors:

*   **Enable iCloud Data on the Web:** `Settings > Apple ID > iCloud > Access iCloud Data on the Web` on your iPhone/iPad.
*   **Disable Advanced Data Protection:** `Settings > Apple ID > iCloud > Advanced Data Protection` on your iPhone/iPad.

## Installation and Usage

Choose your preferred method to get started:

1.  **Download Executable:**  Get the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.28.2) page.
2.  **Use Package Managers:** Install, update, and run through package managers like [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), and [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
3.  **Build from Source:** Build and run the tool from the source code.

For detailed installation instructions, see the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

**Example Usage:**

To synchronize your iCloud photo collection to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> **Important:** Use `icloudpd`, not `icloud` in your commands.

> **Tip:** Customize your sync with command-line parameters. Run `icloudpd --help` for a full list of options.

To authenticate a session:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Experimental Mode

Explore new features in the experimental mode. [Details](EXPERIMENTAL.md)

## Contributing

We welcome contributions! Check out the [contributing guidelines](CONTRIBUTING.md) to get involved.