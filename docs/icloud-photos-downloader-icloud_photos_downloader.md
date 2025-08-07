# iCloud Photos Downloader: Effortlessly Download Your iCloud Photos 

**Tired of being locked into iCloud?**  iCloud Photos Downloader is a powerful command-line tool designed to help you download all your photos and videos from iCloud, giving you complete control over your precious memories. [Get Started](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, across desktops, laptops, and NAS devices.
*   **Multiple Operation Modes:** Choose the best way to download your photos with Copy, Sync, and Move modes.
    *   **Copy:** Download new photos from iCloud (default).
    *   **Sync:** Download new photos and delete local files removed from iCloud (`--auto-delete`).
    *   **Move:** Download new photos and delete photos in iCloud (`--keep-icloud-recent-days`).
*   **Comprehensive Media Support:** Downloads Live Photos (image and video) and RAW images (including RAW+JPEG).
*   **Automatic De-duplication:** Prevents duplicate downloads of photos with the same name.
*   **Continuous Monitoring:**  Option to monitor and automatically download new photos from iCloud (`--watch-with-interval`).
*   **Incremental Download Optimizations:** Efficiently handles incremental runs with `--until-found` and `--recent` options.
*   **Metadata Preservation:** Updates photo metadata (EXIF) for accurate organization (`--set-exif-datetime`).
*   **Easy Installation:** Available via executables, [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), and [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).

## iCloud Prerequisites

Ensure your iCloud account is configured correctly for successful downloads:

*   **Enable "Access iCloud Data on the Web":**  In your iPhone/iPad settings: `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable "Advanced Data Protection":** In your iPhone/iPad settings: `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation

You can install and run `icloudpd` using several methods:

1.  **Download Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2).
2.  **Package Managers:** Install and manage the tool via package managers such as [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), and [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
3.  **Build from Source:** Build and run from the source code.

Detailed installation instructions are available in the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

## Usage

To sync your iCloud photo library to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Important:** Use the `icloudpd` executable, not `icloud`.  Adjust the synchronization logic with command-line parameters - run `icloudpd --help` for a full list.

To authorize a session:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

This feature can verify that your session is authenticated.

## Experimental Mode

Check out the [EXPERIMENTAL.md](EXPERIMENTAL.md) file for features in development.

## Contributing

We welcome contributions!  Please review the [contributing guidelines](CONTRIBUTING.md) for details on how to get involved.