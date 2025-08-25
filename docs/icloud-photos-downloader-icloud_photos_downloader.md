# iCloud Photos Downloader: Download and Back Up Your iCloud Photos Easily

Tired of being locked into Apple's ecosystem? **iCloud Photos Downloader** is a powerful command-line tool that allows you to effortlessly download and back up all your photos and videos from iCloud.  [View the original repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml) [![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting laptops, desktops, and NAS devices.
*   **Multiple Installation Options:** Available as an executable for direct use, as well as through package managers like Docker, PyPI, AUR, and npm.
*   **Versatile Operation Modes:**
    *   **Copy:** Downloads new photos from iCloud (default).
    *   **Sync:** Downloads new photos and deletes local files removed from iCloud (with `--auto-delete`).
    *   **Move:** Downloads new photos and deletes them from iCloud (with `--keep-icloud-recent-days`).
*   **Advanced Support:**  Handles Live Photos (image and video) and RAW images (including RAW+JPEG).
*   **Intelligent Features:**  Automatic de-duplication, and options for incremental downloads.
*   **Metadata Preservation:** Option to update photo metadata (EXIF) with `--set-exif-datetime`.
*   **Continuous Monitoring:** Option to monitor for iCloud changes continuously (`--watch-with-interval`).
*   **Optimized for Efficiency:** Offers `--until-found` and `--recent` options for faster incremental runs.

## iCloud Prerequisites

To ensure successful downloads, configure your iCloud account as follows:

*   **Enable Access iCloud Data on the Web:** Enable this setting on your iPhone/iPad (`Settings > Apple ID > iCloud > Access iCloud Data on the Web`).
*   **Disable Advanced Data Protection:** Disable this setting on your iPhone/iPad (`Settings > Apple ID > iCloud > Advanced Data Protection`).

## Installation

You can install and run `icloudpd` in several ways:

1.  **Executable Download:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.31.0).
2.  **Package Managers:** Utilize package managers such as [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), and [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
3.  **Build from Source:** Build and run the application from the source code.

For detailed installation instructions, please refer to the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

## Usage Examples

**To synchronize your iCloud photo collection with a local directory:**

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> [!IMPORTANT]
> Remember to use `icloudpd` and not `icloud` when running the executable.

> [!TIP]
> Adjust synchronization behavior using the command-line parameters. Use `icloudpd --help` to see the full list of options.

**To authorize a session (and perform 2SA/2FA validation if necessary):**

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

> [!TIP]
> This is also useful to verify and check that the session is still authenticated.

## Contributing

We welcome contributions!  Please review the [contributing guidelines](CONTRIBUTING.md) to learn how to get involved.