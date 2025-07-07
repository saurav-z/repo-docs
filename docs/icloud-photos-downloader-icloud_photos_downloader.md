# iCloud Photos Downloader: Effortlessly Download Your iCloud Photos

Tired of being locked into iCloud? **iCloud Photos Downloader is a powerful command-line tool that lets you download all your iCloud photos and videos to your computer, giving you complete control over your memories.** ([View the original repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS for desktops, laptops, and even NAS devices.
*   **Multiple Download Modes:** Choose the best method to manage your photos:
    *   **Copy:** Downloads new photos from iCloud (default).
    *   **Sync:** Downloads new photos and *deletes* local files removed from iCloud (with `--auto-delete`).
    *   **Move:** Downloads new photos and *deletes* photos from iCloud (with `--keep-icloud-recent-days`).
*   **Comprehensive File Support:** Downloads Live Photos (image and video), RAW images (including RAW+JPEG), and more.
*   **Intelligent Features:**
    *   Automatic de-duplication of photos.
    *   Option to monitor for iCloud changes continuously (`--watch-with-interval`).
    *   Optimizations for incremental downloads (`--until-found` and `--recent`).
    *   Photo metadata (EXIF) updates (`--set-exif-datetime`).
*   **Flexible Installation:** Available via executable downloads, Docker, PyPI, AUR, and npm.

## iCloud Prerequisites

To ensure iCloud Photos Downloader works correctly, please configure your iCloud account as follows:

*   **Enable "Access iCloud Data on the Web":** In your iPhone/iPad settings: `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable "Advanced Data Protection":**  In your iPhone/iPad settings: `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation and Usage

You can install and run iCloud Photos Downloader in several ways:

1.  **Download Executable:** Get the pre-built executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.28.2) page.
2.  **Package Managers:** Utilize your preferred package manager (Docker, PyPI, AUR, npm).  See the [Installation Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for specific instructions.
3.  **Build from Source:** Build and run the tool from the source code.

**Example Usage (Syncing Photos):**

To keep your iCloud photos synchronized to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> [!IMPORTANT]
> Remember to use `icloudpd` and not `icloud` in your commands.

> [!TIP]
> Explore the command-line parameters using `icloudpd --help` for advanced customization.

**Authentication Only**
To independently create and authorize a session (and complete 2SA/2FA validation if needed) on your local system:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

> [!TIP]
> This feature can also be used to check and verify that the session is still authenticated.

## Experimental Mode

Check the [EXPERIMENTAL.md](EXPERIMENTAL.md) file for details on features in the experimental stage.

## Contributing

We welcome contributions!  Please review the [contributing guidelines](CONTRIBUTING.md) to learn how you can help.