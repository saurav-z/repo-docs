# iCloud Photos Downloader: Easily Download and Back Up Your iCloud Photos

**Effortlessly back up and manage your iCloud photos with the powerful and versatile iCloud Photos Downloader.**  This command-line tool allows you to download and manage your entire iCloud photo library on your computer, offering flexible options for backup, syncing, and more.  [Explore the original repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader).

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting laptops, desktops, and NAS devices.
*   **Multiple Download Modes:**
    *   **Copy:** Download new photos from iCloud (default).
    *   **Sync:** Download new photos and delete local files removed from iCloud (`--auto-delete`).
    *   **Move:** Download new photos and delete them from iCloud (`--keep-icloud-recent-days`).
*   **Advanced Media Support:** Handles Live Photos (image and video) and RAW images (including RAW+JPEG).
*   **Smart Features:**
    *   Automatic de-duplication of photos with the same name.
    *   Option to monitor iCloud for changes continuously (`--watch-with-interval`).
    *   Optimizations for incremental downloads (`--until-found` and `--recent` options).
    *   Photo metadata (EXIF) updates (`--set-exif-datetime`).
*   **Flexible Installation:** Available via executable downloads, Docker, PyPI, AUR, and npm.

## iCloud Prerequisites

To ensure successful downloads, your iCloud account requires specific settings:

*   **Enable "Access iCloud Data on the Web":**  `Settings > Apple ID > iCloud > Access iCloud Data on the Web` on your iPhone/iPad.
*   **Disable "Advanced Data Protection":**  `Settings > Apple ID > iCloud > Advanced Data Protection` on your iPhone/iPad.

## Installation and Running

Choose your preferred method to get started:

1.  **Download Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.3) page.
2.  **Package Manager:** Install, update, and run via:
    *   [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker)
    *   [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi)
    *   [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur)
    *   [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm)
3.  **Build from Source:** Build and run the tool from the source code.

For detailed installation instructions, see the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

## Usage Examples

**Synchronize your iCloud photos to your local system:**

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Create and authorize a session:**

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Experimental Mode

Explore cutting-edge features in the [EXPERIMENTAL.md](EXPERIMENTAL.md) documentation.

## Contributing

We welcome contributions!  Review the [contributing guidelines](CONTRIBUTING.md) to learn how you can help improve iCloud Photos Downloader.