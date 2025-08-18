# iCloud Photos Downloader: Securely Download Your iCloud Photos (and Videos)

**Easily and reliably back up your iCloud photos and videos with iCloud Photos Downloader, a powerful command-line tool.**  [View the original repository on GitHub](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting desktops, laptops, and NAS devices.
*   **Flexible Installation:** Download and run the executable directly, or install via package managers (Docker, PyPI, AUR, npm) for easy updates.
*   **Multiple Download Modes:** Choose between Copy (default), Sync (with auto-delete), and Move (delete from iCloud after download) to fit your needs.
*   **Comprehensive Media Support:** Downloads Live Photos (image and video files), RAW images (including RAW+JPEG), and more.
*   **Intelligent Features:** Automatic de-duplication, metadata (EXIF) updates, and incremental download options.
*   **Continuous Monitoring:** Watch for iCloud changes and automatically download new photos with the `--watch-with-interval` option.
*   **Authentication Control:** Allows you to create and authorize a session independently for 2FA/2SA validation.

## iCloud Prerequisites

To ensure a successful download, configure your iCloud account as follows:

*   **Enable Access iCloud Data on the Web:**  `Settings > Apple ID > iCloud > Access iCloud Data on the Web` on your iPhone/iPad.
*   **Disable Advanced Data Protection:** `Settings > Apple ID > iCloud > Advanced Data Protection` on your iPhone/iPad.

## Installation

You can install `icloudpd` in a few ways:

1.  **Download Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.30.0)
2.  **Use a Package Manager:**  Install via [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), or [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
3.  **Build from Source:**  Build and run from the source code.

See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed installation instructions.

## Usage Example

To synchronize your iCloud photo collection to a local directory, run:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Important:** Use the command `icloudpd`, not `icloud`.

## Experimental Mode

Explore new features in the experimental mode before they are released.  See [EXPERIMENTAL.md](EXPERIMENTAL.md) for details.

## Contributing

We welcome contributions!  Check out the [contributing guidelines](CONTRIBUTING.md) to learn how you can help improve iCloud Photos Downloader.