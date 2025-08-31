# Download Your iCloud Photos Easily with iCloud Photos Downloader

**Effortlessly back up your precious memories with iCloud Photos Downloader, a powerful command-line tool designed to download all your iCloud photos and videos.** ([View the original repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, across laptops, desktops, and NAS devices.
*   **Multiple Installation Options:** Install via executable, [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), or [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
*   **Flexible Download Modes:** Choose between Copy, Sync, and Move modes to manage your photos.
*   **Comprehensive Media Support:** Downloads Live Photos (images and videos), RAW images (including RAW+JPEG), and handles duplicates.
*   **Incremental and Automated Downloads:** Supports one-time downloads and continuous monitoring with the `--watch-with-interval` option, optimizing for incremental runs.
*   **Metadata Preservation:** Updates photo metadata (EXIF) with the `--set-exif-datetime` option.
*   **Additional Options:** Explore options with `--help` for advanced features and customization.

## iCloud Prerequisites

Before using iCloud Photos Downloader, ensure your iCloud account is configured with these settings:

*   **Enable "Access iCloud Data on the Web":** Enable in your iPhone / iPad: `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable "Advanced Data Protection":** Disable in your iPhone / iPad: `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation & Usage

You can run `icloudpd` in these ways:

1.  **Download Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.32.1) and run.
2.  **Use Package Manager:** Install and run via [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), or [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
3.  **Build from Source:** Build and run from the source code.

See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for complete installation instructions.

**Example Usage:**

To keep your iCloud photos synchronized to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> \[!IMPORTANT]
> Remember to use `icloudpd` (the executable), not `icloud`.

To authorize a session:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Experimental Mode

Explore new features in the experimental mode before they are released to the main package. See [EXPERIMENTAL.md](EXPERIMENTAL.md) for details.

## Contributing

We welcome contributions!  Please see the [contributing guidelines](CONTRIBUTING.md) to get involved.