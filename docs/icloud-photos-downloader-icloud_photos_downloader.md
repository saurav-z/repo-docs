# iCloud Photos Downloader: Effortlessly Download Your iCloud Photos

Tired of being locked into Apple's ecosystem?  **iCloud Photos Downloader** is a powerful command-line tool that lets you easily download all your precious photos and videos from iCloud, giving you complete control over your memories.  [Check out the original repo!](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml) [![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works flawlessly on Linux, Windows, and macOS - perfect for laptops, desktops, and NAS devices.
*   **Multiple Installation Options:** Download an executable directly, or install via package managers like [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), and [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
*   **Flexible Download Modes:** Choose between Copy, Sync (with auto-delete), and Move options to manage your photos.
*   **Comprehensive Media Support:** Downloads Live Photos (image and video), RAW images (including RAW+JPEG), and handles duplicates automatically.
*   **Automatic Updates:** Options for continuous monitoring and incremental downloads, ensuring your local collection is always up-to-date.
*   **Metadata Preservation:** Includes options for EXIF data preservation, keeping your photo details intact.

## iCloud Prerequisites

To ensure a smooth download experience, please configure your iCloud account:

*   **Enable Web Access:** Enable "Access iCloud Data on the Web" in your iCloud settings on your iPhone/iPad.
*   **Disable Advanced Data Protection:** Disable "Advanced Data Protection" in your iCloud settings on your iPhone/iPad.

## Installation & Running

You can run `icloudpd` in the following ways:

1.  **Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.28.2).
2.  **Package Managers:** Utilize package managers such as [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), or [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
3.  **Build from Source:** Build and run directly from the source code.

See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed instructions.

## Experimental Features

Keep an eye out for new features in "Experimental Mode" before they are released to the main package. [Details](EXPERIMENTAL.md)

## Usage Examples

To keep your iCloud photo collection synchronized to your local system:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> \[!IMPORTANT]
> The executable is `icloudpd`, not `icloud`.

> \[!TIP]
> Customize your synchronization using command-line parameters. Run `icloudpd --help` for a full list.

To independently create and authorize a session (and complete 2SA/2FA validation if needed) on your local system:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```
> \[!TIP]
> This feature can also be used to check and verify that the session is still authenticated.

## Contributing

We welcome contributions!  Check out the [contributing guidelines](CONTRIBUTING.md) to get involved.