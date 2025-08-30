# iCloud Photos Downloader: Easily Download Your iCloud Photos

Tired of being locked into Apple's ecosystem? **iCloud Photos Downloader lets you effortlessly download and back up your entire iCloud photo library to your computer.**  Get started today and take control of your memories! ([See the original repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting desktops, laptops, and even NAS devices.
*   **Multiple Installation Options:** Install via executable, Docker, PyPI, AUR, or npm.
*   **Flexible Download Modes:** Choose from Copy, Sync (with auto-delete), and Move (delete from iCloud) to fit your backup strategy.
*   **Comprehensive Media Support:** Downloads Live Photos (image and video), RAW images (including RAW+JPEG), and more.
*   **Intelligent Features:** Includes automatic de-duplication, metadata (EXIF) updates, and options for incremental and continuous downloads.
*   **Command-Line Driven:** Fully command-line enabled, making it ideal for scripting and automation.

## iCloud Prerequisites

Before using iCloud Photos Downloader, please ensure the following settings are configured in your iCloud account:

*   **Enable Access iCloud Data on the Web:**  `Settings > Apple ID > iCloud > Access iCloud Data on the Web` on your iPhone/iPad.
*   **Disable Advanced Data Protection:**  `Settings > Apple ID > iCloud > Advanced Data Protection` on your iPhone/iPad.

## Installation & Usage

### Installation

You can install `icloudpd` in several ways:

1.  **Download Executable:** Download the executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.32.0) page.
2.  **Package Managers:** Use Docker, PyPI, AUR, or npm. See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed instructions.
3.  **Build from Source:** Build and run the tool from source code.

### Basic Usage

To download your photos and keep them synchronized:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

Remember to use `icloudpd`, *not* `icloud`.  See the output of `icloudpd --help` for all available command-line options.  You can also create an authorization session with:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Experimental Mode

Check out the [EXPERIMENTAL.md](EXPERIMENTAL.md) file for experimental features that are in development.

## Contributing

We welcome contributions!  Please review the [contributing guidelines](CONTRIBUTING.md) to get started.