# iCloud Photos Downloader: Download & Back Up Your iCloud Photos

**Effortlessly back up and manage your precious iCloud photos with the versatile and open-source iCloud Photos Downloader.**  [Explore the original repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Runs seamlessly on Linux, Windows, and macOS, supporting laptops, desktops, and NAS devices.
*   **Multiple Download Modes:** Choose between Copy (default), Sync (with auto-delete), and Move (delete from iCloud).
*   **Comprehensive File Support:** Downloads Live Photos (image and video), RAW images (including RAW+JPEG).
*   **Smart Features:** Includes automatic de-duplication, incremental run optimizations, and EXIF metadata updates.
*   **Flexible Operation:** One-time downloads and continuous monitoring options are available.
*   **Easy Installation:**  Available as an executable and through various package managers ([Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm)).

## iCloud Prerequisites

To ensure proper functionality, configure your iCloud account as follows:

*   **Enable Access iCloud Data on the Web:** `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable Advanced Data Protection:** `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation & Usage

### Installation

Choose your preferred installation method:

1.  **Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.32.2) page.
2.  **Package Managers:** Install via Docker, PyPI, AUR, or npm (see installation documentation).
3.  **Build from Source:**  Build and run the tool from the source code.

Detailed installation instructions can be found in the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

### Basic Usage

Synchronize your iCloud photo collection to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

## Experimental Mode

Explore new features in the [Experimental Mode](EXPERIMENTAL.md).

## Contributing

We welcome contributions! See the [contributing guidelines](CONTRIBUTING.md) to learn how you can get involved.