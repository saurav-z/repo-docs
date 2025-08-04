# iCloud Photos Downloader: Download Your iCloud Photos Easily

**Tired of being locked into Apple's ecosystem?** iCloud Photos Downloader is a versatile command-line tool that allows you to download all your iCloud photos and videos to your computer, giving you full control over your memories. ([Original Repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting laptops, desktops, and NAS devices.
*   **Multiple Installation Options:** Available as an executable, or through package managers, Docker, PyPI, AUR, and npm, allowing flexible installation options.
*   **Flexible Download Modes:** Choose from Copy, Sync, or Move modes to manage your photos, including options for automatically deleting synced files.
*   **Live Photo Support:** Downloads both the image and video components of Live Photos, plus RAW images.
*   **Automatic De-duplication:** Prevents duplicate downloads by automatically identifying and skipping photos with the same name.
*   **Continuous Monitoring:** Option to automatically monitor iCloud for changes and download new photos.
*   **Metadata Preservation:** Option to update photo metadata (EXIF) to preserve details like the date and time the photo was taken.
*   **Other Features:** Includes optimizations for incremental runs, customizable sync logic, and more.

## iCloud Prerequisites

Before you begin, ensure your iCloud account is configured with the following settings to prevent "ACCESS_DENIED" errors:

*   **Enable Access iCloud Data on the Web:**  `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable Advanced Data Protection:**  `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation and Usage

### Installation

You can install `icloudpd` in several ways:

1.  **Download Executable:** Download the pre-built executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2) page.
2.  **Package Managers:** Utilize package managers like Docker, PyPI, AUR, or npm for installation and updates. See the [Installation Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed instructions.
3.  **Build from Source:** Build and run the application from the source code.

### Basic Usage

To synchronize your iCloud photo collection to your local system:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Important Note:** The executable name is `icloudpd`, not `icloud`.

**Tip:** For a full list of available command-line parameters and options, use `icloudpd --help`.

### Authentication

To independently create and authorize a session on your local system:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Contributing

We welcome contributions! Please review the [contributing guidelines](CONTRIBUTING.md) for information on how to get involved.