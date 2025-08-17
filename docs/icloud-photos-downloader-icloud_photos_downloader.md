# iCloud Photos Downloader: Securely Back Up Your iCloud Photos (Command-Line Tool)

**Safeguard your precious memories by effortlessly downloading your iCloud photos and videos with iCloud Photos Downloader, a powerful and versatile command-line tool.**  ([View the Original Repo](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting laptops, desktops, and NAS devices.
*   **Multiple Download Modes:** Choose from Copy, Sync (with auto-delete), and Move (delete from iCloud).
*   **Comprehensive Media Support:** Downloads Live Photos (image and video) and RAW images (including RAW+JPEG).
*   **Intelligent Deduplication:** Automatically avoids downloading duplicate photos with the same names.
*   **Continuous Monitoring:** Option to automatically monitor iCloud for changes and download new photos (`--watch-with-interval`).
*   **Incremental Download Optimizations:** Efficiently handles incremental downloads with options like `--until-found` and `--recent`.
*   **Metadata Preservation:** Option to update photo metadata (EXIF) with `--set-exif-datetime`.
*   **Flexible Installation:** Available as an executable, and installable through package managers such as Docker, PyPI, AUR, and npm.

## iCloud Prerequisites

To ensure a successful download, configure your iCloud account:

*   **Enable Web Access:**  On your iPhone/iPad: `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable Advanced Data Protection:** On your iPhone/iPad: `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation & Usage

### Installation

You can install and use the tool in several ways:

1.  **Download Executable:** Download the executable for your operating system from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.4) page.
2.  **Package Managers:** Install using package managers such as Docker, PyPI, AUR, and npm ([see documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html)).
3.  **Build from Source:** Build and run the tool from the source code.

### Basic Usage

To keep your iCloud photo library synchronized locally:

```bash
icloudpd --directory /your/download/directory --username your@email.com --watch-with-interval 3600
```

> **Important:** The executable is named `icloudpd`, not `icloud`.
>
> **Tip:**  Use `icloudpd --help` for a full list of command-line options to customize your download process.

### Authentication

To create and authorize a session:

```bash
icloudpd --username your@email.com --password your_password --auth-only
```

> **Tip:**  You can also use this command to verify if the current session is still authenticated.

## Experimental Features

Explore new features in the experimental mode; see [EXPERIMENTAL.md](EXPERIMENTAL.md).

## Contributing

We welcome contributions!  Review our [contributing guidelines](CONTRIBUTING.md) to learn how to get involved.