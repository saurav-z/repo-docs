# iCloud Photos Downloader: Back Up Your iCloud Photos with Ease

**Effortlessly download and back up all your iCloud photos with the cross-platform, open-source iCloud Photos Downloader.**

[View the original repository on GitHub](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, across various devices (laptop, desktop, NAS).
*   **Multiple Operation Modes:** Choose between Copy, Sync (with auto-delete), and Move (with iCloud deletion) modes for flexible management.
*   **Comprehensive Media Support:** Downloads Live Photos (image and video), RAW images (including RAW+JPEG).
*   **Automatic De-duplication:** Prevents duplicate downloads with intelligent file name handling.
*   **Continuous Monitoring:**  Keeps your local copy synchronized with iCloud through the `--watch-with-interval` option.
*   **Incremental Downloads:** Optimizations like `--until-found` and `--recent` improve efficiency.
*   **Metadata Preservation:** Option to update photo EXIF data (`--set-exif-datetime`).
*   **Flexible Installation:** Available as an executable, via package managers (Docker, PyPI, AUR, npm), or from source.

## iCloud Prerequisites

To ensure successful downloads, configure your iCloud account as follows:

*   **Enable Web Access:**  `Settings > [Your Name] > iCloud > Access iCloud Data on the Web` on your iPhone/iPad.
*   **Disable Advanced Data Protection:** `Settings > [Your Name] > iCloud > Advanced Data Protection` on your iPhone/iPad.

## Installation

Choose your preferred method:

1.  **Executable:** Download the pre-built executable from the [GitHub Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2) page.
2.  **Package Managers:** Install via [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), or [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
3.  **From Source:** Build and run from the source code.

Detailed installation instructions are available in the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

## Usage Examples

**Keep Your Photos Synced:**

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Authenticate a Session Independently:**

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

> **Important:** Remember to use `icloudpd`, not `icloud`, in your commands.

## Experimental Mode

Explore the latest features in the experimental mode before they are added to the main package: [Details](EXPERIMENTAL.md)

## Contributing

We welcome contributions!  See our [contributing guidelines](CONTRIBUTING.md) to learn how you can help improve iCloud Photos Downloader.