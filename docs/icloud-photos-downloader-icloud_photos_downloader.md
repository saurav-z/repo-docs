# iCloud Photos Downloader: Easily Back Up Your iCloud Photos

**Effortlessly download and back up all your precious photos and videos from iCloud with the powerful and versatile iCloud Photos Downloader.**

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[View the original repository on GitHub](https://github.com/icloud-photos-downloader/icloud_photos_downloader)**

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Multiple Installation Options:** Install via executable, Docker, PyPI, AUR, or npm.
*   **Flexible Download Modes:** Choose between Copy, Sync, and Move modes for versatile photo management.
*   **Live Photo and RAW Support:** Downloads both the image and video components of Live Photos, and supports RAW images including RAW+JPEG.
*   **Automatic De-duplication:** Prevents duplicate downloads based on filenames.
*   **Continuous Monitoring:** Option to watch for iCloud changes and automatically download new content.
*   **Incremental Downloads:** Optimized for efficient incremental downloads with options like `--until-found` and `--recent`.
*   **Metadata Preservation:** Option to update photo metadata (EXIF) using the `--set-exif-datetime` option.
*   **Authentication Flexibility:** Supports independent session creation and 2FA/2SA validation using the `--auth-only` option.

## iCloud Prerequisites

To ensure successful downloads, please configure your iCloud account as follows:

*   **Enable iCloud Data on the Web:** Go to `Settings > Apple ID > iCloud > Access iCloud Data on the Web` on your iPhone/iPad.
*   **Disable Advanced Data Protection:** Go to `Settings > Apple ID > iCloud > Advanced Data Protection` on your iPhone/iPad.

## Installation and Usage

You can download and run `icloudpd` in several ways:

1.  **Download Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2) page and run it.
2.  **Package Managers:** Install, update, and run the application using package managers like [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), or [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
3.  **Build from Source:** Build and run the application directly from the source code.

For detailed installation instructions, refer to the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

**Example Usage (Continuous Synchronization):**

To keep your iCloud photo collection synchronized to your local system:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Important Note:** The executable is named `icloudpd`, not `icloud`.

## Experimental Mode

Explore and test new features in experimental mode. Learn more in [EXPERIMENTAL.md](EXPERIMENTAL.md).

## Contributing

We welcome contributions!  Review the [contributing guidelines](CONTRIBUTING.md) to get involved.