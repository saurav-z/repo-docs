# iCloud Photos Downloader: Download & Back Up Your iCloud Photos Easily

**Tired of being locked into iCloud?** iCloud Photos Downloader is a powerful command-line tool that lets you easily download and back up all your photos and videos from iCloud to your computer, NAS, or other storage solutions. [View the original repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader).

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works on Linux, Windows, and macOS.
*   **Multiple Installation Options:** Download executables, or install via Docker, PyPI, AUR, or npm.
*   **Flexible Download Modes:** Choose between Copy, Sync, and Move modes for different backup strategies.
*   **Live Photo & RAW Support:** Downloads both the image and video components of Live Photos, and handles RAW image formats.
*   **Automatic De-duplication:** Avoids downloading duplicate photos with the same name.
*   **Continuous Monitoring:** Easily monitor and download new photos automatically.
*   **Metadata Preservation:** Optionally updates photo metadata (EXIF data).
*   **Incremental Downloads:** Optimized options for efficient incremental backups.

## iCloud Prerequisites

Before using iCloud Photos Downloader, ensure the following settings are configured in your iCloud account to avoid "ACCESS_DENIED" errors:

*   **Enable "Access iCloud Data on the Web"**:  `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable "Advanced Data Protection"**:  `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation and Usage

### Installation

You have several options for installing `icloudpd`:

*   **Download Executable:** Get the latest executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2) page.
*   **Package Managers:** Install and update using package managers such as Docker, PyPI, AUR, or npm (see [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for details).
*   **Build from Source:** Build and run the tool from the source code.

Detailed installation instructions can be found in the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

### Basic Usage

To download your iCloud photos to a local directory and continuously monitor for changes:

```bash
icloudpd --directory /path/to/your/photos --username your@email.com --watch-with-interval 3600
```

**Important:** Use `icloudpd`, not `icloud`, to run the executable. Run `icloudpd --help` for a full list of command-line options to customize your downloads.

### Authentication-Only Mode

To independently create and authorize a session (and complete 2FA validation if needed):

```bash
icloudpd --username your@email.com --password your_password --auth-only
```

## Experimental Mode

New features are often added to the experimental mode first.  See [EXPERIMENTAL.md](EXPERIMENTAL.md) for more information.

## Contributing

We welcome contributions!  Check out the [contributing guidelines](CONTRIBUTING.md) to learn how you can get involved and help improve iCloud Photos Downloader.