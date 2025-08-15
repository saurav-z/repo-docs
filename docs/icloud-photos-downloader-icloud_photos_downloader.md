# iCloud Photos Downloader: Effortlessly Download Your iCloud Photos

**Easily back up and manage your iCloud photo library with iCloud Photos Downloader, a versatile command-line tool.** ([View the original repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, suitable for laptops, desktops, and NAS devices.
*   **Multiple Download Modes:** Choose between Copy, Sync (with auto-delete), and Move (with iCloud photo deletion).
*   **Live Photo & RAW Support:** Downloads both Live Photos (image & video) and RAW images (including RAW+JPEG).
*   **Automated Deduplication:** Avoids downloading duplicate photos with the same names.
*   **Continuous Monitoring:** Option to automatically monitor for and download new iCloud changes.
*   **Incremental Downloads:** Optimized options for efficient incremental downloads (`--until-found` and `--recent`).
*   **Metadata Preservation:**  Preserves and updates photo metadata (EXIF) with the `--set-exif-datetime` option.
*   **Flexible Installation:** Available as an executable, via package managers, and from source (Docker, PyPI, AUR, npm).

## iCloud Prerequisites

To ensure successful downloads, please configure your iCloud account:

*   **Enable "Access iCloud Data on the Web"**:  Navigate to `Settings > Apple ID > iCloud` on your iPhone/iPad.
*   **Disable "Advanced Data Protection"**: In `Settings > Apple ID > iCloud`, disable "Advanced Data Protection" on your iPhone/iPad.

## Installation

You can install and run `icloudpd` using several methods:

1.  **Download Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.4) page.
2.  **Package Managers:** Install using package managers such as [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), or [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
3.  **Build from Source:** Build and run the tool directly from the source code.

See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed installation instructions.

## Usage Examples

**Syncing your iCloud photos to a local directory:**

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Authenticating a session independently:**

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

> **Important:** Use `icloudpd` and not `icloud` to execute the tool.
> **Tip:** Customize your download behavior using the command-line parameters.  Run `icloudpd --help` for a full list.

## Experimental Mode

Explore cutting-edge features in experimental mode. Learn more in [EXPERIMENTAL.md](EXPERIMENTAL.md).

## Contributing

We welcome contributions! Review the [contributing guidelines](CONTRIBUTING.md) to get involved.