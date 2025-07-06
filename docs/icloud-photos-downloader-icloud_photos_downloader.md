# iCloud Photos Downloader: Securely Download Your iCloud Photos (and More!)

**Easily and reliably download all your photos and videos from iCloud with the versatile command-line tool, iCloud Photos Downloader!** ([Back to Original Repo](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Multiple Download Modes:** Choose between Copy, Sync (with auto-delete), and Move (delete from iCloud after download) options.
*   **Comprehensive File Support:** Downloads Live Photos (image and video), RAW images (including RAW+JPEG), and handles file name de-duplication.
*   **Continuous Monitoring:** Watch for iCloud changes with the `--watch-with-interval` option.
*   **Efficient Downloading:** Optimized for incremental downloads with `--until-found` and `--recent` options.
*   **Metadata Preservation:** Option to update photo metadata (EXIF) with `--set-exif-datetime`.
*   **Flexible Installation:** Install via executables, package managers (Docker, PyPI, AUR, npm), or build from source.

## iCloud Prerequisites

To ensure successful downloads, configure your iCloud account as follows:

*   **Enable Web Access:** In your iPhone/iPad settings: `Settings > [Your Name] > iCloud > Access iCloud Data on the Web`
*   **Disable Advanced Data Protection:** In your iPhone/iPad settings: `Settings > [Your Name] > iCloud > Advanced Data Protection`

## Installation and Running

You can easily get started with iCloud Photos Downloader using one of the following methods:

1.  **Download Executable:** Get the pre-built executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.28.1) section.
2.  **Package Managers:** Install via Docker, PyPI, AUR, or npm.  See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm) for specifics.
3.  **Build from Source:**  Build and run the tool directly from the source code.

Refer to the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed instructions.

## Usage Examples

**Keep your iCloud photos synchronized locally:**

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Create and authorize a session for 2FA validation:**

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Experimental Mode

Explore cutting-edge features in the experimental mode.  See [EXPERIMENTAL.md](EXPERIMENTAL.md) for more details.

## Contributing

We welcome contributions! Review our [contributing guidelines](CONTRIBUTING.md) to learn how you can get involved.