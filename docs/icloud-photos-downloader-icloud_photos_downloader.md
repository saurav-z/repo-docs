# iCloud Photos Downloader: Effortlessly Back Up Your iCloud Photos

Easily back up your precious iCloud photos and videos with the powerful and versatile **iCloud Photos Downloader**, a command-line tool designed to securely download and manage your entire iCloud photo library.  ([Original Repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, making it easy to back up your photos on laptops, desktops, and even NAS devices.
*   **Multiple Download Modes:** Choose the right mode for your needs:
    *   **Copy:** Download new photos from iCloud.
    *   **Sync:** Download new photos and automatically delete local files removed from iCloud.
    *   **Move:** Download new photos and remove them from iCloud.
*   **Comprehensive Media Support:** Handles Live Photos (both image and video files) and RAW image formats (including RAW+JPEG).
*   **Intelligent Features:** Includes automatic de-duplication of photos with the same name and options for continuous monitoring for iCloud changes.
*   **Metadata Preservation:** Keeps photo metadata (EXIF) updated with the `--set-exif-datetime` option.
*   **Efficient Operations:** Offers options for incremental downloads (`--until-found` and `--recent`) for faster backups.
*   **Flexible Installation:** Available via executables, Docker, PyPI, AUR, and npm for easy setup.

## Installation and Running

Get started in minutes with these easy installation options:

*   **Executable Download:** Download a pre-built executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2) page.
*   **Package Managers:** Install and manage through package managers like Docker, PyPI, AUR, or npm.  See [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for installation details.
*   **Build from Source:** For advanced users, build and run the tool directly from the source code.

## iCloud Prerequisites

Before you begin, ensure your iCloud account is configured with these settings to avoid "ACCESS_DENIED" errors:

*   **Enable Web Access:**  On your iPhone/iPad, enable `Settings > [Your Name] > iCloud > Access iCloud Data on the Web`.
*   **Disable Advanced Data Protection:** On your iPhone/iPad, disable `Settings > [Your Name] > iCloud > Advanced Data Protection`.

## Usage Examples

**Synchronize your iCloud photos to a local directory:**

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Create and authorize a session (including 2FA validation):**

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

**Important Note:**  Be sure to use `icloudpd` (the executable name), not `icloud`.

## Experimental Mode

Explore cutting-edge features in the experimental mode.  For details, see the [EXPERIMENTAL.md](EXPERIMENTAL.md) file.

## Contributing

We welcome contributions!  Check out the [contributing guidelines](CONTRIBUTING.md) to get involved.