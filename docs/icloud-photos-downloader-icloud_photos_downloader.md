# Download Your iCloud Photos Easily with iCloud Photos Downloader

Tired of being locked into the iCloud ecosystem? **iCloud Photos Downloader** is a powerful command-line tool that lets you download all your photos and videos from iCloud, giving you complete control over your precious memories. ([Original Repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Multiple Download Modes:** Choose between "Copy", "Sync", and "Move" modes for flexible photo management.
*   **Live Photo Support:** Downloads both images and videos associated with Live Photos.
*   **RAW Image Support:** Includes support for RAW and RAW+JPEG images.
*   **Automatic De-duplication:** Prevents duplicate downloads with the same filenames.
*   **Continuous Monitoring:**  Option to watch for iCloud changes and automatically download new content.
*   **Metadata Preservation:** Updates photo metadata (EXIF) with `--set-exif-datetime` option.
*   **Incremental Downloads:** Optimizations for efficient incremental downloads via `--until-found` and `--recent` options.
*   **Multiple Installation Options:** Available as an executable, Docker image, and through package managers (PyPI, AUR, npm).

## iCloud Prerequisites

Before using iCloud Photos Downloader, ensure your iCloud account is configured with these settings:

*   **Enable "Access iCloud Data on the Web"** in your iPhone/iPad settings.
*   **Disable "Advanced Data Protection"** in your iPhone/iPad settings.

## Installation and Running

You can download and run `icloudpd` in a number of ways:

1.  **Executable Download:** Download a pre-built executable for your operating system from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.3) section.
2.  **Package Managers:** Install and run using package managers like Docker, PyPI, AUR, and npm ([Installation instructions](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html)).
3.  **Build from Source:**  Compile and run the tool from the source code.

For more detailed installation instructions, please refer to the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

## Usage Examples

**Keep Your iCloud Photos Synced:**

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Authenticate a Session Independently:**

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Contributing

We welcome contributions!  Check out the [contributing guidelines](CONTRIBUTING.md) to get started.