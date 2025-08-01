# iCloud Photos Downloader: Easily Download Your iCloud Photos

Tired of being locked into iCloud?  **iCloud Photos Downloader** is a powerful command-line tool that lets you download all your iCloud photos and videos to your computer, giving you full control of your memories.  [**View the original repository on GitHub**](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting laptops, desktops, and NAS devices.
*   **Flexible Installation:** Available as an executable, through package managers, and via Docker, PyPI, AUR, and npm for easy setup.
*   **Multiple Operation Modes:** Choose between Copy, Sync, and Move modes to manage your photos efficiently.
*   **Live Photo & RAW Support:** Downloads both Live Photos (image and video) and RAW images (including RAW+JPEG).
*   **Automatic De-duplication:** Prevents duplicate downloads by automatically skipping photos with the same name.
*   **Continuous Monitoring:** Supports one-time downloads and continuous monitoring for iCloud changes with the `--watch-with-interval` option.
*   **Metadata Preservation:**  Updates photo metadata (EXIF) with the `--set-exif-datetime` option.
*   **Incremental Run Optimization:** Options like `--until-found` and `--recent` for optimized incremental downloads.
*   **And many more!** Discover the full list of features by using the `--help` option.

## iCloud Prerequisites

Before using iCloud Photos Downloader, ensure your iCloud account is configured with the following settings to prevent `ACCESS_DENIED` errors:

*   **Enable "Access iCloud Data on the Web":** On your iPhone/iPad: `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable "Advanced Data Protection":** On your iPhone/iPad: `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation and Usage

You have multiple options for getting started:

1.  **Download Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2) page.
2.  **Package Managers:** Install via Docker, PyPI, AUR, or npm. See the [Installation Guide](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed instructions.
3.  **Build from Source:** Build and run the tool from the source code.

**Basic Usage Example:**

To synchronize your iCloud photo collection to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Important Notes:**

*   Ensure you are using the `icloudpd` executable.
*   Adjust synchronization with command-line parameters. Use `icloudpd --help` for a full list.
*   You can authorize a session with  `icloudpd --username my@email.address --password my_password --auth-only`

## Experimental Mode

Check the [EXPERIMENTAL.md](EXPERIMENTAL.md) file for details on features added to the experimental mode.

## Contributing

We welcome contributions! Please read the [contributing guidelines](CONTRIBUTING.md) to learn how to get involved.