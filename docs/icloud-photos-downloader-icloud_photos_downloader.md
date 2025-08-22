# iCloud Photos Downloader: Download & Back Up Your iCloud Photos Easily

**Effortlessly back up and manage your iCloud photo library with iCloud Photos Downloader, a versatile command-line tool.** (Link to original repo: [https://github.com/icloud-photos-downloader/icloud_photos_downloader](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml) [![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting laptops, desktops, and NAS devices.
*   **Multiple Installation Options:** Download the executable directly, or install via package managers such as Docker, PyPI, AUR, and npm.
*   **Flexible Modes of Operation:**
    *   **Copy:** Downloads new photos from iCloud (default).
    *   **Sync:** Downloads new photos and deletes local files removed from iCloud (with the `--auto-delete` option).
    *   **Move:** Downloads new photos and deletes photos from iCloud (with the `--keep-icloud-recent-days` option).
*   **Comprehensive File Support:** Supports Live Photos (image and video as separate files) and RAW images (including RAW+JPEG).
*   **Intelligent Features:** Automatic de-duplication of photos with the same name.
*   **Continuous Monitoring:** One-time download and the option to monitor for iCloud changes continuously (`--watch-with-interval` option).
*   **Optimized Downloads:** Options for incremental runs (`--until-found` and `--recent` options).
*   **Metadata Preservation:** Photo metadata (EXIF) updates with the `--set-exif-datetime` option.
*   **And many more options, including:**
    *   ... and many more (use `--help` option to get full list)

## iCloud Prerequisites

Before using iCloud Photos Downloader, ensure your iCloud account is configured correctly:

*   **Enable Access iCloud Data on the Web:** On your iPhone / iPad, enable `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable Advanced Data Protection:** On your iPhone /iPad disable `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation & Running

You can run `icloudpd` in three ways:

1.  **Download Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.31.0) page and run it.
2.  **Package Managers:** Install using a package manager ([Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm)).
3.  **Build from Source:** Build and run the tool from its source code.

See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed installation instructions.

## Usage Examples

To synchronize your iCloud photo collection with a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> [!IMPORTANT]
> The executable name is `icloudpd`, not `icloud`.

To independently create and authorize a session (and complete 2SA/2FA validation if needed):

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

> [!TIP]
> This feature can also be used to check and verify that the session is still authenticated.

## Experimental Mode

New features are added to the experimental mode before they graduate into the main package. See [EXPERIMENTAL.md](EXPERIMENTAL.md) for details.

## Contributing

We welcome contributions! Check out the [contributing guidelines](CONTRIBUTING.md) to get started.