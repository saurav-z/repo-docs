# iCloud Photos Downloader: Download Your iCloud Photos with Ease

Tired of being locked into iCloud? **iCloud Photos Downloader is a powerful command-line tool that lets you effortlessly download all your photos and videos from iCloud to your computer.**  [Get the code on GitHub](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works on Linux, Windows, and macOS.
*   **Flexible Installation:** Available as an executable, via Docker, PyPI, AUR, and npm.
*   **Multiple Download Modes:**
    *   **Copy:** Downloads new photos.
    *   **Sync:** Downloads new photos and deletes local files removed from iCloud.
    *   **Move:** Downloads new photos and deletes them from iCloud.
*   **Advanced Media Support:**  Handles Live Photos (image and video) and RAW images (including RAW+JPEG).
*   **Smart Features:** Automatic de-duplication of photos.
*   **Continuous Monitoring:** Option to watch for iCloud changes and download new content automatically.
*   **Metadata Preservation:** Option to update photo metadata (EXIF).
*   **Incremental Downloads:** Optimized for faster downloads with `--until-found` and `--recent` options.

## iCloud Prerequisites

To ensure the iCloud Photos Downloader works correctly, please configure your iCloud account as follows:

*   **Enable "Access iCloud Data on the Web"**: In your iPhone / iPad, go to `Settings > [Your Name] > iCloud > Access iCloud Data on the Web`.
*   **Disable "Advanced Data Protection"**: In your iPhone / iPad, go to `Settings > [Your Name] > iCloud > Advanced Data Protection`.

## Installation and Usage

You can install and run `icloudpd` in the following ways:

1.  **Download Executable:** Download the pre-built executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.28.2) page.
2.  **Package Managers:** Install using Docker, PyPI, AUR, or npm. See the [Installation Guide](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for details.
3.  **Build from Source:** Compile and run the tool directly from the source code.

**Example Usage (Syncing your photos):**

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Important Note:** Use `icloudpd` instead of `icloud` in your commands.

See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/) for more detailed information and command-line options (run `icloudpd --help`).

## Experimental Mode

Explore cutting-edge features in the experimental mode before they make their way into the main package.  See [EXPERIMENTAL.md](EXPERIMENTAL.md) for more info.

## Contributing

We welcome contributions!  Check out the [contributing guidelines](CONTRIBUTING.md) to get started.