# iCloud Photos Downloader: Securely Back Up Your iCloud Photos

**Effortlessly download and back up your precious photos and videos from iCloud with the reliable and versatile iCloud Photos Downloader.** ([Original Repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features:

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, accommodating various devices (laptops, desktops, and NAS).
*   **Multiple Operation Modes:**
    *   **Copy:** Download new photos from iCloud (default).
    *   **Sync:** Download new photos and delete local files removed in iCloud (with `--auto-delete`).
    *   **Move:** Download new photos and delete photos in iCloud (with `--keep-icloud-recent-days`).
*   **Advanced Media Support:** Handles Live Photos (images and videos) and RAW images (including RAW+JPEG).
*   **Intelligent Features:** Automatic de-duplication of photos with the same name.
*   **Continuous Monitoring:** Option to monitor iCloud changes continuously (`--watch-with-interval`).
*   **Optimized Incremental Downloads:**  Supports incremental runs for faster updates (`--until-found` and `--recent`).
*   **Metadata Preservation:** Updates photo metadata (EXIF) with the `--set-exif-datetime` option.
*   **Flexible Installation:** Available as an executable, and through package managers like Docker, PyPI, AUR, and npm.
*   **Session Management:** Easily create and authorize a session for downloading.
*   **Regular Updates:** Releases are planned weekly, offering new features and improvements.

## iCloud Prerequisites

Ensure the following settings are enabled in your iCloud account to ensure successful downloads:

*   **Enable Access iCloud Data on the Web:**  `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable Advanced Data Protection:** `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation and Usage

You can install and run iCloud Photos Downloader in several ways:

1.  **Download Executable:**  Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2).
2.  **Use Package Managers:** Install via Docker, PyPI, AUR, or npm (links provided in the original README).
3.  **Build from Source:** Build and run the tool from the source code.

Detailed installation instructions are available in the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

### Example Usage:

To synchronize your iCloud photo collection to your local system, use:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

## Experimental Mode

Explore cutting-edge features in the experimental mode.  [Details](EXPERIMENTAL.md)

## Contributing

We welcome contributions! Review the [contributing guidelines](CONTRIBUTING.md) to get involved.