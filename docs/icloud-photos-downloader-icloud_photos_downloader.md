# iCloud Photos Downloader: Effortlessly Download Your iCloud Photos

**Tired of iCloud photo storage limits?**  Download all your iCloud photos and videos to your computer or NAS with the versatile and easy-to-use **iCloud Photos Downloader**! [(See original repo)](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting desktops, laptops, and NAS devices.
*   **Flexible Installation:** Available as an executable, Docker image, PyPI package, AUR package, and npm package for easy setup.
*   **Multiple Download Modes:** Choose between Copy, Sync (with auto-delete), and Move (with iCloud deletion) modes to fit your needs.
*   **Comprehensive Media Support:** Downloads Live Photos (image and video), RAW images (including RAW+JPEG).
*   **Intelligent Photo Management:** Automatic de-duplication of photos with the same name.
*   **Automated Updates:**  One-time download or continuous monitoring with the `--watch-with-interval` option.
*   **Optimized for Efficiency:** Supports incremental runs with `--until-found` and `--recent` options.
*   **Metadata Preservation:**  Updates photo metadata (EXIF) using the `--set-exif-datetime` option.
*   **Plus More:** Explore advanced options with the `--help` command for full functionality.

## iCloud Prerequisites

To ensure successful downloads, please configure your iCloud account as follows:

*   **Enable "Access iCloud Data on the Web":**  In your iPhone / iPad settings: `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable "Advanced Data Protection":** In your iPhone / iPad settings: `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation and Usage

You can download and run iCloud Photos Downloader in several ways:

1.  **Download Executable:** Get the pre-built executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2) page.
2.  **Package Managers:** Install via Docker, PyPI, AUR, or npm.  See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for installation instructions.
3.  **Build from Source:** For advanced users, compile and run the tool from the source code.

**Example Usage (Syncing iCloud Photos):**

To automatically keep your local photo library synchronized:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> [!IMPORTANT]
> Remember to use `icloudpd`, not `icloud`.

**Authorize a Session**
You can independantly create and authorize a session (and complete 2SA/2FA validation if needed) on your local system:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Experimental Mode

New features are often tested in experimental mode first. See [EXPERIMENTAL.md](EXPERIMENTAL.md) for details.

## Contributing

We welcome contributions!  Please review the [contributing guidelines](CONTRIBUTING.md) to get involved.