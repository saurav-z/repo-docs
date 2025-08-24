# iCloud Photos Downloader: Download and Back Up Your iCloud Photos Easily

**Easily back up and manage your precious memories with the iCloud Photos Downloader, a powerful command-line tool for downloading your iCloud photos.** Check out the original repository [here](https://github.com/icloud-photos-downloader/icloud_photos_downloader).

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting laptops, desktops, and NAS devices.
*   **Multiple Installation Options:** Install via executable, Docker, PyPI, AUR, or npm for flexible deployment.
*   **Multiple Synchronization Modes:**
    *   **Copy:** Downloads new photos from iCloud.
    *   **Sync:** Downloads new photos and deletes local files removed from iCloud (with the `--auto-delete` option).
    *   **Move:** Downloads new photos and deletes them from iCloud (with the `--keep-icloud-recent-days` option).
*   **Advanced Media Support:** Handles Live Photos (image and video) and RAW images (including RAW+JPEG).
*   **Smart File Management:** Automatic de-duplication of photos with the same name.
*   **Continuous Monitoring:** Option to monitor for iCloud changes continuously (`--watch-with-interval` option).
*   **Incremental Download Options:** Optimizations for incremental runs using `--until-found` and `--recent` options.
*   **Metadata Preservation:** Photo metadata (EXIF) updates (`--set-exif-datetime` option).
*   **Many More Features:** Explore a comprehensive list of features by using the `--help` option.

## Installation and Usage

### Prerequisites for iCloud

Before using iCloud Photos Downloader, ensure your iCloud account is configured with these settings:

*   **Enable Access iCloud Data on the Web:** On your iPhone / iPad, go to `Settings > Apple ID > iCloud > Access iCloud Data on the Web`.
*   **Disable Advanced Data Protection:** On your iPhone / iPad, go to `Settings > Apple ID > iCloud > Advanced Data Protection`.

### Installation

You can install `icloudpd` in three ways:

1.  **Download Executable:** Get the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.31.0).
2.  **Use Package Managers:** Install and manage updates with package managers like Docker, PyPI, AUR, and npm (see [documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html)).
3.  **Build from Source:** Build and run from the source code.

Detailed installation instructions are available in the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

### Example Usage

To automatically synchronize your iCloud photo library to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

Remember to use `icloudpd` and not `icloud` in your commands.

For authentication-only sessions:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Experimental Features

Explore new features in experimental mode. See [EXPERIMENTAL.md](EXPERIMENTAL.md) for details.

## Contributing

We welcome contributions! Review the [contributing guidelines](CONTRIBUTING.md) to get involved.