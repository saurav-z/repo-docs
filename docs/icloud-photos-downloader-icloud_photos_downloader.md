# iCloud Photos Downloader: Download Your iCloud Photos Easily

**Effortlessly back up and manage your precious iCloud photos with the powerful and versatile iCloud Photos Downloader!**

[Get Started with iCloud Photos Downloader](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, including desktops, laptops, and NAS devices.
*   **Multiple Download Modes:** Choose the perfect mode for your needs:
    *   **Copy:** Download new photos from iCloud (default).
    *   **Sync:** Download new photos and delete local files removed from iCloud (`--auto-delete`).
    *   **Move:** Download new photos and delete them from iCloud (`--keep-icloud-recent-days`).
*   **Advanced Media Support:** Handles Live Photos (image and video), RAW images (including RAW+JPEG), and various file types.
*   **Intelligent File Management:** Automatically de-duplicates photos with the same names.
*   **Continuous Monitoring:** Download once or monitor iCloud changes continuously (`--watch-with-interval`).
*   **Incremental Downloads:** Optimized options for faster downloads (`--until-found`, `--recent`).
*   **Metadata Preservation:** Updates photo metadata (EXIF) for accurate file information (`--set-exif-datetime`).
*   **Flexible Installation:** Available as an executable, through package managers (Docker, PyPI, AUR, npm), or from source.

## iCloud Prerequisites

To ensure smooth operation, configure your iCloud account with the following settings:

*   **Enable Access iCloud Data on the Web:**  On your iPhone/iPad, enable `Settings > Apple ID > iCloud > Access iCloud Data on the Web`.
*   **Disable Advanced Data Protection:** On your iPhone/iPad, disable `Settings > Apple ID > iCloud > Advanced Data Protection`.

## Installation and Usage

### Installation Options:

1.  **Executable:** Download the pre-built executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.28.2) page.
2.  **Package Managers:** Install via Docker, PyPI, AUR, or npm. See the [Installation Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed instructions.
3.  **From Source:** Build and run the tool from the source code.

### Basic Usage Example:

To synchronize your iCloud photo collection to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

*   **Important:**  The executable is called `icloudpd`, not `icloud`.
*   For comprehensive options, use `icloudpd --help`.

### Authenticate Session:

To create and authorize a session (including 2SA/2FA if needed):

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Contributing

We welcome contributions!  Review our [contributing guidelines](CONTRIBUTING.md) to get involved in the development of iCloud Photos Downloader.