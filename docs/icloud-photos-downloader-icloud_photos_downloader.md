# Download Your iCloud Photos Easily with iCloud Photos Downloader

Tired of being locked into iCloud? iCloud Photos Downloader is a powerful command-line tool that lets you download all your iCloud photos and videos directly to your computer, freeing you from vendor lock-in. ([View the original repo](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

## Key Features:

*   **Cross-Platform Compatibility:** Works on Linux, Windows, and macOS, supporting desktops, laptops, and even NAS devices.
*   **Multiple Download Modes:** Choose between "Copy," "Sync" (with auto-delete), and "Move" (with iCloud deletion) for flexible management.
*   **Advanced File Handling:** Supports Live Photos (image and video), RAW images (including RAW+JPEG), and automatic de-duplication.
*   **Continuous Monitoring:** Option to watch for iCloud changes continuously, ensuring your local copy stays up-to-date.
*   **Optimized for Incremental Downloads:** Use `--until-found` and `--recent` options for efficient downloads.
*   **Photo Metadata Preservation:** Option to update EXIF data for accurate photo information.
*   **Multiple Installation Options:** Available as an executable, through package managers (Docker, PyPI, AUR, npm), and from source.

## iCloud Prerequisites

Before using iCloud Photos Downloader, ensure your iCloud account is configured with the following settings:

*   **Enable Access iCloud Data on the Web:**  `Settings > Apple ID > iCloud > Access iCloud Data on the Web` on your iPhone/iPad.
*   **Disable Advanced Data Protection:**  `Settings > Apple ID > iCloud > Advanced Data Protection` on your iPhone/iPad.

## Installation

You can download `icloudpd` in a few ways:

1.  **Download Executable:** Get the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.0).
2.  **Package Managers:** Install using Docker, PyPI, AUR, or npm. See the [Installation Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for details.
3.  **Build from Source:** Compile and run the tool directly from its source code.

## Usage Examples

**Synchronize Your Photos:**

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Authenticate a Session:**

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Contribute

We welcome contributions!  Please review the [contributing guidelines](CONTRIBUTING.md) to learn how you can help.