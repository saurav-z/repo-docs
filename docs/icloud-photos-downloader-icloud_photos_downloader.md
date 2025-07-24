# iCloud Photos Downloader: Effortlessly Download Your iCloud Photos

Tired of being locked into iCloud? **iCloud Photos Downloader** is a powerful command-line tool that allows you to easily download all your iCloud photos and videos to your computer.

[View the Original Repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, supporting laptops, desktops, and NAS devices.
*   **Multiple Download Modes:** Choose from Copy, Sync, and Move modes to manage your photo library as you see fit.
*   **Comprehensive Media Support:** Downloads Live Photos (image and video), RAW images (including RAW+JPEG), and maintains photo metadata (EXIF).
*   **Automated Features:** Includes de-duplication, continuous monitoring for iCloud changes, and optimizations for incremental downloads.
*   **Flexible Installation:** Available as an executable, through package managers (Docker, PyPI, AUR, npm), or from source.

## iCloud Prerequisites

Before using iCloud Photos Downloader, ensure your iCloud account is configured:

*   **Enable Access iCloud Data on the Web:** Go to `Settings > Apple ID > iCloud > Access iCloud Data on the Web` on your iPhone/iPad.
*   **Disable Advanced Data Protection:** Go to `Settings > Apple ID > iCloud > Advanced Data Protection` on your iPhone/iPad.

## Installation and Running

Choose your preferred method to get started:

1.  **Executable Download:** Download the pre-built executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases) page.
2.  **Package Managers:** Install using Docker, PyPI, AUR, or npm. See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed instructions.
3.  **Build from Source:** Build and run the tool from its source code.

## Usage Example

To synchronize your iCloud photos to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Important:** Use `icloudpd`, not `icloud` to run the application.

## Experimental Mode

Explore the latest features in the experimental mode. [Learn More](EXPERIMENTAL.md)

## Contributing

We welcome contributions! Review the [contributing guidelines](CONTRIBUTING.md) to get involved and help improve iCloud Photos Downloader.