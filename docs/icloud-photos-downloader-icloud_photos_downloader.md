# Download Your iCloud Photos Easily with iCloud Photos Downloader

Tired of your iCloud photos being locked away? **iCloud Photos Downloader is a versatile command-line tool that empowers you to effortlessly download and back up all your photos and videos from iCloud.**

[Go to the original repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, accommodating laptops, desktops, and NAS devices.
*   **Flexible Installation:** Available as an executable for direct downloading and through package managers like Docker, PyPI, AUR, and npm for easy setup and updates.
*   **Multiple Modes of Operation:**
    *   **Copy:** Downloads new photos from iCloud (default).
    *   **Sync:** Downloads new photos and deletes local files removed from iCloud.
    *   **Move:** Downloads new photos and deletes them from iCloud.
*   **Comprehensive Support:** Handles Live Photos, RAW images (including RAW+JPEG), and photo metadata (EXIF) updates.
*   **Smart Features:** Automatic de-duplication, options for continuous monitoring, incremental downloads, and more.

## Getting Started

### iCloud Prerequisites

Before you begin, ensure your iCloud account is configured as follows to avoid "ACCESS\_DENIED" errors:

*   **Enable Access iCloud Data on the Web:** In your iPhone/iPad settings: `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable Advanced Data Protection:** In your iPhone/iPad settings: `Settings > Apple ID > iCloud > Advanced Data Protection`

### Installation and Usage

Choose your preferred method to install and run:

1.  **Download Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.28.2).
2.  **Package Manager:** Install and update through package managers like Docker, PyPI, AUR, or npm. Detailed instructions can be found in the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).
3.  **Build from Source:** Build and run the tool from the source code.

**Example Usage:**

To synchronize your iCloud photo collection to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

> **Important:** Remember to use the `icloudpd` executable, not `icloud`.

## Experimental Mode

Explore experimental features before they become part of the main package. More details are available in [EXPERIMENTAL.md](EXPERIMENTAL.md).

## Contribute

We welcome contributions! Check out the [contributing guidelines](CONTRIBUTING.md) to get involved.