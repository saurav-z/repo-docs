# iCloud Photos Downloader: Download & Back Up Your iCloud Photos Securely

**Easily back up and manage your precious iCloud photos with the command-line power of iCloud Photos Downloader!**

[Link to Original Repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

This powerful tool provides a robust and reliable way to download and archive your iCloud photo library to your local device.

**Key Features:**

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Multiple Installation Options:** Available as an executable, Docker image, Python package (PyPI), AUR package, and npm package.
*   **Flexible Download Modes:**
    *   **Copy:** Download new photos.
    *   **Sync:** Download new photos and delete locally removed photos from iCloud.
    *   **Move:** Download new photos and delete them from iCloud.
*   **Comprehensive Support:**
    *   Supports Live Photos (image and video files).
    *   Handles RAW images (including RAW+JPEG).
    *   Automatic de-duplication.
    *   Metadata (EXIF) updates.
*   **Efficient Downloading:**
    *   Incremental downloads with `--until-found` and `--recent` options.
    *   Monitor for iCloud changes continuously (`--watch-with-interval` option).
*   **Authentication Management:** Create and authenticate sessions independently with `--auth-only`.

**Quick Start:**

To download your iCloud photos and continuously monitor for changes:

```bash
icloudpd --directory /path/to/your/photos --username your@email.com --watch-with-interval 3600
```

**iCloud Prerequisites:**

To ensure successful downloads, please ensure the following settings are configured in your iCloud account:

*   **Enable "Access iCloud Data on the Web"** in your iPhone/iPad settings (`Settings > Apple ID > iCloud > Access iCloud Data on the Web`).
*   **Disable "Advanced Data Protection"** in your iPhone/iPad settings (`Settings > Apple ID > iCloud > Advanced Data Protection`).

**Installation:**

Choose your preferred method:

*   **Executable:** Download from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.30.0).
*   **Package Managers:**
    *   [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker)
    *   [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi)
    *   [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur)
    *   [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm)

*For detailed installation instructions, please refer to the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).*

**Contribute:**

Help improve iCloud Photos Downloader!  Read the [contributing guidelines](CONTRIBUTING.md) and get involved!