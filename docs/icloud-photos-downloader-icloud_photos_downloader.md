# iCloud Photos Downloader: Effortlessly Back Up Your iCloud Photos

**Tired of being locked into iCloud?** iCloud Photos Downloader is a powerful command-line tool that lets you download all your photos and videos from iCloud to your computer, giving you control over your precious memories.  

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features:

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Flexible Installation:** Available as an executable, through package managers (Docker, PyPI, AUR, npm), and from source.
*   **Multiple Download Modes:**
    *   **Copy:** Download new photos from iCloud (default).
    *   **Sync:** Download new photos and automatically delete local files removed from iCloud.
    *   **Move:** Download new photos and delete them from iCloud.
*   **Advanced Support:** Handles Live Photos (separate image and video files) and RAW images (including RAW+JPEG).
*   **Intelligent Features:**
    *   Automatic de-duplication of photos.
    *   Option to monitor for iCloud changes continuously.
    *   Optimizations for incremental downloads.
    *   Photo metadata (EXIF) updates.
*   **Regular Updates:** The project aims to release new versions weekly to provide new features and ensure compatibility with the Apple ecosystem.

## Installation & Usage

### Prerequisites

Before using iCloud Photos Downloader, ensure the following iCloud settings are enabled:

*   **Enable "Access iCloud Data on the Web"** in your iPhone/iPad settings.
*   **Disable "Advanced Data Protection"** in your iPhone/iPad settings.

### Installation Options

Choose the method that best suits your needs:

1.  **Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.31.0).
2.  **Package Manager:** Install using Docker, PyPI, AUR, or npm. See the [Installation Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for details.
3.  **Build from Source:** Build and run the tool from the source code.

### Basic Usage

To synchronize your iCloud photo library with a local directory:

```bash
icloudpd --directory /path/to/your/photos --username your@email.com --watch-with-interval 3600
```

> **Important:**  Use the `icloudpd` executable, not `icloud`.

> **Tip:**  Explore command-line parameters for advanced synchronization options using `icloudpd --help`.

## Experimental Mode

Try out new features before they are officially released. Check out the [Experimental Mode details](EXPERIMENTAL.md).

## Contributing

We welcome contributions! Check out the [contributing guidelines](CONTRIBUTING.md) to get involved and help improve iCloud Photos Downloader.

[Visit the original repository on GitHub](https://github.com/icloud-photos-downloader/icloud_photos_downloader)