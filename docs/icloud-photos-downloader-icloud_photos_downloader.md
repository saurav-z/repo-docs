# iCloud Photos Downloader: Easily Download Your iCloud Photos

**Tired of being locked into Apple's ecosystem?** iCloud Photos Downloader is a powerful command-line tool that lets you download your entire iCloud photo library to your computer, giving you control and backup options.

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Multiple Installation Options:** Download directly as an executable or via package managers like Docker, PyPI, AUR, and npm.
*   **Flexible Download Modes:** Choose from Copy, Sync (with auto-delete), and Move (download and remove from iCloud) options.
*   **Advanced Media Support:** Downloads Live Photos (image and video), and RAW images (including RAW+JPEG).
*   **Intelligent Handling:** Automatic de-duplication of photos with the same names.
*   **Continuous Monitoring:** Option to monitor for iCloud changes automatically.
*   **Metadata Preservation:** Option to update photo metadata (EXIF).

## Installation and Usage

You can install and run `icloudpd` using several methods:

*   **Download Executable:** Get the pre-built executable for your platform from the [GitHub Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.28.2).
*   **Package Managers:** Install via Docker, PyPI, AUR, or npm. See the [Installation Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed instructions.
*   **Build from Source:** Compile and run the tool directly from the source code.

### Example Usage:

To keep your iCloud photo collection synchronized to a local directory:

```bash
icloudpd --directory /path/to/your/photos --username your@email.com --watch-with-interval 3600
```

## Important iCloud Prerequisites

To ensure smooth operation, make sure your iCloud account is configured as follows:

*   **Enable "Access iCloud Data on the Web"** in your iCloud settings on your iPhone/iPad.
*   **Disable "Advanced Data Protection"** in your iCloud settings on your iPhone/iPad.

## Contributing

We welcome contributions! Please see the [contributing guidelines](CONTRIBUTING.md) for details on how to get involved and help improve iCloud Photos Downloader.

For more detailed information, please refer to the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/) and check out the [Issues](https://github.com/icloud-photos-downloader/icloud_photos_downloader/issues) page.

**[Visit the original repository on GitHub](https://github.com/icloud-photos-downloader/icloud_photos_downloader)**