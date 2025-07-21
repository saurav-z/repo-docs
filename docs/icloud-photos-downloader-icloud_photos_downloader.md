# iCloud Photos Downloader: Securely Download Your iCloud Photos (Command-Line Tool)

Easily back up and manage your precious iCloud photos with the **iCloud Photos Downloader**, a powerful command-line tool.  Access the original source code on [GitHub](https://github.com/icloud-photos-downloader/icloud_photos_downloader).

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Multiple Download Modes:** Choose between Copy, Sync, and Move modes to suit your needs.
*   **Live Photo and RAW Support:** Downloads both image and video components of Live Photos, along with RAW image files.
*   **Automatic De-duplication:** Avoids downloading duplicate photos with the same name.
*   **Continuous Monitoring:** Optionally monitor iCloud for changes and download updates automatically.
*   **Photo Metadata Preservation:**  Preserves EXIF data, ensuring your photo information is intact.
*   **Incremental Downloads:**  Optimize downloads with options like `--until-found` and `--recent`.
*   **Flexible Installation:** Install via executables, package managers (Docker, PyPI, AUR, npm), or build from source.

## iCloud Prerequisites

Before using iCloud Photos Downloader, configure your iCloud account as follows:

*   **Enable Web Access:**  In your iPhone/iPad settings, enable "Access iCloud Data on the Web".
*   **Disable Advanced Data Protection:**  In your iPhone/iPad settings, disable "Advanced Data Protection".

## Installation and Usage

### Installation Options:

1.  **Download Executable:** Get the pre-built executable from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.1) page.
2.  **Package Managers:** Install via Docker, PyPI, AUR, or npm (See [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for details).
3.  **Build from Source:**  Build and run from the source code.

### Basic Usage Example:

To synchronize your iCloud photos to a local directory and monitor for changes every hour:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Important:**  Use the `icloudpd` executable, not `icloud`.  For more options and advanced usage, run `icloudpd --help`.  You can also independently create and authorize a session for your local system:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Experimental Mode

Explore cutting-edge features in the experimental mode. [See details in EXPERIMENTAL.md](EXPERIMENTAL.md)

## Contributing

We welcome contributions!  Check out the [contributing guidelines](CONTRIBUTING.md) to learn how you can help improve iCloud Photos Downloader.