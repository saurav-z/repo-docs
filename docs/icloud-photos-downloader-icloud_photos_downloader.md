# iCloud Photos Downloader: Your Easy Way to Back Up iCloud Photos 

**Tired of being locked into iCloud?** iCloud Photos Downloader is a powerful command-line tool that lets you effortlessly download and back up your entire iCloud photo library to your computer or network storage.  [View the source on GitHub](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Key Features:**

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Multiple Download Modes:** Choose between Copy, Sync (with automatic deletion), and Move modes to fit your needs.
*   **Advanced Media Support:** Downloads Live Photos (image and video) and RAW images (including RAW+JPEG).
*   **Intelligent Deduplication:**  Automatically avoids downloading duplicate photos.
*   **Continuous Monitoring:** Keeps your local backup up-to-date with optional continuous monitoring.
*   **Metadata Preservation:** Preserves and updates photo metadata (EXIF) for accurate organization.
*   **Incremental Downloads:** Optimized for efficient incremental downloads using options like `--until-found` and `--recent`.
*   **Flexible Installation:** Install via Docker, PyPI, AUR, or npm or use the provided executable.

**iCloud Prerequisites**

To ensure iCloud Photos Downloader functions correctly, please configure your iCloud account as follows:

*   **Enable Web Access:** On your iPhone/iPad, navigate to `Settings > [Your Apple ID] > iCloud > Access iCloud Data on the Web`.
*   **Disable Advanced Data Protection:** Disable `Settings > [Your Apple ID] > iCloud > Advanced Data Protection` on your iPhone/iPad.

**Installation & Running**

You have several options for installing and running `icloudpd`:

1.  **Executable Download:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.31.0) page.
2.  **Package Managers:** Utilize package managers like Docker, PyPI, AUR, or npm to install, update, and run the downloader.  See the [Installation Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed instructions.
3.  **Build from Source:** Compile and run the tool directly from the source code.

**Example Usage:**

To synchronize your iCloud photos to a local directory, run:

```bash
icloudpd --directory /data --username your@email.com --watch-with-interval 3600
```

**Important Notes:**

*   The executable name is `icloudpd`, not `icloud`.
*   For detailed command-line options, consult the help: `icloudpd --help`

**Experimental Features:**

Explore experimental features that are in the testing phase.  See the [Experimental Mode Details](EXPERIMENTAL.md) for information on testing and contributing to the tool.

**Contributing**

We welcome contributions! Please review our [contributing guidelines](CONTRIBUTING.md) to get started.