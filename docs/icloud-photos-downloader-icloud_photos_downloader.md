# iCloud Photos Downloader: Download Your iCloud Photos with Ease

Easily back up and manage your precious memories with the iCloud Photos Downloader, a powerful command-line tool. ([Original Repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features:

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Versatile Installation:** Available as an executable, Docker image, or through package managers (PyPI, AUR, npm).
*   **Multiple Download Modes:** Choose between Copy, Sync (with auto-delete), and Move (download and delete from iCloud).
*   **Live Photo and RAW Image Support:** Download your Live Photos (image and video) and RAW images (including RAW+JPEG) without issue.
*   **Intelligent De-duplication:** Automatically handles duplicate photos.
*   **Continuous Monitoring:** Option to automatically monitor iCloud for changes.
*   **Incremental Download Optimizations:**  Efficiently download only new or changed photos with options like `--until-found` and `--recent`.
*   **Metadata Preservation:** Preserve photo EXIF data, and set the date/time in EXIF data.
*   **Command-Line Driven:** Download photos with command-line instructions, ideal for automation.

## iCloud Prerequisites

Before you start, ensure your iCloud account is configured correctly:

*   **Enable iCloud Data on the Web:**  Enable this option in your iCloud settings.
*   **Disable Advanced Data Protection:** Ensure Advanced Data Protection is disabled.

## Installation

You can install and run `icloudpd` in several ways:

1.  **Download Executable:** Get the pre-built executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2) page.
2.  **Package Managers:** Use a package manager such as Docker, PyPI, AUR, or npm (see [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed instructions).
3.  **Build from Source:**  Build and run the tool from its source code.

For detailed installation instructions, refer to the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html).

## Usage

**Example: Synchronize Your Photos**

To continuously sync your iCloud photos to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Note:** The executable name is `icloudpd`, not `icloud`.  Use `icloudpd --help` for a comprehensive list of command-line options.

**Authentication**

To independently create and authorize a session:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

## Experimental Features

Explore cutting-edge features in the experimental mode (see [EXPERIMENTAL.md] for details).

## Contributing

We welcome contributions! Please review the [contributing guidelines](CONTRIBUTING.md) to learn how to get involved.