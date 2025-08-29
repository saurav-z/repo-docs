# iCloud Photos Downloader: Download and Backup Your iCloud Photos Easily

**Effortlessly download and back up your precious iCloud photos with the powerful and versatile iCloud Photos Downloader.** ([View on GitHub](https://github.com/icloud-photos-downloader/icloud_photos_downloader))

This command-line tool empowers you to securely download your entire iCloud photo library to your computer, offering flexibility and control over your photo backup process.

**Key Features:**

*   **Multi-Platform Support:** Works seamlessly on Linux, Windows, and macOS, accommodating various devices like laptops, desktops, and NAS.
*   **Multiple Installation Options:** Install via executable, Docker, PyPI, AUR, or npm, catering to your preferred workflow.
*   **Flexible Download Modes:** Choose between copy, sync (with auto-delete), and move (with iCloud photo deletion).
*   **Live Photo and RAW Support:** Downloads both Live Photos (image and video) and RAW images (including RAW+JPEG).
*   **Intelligent De-duplication:** Automatically avoids downloading duplicate photos.
*   **Continuous Monitoring:** Options for one-time downloads and continuous monitoring for iCloud changes.
*   **Photo Metadata Preservation:** Option to update photo metadata (EXIF) for accurate organization.
*   **Incremental Downloads:** Optimizations with `--until-found` and `--recent` options for efficient downloads.
*   **Comprehensive Functionality:** Explore numerous additional features by using the `--help` command.

## Getting Started

### iCloud Prerequisites

To ensure successful downloads, verify the following settings within your iCloud account:

*   **Enable Access iCloud Data on the Web:** `Settings > Apple ID > iCloud > Access iCloud Data on the Web` on your iPhone/iPad.
*   **Disable Advanced Data Protection:** `Settings > Apple ID > iCloud > Advanced Data Protection` on your iPhone/iPad.

### Installation and Usage

You can install and run iCloud Photos Downloader using several methods:

1.  **Download Executable:** Get the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.31.0).
2.  **Package Managers:** Install via package managers like Docker, PyPI, AUR, and npm. See [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for details.
3.  **Build from Source:** Compile and run the tool directly from the source code.

**Example Usage:**

To synchronize your iCloud photo collection to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Important Note:** Be sure to use the `icloudpd` executable and not `icloud`.

### Experimental Mode

Explore cutting-edge features in the experimental mode before they become part of the main package. More details in [EXPERIMENTAL.md](EXPERIMENTAL.md).

## Contribute

Help improve iCloud Photos Downloader! Review the [contributing guidelines](CONTRIBUTING.md) for more information on how to contribute.