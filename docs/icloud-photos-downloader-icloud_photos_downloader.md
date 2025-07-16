# iCloud Photos Downloader: Securely Download Your iCloud Photos to Your Device

Tired of relying on iCloud for your precious photos? **iCloud Photos Downloader is a powerful, command-line tool that lets you effortlessly download all your iCloud photos and videos to your computer.** Get complete control over your memories and back them up safely and easily.

[![Quality Checks](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Quality%20Checks/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/quality-checks.yml)
[![Build and Package](https://github.com/icloud-photos-downloader/icloud_photos_downloader/workflows/Produce%20Artifacts/badge.svg)](https://github.com/icloud-photos-downloader/icloud_photos_downloader/actions/workflows/produce-artifacts.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Key Features:**

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS, making it accessible for all users.
*   **Multiple Download Modes:** Choose between Copy (default), Sync (delete local files removed from iCloud), and Move (download and delete from iCloud).
*   **Live Photo and RAW Support:** Downloads Live Photos (image and video) and RAW images (including RAW+JPEG) as separate files.
*   **Automatic De-duplication:** Handles photos with the same name efficiently.
*   **Incremental and Continuous Downloading:** One-time download or continuously monitor for iCloud changes.
*   **Metadata Preservation:** Option to update photo metadata (EXIF) for accurate organization.
*   **Flexible Installation Options:** Available as an executable, or through package managers like Docker, PyPI, AUR, and npm.
*   **Watch Mode:** Allows continuous download and syncing of images with iCloud with a set interval.

**Installation and Usage**

You can install and run `icloudpd` in multiple ways:

1.  **Download Executable:** Get the pre-built executable for your platform from the [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.28.2) page.
2.  **Package Managers:** Install via Docker, PyPI, AUR, or npm (see [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for details).
3.  **Build from Source:** Compile and run the tool from the source code.

**Example Usage:**

To keep your iCloud photo collection synchronized to your local system:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Important Notes:**

*   Use `icloudpd` instead of `icloud` as the executable name.
*   For in-depth customization and options, run `icloudpd --help`.

**iCloud Prerequisites:**

To ensure seamless operation, configure your iCloud account as follows:

*   Enable "Access iCloud Data on the Web" on your iPhone/iPad (`Settings > Apple ID > iCloud > Access iCloud Data on the Web`).
*   Disable "Advanced Data Protection" on your iPhone/iPad (`Settings > Apple ID > iCloud > Advanced Data Protection`).

**Experimental Mode**

Check the [EXPERIMENTAL.md](EXPERIMENTAL.md) file for the latest features under testing.

**Contribute**

Your contributions are welcome!  See the [contributing guidelines](CONTRIBUTING.md) to get involved.

**Learn More**

For detailed information, installation instructions, and advanced usage, please refer to the official [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/) and browse the [Issues](https://github.com/icloud-photos-downloader/icloud_photos_downloader/issues) to see how you can help.

**[Back to the Project Repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader)**