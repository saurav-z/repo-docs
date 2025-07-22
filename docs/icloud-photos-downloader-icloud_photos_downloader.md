# iCloud Photos Downloader: Download Your iCloud Photos Easily

Tired of being locked into Apple's ecosystem? **iCloud Photos Downloader** is a versatile command-line tool that lets you effortlessly download all your precious photos and videos from iCloud, giving you complete control over your memories.

[Go to the Original Repository](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, Windows, and macOS.
*   **Multiple Download Modes:** Choose from Copy, Sync, and Move modes to suit your needs.
*   **Supports Various Media Types:** Downloads Live Photos (image and video), RAW images (including RAW+JPEG), and more.
*   **Automatic De-duplication:** Prevents duplicate downloads by automatically identifying and skipping photos with the same name.
*   **Continuous Monitoring:** Option to watch for iCloud changes and download new photos automatically.
*   **Metadata Preservation:** Preserves photo metadata (EXIF) to keep your memories organized.
*   **Flexible Download Options:** Optimized for incremental runs and offers a variety of options to customize your download experience.
*   **Available via Multiple Methods:** Download as an executable, or install via Docker, PyPI, AUR, or npm.

## iCloud Prerequisites

Before you begin, ensure the following settings are enabled in your iCloud account to avoid "ACCESS_DENIED" errors:

*   **Enable Access iCloud Data on the Web:**  `Settings > Apple ID > iCloud > Access iCloud Data on the Web`
*   **Disable Advanced Data Protection:**  `Settings > Apple ID > iCloud > Advanced Data Protection`

## Installation and Usage

You can install and run iCloud Photos Downloader in several ways:

1.  **Download Executable:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.29.2) page.
2.  **Package Manager:** Install and manage with [Docker](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#docker), [PyPI](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#pypi), [AUR](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#aur), or [npm](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html#npm).
3.  **Build from Source:** Build and run the tool from the source code.

**Basic Usage Example:**

To synchronize your iCloud photo collection to a local directory:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

For more detailed information, including advanced options and troubleshooting tips, please refer to the comprehensive [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/).

## Contributing

We welcome contributions! Please review the [contributing guidelines](CONTRIBUTING.md) to get started.