# iCloud Photos Downloader: Download Your iCloud Photos to Any Device

**Effortlessly back up your iCloud photos and videos with the open-source, cross-platform iCloud Photos Downloader.**

[View the original repository on GitHub](https://github.com/icloud-photos-downloader/icloud_photos_downloader)

**Key Features:**

*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux, for desktop, laptop, and NAS devices.
*   **Multiple Installation Options:** Available as a direct executable, Docker image, PyPI package, AUR package, and npm package.
*   **Flexible Download Modes:** Choose from Copy (default), Sync (with automatic deletion), and Move (download and delete from iCloud).
*   **Comprehensive Media Support:** Handles Live Photos, RAW images (including RAW+JPEG), and video.
*   **Smart Features:** Includes automatic de-duplication, metadata (EXIF) updates, and continuous monitoring options.
*   **Efficient Downloading:** Optimize downloads with incremental run options.
*   **Regular Updates:** New versions are released frequently to improve features and address issues.

**iCloud Prerequisites:**

To ensure smooth operation, please configure your iCloud account as follows:

*   **Enable Access iCloud Data on the Web:** In your iPhone/iPad settings, enable `Settings > Apple ID > iCloud > Access iCloud Data on the Web`.
*   **Disable Advanced Data Protection:** On your iPhone/iPad, disable `Settings > Apple ID > iCloud > Advanced Data Protection`.

**Installation and Usage:**

You can download and run the iCloud Photos Downloader in the following ways:

1.  **Executable Download:** Download the executable for your platform from the GitHub [Releases](https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases/tag/v1.31.0) page.
2.  **Package Managers:** Install via Docker, PyPI, AUR, or npm. See the [Documentation](https://icloud-photos-downloader.github.io/icloud_photos_downloader/install.html) for detailed instructions.
3.  **Build from Source:** Compile and run the tool from the source code.

**Example Usage:**

To synchronize your iCloud photo collection with your local system:

```bash
icloudpd --directory /data --username my@email.address --watch-with-interval 3600
```

**Advanced Usage:**

Run `icloudpd --help` to see all command-line parameters and usage options.  Create and authorize a session by using the following command:

```bash
icloudpd --username my@email.address --password my_password --auth-only
```

**Contributing:**

We welcome contributions! Please review our [contributing guidelines](CONTRIBUTING.md) to learn how you can help improve the iCloud Photos Downloader.