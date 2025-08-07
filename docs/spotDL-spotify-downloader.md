# Download Spotify Music with spotDL

**Effortlessly download your favorite Spotify playlists and tracks with spotDL, complete with metadata and high-quality audio.**

[View the spotDL repository on GitHub](https://github.com/spotDL/spotify-downloader)

## Key Features

*   **Batch Downloads:** Download entire Spotify playlists, albums, or individual tracks with ease.
*   **Metadata Embedding:** Automatically adds album art, lyrics, and comprehensive metadata to your downloaded music files.
*   **High-Quality Audio:** Downloads music from YouTube, offering the highest possible bitrate (up to 256 kbps for premium users).
*   **Multiple Operations:** Beyond downloading, also supports saving metadata, syncing directories, getting download URLs, and updating metadata for existing files.
*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux.

## Installation

### Recommended Method: Python

1.  Install spotDL using pip:
    ```bash
    pip install spotdl
    ```
2.  Update spotDL with:
    ```bash
    pip install --upgrade spotdl
    ```

    >   **Note:** On some systems, you might need to use `pip3` instead of `pip`.

<details>
    <summary><strong>Other Installation Options</strong></summary>

*   **Prebuilt Executable:** Download the latest version from the [Releases Tab](https://github.com/spotDL/spotify-downloader/releases).
*   **Termux:**
    ```bash
    curl -L https://raw.githubusercontent.com/spotDL/spotify-downloader/master/scripts/termux.sh | sh
    ```
*   **Arch Linux (AUR):** Use the [AUR package](https://aur.archlinux.org/packages/spotdl/).
*   **Docker:** Instructions are available in the original README.
</details>

### Installing FFmpeg

FFmpeg is essential for spotDL. You can install it to your spotDL installation directory:
`spotdl --download-ffmpeg`

Or install it system-wide using the instructions provided in the original README (Windows, macOS, Linux).

## Usage

Download music using a Spotify URL:

```bash
spotdl [Spotify URL(s)]
```

Or run spotDL as a Python package, if running the script is not working:

```bash
python -m spotdl [Spotify URL(s)]
```

General command structure:

```bash
spotdl [operation] [options] QUERY
```

By default, the operation is `download`.

For a list of all **options** use `spotdl -h`

<details>
<summary><strong>Supported Operations</strong></summary>

*   `save`: Saves metadata without downloading.
    *   Usage: `spotdl save [query] --save-file {filename}.spotdl`
*   `web`: Starts a web interface (limited features).
*   `url`: Gets direct download links.
    *   Usage: `spotdl url [query]`
*   `sync`: Updates directories by comparing with the playlist.
    *   Usage: `spotdl sync [query] --save-file {filename}.spotdl` (creates a sync file)
    *   To update the directory later: `spotdl sync {filename}.spotdl`
*   `meta`: Updates metadata for song files.

</details>

## Music Sourcing and Audio Quality

spotDL downloads music from YouTube to avoid issues related to downloading directly from Spotify.

**Note:** Users are responsible for their actions and potential legal consequences. We do not support unauthorized downloading of copyrighted material.

### Audio Quality

spotDL downloads the highest possible bitrate from YouTube: typically 128 kbps, and up to 256 kbps for YouTube Music premium users.

## Contributing

We welcome contributions! See our [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on getting started.

### Join Our Community

<a href="https://github.com/spotDL/spotify-downloader/graphs/contributors">
  <img class="dark-light" src="https://contrib.rocks/image?repo=spotDL/spotify-downloader&anon=0&columns=25&max=100&r=true" />
</a>

## License

This project is licensed under the [MIT](/LICENSE) License.