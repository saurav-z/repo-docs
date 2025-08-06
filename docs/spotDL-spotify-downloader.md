# spotDL: Download Your Favorite Spotify Music with Ease

Tired of limited music streaming? **spotDL** is the ultimate command-line tool that effortlessly downloads your favorite songs from Spotify, complete with album art, lyrics, and metadata. [Explore the original repository](https://github.com/spotDL/spotify-downloader) for more information and to contribute.

<div align="center">
  <img src="https://img.shields.io/github/license/spotdl/spotify-downloader?color=44CC11&style=flat-square" alt="MIT License">
  <img src="https://img.shields.io/pypi/pyversions/spotDL?color=%2344CC11&style=flat-square" alt="Python Versions">
  <img src="https://img.shields.io/pypi/dw/spotDL?label=downloads@pypi&color=344CC11&style=flat-square" alt="PyPI Downloads">
  <img src="https://img.shields.io/github/contributors/spotDL/spotify-downloader?style=flat-square" alt="Contributors">
  <img src="https://img.shields.io/discord/771628785447337985?label=discord&logo=discord&style=flat-square" alt="Discord">
</div>

## Key Features

*   **Seamless Downloads:** Download music from Spotify playlists and albums directly to your device.
*   **Metadata Magic:** Automatically adds album art, lyrics, and song metadata.
*   **High-Quality Audio:** Downloads the best available quality (up to 256kbps) from YouTube.
*   **Flexible Usage:** Supports various operations, including saving metadata, syncing directories, and updating metadata for existing files.
*   **Cross-Platform:** Works on Windows, macOS, and Linux.

## Installation

For detailed installation instructions, see the [Installation Guide](https://spotdl.readthedocs.io/en/latest/installation.html).

### Python (Recommended)

1.  Install: `pip install spotdl`
2.  Update: `pip install --upgrade spotdl`

    > On some systems, you may need to replace `pip` with `pip3`.

<details>
    <summary style="font-size:1.25em"><strong>Other Options</strong></summary>

*   **Prebuilt Executable:** Download from the [Releases Tab](https://github.com/spotDL/spotify-downloader/releases).
*   **Termux:** `curl -L https://raw.githubusercontent.com/spotDL/spotify-downloader/master/scripts/termux.sh | sh`
*   **Arch Linux (AUR):** Use the [AUR package](https://aur.archlinux.org/packages/spotdl/).
*   **Docker:** Build and run using the provided Dockerfile.

</details>

### Installing FFmpeg

FFmpeg is required for spotDL.  If you only need FFmpeg for spotDL, install it to the spotdl installation directory:
`spotdl --download-ffmpeg`

For system-wide installation, follow these guides:

*   [Windows Tutorial](https://windowsloop.com/install-ffmpeg-windows-10/)
*   OSX - `brew install ffmpeg`
*   Linux - `sudo apt install ffmpeg` or use your distro's package manager

## Usage

Basic usage:

```bash
spotdl [urls]
```

To run spotDL as a package if running it as a script doesn't work:

```bash
python -m spotdl [urls]
```

General usage:

```bash
spotdl [operation] [options] QUERY
```

The default **operation** is `download`. The **query** is typically a Spotify URL. For all **options**, run `spotdl -h`.

<details>
<summary style="font-size:1em"><strong>Supported Operations</strong></summary>

*   `save`: Save metadata only.
    *   Usage: `spotdl save [query] --save-file {filename}.spotdl`
*   `web`: Launch a web interface (limited features).
*   `url`: Get direct download links.
    *   Usage: `spotdl url [query]`
*   `sync`: Update directories based on a playlist.
    *   Usage: `spotdl sync [query] --save-file {filename}.spotdl`
*   `meta`: Update metadata for existing files.

</details>

## Music Sourcing and Audio Quality

spotDL uses YouTube for music downloads, aiming for the highest available bitrate (up to 256kbps).

> **Disclaimer:** Users are responsible for their actions. We do not support unauthorized downloading of copyrighted material.

## Contributing

Interested in contributing?  See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for resources and setup instructions.

### Join the Community

<a href="https://github.com/spotDL/spotify-downloader/graphs/contributors">
  <img class="dark-light" src="https://contrib.rocks/image?repo=spotDL/spotify-downloader&anon=0&columns=25&max=100&r=true" alt="Contributors">
</a>

## License

This project is licensed under the [MIT](/LICENSE) License.