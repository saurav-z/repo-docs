# Download Your Favorite Spotify Music with spotDL

**spotDL** is a powerful command-line tool that lets you download music from Spotify playlists, extracting songs from YouTube along with album art, lyrics, and metadata.  Get ready to enjoy your favorite tunes offline! [(See the original repository on GitHub)](https://github.com/spotDL/spotify-downloader)

[![MIT License](https://img.shields.io/github/license/spotdl/spotify-downloader?color=44CC11&style=flat-square)](https://github.com/spotDL/spotify-downloader/blob/master/LICENSE)
[![PyPI version](https://img.shields.io/pypi/pyversions/spotdl?color=%2344CC11&style=flat-square)](https://pypi.org/project/spotdl/)
[![PyPi downloads](https://img.shields.io/pypi/dw/spotDL?label=downloads@pypi&color=344CC11&style=flat-square)](https://pypi.org/project/spotdl/)
![Contributors](https://img.shields.io/github/contributors/spotDL/spotify-downloader?style=flat-square)
[![Discord](https://img.shields.io/discord/771628785447337985?label=discord&logo=discord&style=flat-square)](https://discord.gg/xCa23pwJWY)

______________________________________________________________________

**[Read the full documentation](https://spotdl.readthedocs.io)**

## Key Features

*   **Effortless Downloads:** Easily download music from Spotify playlists using YouTube as a source.
*   **Metadata & Lyrics:** Automatically includes album art, lyrics, and song metadata.
*   **Multiple Operations:**  Download, save metadata only, get direct download URLs, and synchronize directories.
*   **High-Quality Audio:** Downloads audio at the highest available bitrate (up to 256 kbps for YouTube Music Premium users).
*   **Flexible Installation:** Supports Python `pip`, prebuilt executables, and other methods.
*   **Command-Line Convenience:**  Simple and efficient command-line interface.
*   **Community-Driven:** Actively maintained open-source project with a supportive community.

## Installation

### Python (Recommended)

1.  Install spotDL: `pip install spotdl`
2.  Update spotDL: `pip install --upgrade spotdl`

    > On some systems, you might need to use `pip3` instead of `pip`.

### Other Installation Options

*   **Prebuilt Executable:** Download from the [Releases](https://github.com/spotDL/spotify-downloader/releases) tab.
*   **Termux:**  `curl -L https://raw.githubusercontent.com/spotDL/spotify-downloader/master/scripts/termux.sh | sh`
*   **Arch Linux (AUR):** Use the [AUR package](https://aur.archlinux.org/packages/spotdl/).
*   **Docker:** Build and run a Docker container (see the original README for details).

### Installing FFmpeg

FFmpeg is required for spotDL.  Install it by running: `spotdl --download-ffmpeg` (recommended).

Alternatively, install it system-wide:

*   Windows: Follow a [Windows Tutorial](https://windowsloop.com/install-ffmpeg-windows-10/)
*   OSX: `brew install ffmpeg`
*   Linux: `sudo apt install ffmpeg` (or your distro's package manager)

## Usage

Basic usage:

```bash
spotdl [Spotify URL or query]
```

For advanced usage:

```bash
spotdl [operation] [options] [query]
```

Operations include:

*   `download`:  (Default) Downloads songs with metadata.
*   `save`: Saves only metadata (e.g., `spotdl save [query] --save-file {filename}.spotdl`).
*   `web`: Starts a web interface (limited features).
*   `url`: Gets direct download links.
*   `sync`: Synchronizes a directory with a playlist (e.g., `spotdl sync [query] --save-file {filename}.spotdl`).
*   `meta`: Updates metadata for existing song files.

Use `spotdl -h` for a complete list of options.

## Music Sourcing and Audio Quality

spotDL uses YouTube as a source for music downloads. It downloads audio at the highest available bitrate (up to 256 kbps for YouTube Music Premium users).

**Disclaimer:** Users are responsible for their actions and potential legal consequences. We do not support unauthorized downloading of copyrighted material and take no responsibility for user actions.

## Contributing

Interested in contributing? Check out our [CONTRIBUTING.md](docs/CONTRIBUTING.md) to find
resources around contributing along with a guide on how to set up a development environment.

### Join our amazing community as a code contributor

<a href="https://github.com/spotDL/spotify-downloader/graphs/contributors">
  <img class="dark-light" src="https://contrib.rocks/image?repo=spotDL/spotify-downloader&anon=0&columns=25&max=100&r=true" />
</a>

## License

This project is licensed under the [MIT](/LICENSE) License.