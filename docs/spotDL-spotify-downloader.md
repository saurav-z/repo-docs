# spotDL: Download Spotify Music with Ease

**Tired of complicated music downloaders?** spotDL simplifies downloading your favorite songs from Spotify playlists by leveraging the power of YouTube, along with album art, lyrics, and metadata.

[![MIT License](https://img.shields.io/github/license/spotdl/spotify-downloader?color=44CC11&style=flat-square)](https://github.com/spotDL/spotify-downloader/blob/master/LICENSE)
[![PyPI version](https://img.shields.io/pypi/pyversions/spotDL?color=%2344CC11&style=flat-square)](https://pypi.org/project/spotdl/)
[![PyPi downloads](https://img.shields.io/pypi/dw/spotDL?label=downloads@pypi&color=344CC11&style=flat-square)](https://pypi.org/project/spotdl/)
![Contributors](https://img.shields.io/github/contributors/spotDL/spotify-downloader?style=flat-square)
[![Discord](https://img.shields.io/discord/771628785447337985?label=discord&logo=discord&style=flat-square)](https://discord.gg/xCa23pwJWY)

> **[View the original repository on GitHub](https://github.com/spotDL/spotify-downloader)**

## Key Features

*   **Effortless Downloads:** Easily download songs from Spotify playlists.
*   **High-Quality Audio:** Downloads from YouTube, aiming for the best available bitrate (up to 256 kbps).
*   **Metadata & Lyrics:** Automatically includes album art, lyrics, and song metadata.
*   **Multiple Operations:** Download, save metadata, get direct download URLs, sync directories, and update metadata.
*   **Community Driven:** Actively maintained and supported by a vibrant community.

## Installation

### Python (Recommended)

1.  Install with pip: `pip install spotdl`
2.  Update with: `pip install --upgrade spotdl`

    >  You may need to use `pip3` on some systems.

### Other Installation Options

*   **Prebuilt Executable:** Download from the [Releases Tab](https://github.com/spotDL/spotify-downloader/releases)
*   **Termux:** `curl -L https://raw.githubusercontent.com/spotDL/spotify-downloader/master/scripts/termux.sh | sh`
*   **Arch Linux (AUR):** [AUR package](https://aur.archlinux.org/packages/spotdl/)
*   **Docker:** Instructions in the original README.
*   **Build from source:** Instructions in the original README.

### Installing FFmpeg

FFmpeg is required. The easiest method is: `spotdl --download-ffmpeg`

Alternatively, install FFmpeg system-wide:

*   **Windows:** Instructions in the original README.
*   **macOS:** `brew install ffmpeg`
*   **Linux:** `sudo apt install ffmpeg` or use your distribution's package manager.

## Usage

```sh
spotdl [urls]
```

Or run as a package:

```sh
python -m spotdl [urls]
```

General Syntax:

```sh
spotdl [operation] [options] QUERY
```

The default **operation** is `download`. The **query** is typically Spotify URLs.  For a full list of **options**, use `spotdl -h`.

### Supported Operations

*   **`download` (Default):** Downloads songs and embeds metadata.
*   **`save`:** Saves only metadata. Usage: `spotdl save [query] --save-file {filename}.spotdl`
*   **`web`:** Starts a web interface (limited features).
*   **`url`:** Gets direct download links. Usage: `spotdl url [query]`
*   **`sync`:** Updates directories by comparing with playlists. Usage: `spotdl sync [query] --save-file {filename}.spotdl`. Use the created file to update the directory in the future: `spotdl sync {filename}.spotdl`
*   **`meta`:** Updates metadata for song files.

## Music Sourcing and Audio Quality

spotDL sources music from YouTube.  It aims to download the highest available bitrate (up to 256 kbps).

> **Disclaimer:** Users are responsible for their actions. Unauthorized downloading of copyrighted material is not supported.

## Contributing

Interested in contributing?  See [CONTRIBUTING.md](docs/CONTRIBUTING.md)

### Join our code contributor community

<a href="https://github.com/spotDL/spotify-downloader/graphs/contributors">
  <img class="dark-light" src="https://contrib.rocks/image?repo=spotDL/spotify-downloader&anon=0&columns=25&max=100&r=true" />
</a>

## License

This project is licensed under the [MIT](/LICENSE) License.