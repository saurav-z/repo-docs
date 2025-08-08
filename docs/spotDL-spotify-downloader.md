# spotDL: Download Music from Spotify with Ease

Tired of juggling multiple apps to enjoy your favorite Spotify playlists offline? spotDL is your all-in-one solution for downloading Spotify music with album art, lyrics, and metadata, all from the command line. [(Back to original repo)](https://github.com/spotDL/spotify-downloader)

**Key Features:**

*   🎶 **Effortless Downloads:** Download entire Spotify playlists or individual tracks with a simple command.
*   🖼️ **Metadata Magic:** Automatically embeds album art, lyrics, and other essential metadata.
*   🎬 **YouTube Sourcing:** Reliably finds music sources on YouTube, ensuring broad compatibility.
*   🎧 **Flexible Audio Quality:** Downloads the highest possible bitrate available on YouTube (up to 256kbps for premium users).
*   🔄 **Playlist Synchronization:** Keeps your downloaded music in sync with your Spotify playlists.
*   💻 **Cross-Platform:** Works seamlessly on Windows, macOS, and Linux.
*   🌐 **Web Interface (Optional):** Offers a web interface for simpler downloads.
*   ⚙️ **Advanced Operations:** Supports various operations like saving metadata, generating download links, and updating metadata.

## Installation

Follow the installation guide for detailed instructions.

*   **Python (Recommended):**
    *   Install: `pip install spotdl`
    *   Update: `pip install --upgrade spotdl`

    > Note:  You may need to use `pip3` instead of `pip` on some systems.

*   **Other Options:**
    *   Prebuilt Executable (from [Releases](https://github.com/spotDL/spotify-downloader/releases))
    *   Termux: `curl -L https://raw.githubusercontent.com/spotDL/spotify-downloader/master/scripts/termux.sh | sh`
    *   Arch Linux (AUR):  [spotdl AUR package](https://aur.archlinux.org/packages/spotdl/)
    *   Docker:  Build and run container (see original README for instructions).

### Installing FFmpeg

FFmpeg is required for spotDL. Install as follows.

*   **Recommended:** `spotdl --download-ffmpeg` (installs to spotDL directory)
*   **System-wide:** (Follow instructions for your OS)
    *   Windows: [FFmpeg on Windows](https://windowsloop.com/install-ffmpeg-windows-10/)
    *   macOS: `brew install ffmpeg`
    *   Linux: `sudo apt install ffmpeg` or use your distribution's package manager

## Usage

Basic usage:

```sh
spotdl [spotify_url_or_query]
```

Alternatively, use spotDL as a package:

```sh
python -m spotdl [spotify_url_or_query]
```

General usage with operations and options:

```sh
spotdl [operation] [options] QUERY
```

Supported operations:

*   `download` (default): Downloads songs from YouTube, embedding metadata.
*   `save`: Saves metadata without downloading files.
*   `web`: Launches a web interface (limited features).
*   `url`: Retrieves direct download links for songs.
*   `sync`:  Synchronizes a local directory with a Spotify playlist (downloads new songs, removes deleted ones).
*   `meta`: Updates metadata for existing song files.

## Music Sourcing and Audio Quality

spotDL utilizes YouTube as its primary music source.

*   **Audio Quality:** Downloads the highest possible bitrate available on YouTube (up to 256kbps for YouTube Music Premium users).
*   See the [Audio Formats and Quality](docs/usage.md#audio-formats-and-quality) page for more details.

> **Disclaimer:**  Users are responsible for their actions and legal compliance regarding copyright.  spotDL does not condone unauthorized downloading of copyrighted material.

## Contributing

Contribute to the project! Check out the [CONTRIBUTING.md](docs/CONTRIBUTING.md) for information on how to set up a development environment.

### Code Contributors

<a href="https://github.com/spotDL/spotify-downloader/graphs/contributors">
  <img class="dark-light" src="https://contrib.rocks/image?repo=spotDL/spotify-downloader&anon=0&columns=25&max=100&r=true" />
</a>

## License

This project is licensed under the [MIT License](/LICENSE).