[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# YouTube-dl: The Ultimate Command-Line Video Downloader

Tired of buffering?  **Download videos from YouTube and thousands of other sites with youtube-dl**, a versatile command-line tool that puts you in control of your media.  

[Get Started with youtube-dl](https://github.com/ytdl-org/youtube-dl)

**Key Features:**

*   **Broad Site Support:** Works with YouTube and many other video platforms.
*   **Format Selection:** Choose your preferred video and audio formats.
*   **Playlist Downloads:** Download entire playlists with ease.
*   **Customization:** Output filenames, set download limits, and more.
*   **Cross-Platform:** Compatible with Windows, macOS, and Linux.
*   **Flexible:** Supports a wide array of options for advanced control.

## Installation

Choose your preferred method for installing `youtube-dl`:

*   **UNIX (Linux, macOS, etc.):**
    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```
    *(If `curl` is unavailable, use `wget` instead.)*

*   **Windows:** Download the [`.exe file`](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory included in your system's [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) (excluding `%SYSTEMROOT%\System32`).

*   **Using pip:**
    ```bash
    sudo -H pip install --upgrade youtube-dl
    ```
    *(Updates existing installations)*

*   **macOS (Homebrew):**
    ```bash
    brew install youtube-dl
    ```

*   **macOS (MacPorts):**
    ```bash
    sudo port install youtube-dl
    ```

*   **From Source:** See [Developer Instructions](#developer-instructions) for building from the git repository.

For detailed installation options, including PGP signatures, consult the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## Description

`youtube-dl` is a powerful, open-source command-line utility built with Python for downloading videos from a vast number of websites.  It's platform-agnostic, ensuring it works seamlessly on your preferred operating system (Unix-like, Windows, and macOS).  Released to the public domain, it's free to use, modify, and redistribute.

## Usage

To download a video, use the following basic command:

```bash
youtube-dl [OPTIONS] URL [URL...]
```

See the [OPTIONS](#options) section below for extensive configuration options.

## Options

Explore the range of `youtube-dl` options to customize your downloads.

*   **General Options:**
    *   `-h`, `--help`: Print help text.
    *   `--version`: Print program version.
    *   `-U`, `--update`: Update `youtube-dl`.
    *   `-i`, `--ignore-errors`: Continue on download errors.
    *   `--abort-on-error`: Abort further downloads on error.
    *   `--list-extractors`: List supported extractors.

*   **Network Options:**
    *   `--proxy URL`: Use a proxy server.
    *   `--socket-timeout SECONDS`: Set socket timeout.

*   **Geo Restriction Options:**
    *   `--geo-bypass`: Bypass geo-restrictions.
    *   `--geo-bypass-country CODE`: Force geo-bypass with a country code.

*   **Video Selection Options:**
    *   `--playlist-start NUMBER`: Start playlist download at a specific video.
    *   `--playlist-end NUMBER`: End playlist download at a specific video.
    *   `--match-title REGEX`: Download videos matching title regex.
    *   `--reject-title REGEX`: Skip videos matching title regex.

*   **Download Options:**
    *   `-r`, `--limit-rate RATE`: Limit download rate.
    *   `-R`, `--retries RETRIES`: Set retry attempts.
    *   `-a`, `--batch-file FILE`: Download from a list of URLs.

*   **Filesystem Options:**
    *   `-o`, `--output TEMPLATE`: Set output filename template.
    *   `-w`, `--no-overwrites`: Don't overwrite files.
    *   `-c`, `--continue`: Resume partially downloaded files.

*   **Thumbnail Options:**
    *   `--write-thumbnail`: Write thumbnail image.

*   **Verbosity / Simulation Options:**
    *   `-q`, `--quiet`: Quiet mode.
    *   `-s`, `--simulate`: Simulate download.
    *   `-v`, `--verbose`: Verbose mode.

*   **Video Format Options:**
    *   `-f`, `--format FORMAT`: Select video format (see [FORMAT SELECTION](#format-selection)).
    *   `--all-formats`: Download all available formats.

*   **Subtitle Options:**
    *   `--write-sub`: Download subtitles.
    *   `--sub-lang LANGS`: Specify subtitle languages.

*   **Authentication Options:**
    *   `-u`, `--username USERNAME`: Login with a username.
    *   `-p`, `--password PASSWORD`: Login with a password.
    *   `-n`, `--netrc`: Use .netrc authentication data.

*   **Post-processing Options:**
    *   `-x`, `--extract-audio`: Extract audio from video.
    *   `--audio-format FORMAT`: Specify audio format.

*   **Configuration:**
    *   Configuration can be set in a configuration file (e.g., `/etc/youtube-dl.conf` on Linux, or `~/.config/youtube-dl/config`). Use `--ignore-config` to disable and `--config-location` to specify a custom config file.

Refer to the detailed original documentation (linked at the top) for a complete list of options.

## Configuration

Customize `youtube-dl` behavior using a configuration file.  On Linux and macOS, this is typically found at `/etc/youtube-dl.conf` (system-wide) and `~/.config/youtube-dl/config` (user-specific). On Windows, it's `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`.

## Output Template

The `-o` option lets you create custom output filenames.  Use special sequences like `%(title)s`, `%(id)s`, and `%(ext)s` in the template. See the detailed [OUTPUT TEMPLATE](https://github.com/ytdl-org/youtube-dl#output-template) section of the original documentation for a full list and examples.

## Format Selection

Use the `-f` or `--format` options for fine-grained control over video and audio formats.

*   `-f 22`:  Download format with code 22.
*   `-f webm`:  Download the best quality format in `webm` format.
*   `-f best`:  Download the best overall quality (video + audio).
*   `-f worstvideo`: Download the lowest quality video-only format.
*   `-f 22/17/18`:  Download 22 if available, then 17, then 18.
*   `-f 22,17,18`:  Download formats 22, 17, and 18.
*   `-f "best[height=720]"`: Filter formats based on properties.
*   `-f bestvideo+bestaudio`: Merge best video and audio formats.

See the original [FORMAT SELECTION](https://github.com/ytdl-org/youtube-dl#format-selection) section for complete details.

## Video Selection

Refine your downloads by date using the `--date`, `--datebefore`, and `--dateafter` options:

```bash
# Download videos uploaded in the last 6 months
youtube-dl --dateafter now-6months
```

## FAQ

Find answers to common questions in the [FAQ](https://github.com/ytdl-org/youtube-dl#faq) section of the original documentation. Topics include:

*   How to update youtube-dl.
*   Common errors and solutions.
*   Using ffmpeg/avconv.
*   Streaming directly to a media player.
*   Downloading only new videos from a playlist.

## Developer Instructions

For instructions on contributing, see the [DEVELOPER INSTRUCTIONS](#developer-instructions) section.

## Embedding YouTube-dl

Integrate `youtube-dl` functionality into your Python scripts. See the [EMBEDDING YOUTUBE-DL](#embedding-youtube-dl) section for examples.

## Bugs

Report bugs and suggestions in the [issue tracker](https://github.com/ytdl-org/youtube-dl/issues). Include the full output of `youtube-dl -v <your command line>`.

## Copyright

youtube-dl is released into the public domain.

---