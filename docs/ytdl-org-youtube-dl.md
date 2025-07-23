[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: The Ultimate Command-Line Video Downloader

**Download videos from YouTube and hundreds of other sites with ease using the powerful and versatile youtube-dl!**

[Find the original project on GitHub](https://github.com/ytdl-org/youtube-dl)

**Key Features:**

*   **Wide Site Support:** Download from YouTube, plus hundreds of other video and audio platforms.
*   **Format Selection:** Choose your preferred video and audio formats, or let youtube-dl select the best available.
*   **Playlist & Channel Downloads:** Easily download entire playlists or channels.
*   **Metadata Extraction:** Retrieve video titles, descriptions, thumbnails, and more.
*   **Customizable Output:** Control filenames, directory structure, and output templates.
*   **Download Resuming:** Resume interrupted downloads.
*   **Subtitle Support:** Download and convert subtitles.
*   **Authentication:** Supports login for authenticated content.
*   **Post-processing:** Convert videos to audio, embed subtitles, and add metadata.

## Installation

Choose your operating system for installation instructions:

### UNIX-like systems (Linux, macOS, etc.)

1.  **Using `curl`:**

    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```

2.  **Using `wget` (if `curl` is unavailable):**

    ```bash
    sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```

### Windows

1.  **Download the executable:** [Download youtube-dl.exe](https://yt-dl.org/latest/youtube-dl.exe)
2.  **Place it in your PATH:** Put `youtube-dl.exe` in a directory included in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29), *except* `C:\Windows\System32`.  A common choice is `C:\Users\<YourUsername>\bin` or similar.
3.  **Alternative: Using `pip`:**

    ```bash
    sudo -H pip install --upgrade youtube-dl
    ```

    (This will update youtube-dl if already installed) See the [pypi page](https://pypi.python.org/pypi/youtube_dl) for more information.
4.  **Alternative: Homebrew (macOS):**

    ```bash
    brew install youtube-dl
    ```

5.  **Alternative: MacPorts (macOS):**

    ```bash
    sudo port install youtube-dl
    ```

For additional installation options, including PGP signatures, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html). Refer to the [developer instructions](#developer-instructions) if you wish to work directly with the Git repository.

## Description

youtube-dl is a command-line program designed to download videos from YouTube.com and a vast array of other websites. Written in Python, it's cross-platform compatible, working seamlessly on Unix-based systems, Windows, and macOS. It is released into the public domain.  This gives you the freedom to modify, redistribute, and use the software as you see fit.

### Usage

```bash
youtube-dl [OPTIONS] URL [URL...]
```

## Key Options

For detailed option descriptions, refer to the complete [OPTIONS](#options) section in the original README. Here's a quick overview:

*   `-h, --help`: Display help information.
*   `-U, --update`: Update youtube-dl to the latest version.
*   `-o, --output TEMPLATE`: Specify output filename.
*   `-f, --format FORMAT`: Select video format.
*   `-x, --extract-audio`: Extract audio from video.
*   `--list-formats`: List available formats for a video.

## Configuration

Customize youtube-dl's behavior using a configuration file.  On Linux/macOS, the system-wide config is at `/etc/youtube-dl.conf`, and user-specific settings are in `~/.config/youtube-dl/config`. On Windows, user-specific settings are located in `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`. Create the file yourself if it doesn't exist.  You can disable the config file with `--ignore-config`.

*   **Example Configuration:**

    ```
    # Lines starting with # are comments

    # Always extract audio
    -x

    # Do not copy the mtime
    --no-mtime

    # Use this proxy
    --proxy 127.0.0.1:3128

    # Save all videos under Movies directory in your home directory
    -o ~/Movies/%(title)s.%(ext)s
    ```

## Output Template

Control how your downloaded files are named and organized with output templates. The `-o` option allows you to use special sequences that are replaced during the download process.

*   **Example:**
    ```bash
    youtube-dl -o '%(title)s-%(id)s.%(ext)s' https://www.youtube.com/watch?v=BaW_jenozKc
    ```

    This will download a video with the title "youtube-dl test video" and ID "BaW_jenozKcj" and create a file like `youtube-dl test video-BaW_jenozKcj.mp4` in the current directory.

For more details, explore the full [OUTPUT TEMPLATE](#output-template) section.

## Format Selection

Choose the perfect video format for your needs.  Use the `--format` (or `-f`) option with a format selector.

*   **Example:**
    ```bash
    youtube-dl -f 22 https://www.youtube.com/watch?v=BaW_jenozKc
    ```
    This downloads the video using the format code 22.

*   **Examples:**
    ```bash
    # Download the best available mp4 format:
    youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'

    # Download best format available but no better than 480p
    youtube-dl -f 'bestvideo[height<=480]+bestaudio/best[height<=480]'
    ```

*   Refer to the [FORMAT SELECTION](#format-selection) section for complete details on format selection.

## Video Selection

Filter your downloads using various criteria:

*   `--date DATE`: Download videos uploaded on a specific date (YYYYMMDD).
*   `--dateafter DATE`: Download videos uploaded on or after a date (YYYYMMDD or relative, e.g., `now-6months`).
*   `--datebefore DATE`: Download videos uploaded on or before a date (YYYYMMDD or relative).

*   **Example:**
    ```bash
    youtube-dl --dateafter now-6months https://www.youtube.com/playlist?list=PLwiyx1dc3P2JR9N8gQaQN_BCvlSlap7re
    ```
    Downloads videos from the playlist that were uploaded in the last 6 months.

Consult the [VIDEO SELECTION](#video-selection) section in the original README for more filtering options.

## FAQ

Find answers to common questions, including updating, error messages, and other usage scenarios, in the [FAQ](#faq) section of the original README.

## Developer Instructions

For those looking to contribute, the [DEVELOPER INSTRUCTIONS](#developer-instructions) section of the original README provides details on how to set up and contribute to the project.

## Bugs

Report any bugs or suggestions via the issue tracker:  <https://github.com/ytdl-org/youtube-dl/issues>

Refer to the [BUGS](#bugs) section in the original README for instructions on reporting bugs and providing the necessary information to resolve them.

## Copyright

This project is released into the public domain.