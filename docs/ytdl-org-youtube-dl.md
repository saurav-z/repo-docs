[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Download Videos from YouTube and Beyond

**Download videos from YouTube and hundreds of other sites with ease using youtube-dl!**  [Visit the official repository](https://github.com/ytdl-org/youtube-dl) to get started.

## Key Features

*   **Wide Site Support:** Download videos from YouTube, Vimeo, Dailymotion, and hundreds of other video platforms.
*   **Format Selection:** Choose from various video formats, resolutions, and qualities.
*   **Playlist Support:** Download entire playlists with a single command.
*   **Metadata Extraction:** Automatically extracts video titles, descriptions, and other metadata.
*   **Customization:** Extensive options for file naming, output formats, and more.
*   **Cross-Platform:** Works on Linux, macOS, Windows, and other operating systems.
*   **Open Source:** Released into the public domain, allowing modification and redistribution.

## Installation

### Unix (Linux, macOS, etc.)

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

1.  Download the latest [.exe file](https://yt-dl.org/latest/youtube-dl.exe).
2.  Place the `.exe` in a directory included in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29), **except** `C:\Windows\System32`.

### Other Installation Methods

*   **pip:**  `sudo -H pip install --upgrade youtube-dl` (updates if already installed)
*   **Homebrew (macOS):** `brew install youtube-dl`
*   **MacPorts (macOS):** `sudo port install youtube-dl`

**For more installation options, including PGP signatures, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).**

## Usage

To download a video, simply use the command:

```bash
youtube-dl [OPTIONS] URL [URL...]
```

Replace `[OPTIONS]` with any of the available command-line options (see below), and `URL` with the video or playlist URL you want to download.

## Command-Line Options

A vast array of options allows you to customize your downloads. Below is a summary of key options, with full details in the original [README](https://github.com/ytdl-org/youtube-dl).

*   **`-h`, `--help`**:  Print help text and exit.
*   **`-U`, `--update`**: Update youtube-dl to the latest version.  (Run with `sudo` if necessary)
*   **`-i`, `--ignore-errors`**:  Continue downloading even if errors occur.
*   **`--proxy URL`**: Use a proxy server for downloads.
*   **`--format FORMAT`, `-f FORMAT`**:  Specify video format (e.g., `-f best`, `-f 22`).
*   **`-o TEMPLATE`, `--output TEMPLATE`**:  Set the output filename template (see "OUTPUT TEMPLATE" below).
*   **`--write-sub`**: Write subtitle file.
*   **`-x`, `--extract-audio`**: Convert video files to audio-only files (requires ffmpeg/avconv).

### Network Options

*   **`--proxy URL`**:  Use the specified HTTP/HTTPS/SOCKS proxy.
*   **`--socket-timeout SECONDS`**: Set the connection timeout in seconds.
*   **`-4`, `--force-ipv4`**: Force all connections via IPv4.
*   **`-6`, `--force-ipv6`**: Force all connections via IPv6.

### Video Selection

*   **`--playlist-start NUMBER`**: Start at a specific playlist video.
*   **`--playlist-end NUMBER`**: End at a specific playlist video.
*   **`--playlist-items ITEM_SPEC`**: Download specific playlist items (e.g., `--playlist-items 1,3,5-7`).
*   **`--match-title REGEX`**: Only download videos with matching titles.
*   **`--reject-title REGEX`**: Skip downloads for videos with matching titles.
*   **`--date DATE`**: Download videos uploaded on a specific date.

### Download Options

*   **`-r RATE`, `--limit-rate RATE`**:  Limit the download rate (e.g., `50K`, `4.2M`).
*   **`-R RETRIES`, `--retries RETRIES`**:  Set the number of download retries.
*   **`-c`, `--continue`**: Resume partially downloaded files.
*   **`--external-downloader COMMAND`**: Use an external downloader (e.g., `aria2c`, `ffmpeg`).

### Filesystem Options

*   **`-a FILE`, `--batch-file FILE`**:  Download from a file containing URLs.
*   **`-o TEMPLATE`, `--output TEMPLATE`**: Customize the output filename.
*   **`-w`, `--no-overwrites`**: Don't overwrite existing files.

### Verbosity / Simulation Options

*   **`-q`, `--quiet`**:  Activate quiet mode.
*   **`-s`, `--simulate`**:  Simulate the download (don't download the video).
*   **`-v`, `--verbose`**:  Print detailed debugging information.

### Video Format Options

*   **`-f FORMAT`, `--format FORMAT`**: Select the desired video format (see "FORMAT SELECTION" below).
*   **`-F`, `--list-formats`**: List all available formats for a video.

## Output Template

The `-o` or `--output` option allows you to define custom output filenames using a template.  Use special sequences that are replaced with video metadata.  Refer to the original README for the complete list of available template keywords and examples.

### Examples:

*   `youtube-dl -o '%(title)s-%(id)s.%(ext)s' "VIDEO_URL"`
*   `youtube-dl -o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' "PLAYLIST_URL"`

## Format Selection

The `-f` or `--format` option is used to specify the desired video format.  Use format codes or file extensions, with options for prioritizing formats.

### Examples:

*   `-f best`:  Download the best available quality (video and audio in one file).
*   `-f 22`:  Download a specific format code.
*   `-f webm`: Download the best webm file.
*   `-f bestvideo+bestaudio`: Download best video and audio and merge into one file.
*   `-f "best[height<=?1080]+bestaudio/best"`: Download best quality no greater than 1080p.

## Video Selection

Use options like `--date`, `--datebefore`, and `--dateafter` to filter videos by upload date.

## Configuration

Customize youtube-dl by placing command-line options in a configuration file.  On Linux/macOS, the user-wide config file is `~/.config/youtube-dl/config`.  On Windows, it's  `%APPDATA%\youtube-dl\config.txt`.

## FAQ

For common questions and troubleshooting, please refer to the [FAQ section](https://github.com/ytdl-org/youtube-dl/blob/master/README.md#faq) in the original README. This includes how to update, how to pass cookies, and how to stream to media players.

## Bugs

Report bugs and suggest improvements in the [issue tracker](https://github.com/ytdl-org/youtube-dl/issues).  Include the full output of `youtube-dl -v YOUR_URL_HERE` for bug reports.

## Developer Instructions

If you wish to contribute, detailed developer instructions, including information about adding support for new sites, are available in the original README.

## Copyright

youtube-dl is released into the public domain by the copyright holders.