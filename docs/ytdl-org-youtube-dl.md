[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Download Videos from YouTube and Beyond

Tired of buffering? **youtube-dl is your command-line friend for downloading videos from YouTube.com and thousands of other sites!**

## Key Features

*   **Broad Site Support:** Download from YouTube, Vimeo, and over a thousand more video and audio sites.
*   **Format Selection:** Choose the best video and audio quality, or download multiple formats.
*   **Playlist & Channel Downloads:** Effortlessly download entire playlists, channels, or selections.
*   **Customization:** Fine-tune your downloads with options for output format, file naming, and more.
*   **Cross-Platform:** Works on Windows, macOS, Linux, and other Unix-like systems.
*   **Easy to Update:** Keep your tool up-to-date with a simple command.

## Installation

Choose your platform:

*   **Unix (Linux, macOS, etc.):**
    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```
    (If you don't have curl, use `wget` instead)
*   **Windows:** Download the [executable](https://yt-dl.org/latest/youtube-dl.exe) and place it in a folder in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) (but *not* `C:\Windows\System32`).
*   **Package Managers:** Use `pip install --upgrade youtube-dl`, `brew install youtube-dl` (macOS with Homebrew), or your distribution's package manager.

For advanced installation options like PGP signatures, visit the [Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## Core Functionality

**youtube-dl** is a powerful command-line tool for downloading videos. Its basic usage is simple:

```bash
youtube-dl [OPTIONS] URL [URL...]
```

Replace `URL` with the video's web address (e.g., `https://www.youtube.com/watch?v=dQw4w9WgXcQ`).  You can specify multiple URLs to download multiple videos.  Use the following `OPTIONS` to customize your downloads:

### Core Options

*   `-U`, `--update`: Updates youtube-dl to the latest version.
*   `-h`, `--help`:  Displays help information and exits.
*   `-v`, `--verbose`:  Prints detailed debugging information.
*   `--version`: Prints the program version and exits.

### Network Options

*   `--proxy URL`: Use a proxy server for the download.
*   `--socket-timeout SECONDS`: Set the timeout for network operations.
*   `-4, --force-ipv4`: Force IPv4 connections.
*   `-6, --force-ipv6`: Force IPv6 connections.

### Geo Restriction

*   `--geo-bypass`: Bypass geo-restrictions.
*   `--geo-bypass-country CODE`: Force bypass with a 2-letter ISO code.

### Video Selection

*   `--playlist-start NUMBER`: Start downloading a playlist from a specific video number.
*   `--playlist-end NUMBER`: End the playlist download at a specific video number.
*   `--playlist-items ITEM_SPEC`: Download specific items from a playlist.
*   `--match-title REGEX`: Download only videos with titles matching a regular expression.
*   `--reject-title REGEX`: Skip download for matching titles.
*   `--max-downloads NUMBER`: Limit the number of videos to download.
*   `--min-views COUNT`: Filter videos by minimum views.
*   `--max-views COUNT`: Filter videos by maximum views.
*   `--date DATE`: Download videos uploaded on a specific date.
*   `--datebefore DATE`: Download videos uploaded before a specific date.
*   `--dateafter DATE`: Download videos uploaded after a specific date.

### Download Options

*   `-r, --limit-rate RATE`: Set a download speed limit.
*   `-R, --retries RETRIES`: Set the number of download retries.
*   `-c, --continue`: Resume interrupted downloads.
*   `-w, --no-overwrites`: Prevent overwriting existing files.

### Filesystem Options

*   `-a, --batch-file FILE`: Download from a file containing URLs.
*   `-o, --output TEMPLATE`: Set the output filename template. See the [Output Template](#output-template) section for details.
*   `--restrict-filenames`: Restrict filenames to ASCII characters and avoid spaces.
*   `--write-description`: Write video description to a `.description` file.
*   `--write-info-json`: Write video metadata to a `.info.json` file.

### Thumbnail Options

*   `--write-thumbnail`: Write the thumbnail image to disk.
*   `--list-thumbnails`: List available thumbnail formats.

### Verbosity / Simulation Options

*   `-q, --quiet`:  Suppress most output.
*   `-s, --simulate`: Simulate the download; do not write to disk.
*   `-g, --get-url`:  Print the direct URL of the video.

### Workarounds

*   `--user-agent UA`: Set a custom user agent.
*   `--cookies FILE`: Load cookies from a file.

### Video Format Options

*   `-f, --format FORMAT`: Select video format. See the [Format Selection](#format-selection) section for details.
*   `-F, --list-formats`:  List available formats for a video.
*   `--all-formats`: Download all available formats.

### Subtitle Options

*   `--write-sub`: Write subtitles to a file.
*   `--sub-lang LANGS`: Download subtitles in specific languages.

### Authentication Options

*   `-u, --username USERNAME`: Log in with a username.
*   `-p, --password PASSWORD`: Log in with a password.
*   `-n, --netrc`: Use .netrc file for authentication.

### Adobe Pass Options

*   `--ap-mso MSO`: Adobe Pass multiple-system operator (TV provider) identifier.

### Post-processing Options

*   `-x, --extract-audio`: Extract audio from the video.
*   `--audio-format FORMAT`:  Specify the audio format.
*   `--audio-quality QUALITY`:  Specify the audio quality for audio extraction.

## Output Template

The `-o` option controls the output filename.  It uses placeholders:

*   `%(id)s`: Video ID
*   `%(title)s`: Video Title
*   `%(ext)s`: File Extension
*   ... (and many more; see the full [README](https://github.com/ytdl-org/youtube-dl) for a complete list)

**Examples:**

*   `-o "%(title)s-%(id)s.%(ext)s"`:  Creates a filename like `My Video-12345.mp4`.
*   `-o "Playlist Name/%(playlist_index)s - %(title)s.%(ext)s"`:  Organizes downloads into playlist-specific folders.

## Format Selection

Use `-f` (or `--format`) to select video formats:

*   `-f best`:  Download the best available quality (usually defaults to this).
*   `-f 22`: Download a specific format code (use `-F` to list codes).
*   `-f mp4`: Download the best MP4 format.
*   `-f bestvideo+bestaudio`: Download best video and audio separately, and merge them.
*   `-f "best[height<=720]"`: Select a format with a height no more than 720.
*   `-f "best[ext=mp4]"`: Select a format with .mp4 extension.

## Video Selection

You can download videos based on date range:

```bash
youtube-dl --dateafter "20230101" --datebefore "20230630" "URL"
```

## Configuration

Customize youtube-dl's behavior by putting options in a configuration file.  On Linux/macOS, this is `~/.config/youtube-dl/config`. On Windows, this is `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`.

**Example config file:**

```
# Always extract audio
-x
# Use this proxy
--proxy 127.0.0.1:3128
# Save videos to a specific folder
-o ~/Videos/%(title)s.%(ext)s
```

## FAQ

Common questions and answers can be found in the [full README](https://github.com/ytdl-org/youtube-dl#faq).

## Bugs & Support

Report issues on the [GitHub issue tracker](https://github.com/ytdl-org/youtube-dl/issues).  Include the full output of `youtube-dl -v [your command]` for detailed debugging.

## Development

See the [full README](https://github.com/ytdl-org/youtube-dl#developer-instructions) for instructions on contributing.

## Copyright

Released into the public domain. See the [full README](https://github.com/ytdl-org/youtube-dl#copyright) for details.