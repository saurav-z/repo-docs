[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# YouTube-DL: The Ultimate Command-Line Video Downloader

**Download videos from YouTube and hundreds of other sites with ease using the versatile and open-source `youtube-dl`!**

*   **[Installation](#installation)**: Get started quickly with simple installation instructions for various operating systems.
*   **[Key Features](#key-features)**: Discover the power of `youtube-dl` with its advanced features.
*   **[Options](#options)**: Customize your downloads with a wide array of command-line options.
*   **[Configuration](#configuration)**: Configure `youtube-dl` with a configuration file.
*   **[Output Template](#output-template)**: Control output filenames with flexible templates.
*   **[Format Selection](#format-selection)**: Select the best video and audio formats for your needs.
*   **[Video Selection](#video-selection)**: Filter videos based on various criteria like date, title, and views.
*   **[FAQ](#faq)**: Find answers to common questions and troubleshooting tips.
*   **[Developer Instructions](#developer-instructions)**: Contribute to the project with instructions for developers.
*   **[Embedding youtube-dl](#embedding-youtube-dl)**: Integrate `youtube-dl` into your Python projects.
*   **[Bugs](#bugs)**: Learn how to report bugs and get help.

## Key Features

*   **Broad Site Support**: Download videos from YouTube, Vimeo, and many other popular video platforms.
*   **Format Selection**: Choose your preferred video and audio quality and format.
*   **Playlist and Channel Downloads**: Download entire playlists or all videos from a channel.
*   **Metadata Preservation**: Automatically include video metadata like title, description, and more.
*   **Download Customization**: Control output file names, and skip already downloaded files.
*   **Cross-Platform Compatibility**: Runs on Windows, macOS, Linux, and other Unix-like systems.
*   **Open Source and Customizable**: Modify, redistribute, and use the software as you wish.
*   **Extensible**: Developers can add support for new sites through extraction plugins.

## Installation

### Unix (Linux, macOS, etc.)

Install `youtube-dl` directly using `curl`:

    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl

If you don't have `curl`, use `wget`:

    sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl

### Windows

Download the executable: [youtube-dl.exe](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory on your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29), except `C:\Windows\System32`.

### Python Package Manager

Install or update using pip:

    sudo -H pip install --upgrade youtube-dl

### macOS (Homebrew)

Install with Homebrew:

    brew install youtube-dl

### macOS (MacPorts)

Install with MacPorts:

    sudo port install youtube-dl

For alternative options, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html), or refer to the [developer instructions](#developer-instructions) to work with the git repository.

## Options

Use `youtube-dl [OPTIONS] URL [URL...]` to download videos.  Below are some of the available options, though many more exist.

*   `-U, --update`: Update youtube-dl to the latest version.
*   `-i, --ignore-errors`: Continue downloading even if errors occur.
*   `--proxy URL`: Use a proxy for downloads.
*   `-o, --output TEMPLATE`: Set the output filename template.
*   `-f, --format FORMAT`: Select the video format (see [Format Selection](#format-selection)).
*   `--write-sub`, `--write-auto-sub`: Download subtitles.
*   `-x, --extract-audio`: Extract audio from video.
*   `-u, --username USERNAME -p, --password PASSWORD`: Provide login credentials.

See [OPTIONS](#options) in the original README for a complete list.

## Configuration

Configure `youtube-dl` by creating a configuration file:

*   **Linux/macOS**: `/etc/youtube-dl.conf` (system-wide) or `~/.config/youtube-dl/config` (user-specific).
*   **Windows**: `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`.

Use the same command-line options within the configuration file, one per line.  Example:

```
# Always extract audio
-x

# Do not copy the mtime
--no-mtime

# Use this proxy
--proxy 127.0.0.1:3128

# Save all videos under Movies directory in your home directory
-o ~/Movies/%(title)s.%(ext)s
```

Use `--ignore-config` to disable or `--config-location` to specify a custom configuration file.  See the original [configuration section](https://github.com/ytdl-org/youtube-dl#configuration) for more details.

## Output Template

Customize the output filename with the `-o` option and template sequences. The basic usage is not to set any template arguments when downloading a single file, like in `youtube-dl -o funny_video.flv "https://some/video"`. These sequences are supported (see [Output Template](https://github.com/ytdl-org/youtube-dl#output-template) for a comprehensive list):

*   `id`: Video ID
*   `title`: Video title
*   `ext`: File extension
*   `playlist`: Playlist name
*   `playlist_index`: Index in playlist (padded)

Example: `-o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s'`

## Format Selection

Use `-f` or `--format` to select video formats.  Get a list with `-F` or `--list-formats`.

*   **Specific Format**:  `-f 22` (format code 22).
*   **File Extension**: `-f mp4` (best mp4).
*   **Predefined Choices**:  `best`, `worst`, `bestvideo`, `bestaudio`.
*   **Preferences**: `-f 22/17/18` (choose in order).
*   **Multiple Formats**: `-f 22,17,18` (download all).
*   **Filtering**: `-f "best[height=720]"` (videos with height 720 or videos with no height information) or  `-f "[filesize>10M]"`.
*   **Merging (Requires ffmpeg/avconv)**: `-f bestvideo+bestaudio`.

See [Format Selection](https://github.com/ytdl-org/youtube-dl#format-selection) in the original README for more.

## Video Selection

Use options to filter videos by various criteria:

*   `--date YYYYMMDD`: Download videos uploaded on a specific date.
*   `--datebefore DATE`: Download videos uploaded on or before a date.
*   `--dateafter DATE`: Download videos uploaded on or after a date.
*   `--match-title REGEX`: Download videos with matching titles.
*   `--reject-title REGEX`: Skip videos with matching titles.
*   `--min-views COUNT`, `--max-views COUNT`: Filter by video view count.

See [Video Selection](https://github.com/ytdl-org/youtube-dl#video-selection) in the original README for details.

## FAQ

See the [FAQ](#faq) section for answers to common questions like updating youtube-dl, common error messages, and more.

## Developer Instructions

See the [Developer Instructions](#developer-instructions) section for information on contributing to `youtube-dl`.

## Embedding youtube-dl

Integrate `youtube-dl` into your Python projects.

```python
from __future__ import unicode_literals
import youtube_dl

ydl_opts = {}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=BaW_jenozKc'])
```

See [Embedding youtube-dl](#embedding-youtube-dl) for more examples and the available options.

## Bugs

Report bugs and suggestions through the [issue tracker](https://github.com/ytdl-org/youtube-dl/issues).  Be sure to include the full output with `-v` (verbose mode).

**[Visit the YouTube-DL repository for more details and to get involved!](https://github.com/ytdl-org/youtube-dl)**