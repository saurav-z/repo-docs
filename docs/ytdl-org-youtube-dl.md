[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Download Videos from YouTube and Beyond

Tired of buffering? **youtube-dl is your go-to command-line tool for downloading videos from YouTube and thousands of other sites.**

[Visit the official repository](https://github.com/ytdl-org/youtube-dl) for the latest updates.

## Key Features

*   **Wide Site Support:** Download videos from YouTube, Vimeo, Dailymotion, and a vast array of other video platforms.
*   **Format Selection:** Choose your preferred video and audio formats, quality, and resolution.
*   **Playlist & Channel Downloads:** Easily download entire playlists or all videos from a channel.
*   **Customizable Output:** Control filenames, output directories, and metadata.
*   **Subtitle Support:** Download subtitles in various formats and languages.
*   **Authentication:** Supports login for sites requiring it.
*   **Cross-Platform Compatibility:** Works on Windows, macOS, and Linux.
*   **Actively Maintained:** Benefit from frequent updates to support new sites and features.

## Installation

Choose your preferred method for installing youtube-dl:

*   **Unix (Linux, macOS, etc.):**

    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```
    If `curl` is unavailable, use `wget` instead:
    ```bash
    sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```

*   **Windows:**
    *   Download the executable: [Download .exe File](https://yt-dl.org/latest/youtube-dl.exe)
    *   Place the `youtube-dl.exe` file in a directory included in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) environment variable (but not in `%SYSTEMROOT%\System32`).

*   **Pip:**

    ```bash
    sudo -H pip install --upgrade youtube-dl
    ```

*   **Homebrew (macOS):**

    ```bash
    brew install youtube-dl
    ```

*   **MacPorts (macOS):**

    ```bash
    sudo port install youtube-dl
    ```

*   For advanced installation options, including PGP signatures, refer to the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## Usage

Download a video:

```bash
youtube-dl "VIDEO_URL"
```

Download a playlist:

```bash
youtube-dl "PLAYLIST_URL"
```

To view available options, run `youtube-dl --help`.

## Common Options

*   `-U, --update`: Update youtube-dl to the latest version.
*   `-f, --format FORMAT`: Specify video format (e.g., `-f mp4`, `-f "bestvideo+bestaudio"`).  See [Format Selection](#format-selection) for details.
*   `-o, --output TEMPLATE`: Set output filename and directory. See [Output Template](#output-template) for details.
*   `--list-formats`: List available formats for a video.
*   `--write-sub`: Download subtitles.
*   `--sub-lang LANGS`: Specify subtitle languages.
*   `--proxy URL`: Use a proxy.
*   `--username USERNAME` and `--password PASSWORD`: Login to sites.

## Format Selection

Use the `--format` (or `-f`) option to control the downloaded video format.

*   `best`:  Download the best available quality.
*   `worst`: Download the lowest quality.
*   `22`: Download format code 22 (e.g., 720p MP4).
*   `webm`: Download the best WebM format.
*   `bestvideo+bestaudio`:  Merge best video and audio streams.
*   `bestvideo[height<=720]+bestaudio/best`: Prioritize 720p or lower resolution with the best audio quality.

See the full documentation for more [Format Selection Examples](#format-selection-examples).

## Output Template

Use the `-o` option to customize the output filename.

Examples:
```bash
# Basic
youtube-dl -o "my_video.mp4" "VIDEO_URL"

# Title and ID
youtube-dl -o '%(title)s-%(id)s.%(ext)s' "VIDEO_URL"

# Playlists
youtube-dl -o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' "PLAYLIST_URL"
```

Refer to the full documentation for the [Output Template examples](#output-template-examples) and a list of available variables.

## Configuration

Configure youtube-dl using a configuration file:

*   **Linux/macOS:** `/etc/youtube-dl.conf` (system-wide) or `~/.config/youtube-dl/config` (user-specific).
*   **Windows:** `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`.

## Advanced Features

*   [Network Options](#network-options) for proxy and IP configuration.
*   [Geo Restriction](#geo-restriction) bypass options.
*   [Video Selection](#video-selection) for filtering videos within playlists.
*   [Download Options](#download-options) for controlling download behavior.
*   [Filesystem Options](#filesystem-options) for file management.
*   [Thumbnail Options](#thumbnail-options)
*   [Verbosity / Simulation Options](#verbosity--simulation-options) for debugging.
*   [Workarounds](#workarounds) for common issues.
*   [Subtitle Options](#subtitle-options)
*   [Authentication Options](#authentication-options) for login.
*   [Adobe Pass Options](#adobe-pass-options)
*   [Post-processing Options](#post-processing-options)

## Frequently Asked Questions (FAQ)

Get answers to common questions:

*   [How do I update youtube-dl?](#how-do-i-update-youtube-dl)
*   [youtube-dl is extremely slow to start on Windows?](#youtube-dl-is-extremely-slow-to-start-on-windows)
*   [I'm getting an error `Unable to extract OpenGraph title` on YouTube playlists?](#im-getting-an-error-unable-to-extract-opengraph-title-on-youtube-playlists)
*   [I'm getting an error when trying to use output template?](#im-getting-an-error-when-trying-to-use-output-template-error-using-output-template-conflicts-with-using-title-video-id-or-auto-number)
*   [Do I always have to pass `-citw`?](#do-i-always-have-to-pass--citw)
*   [Can you please put the `-b` option back?](#can-you-please-put-the--b-option-back)
*   [I get HTTP error 402 when trying to download a video. What's this?](#i-get-http-error-402-when-trying-to-download-a-video-whats-this)
*   [Do I need any other programs?](#do-i-need-any-other-programs)
*   [I have downloaded a video but how can I play it?](#i-have-downloaded-a-video-but-how-can-i-play-it)
*   [I extracted a video URL with `-g`, but it does not play on another machine / in my web browser.](#i-extracted-a-video-url-with--g-but-it-does-not-play-on-another-machine--in-my-web-browser)
*   [ERROR: no fmt_url_map or conn information found in video info](#error-no-fmt_url_map-or-conn-information-found-in-video-info)
*   [ERROR: unable to download video](#error-unable-to-download-video)
*   [Video URL contains an ampersand and I'm getting some strange output `[1] 2839` or `'v' is not recognized as an internal or external command`](#video-url-contains-an-ampersand-and-im-getting-some-strange-output--1-2839-or-v-is-not-recognized-as-an-internal-or-external-command)
*   [ExtractorError: Could not find JS function u'OF'](#extractorerror-could-not-find-js-function-uof)
*   [HTTP Error 429: Too Many Requests or 402: Payment Required](#http-error-429-too-many-requests-or-402-payment-required)
*   [SyntaxError: Non-ASCII character](#syntaxerror-non-ascii-character)
*   [What is this binary file? Where has the code gone?](#what-is-this-binary-file-where-has-the-code-gone)
*   [The exe throws an error due to missing `MSVCR100.dll`](#the-exe-throws-an-error-due-to-missing-msvcr100dll)
*   [On Windows, how should I set up ffmpeg and youtube-dl? Where should I put the exe files?](#on-windows-how-should-i-set-up-ffmpeg-and-youtube-dl-where-should-i-put-the-exe-files)
*   [How do I put downloads into a specific folder?](#how-do-i-put-downloads-into-a-specific-folder)
*   [How do I download a video starting with a `-`?](#how-do-i-download-a-video-starting-with-a--)
*   [How do I pass cookies to youtube-dl?](#how-do-i-pass-cookies-to-youtube-dl)
*   [How do I stream directly to media player?](#how-do-i-stream-directly-to-media-player)
*   [How do I download only new videos from a playlist?](#how-do-i-download-only-new-videos-from-a-playlist)
*   [Should I add `--hls-prefer-native` into my config?](#should-i-add--hls-prefer-native-into-my-config)
*   [Can you add support for this anime video site, or site which shows current movies for free?](#can-you-add-support-for-this-anime-video-site-or-site-which-shows-current-movies-for-free)
*   [How can I speed up work on my issue?](#how-can-i-speed-up-work-on-my-issue)
*   [How can I detect whether a given URL is supported by youtube-dl?](#how-can-i-detect-whether-a-given-url-is-supported-by-youtube-dl)
## Bugs

Report bugs and suggestions in the [issue tracker](https://github.com/ytdl-org/youtube-dl/issues).  Please include the full output of `youtube-dl -v YOUR_URL_HERE` in your report.  See the [Bug reporting instructions](#bugs) for more information.

## Developer Instructions
*   [Developer Instructions](#developer-instructions)
*   [Adding support for a new site](#adding-support-for-a-new-site)
*   [youtube-dl coding conventions](#youtube-dl-coding-conventions)

## Legal

youtube-dl is released into the public domain.