[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Your Go-To Tool for Downloading Videos from the Web

**Download videos from YouTube and hundreds of other sites with ease using the versatile command-line tool, youtube-dl, empowering you to save your favorite content locally for offline enjoyment.**

*   [Installation](#installation)
*   [Key Features](#key-features)
*   [Options](#options)
*   [Configuration](#configuration)
*   [Output Template](#output-template)
*   [Format Selection](#format-selection)
*   [Video Selection](#video-selection)
*   [FAQ](#faq)
*   [Developer Instructions](#developer-instructions)
*   [Embedding youtube-dl](#embedding-youtube-dl)
*   [Bugs](#bugs)
*   [Copyright](#copyright)

## Installation

Get started with youtube-dl quickly and easily on various operating systems.  Visit the [youtube-dl download page](https://ytdl-org.github.io/youtube-dl/download.html) for detailed installation instructions, including PGP signatures.

### UNIX (Linux, macOS, etc.)

```bash
sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

or

```bash
sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

### Windows

Download the .exe file from [here](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory within your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29), *except* for `%SYSTEMROOT%\System32`.

### Using pip

```bash
sudo -H pip install --upgrade youtube-dl
```

### macOS (Homebrew or MacPorts)

```bash
brew install youtube-dl
```

or

```bash
sudo port install youtube-dl
```

## Key Features

*   **Broad Site Support:** Download videos from YouTube, plus hundreds of other video platforms ([see supported sites](https://ytdl-org.github.io/youtube-dl/supportedsites.html)).
*   **Flexible Format Selection:** Choose your preferred video and audio formats or let youtube-dl select the best available.
*   **Playlist and Channel Downloads:** Download entire playlists and channels with a single command.
*   **Customizable Output:** Control filenames and directory structures using output templates.
*   **Subtitle Support:** Download and embed subtitles.
*   **Cross-Platform:** Works on Windows, macOS, and Linux.
*   **Open Source:** Free to use, modify, and redistribute under the public domain.
*   **Command-Line Interface:** Easy to use from the command line, making it ideal for automation.

## Options

youtube-dl offers a vast array of options to customize your downloads. Some of the most used include:

*   `-U, --update`: Update youtube-dl to the latest version.
*   `-o, --output TEMPLATE`:  Specify the output filename template.
*   `-f, --format FORMAT`: Select video format.
*   `-x, --extract-audio`: Convert video files to audio-only files.
*   `--proxy URL`: Use a proxy for downloads.
*   `-i, --ignore-errors`: Continue on download errors.
*   See the full [OPTIONS](#options) section below for a complete list.

The full list of options is available below.

## Configuration

Customize youtube-dl's behavior using configuration files.  Create configuration files in:

*   `/etc/youtube-dl.conf` (system-wide on Linux/macOS)
*   `~/.config/youtube-dl/config` (user-specific on Linux/macOS)
*   `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf` (user-specific on Windows).

Examples:

```bash
# Comment
-x
--no-mtime
--proxy 127.0.0.1:3128
-o ~/Movies/%(title)s.%(ext)s
```

Use `--ignore-config` or `--config-location` for specific configurations.

## Output Template

The `-o` option allows you to create custom output filenames. Use special sequences that will be replaced when downloading each video. The special sequences may be formatted according to [python string formatting operations](https://docs.python.org/2/library/stdtypes.html#string-formatting).

```bash
youtube-dl -o '%(title)s-%(id)s.%(ext)s' [URL]
```

### Output Template Examples

```bash
# Simple file name
$ youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc

# Restrict filenames
$ youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc --restrict-filenames

# Download YouTube playlist videos in separate directory indexed by video order in a playlist
$ youtube-dl -o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' https://www.youtube.com/playlist?list=PLwiyx1dc3P2JR9N8gQaQN_BCvlSlap7re

# Download all playlists of YouTube channel/user keeping each playlist in separate directory:
$ youtube-dl -o '%(uploader)s/%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' https://www.youtube.com/user/TheLinuxFoundation/playlists

# Download Udemy course keeping each chapter in separate directory under MyVideos directory in your home
$ youtube-dl -u user -p password -o '~/MyVideos/%(playlist)s/%(chapter_number)s - %(chapter)s/%(title)s.%(ext)s' https://www.udemy.com/java-tutorial/

# Download entire series season keeping each series and each season in separate directory under C:/MyVideos
$ youtube-dl -o "C:/MyVideos/%(series)s/%(season_number)s - %(season)s/%(episode_number)s - %(episode)s.%(ext)s" https://videomore.ru/kino_v_detalayah/5_sezon/367617

# Stream the video being downloaded to stdout
$ youtube-dl -o - BaW_jenozKc
```

## Format Selection

Use `--format FORMAT` (or `-f FORMAT`) to specify the video and audio formats you want to download.

### Format Selection Examples

```bash
# Download best mp4 format available or any other best if no mp4 available
$ youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'

# Download best format available but no better than 480p
$ youtube-dl -f 'bestvideo[height<=480]+bestaudio/best[height<=480]'

# Download best video only format but no bigger than 50 MB
$ youtube-dl -f 'best[filesize<50M]'

# Download best format available via direct link over HTTP/HTTPS protocol
$ youtube-dl -f '(bestvideo+bestaudio/best)[protocol^=http]'

# Download the best video format and the best audio format without merging them
$ youtube-dl -f 'bestvideo,bestaudio' -o '%(title)s.f%(format_id)s.%(ext)s'
```
Note that in the last example, an output template is recommended as bestvideo and bestaudio may have the same file name.

## Video Selection

Filter videos based on various criteria.

```bash
# Download only the videos uploaded in the last 6 months
$ youtube-dl --dateafter now-6months

# Download only the videos uploaded on January 1, 1970
$ youtube-dl --date 19700101

# Download only the videos uploaded in the 200x decade
$ youtube-dl --dateafter 20000101 --datebefore 20091231
```

## FAQ

Find answers to common questions:

*   [How do I update youtube-dl?](#how-do-i-update-youtube-dl)
*   [youtube-dl is extremely slow to start on Windows](#youtube-dl-is-extremely-slow-to-start-on-windows)
*   [I'm getting an error `Unable to extract OpenGraph title` on YouTube playlists](#im-getting-an-error-unable-to-extract-opengraph-title-on-youtube-playlists)
*   [I'm getting an error when trying to use output template: `error: using output template conflicts with using title, video ID or auto number`](#im-getting-an-error-when-trying-to-use-output-template-error-using-output-template-conflicts-with-using-title-video-id-or-auto-number)
*   [Do I always have to pass `-citw`?](#do-i-always-have-to-pass--citw)
*   [Can you please put the `-b` option back?](#can-you-please-put-the--b-option-back)
*   [I get HTTP error 402 when trying to download a video. What's this?](#i-get-http-error-402-when-trying-to-download-a-video-whats-this)
*   [Do I need any other programs?](#do-i-need-any-other-programs)
*   [I have downloaded a video but how can I play it?](#i-have-downloaded-a-video-but-how-can-i-play-it)
*   [I extracted a video URL with `-g`, but it does not play on another machine / in my web browser.](#i-extracted-a-video-url-with--g-but-it-does-not-play-on-another-machine--in-my-web-browser)
*   [ERROR: no fmt_url_map or conn information found in video info](#error-no-fmt_url_map-or-conn-information-found-in-video-info)
*   [ERROR: unable to download video](#error-unable-to-download-video)
*   [Video URL contains an ampersand and I'm getting some strange output `[1] 2839` or `'v' is not recognized as an internal or external command`](#video-url-contains-an-ampersand-and-im-getting-some-strange-output-1-2839-or-v-is-not-recognized-as-an-internal-or-external-command)
*   [ExtractorError: Could not find JS function u'OF'](#extractorerror-could-not-find-js-function-uof)
*   [HTTP Error 429: Too Many Requests or 402: Payment Required](#http-error-429-too-many-requests-or-402-payment-required)
*   [SyntaxError: Non-ASCII character](#syntaxerror-non-ascii-character)
*   [What is this binary file? Where has the code gone?](#what-is-this-binary-file-where-has-the-code-gone)
*   [The exe throws an error due to missing `MSVCR100.dll`](#the-exe-throws-an-error-due-to-missing-msvcr100dll)
*   [On Windows, how should I set up ffmpeg and youtube-dl? Where should I put the exe files?](#on-windows-how-should-i-set-up-ffmpeg-and-youtube-dl-where-should-i-put-the-exe-files)
*   [How do I put downloads into a specific folder?](#how-do-i-put-downloads-into-a-specific-folder)
*   [How do I download a video starting with a `-`?](#how-do-i-download-a-video-starting-with-a-)
*   [How do I pass cookies to youtube-dl?](#how-do-i-pass-cookies-to-youtube-dl)
*   [How do I stream directly to media player?](#how-do-i-stream-directly-to-media-player)
*   [How do I download only new videos from a playlist?](#how-do-i-download-only-new-videos-from-a-playlist)
*   [Should I add `--hls-prefer-native` into my config?](#should-i-add--hls-prefer-native-into-my-config)
*   [Can you add support for this anime video site, or site which shows current movies for free?](#can-you-add-support-for-this-anime-video-site-or-site-which-shows-current-movies-for-free)
*   [How can I speed up work on my issue?](#how-can-i-speed-up-work-on-my-issue)
*   [How can I detect whether a given URL is supported by youtube-dl?](#how-can-i-detect-whether-a-given-url-is-supported-by-youtube-dl)
*   [Why do I need to go through that much red tape when filing bugs?](#why-do-i-need-to-go-through-that-much-red-tape-when-filing-bugs)

## Developer Instructions

For information on contributing and building youtube-dl, see the [Developer Instructions](#developer-instructions) section.

### Adding support for a new site

Please follow the [Adding support for a new site](#adding-support-for-a-new-site) section.

## Embedding youtube-dl

Learn how to embed youtube-dl in your Python projects in the [Embedding youtube-dl](#embedding-youtube-dl) section.

## Bugs

Report bugs and suggestions [here](https://github.com/ytdl-org/youtube-dl/issues).  Include the full output of `youtube-dl -v [your command]` in your report. Read more at the [bugs](#bugs) section.

## Copyright

youtube-dl is released into the public domain. This README file is likewise released into the public domain.