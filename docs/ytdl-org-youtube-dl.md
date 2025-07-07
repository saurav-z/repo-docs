[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Download Videos from YouTube and Beyond

**Need to save videos from the internet?** [youtube-dl](https://github.com/ytdl-org/youtube-dl) is a powerful, cross-platform command-line tool that lets you download videos from YouTube and thousands of other websites!

## Key Features

*   **Broad Site Support:** Works with YouTube, Vimeo, Dailymotion, and **thousands** of other sites.
*   **Format Selection:** Choose the video quality, resolution, and format that suits your needs.
*   **Playlist and Channel Downloads:** Download entire playlists or all videos from a channel with ease.
*   **Download Customization:**  Control download speed, number of retries, and more.
*   **Metadata and Subtitles:**  Automatically download video metadata, subtitles, and even thumbnails.
*   **Cross-Platform:**  Works seamlessly on Windows, macOS, and Linux.
*   **Flexible Output:** Customize filenames with output templates.
*   **Post-Processing:** Convert videos to audio (MP3, etc.) and embed subtitles.
*   **Open Source & Free:** Released into the public domain, so you can modify it, redistribute it, and use it however you like.

## Table of Contents

*   [Installation](#installation)
*   [Description](#description)
*   [Options](#options)
    *   [Network Options](#network-options)
    *   [Geo Restriction](#geo-restriction)
    *   [Video Selection](#video-selection)
    *   [Download Options](#download-options)
    *   [Filesystem Options](#filesystem-options)
    *   [Thumbnail Options](#thumbnail-options)
    *   [Verbosity / Simulation Options](#verbosity--simulation-options)
    *   [Workarounds](#workarounds)
    *   [Video Format Options](#video-format-options)
    *   [Subtitle Options](#subtitle-options)
    *   [Authentication Options](#authentication-options)
    *   [Adobe Pass Options](#adobe-pass-options)
    *   [Post-processing Options](#post-processing-options)
*   [Configuration](#configuration)
    *   [Authentication with .netrc file](#authentication-with-.netrc-file)
*   [Output Template](#output-template)
    *   [Output Template Examples](#output-template-examples)
*   [Format Selection](#format-selection)
    *   [Format Selection Examples](#format-selection-examples)
*   [Video Selection](#video-selection)
*   [FAQ](#faq)
*   [Developer Instructions](#developer-instructions)
    *   [Adding Support for a New Site](#adding-support-for-a-new-site)
    *   [youtube-dl coding conventions](#youtube-dl-coding-conventions)
*   [Embedding youtube-dl](#embedding-youtube-dl)
*   [Bugs](#bugs)
*   [Copyright](#copyright)

## Installation

### Unix (Linux, macOS, etc.)

Install for all users (requires `sudo`):

```bash
sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

If you don't have `curl`, use `wget`:

```bash
sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

### Windows

Download the [latest .exe file](https://yt-dl.org/latest/youtube-dl.exe) and place it in a folder within your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) (e.g., `C:\Windows`).  **Do not** put it in `%SYSTEMROOT%\System32`.

###  Using `pip`

```bash
sudo -H pip install --upgrade youtube-dl
```

This command updates `youtube-dl` if it is already installed. For more information, see the [pypi page](https://pypi.python.org/pypi/youtube_dl).

### macOS (Homebrew)

```bash
brew install youtube-dl
```

### macOS (MacPorts)

```bash
sudo port install youtube-dl
```

For advanced installation options, including PGP signatures, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).  Alternatively, consult the [developer instructions](#developer-instructions) if you'd like to work with the git repository.

## Description

**youtube-dl** is a command-line program designed to download videos from YouTube.com and many other websites. It leverages the Python interpreter (version 2.6, 2.7, or 3.2+) and is platform-independent, working on Unix, Windows, and macOS. It is released into the public domain, allowing you to modify and redistribute it freely.

Basic usage:

```bash
youtube-dl [OPTIONS] URL [URL...]
```

## Options

Use `youtube-dl -h` for a full list of options. Below are the main categories.

### Network Options

*   `--proxy URL`: Use the specified HTTP/HTTPS/SOCKS proxy.
*   `--socket-timeout SECONDS`: Timeout in seconds.
*   `--source-address IP`: Client-side IP address to bind to.
*   `-4, --force-ipv4`: Force IPv4.
*   `-6, --force-ipv6`: Force IPv6.

### Geo Restriction

*   `--geo-verification-proxy URL`: Use this proxy to verify the IP address for some geo-restricted sites.
*   `--geo-bypass`: Bypass geographic restriction via faking X-Forwarded-For HTTP header
*   `--no-geo-bypass`: Do not bypass geographic restriction via faking X-Forwarded-For HTTP header
*   `--geo-bypass-country CODE`: Force bypass geographic restriction with explicitly provided two-letter ISO 3166-2 country code
*   `--geo-bypass-ip-block IP_BLOCK`: Force bypass geographic restriction with explicitly provided IP block in CIDR notation

### Video Selection

*   `--playlist-start NUMBER`: Playlist video start (default: 1).
*   `--playlist-end NUMBER`: Playlist video end (default: last).
*   `--playlist-items ITEM_SPEC`: Select specific playlist items (e.g., `--playlist-items 1,3,5-7`).
*   `--match-title REGEX`: Download only matching titles.
*   `--reject-title REGEX`: Skip download for matching titles.
*   `--max-downloads NUMBER`: Abort after downloading this many files.
*   `--min-filesize SIZE`:  Don't download videos smaller than SIZE.
*   `--max-filesize SIZE`: Don't download videos larger than SIZE.
*   `--date DATE`: Download only videos uploaded on this date.
*   `--datebefore DATE`: Download only videos uploaded on or before this date.
*   `--dateafter DATE`: Download only videos uploaded on or after this date.
*   `--min-views COUNT`: Don't download videos with fewer than COUNT views.
*   `--max-views COUNT`: Don't download videos with more than COUNT views.
*   `--match-filter FILTER`: Generic video filter (see [OUTPUT TEMPLATE](#output-template) for available keys)
*   `--no-playlist`: Download only the video if the URL refers to both a video and playlist.
*   `--yes-playlist`: Download the playlist, if the URL refers to both a video and a playlist.
*   `--age-limit YEARS`: Download only videos suitable for the given age
*   `--download-archive FILE`: Download only videos not listed in the archive file.
*   `--include-ads`: Download advertisements as well (experimental)

### Download Options

*   `-r, --limit-rate RATE`: Max download rate (e.g., `50K` or `4.2M`).
*   `-R, --retries RETRIES`: Number of retries (default: 10), or "infinite".
*   `--fragment-retries RETRIES`: Number of retries for a fragment (default: 10), or "infinite" (DASH, hlsnative and ISM)
*   `--skip-unavailable-fragments`: Skip unavailable fragments (DASH, hlsnative and ISM)
*   `--abort-on-unavailable-fragment`: Abort downloading when some fragment is not available
*   `--keep-fragments`: Keep downloaded fragments on disk after downloading is finished; fragments are erased by default
*   `--buffer-size SIZE`: Download buffer size (default: 1024).
*   `--no-resize-buffer`: Do not automatically adjust the buffer size.
*   `--http-chunk-size SIZE`: Chunk size for HTTP downloading (e.g. 10M).
*   `--playlist-reverse`: Download playlist videos in reverse order.
*   `--playlist-random`: Download playlist videos in random order.
*   `--xattr-set-filesize`: Set file xattribute `ytdl.filesize` with expected file size
*   `--hls-prefer-native`: Use the native HLS downloader instead of ffmpeg
*   `--hls-prefer-ffmpeg`: Use ffmpeg instead of the native HLS downloader
*   `--hls-use-mpegts`: Use the mpegts container for HLS videos, allowing to play the video while downloading (some players may not be able to play it)
*   `--external-downloader COMMAND`: Use an external downloader (aria2c, avconv, axel, cURL, ffmpeg, httpie, wget).
*   `--external-downloader-args ARGS`: Give these arguments to the external downloader

### Filesystem Options

*   `-a, --batch-file FILE`: Download URLs from a file (one URL per line).
*   `--id`: Use only video ID in file name.
*   `-o, --output TEMPLATE`: Output filename template (see [OUTPUT TEMPLATE](#output-template)).
*   `--output-na-placeholder PLACEHOLDER`: Placeholder value for unavailable meta fields in output filename template
*   `--autonumber-start NUMBER`: Start value for `%(autonumber)s` (default: 1).
*   `--restrict-filenames`: Restrict filenames to ASCII and avoid special characters.
*   `-w, --no-overwrites`: Don't overwrite files.
*   `-c, --continue`: Resume partially downloaded files.
*   `--no-continue`: Don't resume.
*   `--no-part`: Don't use `.part` files.
*   `--no-mtime`: Don't set the file modification time.
*   `--write-description`: Write video description to a `.description` file.
*   `--write-info-json`: Write video metadata to a `.info.json` file.
*   `--write-annotations`: Write video annotations to a `.annotations.xml` file.
*   `--load-info-json FILE`: Load video info from a `.info.json` file.
*   `--cookies FILE`: Read cookies from this file.
*   `--cache-dir DIR`: Cache downloaded information.
*   `--no-cache-dir`: Disable caching.
*   `--rm-cache-dir`: Delete all cache files.

### Thumbnail Options

*   `--write-thumbnail`: Write thumbnail image to disk.
*   `--write-all-thumbnails`: Write all thumbnail image formats to disk.
*   `--list-thumbnails`: List available thumbnail formats.

### Verbosity / Simulation Options

*   `-q, --quiet`: Quiet mode.
*   `--no-warnings`: Ignore warnings.
*   `-s, --simulate`: Simulate only.
*   `--skip-download`: Skip download.
*   `-g, --get-url`: Simulate, print URL.
*   `-e, --get-title`: Simulate, print title.
*   `--get-id`: Simulate, print ID.
*   `--get-thumbnail`: Simulate, print thumbnail URL.
*   `--get-description`: Simulate, print video description.
*   `--get-duration`: Simulate, print video length.
*   `--get-filename`: Simulate, print output filename.
*   `--get-format`: Simulate, print output format.
*   `-j, --dump-json`: Simulate, print JSON.
*   `-J, --dump-single-json`: Simulate, print JSON for each command-line argument.
*   `--print-json`: Print video information as JSON (download continues).
*   `--newline`: Output progress bar as new lines.
*   `--no-progress`: Don't print progress bar.
*   `--console-title`: Display progress in console titlebar.
*   `-v, --verbose`: Print debugging information.
*   `--dump-pages`: Print downloaded pages for debugging.
*   `--write-pages`: Write downloaded intermediary pages to files.
*   `--print-traffic`: Display sent and read HTTP traffic.
*   `-C, --call-home`: Contact youtube-dl server for debugging.
*   `--no-call-home`: Do NOT contact the youtube-dl server for debugging.

### Workarounds

*   `--encoding ENCODING`: Force the specified encoding.
*   `--no-check-certificate`: Suppress HTTPS certificate validation.
*   `--prefer-insecure`: Use unencrypted connection (YouTube only).
*   `--user-agent UA`: Specify a custom user agent.
*   `--referer URL`: Specify a custom referer.
*   `--add-header FIELD:VALUE`: Specify a custom HTTP header.
*   `--bidi-workaround`: Work around bidirectional text support.
*   `--sleep-interval SECONDS`: Number of seconds to sleep before each download.
*   `--max-sleep-interval SECONDS`: Upper bound of a range for randomized sleep before each download.

### Video Format Options

*   `-f, --format FORMAT`: Video format code (see [FORMAT SELECTION](#format-selection)).
*   `--all-formats`: Download all available video formats.
*   `--prefer-free-formats`: Prefer free video formats.
*   `-F, --list-formats`: List all available formats.
*   `--youtube-skip-dash-manifest`: Do not download the DASH manifests and related data on YouTube videos.
*   `--merge-output-format FORMAT`: If a merge is required, output to a given container format (mkv, mp4, ogg, webm, flv).

### Subtitle Options

*   `--write-sub`: Write subtitle file.
*   `--write-auto-sub`: Write automatically generated subtitle file (YouTube only).
*   `--all-subs`: Download all available subtitles.
*   `--list-subs`: List available subtitles.
*   `--sub-format FORMAT`: Subtitle format (e.g., `srt` or `ass/srt/best`).
*   `--sub-lang LANGS`: Languages of the subtitles to download.

### Authentication Options

*   `-u, --username USERNAME`: Login with this account ID.
*   `-p, --password PASSWORD`: Account password.
*   `-2, --twofactor TWOFACTOR`: Two-factor authentication code.
*   `-n, --netrc`: Use `.netrc` authentication data.
*   `--video-password PASSWORD`: Video password (Vimeo, Youku).

### Adobe Pass Options

*   `--ap-mso MSO`: Adobe Pass multiple-system operator identifier.
*   `--ap-username USERNAME`: Multiple-system operator account login.
*   `--ap-password PASSWORD`: Multiple-system operator account password.
*   `--ap-list-mso`: List all supported multiple-system operators.

### Post-processing Options

*   `-x, --extract-audio`: Convert video to audio (requires `ffmpeg`/`avconv`).
*   `--audio-format FORMAT`: Audio format (e.g., `mp3`, `m4a`, `flac`, `wav`).
*   `--audio-quality QUALITY`: Audio quality (0-9 for VBR, bitrate like `128k`).
*   `--recode-video FORMAT`: Encode video to another format.
*   `--postprocessor-args ARGS`: Give arguments to the postprocessor.
*   `-k, --keep-video`: Keep the video file after post-processing.
*   `--no-post-overwrites`: Don't overwrite post-processed files.
*   `--embed-subs`: Embed subtitles in the video.
*   `--embed-thumbnail`: Embed thumbnail in the audio as cover art.
*   `--add-metadata`: Write metadata to the video file.
*   `--metadata-from-title FORMAT`: Parse additional metadata like song title / artist from the video title.
*   `--xattrs`: Write metadata to the video file's xattrs.
*   `--fixup POLICY`: Automatically correct known faults of the file.
*   `--prefer-avconv`: Prefer `avconv` over `ffmpeg`.
*   `--prefer-ffmpeg`: Prefer `ffmpeg` over `avconv`.
*   `--ffmpeg-location PATH`: Location of `ffmpeg`/`avconv`.
*   `--exec CMD`: Execute a command on the file after downloading and post-processing.
*   `--convert-subs FORMAT`: Convert the subtitles to other format (currently supported: srt|ass|vtt|lrc)

## Configuration

You can configure youtube-dl by adding command-line options to a configuration file.

*   **Linux/macOS:** `/etc/youtube-dl.conf` (system-wide) and `~/.config/youtube-dl/config` (user-specific).
*   **Windows:** `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`.

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

Use `--ignore-config` to disable the configuration file.  Use `--config-location` to specify a custom configuration file.

### Authentication with `.netrc` file

Configure credentials for extractors that support authentication using a [.netrc file](https://stackoverflow.com/tags/.netrc/info) in your `$HOME`.

Create a `.netrc` file and set permissions:

```bash
touch $HOME/.netrc
chmod a-rwx,u+rw $HOME/.netrc
```

Add your login credentials:

```
machine <extractor> login <login> password <password>
```

For example:

```
machine youtube login myaccount@gmail.com password my_youtube_password
machine twitch login my_twitch_account_name password my_twitch_password
```

Activate authentication with the `.netrc` file: pass `--netrc` to youtube-dl or add it to your configuration file.

On Windows: setup the `%HOME%` environment variable manually:
```
set HOME=%USERPROFILE%
```

## Output Template

The `-o` option allows users to indicate a template for the output file names.

**tl;dr:** Output template examples [below](#output-template-examples).

Special sequences are replaced when downloading videos, formatted by [python string formatting operations](https://docs.python.org/2/library/stdtypes.html#string-formatting) such as `%(NAME)s`.

*   `id`: Video identifier
*   `title`: Video title
*   `url`: Video URL
*   `ext`: Video filename extension
*   `alt_title`
*   `display_id`
*   `uploader`
*   `license`
*   `creator`
*   `release_date`
*   `timestamp`
*   `upload_date`
*   `uploader_id`
*   `channel`
*   `channel_id`
*   `location`
*   `duration`
*   `view_count`
*   `like_count`
*   `dislike_count`
*   `repost_count`
*   `average_rating`
*   `comment_count`
*   `age_limit`
*   `is_live`
*   `start_time`
*   `end_time`
*   `format`
*   `format_id`
*   `format_note`
*   `width`
*   `height`
*   `resolution`
*   `tbr`
*   `abr`
*   `acodec`
*   `asr`
*   `vbr`
*   `fps`
*   `vcodec`
*   `container`
*   `filesize`
*   `filesize_approx`
*   `protocol`
*   `extractor`
*   `extractor_key`
*   `epoch`
*   `autonumber`
*   `playlist`
*   `playlist_index`
*   `playlist_id`
*   `playlist_title`
*   `playlist_uploader`
*   `playlist_uploader_id`
*   `chapter`
*   `chapter_number`
*   `chapter_id`
*   `series`
*   `season`
*   `season_number`
*   `season_id`
*   `episode`
*   `episode_number`
*   `episode_id`
*   `track`
*   `track_number`
*   `track_id`
*   `artist`
*   `genre`
*   `album`
*   `album_type`
*   `album_artist`
*   `disc_number`
*   `release_year`

Sequences not available will be replaced by `--output-na-placeholder` (default: `NA`).
Use `%%` to output percent literals.

The default template is `%(title)s-%(id)s.%(ext)s`.

To avoid issues with special characters, use the `--restrict-filenames` flag.

#### Output Template Examples

```bash
# A simple file name
youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc --restrict-filenames

# Download YouTube playlist videos in separate directory indexed by video order in a playlist
youtube-dl -o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' https://www.youtube.com/playlist?list=PLwiyx1dc3P2JR9N8gQaQN_BCvlSlap7re

# Download all playlists of YouTube channel/user keeping each playlist in separate directory:
youtube-dl -o '%(uploader)s/%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' https://www.youtube.com/user/TheLinuxFoundation/playlists

# Download Udemy course keeping each chapter in separate directory under MyVideos directory in your home
youtube-dl -u user -p password -o '~/MyVideos/%(playlist)s/%(chapter_number)s - %(chapter)s/%(title)s.%(ext)s' https://www.udemy.com/java-tutorial/

# Download entire series season keeping each series and each season in separate directory under C:/MyVideos
youtube-dl -o "C:/MyVideos/%(series)s/%(season_number)s - %(season)s/%(episode_number)s - %(episode)s.%(ext)s" https://videomore.ru/kino_v_detalayah/5_sezon/367617

# Stream the video being downloaded to stdout
youtube-dl -o - BaW_jenozKc
```

## Format Selection

Download videos in specific formats with `-f FORMAT`.

**tl;dr:** Format selection examples [below](#format-selection-examples).

Use `--list-formats` or `-F` to list available formats.

*   `best`: Best quality video with audio.
*   `worst`: Worst quality video with audio.
*   `bestvideo`: Best video-only format.
*   `worstvideo`: Worst video-only format.
*   `bestaudio`: Best audio-only format.
*   `worstaudio`: Worst audio-only format.

Format precedence is specified with slashes, e.g., `-f 22/17/18`.

Use a comma to download multiple formats: `-f 22,17,18`.

Filter formats with conditions in brackets, e.g., `-f "best[height=720]"`.
Numeric comparisons: `<` , `<=`, `>`, `>=`, `=`, `!=`
String comparisons: `=`, `^=`, `$=`, `*=`, `!` (negation)

*   `filesize`
*   `width`
*   `height`
*   `tbr`
*   `abr`
*   `vbr`
*   `asr`
*   `fps`
*   `ext`
*   `acodec`
*   `vcodec`
*   `container`
*   `protocol`
*   `format_id`
*   `language`

Merge video and audio with `-f <video-format>+<audio-format>` (requires ffmpeg or avconv).

#### Format Selection Examples

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

## Video Selection

Filter videos by upload date with `--date`, `--datebefore`, and `--dateafter`.
Date formats: `YYYYMMDD` or `(now|today)[+-][0-9](day|week|month|year)(s)?`.

```bash
# Download only the videos uploaded in the last 6 months
$ youtube-dl --dateafter now-6months

# Download only the videos uploaded on January 1, 1970
$ youtube-dl --date 19700101

$ # Download only the videos uploaded in the 200x decade
$ youtube-dl --dateafter 20000101 --datebefore 20091231
```

## FAQ

A collection of frequently asked questions can be found within the original README [here](https://github.com/ytdl-org/youtube-dl#faq).

## Developer Instructions

See [original README](https://github.com/ytdl-org/youtube-dl) for comprehensive developer instructions.

### Adding Support for a New Site

1.  [Fork the repository](https://github.com/ytdl-org/youtube-dl/fork).
2.  Clone the source code.
3.  Create a new branch: `git checkout -b yourextractor`.
4.  Create an extractor file, e.g., `youtube_dl/extractor/yourextractor.py`, using this [template](#adding-support-for-a-new-site) as a starting point.
5.  Add an import in `youtube_dl/extractor/extractors.py`.
6.  Run tests: `python test/test_download.py TestDownload.test_YourExtractor`.
7.  Use the available [helper methods](https://github.com/ytdl-org/youtube-dl/blob/master/youtube_dl/extractor/common.py) to implement metadata extraction.
8.  Adhere to [youtube-dl coding conventions](#youtube-dl-coding-conventions).
9.  Test code: `flake8 youtube_dl/extractor/yourextractor.py`.
10. Ensure the code works across supported Python versions.
11. Commit, push, and [create a pull request](https://help.github.com/articles/creating-a-pull-request).

### youtube-dl coding conventions

See the [original README](https://github.com/ytdl-org/youtube-dl#youtube-dl-coding-conventions) for full coding conventions.

## Embedding youtube-dl

Use `youtube_dl` from a Python program.

```python
from __future__ import unicode_literals
import youtube_dl

ydl_opts = {}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=BaW_jenozKc'])
```

Customize behavior using options documented in [`youtube_dl/YoutubeDL.py`](https://github.com/ytdl-org/youtube-dl/blob/3e4cedf9e8cd3157df2457df7274d0c842421945/youtube_dl/YoutubeDL.py#L137-L312).

## Bugs

Report bugs and suggestions in the issue tracker: <https://github.com/ytdl-org/youtube-dl/issues> (or <https://yt-dl.org/bug>).

Follow the [bug reporting instructions](#bugs) in the original README when filing an issue.

## Copyright

youtube-dl is released into the public domain by the copyright holders.