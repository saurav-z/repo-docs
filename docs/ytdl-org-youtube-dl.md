[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Download Videos from YouTube and Beyond

**youtube-dl is a versatile command-line tool for downloading videos from YouTube and numerous other video platforms, empowering users with the ability to save their favorite content.**  [Visit the official repository](https://github.com/ytdl-org/youtube-dl) for more information and to contribute.

## Key Features

*   **Wide Platform Support:** Downloads videos from YouTube, Vimeo, Dailymotion, and hundreds of other sites.
*   **Format Selection:** Choose from a variety of video and audio formats and qualities.
*   **Playlist and Channel Downloads:** Easily download entire playlists or all videos from a channel.
*   **Customizable Output:** Control file names, output directories, and other settings using templates.
*   **Subtitle Support:** Download subtitles in various languages and formats.
*   **Cross-Platform Compatibility:** Works on Linux, macOS, Windows, and other operating systems with Python installed.
*   **Metadata Handling:** Preserve video metadata, including title, description, and more.
*   **Advanced Options:** Fine-tune downloads with features like proxy support, rate limiting, and download resuming.

## Table of Contents

*   [Installation](#installation)
    *   [UNIX (Linux, macOS, etc.)](#installation-unix)
    *   [Windows](#installation-windows)
    *   [Using pip](#installation-pip)
    *   [macOS (Homebrew & MacPorts)](#installation-macos)
    *   [Developer Installation](#developer-instructions)
    *   [Download Page](#youtube-dl-download-page)
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
    *   [Opening a bug report or suggestion](#opening-a-bug-report-or-suggestion)
*   [Copyright](#copyright)

### Installation

#### Installation (UNIX)

To install youtube-dl on UNIX-based systems (Linux, macOS, etc.):

```bash
sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

If you don't have `curl`, use `wget`:

```bash
sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

#### Installation (Windows)

For Windows users, [download an executable](https://yt-dl.org/latest/youtube-dl.exe) and place it in any directory within your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) environment variable, excluding `%SYSTEMROOT%\System32`.

#### Installation (Using pip)

You can also install or update youtube-dl using `pip`:

```bash
sudo -H pip install --upgrade youtube-dl
```

Visit the [pypi page](https://pypi.python.org/pypi/youtube_dl) for further details.

#### Installation (macOS - Homebrew & MacPorts)

macOS users can utilize package managers:

*   **Homebrew:**

    ```bash
    brew install youtube-dl
    ```
*   **MacPorts:**

    ```bash
    sudo port install youtube-dl
    ```

#### Developer Instructions

Refer to the [developer instructions](#developer-instructions) to learn how to work with the Git repository.

#### youtube-dl Download Page

Explore further installation options, including PGP signatures, on the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

### Description

**youtube-dl** is a versatile, command-line program designed to download videos from YouTube and numerous other websites. It's written in Python, making it platform-independent and compatible with Unix-like systems, Windows, and macOS. This open-source tool is released to the public domain, enabling users to modify, redistribute, and use it as they see fit.

### Options

Use the `youtube-dl [OPTIONS] URL [URL...]` command structure.

*   `-h`, `--help`: Displays help text and exits.
*   `--version`: Displays program version and exits.
*   `-U`, `--update`: Updates youtube-dl to the latest version (requires appropriate permissions).
*   `-i`, `--ignore-errors`: Continues downloads even with errors (e.g., for unavailable videos in a playlist).
*   `--abort-on-error`: Halts downloading upon encountering an error.
*   `--dump-user-agent`: Shows the current browser user agent.
*   `--list-extractors`: Lists all supported extractors.
*   `--extractor-descriptions`: Shows descriptions of extractors.
*   `--force-generic-extractor`: Forces use of the generic extractor.
*   `--default-search PREFIX`: Defines the prefix for unspecified URLs.  Options: "auto", "auto_warning", "error", "fixup_error".
*   `--ignore-config`: Prevents reading configuration files.
*   `--config-location PATH`: Specifies the configuration file location.
*   `--flat-playlist`: Lists playlist videos without extracting them.
*   `--mark-watched`: Marks YouTube videos as watched.
*   `--no-mark-watched`: Disables marking YouTube videos as watched.
*   `--no-color`: Disables color output.

#### Network Options

*   `--proxy URL`: Uses the specified HTTP/HTTPS/SOCKS proxy. For SOCKS, use the appropriate scheme (e.g., `socks5://127.0.0.1:1080/`).  Use `--proxy ""` for direct connection.
*   `--socket-timeout SECONDS`: Sets the socket timeout in seconds.
*   `--source-address IP`: Sets the client-side IP address to bind to.
*   `-4`, `--force-ipv4`: Forces IPv4 connections.
*   `-6`, `--force-ipv6`: Forces IPv6 connections.

#### Geo Restriction

*   `--geo-verification-proxy URL`: Uses a proxy to verify the IP address for geo-restricted sites; downloads use the proxy specified by `--proxy`.
*   `--geo-bypass`: Bypasses geo-restrictions using a fake `X-Forwarded-For` header.
*   `--no-geo-bypass`: Disables geo-bypass.
*   `--geo-bypass-country CODE`: Forces geo-restriction bypass with a two-letter ISO 3166-2 country code.
*   `--geo-bypass-ip-block IP_BLOCK`: Forces geo-restriction bypass with an IP block in CIDR notation.

#### Video Selection

*   `--playlist-start NUMBER`: Starts downloading from a specified playlist video number (default: 1).
*   `--playlist-end NUMBER`: Ends downloading at a specified playlist video number (default: last).
*   `--playlist-items ITEM_SPEC`: Downloads specific playlist items (e.g., `--playlist-items 1,2,5-8`).
*   `--match-title REGEX`: Downloads videos with matching titles (regex or substring).
*   `--reject-title REGEX`: Skips downloads with matching titles.
*   `--max-downloads NUMBER`: Limits the number of downloaded files.
*   `--min-filesize SIZE`: Excludes videos smaller than a specified size (e.g., 50k, 44.6m).
*   `--max-filesize SIZE`: Excludes videos larger than a specified size.
*   `--date DATE`: Downloads videos uploaded on a specific date.
*   `--datebefore DATE`: Downloads videos uploaded on or before a date.
*   `--dateafter DATE`: Downloads videos uploaded on or after a date.
*   `--min-views COUNT`: Excludes videos with fewer views.
*   `--max-views COUNT`: Excludes videos with more views.
*   `--match-filter FILTER`: Uses a generic video filter (e.g., `--match-filter "like_count > 100 & dislike_count <? 50 & description"`).
*   `--no-playlist`: Downloads the video if the URL refers to both a video and a playlist.
*   `--yes-playlist`: Downloads the playlist if the URL refers to both a video and a playlist.
*   `--age-limit YEARS`: Filters videos based on age restrictions.
*   `--download-archive FILE`: Downloads only videos not listed in the archive file.
*   `--include-ads`: Downloads advertisements (experimental).

#### Download Options

*   `-r`, `--limit-rate RATE`: Sets the maximum download rate in bytes per second (e.g., 50K, 4.2M).
*   `-R`, `--retries RETRIES`: Sets the number of download retries (default: 10, or "infinite").
*   `--fragment-retries RETRIES`: Sets the number of retries for fragments (default: 10, or "infinite").
*   `--skip-unavailable-fragments`: Skips unavailable fragments.
*   `--abort-on-unavailable-fragment`: Aborts if a fragment is unavailable.
*   `--keep-fragments`: Keeps downloaded fragments on disk.
*   `--buffer-size SIZE`: Sets the download buffer size (default: 1024).
*   `--no-resize-buffer`: Disables automatic buffer size adjustment.
*   `--http-chunk-size SIZE`: Sets the chunk size for chunk-based HTTP downloads (disabled by default).
*   `--playlist-reverse`: Downloads playlist videos in reverse order.
*   `--playlist-random`: Downloads playlist videos in random order.
*   `--xattr-set-filesize`: Sets file xattribute `ytdl.filesize` with the expected file size.
*   `--hls-prefer-native`: Uses the native HLS downloader instead of ffmpeg.
*   `--hls-prefer-ffmpeg`: Uses ffmpeg for HLS downloads.
*   `--hls-use-mpegts`: Uses the mpegts container for HLS videos.
*   `--external-downloader COMMAND`: Uses an external downloader (e.g., aria2c, avconv, etc.).
*   `--external-downloader-args ARGS`: Passes arguments to the external downloader.

#### Filesystem Options

*   `-a`, `--batch-file FILE`: Reads URLs from a file (one per line).
*   `--id`: Uses only the video ID in the file name.
*   `-o`, `--output TEMPLATE`: Specifies the output file name template (see [Output Template](#output-template)).
*   `--output-na-placeholder PLACEHOLDER`: Placeholder for missing metadata (default: "NA").
*   `--autonumber-start NUMBER`: Sets the starting value for `%(autonumber)s` (default: 1).
*   `--restrict-filenames`: Restricts file names to ASCII characters and avoids spaces and "&".
*   `-w`, `--no-overwrites`: Prevents overwriting files.
*   `-c`, `--continue`: Resumes partial downloads.
*   `--no-continue`: Disables download resuming.
*   `--no-part`: Disables the use of `.part` files.
*   `--no-mtime`: Prevents setting the file modification time.
*   `--write-description`: Writes video description to a `.description` file.
*   `--write-info-json`: Writes video metadata to a `.info.json` file.
*   `--write-annotations`: Writes video annotations to a `.annotations.xml` file.
*   `--load-info-json FILE`: Loads video information from a `.info.json` file.
*   `--cookies FILE`: Reads cookies from a file.
*   `--cache-dir DIR`: Specifies the cache directory (default: `$XDG_CACHE_HOME/youtube-dl` or `~/.cache/youtube-dl`).
*   `--no-cache-dir`: Disables filesystem caching.
*   `--rm-cache-dir`: Deletes the cache directory.

#### Thumbnail Options

*   `--write-thumbnail`: Writes the thumbnail image to disk.
*   `--write-all-thumbnails`: Writes all available thumbnail formats.
*   `--list-thumbnails`: Lists available thumbnail formats.

#### Verbosity / Simulation Options

*   `-q`, `--quiet`: Enables quiet mode.
*   `--no-warnings`: Suppresses warnings.
*   `-s`, `--simulate`: Simulates a download without saving.
*   `--skip-download`: Skips the video download.
*   `-g`, `--get-url`: Simulates and prints the URL.
*   `-e`, `--get-title`: Simulates and prints the title.
*   `--get-id`: Simulates and prints the ID.
*   `--get-thumbnail`: Simulates and prints the thumbnail URL.
*   `--get-description`: Simulates and prints the video description.
*   `--get-duration`: Simulates and prints the video length.
*   `--get-filename`: Simulates and prints the output file name.
*   `--get-format`: Simulates and prints the output format.
*   `-j`, `--dump-json`: Simulates and prints JSON information.
*   `-J`, `--dump-single-json`: Simulates and prints JSON information for each command-line argument.
*   `--print-json`: Prints video information as JSON during the download.
*   `--newline`: Outputs the progress bar on new lines.
*   `--no-progress`: Disables the progress bar.
*   `--console-title`: Displays progress in the console title bar.
*   `-v`, `--verbose`: Enables verbose output.
*   `--dump-pages`: Prints downloaded pages (base64 encoded) for debugging.
*   `--write-pages`: Writes downloaded pages to files for debugging.
*   `--print-traffic`: Displays HTTP traffic.
*   `-C`, `--call-home`: Contacts the youtube-dl server for debugging.
*   `--no-call-home`: Disables contacting the youtube-dl server.

#### Workarounds

*   `--encoding ENCODING`: Forces a specific encoding (experimental).
*   `--no-check-certificate`: Suppresses HTTPS certificate validation.
*   `--prefer-insecure`: Uses an unencrypted connection for video information.
*   `--user-agent UA`: Specifies a custom user agent.
*   `--referer URL`: Specifies a custom referrer URL.
*   `--add-header FIELD:VALUE`: Adds a custom HTTP header.
*   `--bidi-workaround`: Works around terminals lacking bidirectional text support.
*   `--sleep-interval SECONDS`: Sets the sleep interval before downloads.
*   `--max-sleep-interval SECONDS`: Sets the maximum sleep interval.

#### Video Format Options

*   `-f`, `--format FORMAT`: Specifies the video format code (see [Format Selection](#format-selection)).
*   `--all-formats`: Downloads all available video formats.
*   `--prefer-free-formats`: Prioritizes free video formats.
*   `-F`, `--list-formats`: Lists all available formats.
*   `--youtube-skip-dash-manifest`: Disables DASH manifest downloads on YouTube.
*   `--merge-output-format FORMAT`: Specifies the container format for merging (e.g., `mkv`, `mp4`).

#### Subtitle Options

*   `--write-sub`: Writes subtitle files.
*   `--write-auto-sub`: Writes automatically generated subtitles (YouTube only).
*   `--all-subs`: Downloads all available subtitles.
*   `--list-subs`: Lists available subtitles.
*   `--sub-format FORMAT`: Specifies the subtitle format (e.g., `srt`, `ass/srt/best`).
*   `--sub-lang LANGS`: Specifies the subtitle languages (separated by commas).

#### Authentication Options

*   `-u`, `--username USERNAME`: Logs in with a specific account.
*   `-p`, `--password PASSWORD`: Specifies the account password.
*   `-2`, `--twofactor TWOFACTOR`: Provides a two-factor authentication code.
*   `-n`, `--netrc`: Uses .netrc authentication data.
*   `--video-password PASSWORD`: Sets a video password (e.g., for Vimeo, Youku).

#### Adobe Pass Options

*   `--ap-mso MSO`: Specifies the Adobe Pass multiple-system operator identifier.
*   `--ap-username USERNAME`: Specifies the multiple-system operator account username.
*   `--ap-password PASSWORD`: Specifies the multiple-system operator account password.
*   `--ap-list-mso`: Lists supported multiple-system operators.

#### Post-processing Options

*   `-x`, `--extract-audio`: Converts video files to audio-only (requires ffmpeg/avconv).
*   `--audio-format FORMAT`: Specifies the audio format (e.g., `mp3`, `wav`).
*   `--audio-quality QUALITY`: Specifies the audio quality (0-9 for VBR or bitrate like 128K).
*   `--recode-video FORMAT`: Re-encodes the video to another format.
*   `--postprocessor-args ARGS`: Passes arguments to the postprocessor.
*   `-k`, `--keep-video`: Keeps the video file after post-processing.
*   `--no-post-overwrites`: Prevents overwriting post-processed files.
*   `--embed-subs`: Embeds subtitles in the video (for mp4, webm, and mkv).
*   `--embed-thumbnail`: Embeds the thumbnail in the audio as cover art.
*   `--add-metadata`: Writes metadata to the video file.
*   `--metadata-from-title FORMAT`: Parses metadata from the video title.
*   `--xattrs`: Writes metadata to video file xattrs.
*   `--fixup POLICY`: Automatically corrects known file faults.
*   `--prefer-avconv`:  Prefers avconv over ffmpeg for post-processing.
*   `--prefer-ffmpeg`:  Prefers ffmpeg over avconv for post-processing (default).
*   `--ffmpeg-location PATH`: Sets the ffmpeg/avconv binary location.
*   `--exec CMD`: Executes a command after download and post-processing.
*   `--convert-subs FORMAT`: Converts subtitles to other formats (e.g., `srt`, `ass`).

### Configuration

Configure youtube-dl by adding command-line options to a configuration file.

*   **Linux/macOS:** System-wide: `/etc/youtube-dl.conf`, User-specific: `~/.config/youtube-dl/config`.
*   **Windows:**  `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`.

Use `--ignore-config` to disable the configuration file and `--config-location` to specify a custom config.

#### Authentication with .netrc file

For extractors that support authentication:

1.  Create a `.netrc` file in your `$HOME` directory:

```bash
touch $HOME/.netrc
chmod a-rwx,u+rw $HOME/.netrc
```

2.  Add credentials in this format:

```
machine <extractor> login <login> password <password>
```

3.  Activate with `--netrc` or by adding it to the configuration file.

On Windows, set the `%HOME%` environment variable:

```
set HOME=%USERPROFILE%
```

### Output Template

Use `-o` with output templates for custom file names.

*   **Basic usage:**  `youtube-dl -o funny_video.flv "https://some/video"`
*   **Special sequences (e.g., `%(title)s-%(id)s.%(ext)s`)**: The template can use python string formatting operators and contains special sequences for filename formatting, such as:
    *   `id`, `title`, `url`, `ext`, `alt_title`, `display_id`, `uploader`, `license`, `creator`, `release_date`, `timestamp`, `upload_date`, `uploader_id`, `channel`, `channel_id`, `location`, `duration`, `view_count`, `like_count`, `dislike_count`, `repost_count`, `average_rating`, `comment_count`, `age_limit`, `is_live`, `start_time`, `end_time`, `format`, `format_id`, `format_note`, `width`, `height`, `resolution`, `tbr`, `abr`, `acodec`, `asr`, `vbr`, `fps`, `vcodec`, `container`, `filesize`, `filesize_approx`, `protocol`, `extractor`, `extractor_key`, `epoch`, `autonumber`, `playlist`, `playlist_index`, `playlist_id`, `playlist_title`, `playlist_uploader`, `playlist_uploader_id`, `chapter`, `chapter_number`, `chapter_id`, `series`, `season`, `season_number`, `season_id`, `episode`, `episode_number`, `episode_id`, `track`, `track_number`, `track_id`, `artist`, `genre`, `album`, `album_type`, `album_artist`, `disc_number`, `release_year`.
*   **Numeric formatting:**  `%(view_count)05d`
*   **Hierarchical paths:** `-o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s'`
*   **Percent literals:** `%%`
*   **Stdout:** `-o -`
*   **Restrict filenames:** `--restrict-filenames`

#### Output Template Examples

```bash
# Create a filename
youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc

# Create a filename
youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc --restrict-filenames

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

### Format Selection

*   **Default:** Downloads the best available quality.
*   **Specific format:** `-f 22`
*   **File extension:** `-f webm`
*   **Special names:** `best`, `worst`, `bestvideo`, `worstvideo`, `bestaudio`, `worstaudio`
*   **Preference:** `-f 22/17/18`
*   **Multiple formats:** `-f 22,17,18`
*   **Filter by condition:** `-f "best[height=720]"` or `-f "[filesize>10M]"`
*   **Numeric meta fields:**  `filesize`, `width`, `height`, `tbr`, `abr`, `vbr`, `asr`, `fps`. Supported Comparisons: `<`, `<=`, `>`, `>=`, `=`, `!=`.
*   **String meta fields:** `ext`, `acodec`, `vcodec`, `container`, `protocol`, `format_id`, `language`.  Supported Comparisons: `=`, `^=`, `$*=`, `*=`.
*   **Merge video and audio:** `-f <video-format>+<audio-format>`
*   **Group formats:** `-f '(mp4,webm)[height<480]'`

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

### Video Selection

You can filter videos based on upload date, using the options `--date`, `--datebefore`, or `--dateafter`. These accept dates in two formats:

*   Absolute dates: Dates in the format `YYYYMMDD`.
*   Relative dates: Dates in the format `(now|today)[+-][0-9](day|week|month|year)(s)?`.

Examples:

```bash
# Download only the videos uploaded in the last 6 months
$ youtube-dl --dateafter now-6months

# Download only the videos uploaded on January 1, 1970
$ youtube-dl --date 19700101

$ # Download only the videos uploaded in the 200x decade
$ youtube-dl --dateafter 20000101 --datebefore 20091231
```

### FAQ

Common questions and their solutions are available in the [FAQ section](README.md#faq) of the README.

### Developer Instructions

Detailed instructions are provided in the [Developer Instructions section](README.md#developer-instructions).

#### Adding Support for a New Site

1.  [Fork the repository](https://github.com/ytdl-org/youtube-dl/fork).
2.  Clone the repository:

    ```bash
    git clone git@github.com:YOUR_GITHUB_USERNAME/youtube-dl.git
    ```
3.  Create a new branch:

    ```bash
    cd youtube-dl
    git checkout -b yourextractor
    ```
4.  Create a new extractor file (`youtube_dl/extractor/yourextractor.py`) with the provided template.
5.  Add an import in `youtube_dl/extractor/extractors.py`.
6.  Run tests: `python test/test_download.py TestDownload.test_YourExtractor`.
7.  Follow the detailed instructions for code development.
8.  Ensure code follows [youtube-dl coding conventions](#youtube-dl-coding-conventions) and run Flake8.
9.  Test under Python versions 2.6, 2.7, and 3.2+.
10. Commit, push, and create a pull request.

#### youtube-dl coding conventions

Follow the guidelines for writing idiomatic, robust, and future-proof extractor code.

### Embedding youtube-dl

Embed youtube-dl in your Python program:

```python
from __future__ import unicode_literals
import youtube_dl

ydl_opts = {}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=BaW_jenozKc'])
```

### Bugs

Report bugs and suggestions in the issue tracker: <https://github.com/ytdl-org/youtube-dl/issues>.

#### Opening a bug report or suggestion

Follow instructions in the issue tracker.

*   Include the full output from `youtube-dl -v YOUR_URL_HERE`
*   Ensure the issue is not already documented.
*   Use the latest version: `youtube-dl -U`.
*   Explain why existing options are insufficient.
*   Provide sufficient context.
*   Report one problem per issue.
*   Is the question about youtube-dl?

### Copyright

youtube-dl is released into the public domain.
This README file was originally written by [Daniel Bolton](https://github.com/dbbolton) and is also released into the public domain.