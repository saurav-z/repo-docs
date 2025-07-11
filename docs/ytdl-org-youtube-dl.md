[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# YouTube-DL: Download Videos from YouTube and Beyond!

**Tired of buffering? Download videos from YouTube and numerous other sites with the powerful command-line tool, [youtube-dl](https://github.com/ytdl-org/youtube-dl).** This versatile program lets you save your favorite videos for offline viewing, extract audio, and more.

## Key Features:

*   **Wide Site Support:** Download from YouTube, plus hundreds of other video platforms.  See the full list of [supported sites](https://ytdl-org.github.io/youtube-dl/supportedsites.html).
*   **Format Selection:** Choose your preferred video and audio formats, resolutions, and quality settings.
*   **Playlist and Channel Downloads:** Easily download entire playlists and channels.
*   **Metadata Extraction:** Automatically retrieve and save video titles, descriptions, and other metadata.
*   **Customization:** Configure output filenames, download directories, and more using command-line options or configuration files.
*   **Post-Processing:** Convert videos to audio (MP3, WAV, etc.), embed subtitles, and add metadata.
*   **Cross-Platform:** Works seamlessly on Linux, macOS, and Windows.
*   **Active Community:** Benefit from ongoing updates and community support.

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

Installation instructions vary depending on your operating system.

**For UNIX-like systems (Linux, macOS, etc.):**

```bash
sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

If you don't have `curl`, use `wget`:

```bash
sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

**For Windows:**

*   [Download an .exe file](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory on your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) *except* `%SYSTEMROOT%\System32`.

**Alternative Installation Methods:**

*   **Pip:**  `sudo -H pip install --upgrade youtube-dl` (updates if already installed). See the [pypi page](https://pypi.python.org/pypi/youtube_dl).
*   **Homebrew (macOS):** `brew install youtube-dl`
*   **MacPorts (macOS):** `sudo port install youtube-dl`

For more options (including PGP signatures) see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## Description

**youtube-dl** is a powerful, command-line tool designed to download videos from YouTube.com and hundreds of other video-hosting websites. It leverages the Python interpreter (version 2.6, 2.7, or 3.2+) and is compatible across various platforms, including Unix-based systems, Windows, and macOS. It is released to the public domain.

## Options

Use `youtube-dl -h` to see a comprehensive list of options.

```
youtube-dl [OPTIONS] URL [URL...]
```

A brief overview follows:

*   `-h, --help`: Print help text and exit.
*   `--version`: Print program version and exit.
*   `-U, --update`: Update youtube-dl to the latest version.
*   `-i, --ignore-errors`: Continue on download errors.
*   `--abort-on-error`: Abort downloading further videos if an error occurs.
*   `--dump-user-agent`: Display the current browser identification.
*   `--list-extractors`: List all supported extractors.
*   `--extractor-descriptions`: Output descriptions of all supported extractors.
*   `--force-generic-extractor`: Force extraction to use the generic extractor.
*   `--default-search PREFIX`: Use this prefix for unqualified URLs.
*   `--ignore-config`: Do not read configuration files.
*   `--config-location PATH`: Location of the configuration file.
*   `--flat-playlist`: Do not extract the videos of a playlist, only list them.
*   `--mark-watched`: Mark videos watched (YouTube only).
*   `--no-mark-watched`: Do not mark videos watched (YouTube only).
*   `--no-color`: Do not emit color codes in output.

### Network Options:

*   `--proxy URL`: Use the specified HTTP/HTTPS/SOCKS proxy.
*   `--socket-timeout SECONDS`: Time to wait before giving up, in seconds.
*   `--source-address IP`: Client-side IP address to bind to.
*   `-4, --force-ipv4`: Make all connections via IPv4.
*   `-6, --force-ipv6`: Make all connections via IPv6.

### Geo Restriction:

*   `--geo-verification-proxy URL`: Use this proxy to verify the IP address for some geo-restricted sites.
*   `--geo-bypass`: Bypass geographic restriction via faking X-Forwarded-For HTTP header.
*   `--no-geo-bypass`: Do not bypass geographic restriction via faking X-Forwarded-For HTTP header.
*   `--geo-bypass-country CODE`: Force bypass geographic restriction with explicitly provided two-letter ISO 3166-2 country code.
*   `--geo-bypass-ip-block IP_BLOCK`: Force bypass geographic restriction with explicitly provided IP block in CIDR notation.

### Video Selection:

*   `--playlist-start NUMBER`: Playlist video to start at (default is 1).
*   `--playlist-end NUMBER`: Playlist video to end at (default is last).
*   `--playlist-items ITEM_SPEC`: Playlist video items to download.
*   `--match-title REGEX`: Download only matching titles.
*   `--reject-title REGEX`: Skip download for matching titles.
*   `--max-downloads NUMBER`: Abort after downloading NUMBER files.
*   `--min-filesize SIZE`: Do not download any videos smaller than SIZE (e.g. 50k or 44.6m).
*   `--max-filesize SIZE`: Do not download any videos larger than SIZE (e.g. 50k or 44.6m).
*   `--date DATE`: Download only videos uploaded in this date.
*   `--datebefore DATE`: Download only videos uploaded on or before this date (i.e. inclusive).
*   `--dateafter DATE`: Download only videos uploaded on or after this date (i.e. inclusive).
*   `--min-views COUNT`: Do not download any videos with less than COUNT views.
*   `--max-views COUNT`: Do not download any videos with more than COUNT views.
*   `--match-filter FILTER`: Generic video filter.
*   `--no-playlist`: Download only the video, if the URL refers to a video and a playlist.
*   `--yes-playlist`: Download the playlist, if the URL refers to a video and a playlist.
*   `--age-limit YEARS`: Download only videos suitable for the given age.
*   `--download-archive FILE`: Download only videos not listed in the archive file.
*   `--include-ads`: Download advertisements as well (experimental).

### Download Options:

*   `-r, --limit-rate RATE`: Maximum download rate in bytes per second (e.g. 50K or 4.2M).
*   `-R, --retries RETRIES`: Number of retries (default is 10), or "infinite".
*   `--fragment-retries RETRIES`: Number of retries for a fragment (default is 10), or "infinite".
*   `--skip-unavailable-fragments`: Skip unavailable fragments.
*   `--abort-on-unavailable-fragment`: Abort downloading when some fragment is not available.
*   `--keep-fragments`: Keep downloaded fragments on disk after downloading is finished.
*   `--buffer-size SIZE`: Size of download buffer (e.g. 1024 or 16K) (default is 1024).
*   `--no-resize-buffer`: Do not automatically adjust the buffer size.
*   `--http-chunk-size SIZE`: Size of a chunk for chunk-based HTTP downloading (e.g. 10485760 or 10M) (default is disabled).
*   `--playlist-reverse`: Download playlist videos in reverse order.
*   `--playlist-random`: Download playlist videos in random order.
*   `--xattr-set-filesize`: Set file xattribute ytdl.filesize with expected file size.
*   `--hls-prefer-native`: Use the native HLS downloader instead of ffmpeg.
*   `--hls-prefer-ffmpeg`: Use ffmpeg instead of the native HLS downloader.
*   `--hls-use-mpegts`: Use the mpegts container for HLS videos, allowing to play the video while downloading.
*   `--external-downloader COMMAND`: Use the specified external downloader.
*   `--external-downloader-args ARGS`: Give these arguments to the external downloader.

### Filesystem Options:

*   `-a, --batch-file FILE`: File containing URLs to download ('-' for stdin), one URL per line.
*   `--id`: Use only video ID in file name.
*   `-o, --output TEMPLATE`: Output filename template.
*   `--output-na-placeholder PLACEHOLDER`: Placeholder value for unavailable meta fields in output filename template (default is "NA").
*   `--autonumber-start NUMBER`: Specify the start value for %(autonumber)s (default is 1).
*   `--restrict-filenames`: Restrict filenames to only ASCII characters.
*   `-w, --no-overwrites`: Do not overwrite files.
*   `-c, --continue`: Force resume of partially downloaded files.
*   `--no-continue`: Do not resume partially downloaded files.
*   `--no-part`: Do not use .part files - write directly into output file.
*   `--no-mtime`: Do not use the Last-modified header to set the file modification time.
*   `--write-description`: Write video description to a .description file.
*   `--write-info-json`: Write video metadata to a .info.json file.
*   `--write-annotations`: Write video annotations to a .annotations.xml file.
*   `--load-info-json FILE`: JSON file containing the video information.
*   `--cookies FILE`: File to read cookies from and dump cookie jar in.
*   `--cache-dir DIR`: Location in the filesystem where youtube-dl can store some downloaded information permanently.
*   `--no-cache-dir`: Disable filesystem caching.
*   `--rm-cache-dir`: Delete all filesystem cache files.

### Thumbnail Options:

*   `--write-thumbnail`: Write thumbnail image to disk.
*   `--write-all-thumbnails`: Write all thumbnail image formats to disk.
*   `--list-thumbnails`: Simulate and list all available thumbnail formats.

### Verbosity / Simulation Options:

*   `-q, --quiet`: Activate quiet mode.
*   `--no-warnings`: Ignore warnings.
*   `-s, --simulate`: Do not download the video and do not write anything to disk.
*   `--skip-download`: Do not download the video.
*   `-g, --get-url`: Simulate, quiet but print URL.
*   `-e, --get-title`: Simulate, quiet but print title.
*   `--get-id`: Simulate, quiet but print id.
*   `--get-thumbnail`: Simulate, quiet but print thumbnail URL.
*   `--get-description`: Simulate, quiet but print video description.
*   `--get-duration`: Simulate, quiet but print video length.
*   `--get-filename`: Simulate, quiet but print output filename.
*   `--get-format`: Simulate, quiet but print output format.
*   `-j, --dump-json`: Simulate, quiet but print JSON information.
*   `-J, --dump-single-json`: Simulate, quiet but print JSON information for each command-line argument.
*   `--print-json`: Be quiet and print the video information as JSON (video is still being downloaded).
*   `--newline`: Output progress bar as new lines.
*   `--no-progress`: Do not print progress bar.
*   `--console-title`: Display progress in console titlebar.
*   `-v, --verbose`: Print various debugging information.
*   `--dump-pages`: Print downloaded pages encoded using base64 to debug problems.
*   `--write-pages`: Write downloaded intermediary pages to files in the current directory to debug problems.
*   `--print-traffic`: Display sent and read HTTP traffic.
*   `-C, --call-home`: Contact the youtube-dl server for debugging.
*   `--no-call-home`: Do NOT contact the youtube-dl server for debugging.

### Workarounds:

*   `--encoding ENCODING`: Force the specified encoding.
*   `--no-check-certificate`: Suppress HTTPS certificate validation.
*   `--prefer-insecure`: Use an unencrypted connection to retrieve information about the video.
*   `--user-agent UA`: Specify a custom user agent.
*   `--referer URL`: Specify a custom referer.
*   `--add-header FIELD:VALUE`: Specify a custom HTTP header and its value.
*   `--bidi-workaround`: Work around terminals that lack bidirectional text support.
*   `--sleep-interval SECONDS`: Number of seconds to sleep before each download.
*   `--max-sleep-interval SECONDS`: Upper bound of a range for randomized sleep before each download.

### Video Format Options:

*   `-f, --format FORMAT`: Video format code.
*   `--all-formats`: Download all available video formats.
*   `--prefer-free-formats`: Prefer free video formats unless a specific one is requested.
*   `-F, --list-formats`: List all available formats of requested videos.
*   `--youtube-skip-dash-manifest`: Do not download the DASH manifests and related data on YouTube videos.
*   `--merge-output-format FORMAT`: If a merge is required, output to given container format.

### Subtitle Options:

*   `--write-sub`: Write subtitle file.
*   `--write-auto-sub`: Write automatically generated subtitle file (YouTube only).
*   `--all-subs`: Download all the available subtitles of the video.
*   `--list-subs`: List all available subtitles for the video.
*   `--sub-format FORMAT`: Subtitle format.
*   `--sub-lang LANGS`: Languages of the subtitles to download (optional).

### Authentication Options:

*   `-u, --username USERNAME`: Login with this account ID.
*   `-p, --password PASSWORD`: Account password.
*   `-2, --twofactor TWOFACTOR`: Two-factor authentication code.
*   `-n, --netrc`: Use .netrc authentication data.
*   `--video-password PASSWORD`: Video password (vimeo, youku).

### Adobe Pass Options:

*   `--ap-mso MSO`: Adobe Pass multiple-system operator (TV provider) identifier.
*   `--ap-username USERNAME`: Multiple-system operator account login.
*   `--ap-password PASSWORD`: Multiple-system operator account password.
*   `--ap-list-mso`: List all supported multiple-system operators.

### Post-processing Options:

*   `-x, --extract-audio`: Convert video files to audio-only files (requires ffmpeg/avconv).
*   `--audio-format FORMAT`: Specify audio format.
*   `--audio-quality QUALITY`: Specify ffmpeg/avconv audio quality.
*   `--recode-video FORMAT`: Encode the video to another format.
*   `--postprocessor-args ARGS`: Give these arguments to the postprocessor.
*   `-k, --keep-video`: Keep the video file on disk after the post-processing.
*   `--no-post-overwrites`: Do not overwrite post-processed files.
*   `--embed-subs`: Embed subtitles in the video.
*   `--embed-thumbnail`: Embed thumbnail in the audio as cover art.
*   `--add-metadata`: Write metadata to the video file.
*   `--metadata-from-title FORMAT`: Parse additional metadata from the video title.
*   `--xattrs`: Write metadata to the video file's xattrs (using dublin core and xdg standards).
*   `--fixup POLICY`: Automatically correct known faults of the file.
*   `--prefer-avconv`: Prefer avconv over ffmpeg for running the postprocessors.
*   `--prefer-ffmpeg`: Prefer ffmpeg over avconv for running the postprocessors (default).
*   `--ffmpeg-location PATH`: Location of the ffmpeg/avconv binary.
*   `--exec CMD`: Execute a command on the file after downloading and post-processing.
*   `--convert-subs FORMAT`: Convert the subtitles to other format.

## Configuration

youtube-dl can be configured using a configuration file, allowing you to set default options. On Linux and macOS, the system-wide file is at `/etc/youtube-dl.conf` and the user-specific file is `~/.config/youtube-dl/config`. On Windows, the user configuration is `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`.

Example Configuration (extract audio, don't copy mtime, use a proxy, save under a specific dir):

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

Use `--ignore-config` to disable the configuration file for a specific run, and `--config-location` to specify a custom configuration file.

### Authentication with .netrc file

Configure automatic credentials storage for extractors that support authentication using a [`.netrc` file](https://stackoverflow.com/tags/.netrc/info) and the `--netrc` option.

1.  Create a `.netrc` file in your `$HOME` directory, and restrict permissions:

```bash
touch $HOME/.netrc
chmod a-rwx,u+rw $HOME/.netrc
```

2.  Add credentials for an extractor (e.g., `youtube`) in the format:

```
machine <extractor> login <login> password <password>
```

For example:
```
machine youtube login myaccount@gmail.com password my_youtube_password
machine twitch login my_twitch_account_name password my_twitch_password
```
3. To activate authentication, pass `--netrc` to youtube-dl or place it in the [configuration file](#configuration).

## Output Template

The `-o` option controls the output filename. It can contain special sequences that are replaced with video metadata. See the original documentation in the README for a complete list of available keys.

*   `id`: Video identifier
*   `title`: Video title
*   `url`: Video URL
*   `ext`: Video filename extension
*   ...and many more.

#### Output Template Examples

```bash
# Simple file name
$ youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc
youtube-dl test video ''_ä↭𝕐.mp4    # All kinds of weird characters

$ youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc --restrict-filenames
youtube-dl_test_video_.mp4          # A simple file name

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

The `-f` or `--format` option allows you to specify the desired video format. Use `--list-formats` or `-F` to see available formats for a video.

*   `best`: Best quality format represented by a single file with video and audio.
*   `worst`: Worst quality format represented by a single file with video and audio.
*   `bestvideo`: Best quality video-only format.
*   `worstvideo`: Worst quality video-only format.
*   `bestaudio`: Best quality audio only-format.
*   `worstaudio`: Worst quality audio only-format.

You can combine formats using `/` for preference (e.g., `-f 22/18`) and `,` for multiple formats.

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

## Video Selection

Use options like `--date`, `--datebefore`, and `--dateafter` to filter videos by upload date.

Examples:

```bash
# Download only the videos uploaded in the last 6 months
$ youtube-dl --dateafter now-6months

# Download only the videos uploaded on January 1, 1970
$ youtube-dl --date 19700101

$ # Download only the videos uploaded in the 200x decade
$ youtube-dl --dateafter 20000101 --datebefore 20091231
```

## FAQ

A list of commonly asked questions, including how to update youtube-dl, how to handle various errors, and more. See the original README for an extensive FAQ.

## Developer Instructions

For developers, the original README provides detailed instructions on how to build, test, and contribute to youtube-dl.

### Adding Support for a New Site

Follow these steps to add support for a new website:

1.  [Fork the repository](https://github.com/ytdl-org/youtube-dl/fork)
2.  Check out the source code.
3.  Create a new branch.
4.  Create an extractor file (e.g., `youtube_dl/extractor/yourextractor.py`).
5.  Add an import in [`youtube_dl/extractor/extractors.py`](https://github.com/ytdl-org/youtube-dl/blob/master/youtube_dl/extractor/extractors.py).
6.  Write the extractor code using a template, the testing framework, and coding conventions.
7.  Run tests.
8.  Commit and push your changes.
9.  [Create a pull request](https://help.github.com/articles/creating-a-pull-request).

### youtube-dl coding conventions

Adhere to the coding conventions detailed in the original README.

## Embedding youtube-dl

The program can be used in other Python applications. Instructions on how to import and embed youtube-dl are in the original README.

## Bugs

Report bugs and suggestions in the [issue tracker](https://github.com/ytdl-org/youtube-dl/issues) and follow the bug reporting instructions.

## Copyright

youtube-dl is released into the public domain by the copyright holders.