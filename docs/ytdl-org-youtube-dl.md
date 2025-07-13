[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Download Videos from YouTube and Beyond

**youtube-dl is a powerful command-line tool that allows you to download videos from YouTube and thousands of other video and music streaming platforms, offering extensive customization and control.**  ([Visit the original repository](https://github.com/ytdl-org/youtube-dl))

## Key Features

*   **Broad Platform Support:** Download videos from YouTube, plus 1000s of other sites.
*   **Format Selection:** Choose the video and audio quality that suits your needs.
*   **Playlist & Channel Downloads:** Download entire playlists, channels, or specific videos.
*   **Customizable Output:**  Control file names, formats, and directory structures.
*   **Metadata Extraction:**  Retrieve video titles, descriptions, and other metadata.
*   **Subtitle Support:** Download and convert subtitles in various formats.
*   **Post-Processing:** Convert and process downloaded videos with ffmpeg/avconv.
*   **Cross-Platform:** Works on Linux, macOS, Windows, and other operating systems.
*   **Easy to Update:** Keeps itself up-to-date with a simple command.

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
    *   [.netrc Authentication](#authentication-with-netrc-file)
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

youtube-dl is easy to install on any operating system.

**UNIX (Linux, macOS, etc.):**

1.  **Using `curl`:**

    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```

2.  **Using `wget` (if you don't have `curl`):**

    ```bash
    sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```

**Windows:**

1.  [Download the `.exe` file](https://yt-dl.org/latest/youtube-dl.exe).
2.  Place it in any directory included in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29), *except* for `%SYSTEMROOT%\System32`.

**Using `pip`:**

    ```bash
    sudo -H pip install --upgrade youtube-dl
    ```

**macOS (using Homebrew):**

    ```bash
    brew install youtube-dl
    ```

**macOS (using MacPorts):**

    ```bash
    sudo port install youtube-dl
    ```

**Alternative Installation:** Refer to the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html) for further options, including PGP signatures or refer to the [developer instructions](#developer-instructions).

## Description

youtube-dl is a versatile command-line tool designed to download videos from YouTube and a multitude of other websites. It's written in Python and works across various platforms, including Unix-like systems, Windows, and macOS.  It's free to use and modify under the public domain.

## Options

youtube-dl offers a comprehensive set of options to customize your downloads. Here are some of the key categories:

*   `-h`, `--help`: Print help text and exit.
*   `--version`: Print program version and exit.
*   `-U`, `--update`: Update this program to latest version. Make sure that you have sufficient permissions (run with sudo if needed)
*   `-i`, `--ignore-errors`: Continue on download errors, for example to skip unavailable videos in a playlist
*   `--abort-on-error`: Abort downloading of further videos (in the playlist or the command line) if an error occurs
*   `--dump-user-agent`: Display the current browser identification
*   `--list-extractors`: List all supported extractors
*   `--extractor-descriptions`: Output descriptions of all supported extractors
*   `--force-generic-extractor`: Force extraction to use the generic extractor
*   `--default-search PREFIX`: Use this prefix for unqualified URLs. For example "gvsearch2:" downloads two videos from google videos for youtube-dl "large apple". Use the value "auto" to let youtube-dl guess ("auto_warning" to emit a warning when guessing). "error" just throws an error. The default value "fixup_error" repairs broken URLs, but emits an error if this is not possible instead of searching.
*   `--ignore-config`: Do not read configuration files. When given in the global configuration file /etc/youtube-dl.conf: Do not read the user configuration in ~/.config/youtube-dl/config (%APPDATA%/youtube-dl/config.txt on Windows)
*   `--config-location PATH`: Location of the configuration file; either the path to the config or its containing directory.
*   `--flat-playlist`: Do not extract the videos of a playlist, only list them.
*   `--mark-watched`: Mark videos watched (YouTube only)
*   `--no-mark-watched`: Do not mark videos watched (YouTube only)
*   `--no-color`: Do not emit color codes in output

### Network Options

*   `--proxy URL`: Use the specified HTTP/HTTPS/SOCKS proxy.
*   `--socket-timeout SECONDS`: Time to wait before giving up, in seconds.
*   `--source-address IP`: Client-side IP address to bind to.
*   `-4`, `--force-ipv4`: Make all connections via IPv4.
*   `-6`, `--force-ipv6`: Make all connections via IPv6.

### Geo Restriction

*   `--geo-verification-proxy URL`: Use this proxy to verify the IP address for some geo-restricted sites.
*   `--geo-bypass`: Bypass geographic restriction via faking X-Forwarded-For HTTP header.
*   `--no-geo-bypass`: Do not bypass geographic restriction via faking X-Forwarded-For HTTP header.
*   `--geo-bypass-country CODE`: Force bypass geographic restriction with explicitly provided two-letter ISO 3166-2 country code.
*   `--geo-bypass-ip-block IP_BLOCK`: Force bypass geographic restriction with explicitly provided IP block in CIDR notation.

### Video Selection

*   `--playlist-start NUMBER`: Playlist video to start at (default is 1).
*   `--playlist-end NUMBER`: Playlist video to end at (default is last).
*   `--playlist-items ITEM_SPEC`: Playlist video items to download.
*   `--match-title REGEX`: Download only matching titles (regex or caseless sub-string).
*   `--reject-title REGEX`: Skip download for matching titles (regex or caseless sub-string).
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
*   `--download-archive FILE`: Download only videos not listed in the archive file. Record the IDs of all downloaded videos in it.
*   `--include-ads`: Download advertisements as well (experimental).

### Download Options

*   `-r`, `--limit-rate RATE`: Maximum download rate in bytes per second (e.g. 50K or 4.2M).
*   `-R`, `--retries RETRIES`: Number of retries (default is 10), or "infinite".
*   `--fragment-retries RETRIES`: Number of retries for a fragment (default is 10), or "infinite" (DASH, hlsnative and ISM).
*   `--skip-unavailable-fragments`: Skip unavailable fragments (DASH, hlsnative and ISM).
*   `--abort-on-unavailable-fragment`: Abort downloading when some fragment is not available.
*   `--keep-fragments`: Keep downloaded fragments on disk after downloading is finished; fragments are erased by default.
*   `--buffer-size SIZE`: Size of download buffer (e.g. 1024 or 16K) (default is 1024).
*   `--no-resize-buffer`: Do not automatically adjust the buffer size. By default, the buffer size is automatically resized from an initial value of SIZE.
*   `--http-chunk-size SIZE`: Size of a chunk for chunk-based HTTP downloading (e.g. 10485760 or 10M) (default is disabled).
*   `--playlist-reverse`: Download playlist videos in reverse order.
*   `--playlist-random`: Download playlist videos in random order.
*   `--xattr-set-filesize`: Set file xattribute ytdl.filesize with expected file size.
*   `--hls-prefer-native`: Use the native HLS downloader instead of ffmpeg.
*   `--hls-prefer-ffmpeg`: Use ffmpeg instead of the native HLS downloader.
*   `--hls-use-mpegts`: Use the mpegts container for HLS videos, allowing to play the video while downloading (some players may not be able to play it).
*   `--external-downloader COMMAND`: Use the specified external downloader.
*   `--external-downloader-args ARGS`: Give these arguments to the external downloader.

### Filesystem Options

*   `-a`, `--batch-file FILE`: File containing URLs to download ('-' for stdin), one URL per line.
*   `--id`: Use only video ID in file name.
*   `-o`, `--output TEMPLATE`: Output filename template, see the "OUTPUT TEMPLATE" for all the info.
*   `--output-na-placeholder PLACEHOLDER`: Placeholder value for unavailable meta fields in output filename template (default is "NA").
*   `--autonumber-start NUMBER`: Specify the start value for %(autonumber)s (default is 1).
*   `--restrict-filenames`: Restrict filenames to only ASCII characters, and avoid "&" and spaces in filenames.
*   `-w`, `--no-overwrites`: Do not overwrite files.
*   `-c`, `--continue`: Force resume of partially downloaded files.
*   `--no-continue`: Do not resume partially downloaded files (restart from beginning).
*   `--no-part`: Do not use .part files - write directly into output file.
*   `--no-mtime`: Do not use the Last-modified header to set the file modification time.
*   `--write-description`: Write video description to a .description file.
*   `--write-info-json`: Write video metadata to a .info.json file.
*   `--write-annotations`: Write video annotations to a .annotations.xml file.
*   `--load-info-json FILE`: JSON file containing the video information (created with the "--write-info-json" option).
*   `--cookies FILE`: File to read cookies from and dump cookie jar in.
*   `--cache-dir DIR`: Location in the filesystem where youtube-dl can store some downloaded information permanently.
*   `--no-cache-dir`: Disable filesystem caching.
*   `--rm-cache-dir`: Delete all filesystem cache files.

### Thumbnail Options

*   `--write-thumbnail`: Write thumbnail image to disk.
*   `--write-all-thumbnails`: Write all thumbnail image formats to disk.
*   `--list-thumbnails`: Simulate and list all available thumbnail formats.

### Verbosity / Simulation Options

*   `-q`, `--quiet`: Activate quiet mode.
*   `--no-warnings`: Ignore warnings.
*   `-s`, `--simulate`: Do not download the video and do not write anything to disk.
*   `--skip-download`: Do not download the video.
*   `-g`, `--get-url`: Simulate, quiet but print URL.
*   `-e`, `--get-title`: Simulate, quiet but print title.
*   `--get-id`: Simulate, quiet but print id.
*   `--get-thumbnail`: Simulate, quiet but print thumbnail URL.
*   `--get-description`: Simulate, quiet but print video description.
*   `--get-duration`: Simulate, quiet but print video length.
*   `--get-filename`: Simulate, quiet but print output filename.
*   `--get-format`: Simulate, quiet but print output format.
*   `-j`, `--dump-json`: Simulate, quiet but print JSON information.
*   `-J`, `--dump-single-json`: Simulate, quiet but print JSON information for each command-line argument.
*   `--print-json`: Be quiet and print the video information as JSON (video is still being downloaded).
*   `--newline`: Output progress bar as new lines.
*   `--no-progress`: Do not print progress bar.
*   `--console-title`: Display progress in console titlebar.
*   `-v`, `--verbose`: Print various debugging information.
*   `--dump-pages`: Print downloaded pages encoded using base64 to debug problems (very verbose).
*   `--write-pages`: Write downloaded intermediary pages to files in the current directory to debug problems.
*   `--print-traffic`: Display sent and read HTTP traffic.
*   `-C`, `--call-home`: Contact the youtube-dl server for debugging.
*   `--no-call-home`: Do NOT contact the youtube-dl server for debugging.

### Workarounds

*   `--encoding ENCODING`: Force the specified encoding (experimental).
*   `--no-check-certificate`: Suppress HTTPS certificate validation.
*   `--prefer-insecure`: Use an unencrypted connection to retrieve information about the video.
*   `--user-agent UA`: Specify a custom user agent.
*   `--referer URL`: Specify a custom referer, use if the video access is restricted to one domain.
*   `--add-header FIELD:VALUE`: Specify a custom HTTP header and its value, separated by a colon ':'.
*   `--bidi-workaround`: Work around terminals that lack bidirectional text support.
*   `--sleep-interval SECONDS`: Number of seconds to sleep before each download.
*   `--max-sleep-interval SECONDS`: Upper bound of a range for randomized sleep before each download.

### Video Format Options

*   `-f`, `--format FORMAT`: Video format code, see the "FORMAT SELECTION" for all the info.
*   `--all-formats`: Download all available video formats.
*   `--prefer-free-formats`: Prefer free video formats unless a specific one is requested.
*   `-F`, `--list-formats`: List all available formats of requested videos.
*   `--youtube-skip-dash-manifest`: Do not download the DASH manifests and related data on YouTube videos.
*   `--merge-output-format FORMAT`: If a merge is required (e.g. bestvideo+bestaudio), output to given container format.

### Subtitle Options

*   `--write-sub`: Write subtitle file.
*   `--write-auto-sub`: Write automatically generated subtitle file (YouTube only).
*   `--all-subs`: Download all the available subtitles of the video.
*   `--list-subs`: List all available subtitles for the video.
*   `--sub-format FORMAT`: Subtitle format.
*   `--sub-lang LANGS`: Languages of the subtitles to download.

### Authentication Options

*   `-u`, `--username USERNAME`: Login with this account ID.
*   `-p`, `--password PASSWORD`: Account password.
*   `-2`, `--twofactor TWOFACTOR`: Two-factor authentication code.
*   `-n`, `--netrc`: Use .netrc authentication data.
*   `--video-password PASSWORD`: Video password (vimeo, youku).

### Adobe Pass Options

*   `--ap-mso MSO`: Adobe Pass multiple-system operator (TV provider) identifier.
*   `--ap-username USERNAME`: Multiple-system operator account login.
*   `--ap-password PASSWORD`: Multiple-system operator account password.
*   `--ap-list-mso`: List all supported multiple-system operators.

### Post-processing Options

*   `-x`, `--extract-audio`: Convert video files to audio-only files (requires ffmpeg/avconv and ffprobe/avprobe).
*   `--audio-format FORMAT`: Specify audio format.
*   `--audio-quality QUALITY`: Specify ffmpeg/avconv audio quality.
*   `--recode-video FORMAT`: Encode the video to another format if necessary.
*   `--postprocessor-args ARGS`: Give these arguments to the postprocessor.
*   `-k`, `--keep-video`: Keep the video file on disk after the post-processing; the video is erased by default.
*   `--no-post-overwrites`: Do not overwrite post-processed files; the post-processed files are overwritten by default.
*   `--embed-subs`: Embed subtitles in the video (only for mp4, webm and mkv videos).
*   `--embed-thumbnail`: Embed thumbnail in the audio as cover art.
*   `--add-metadata`: Write metadata to the video file.
*   `--metadata-from-title FORMAT`: Parse additional metadata like song title / artist from the video title.
*   `--xattrs`: Write metadata to the video file's xattrs (using dublin core and xdg standards).
*   `--fixup POLICY`: Automatically correct known faults of the file.
*   `--prefer-avconv`: Prefer avconv over ffmpeg for running the postprocessors.
*   `--prefer-ffmpeg`: Prefer ffmpeg over avconv for running the postprocessors (default).
*   `--ffmpeg-location PATH`: Location of the ffmpeg/avconv binary.
*   `--exec CMD`: Execute a command on the file after downloading and post-processing.
*   `--convert-subs FORMAT`: Convert the subtitles to other format (currently supported: srt|ass|vtt|lrc).

## Configuration

youtube-dl can be configured through a configuration file.

**Linux/macOS:**

*   System-wide: `/etc/youtube-dl.conf`
*   User-specific: `~/.config/youtube-dl/config`

**Windows:**

*   User-specific: `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`

To disable the configuration file for a particular run use `--ignore-config`.
To use a custom configuration file for a particular run use `--config-location`.

### .netrc Authentication

youtube-dl supports automatic credential storage using a `.netrc` file for extractors that support authentication.  You should pass `--netrc` to youtube-dl or place it in the [configuration file](#configuration).

1.  Create a `.netrc` file (e.g., `touch $HOME/.netrc`) and restrict permissions (e.g., `chmod a-rwx,u+rw $HOME/.netrc`).
2.  Add credentials for each extractor:

    ```
    machine <extractor> login <login> password <password>
    ```

    Example:

    ```
    machine youtube login myaccount@gmail.com password my_youtube_password
    machine twitch login my_twitch_account_name password my_twitch_password
    ```

    On Windows, you may also need to setup the `%HOME%` environment variable manually.

    ```
    set HOME=%USERPROFILE%
    ```

## Output Template

The `-o` option allows you to customize output filenames.

*   **`%(id)s`**:  Video identifier.
*   **`%(title)s`**: Video title.
*   **`%(url)s`**: Video URL.
*   **`%(ext)s`**: Video filename extension.
*   **`%(playlist)s`**: Playlist name.
*   **`%(playlist_index)s`**: Video index in the playlist.
*   ...and many more metadata fields.

Use `%` followed by a name within parentheses, such as `%(title)s` or `%(id)s`. Numeric sequences can be formatted (e.g., `%(view_count)05d` to pad with leading zeros). Use `%%` for a literal percent sign.

**Default Template:** `%(title)s-%(id)s.%(ext)s`

### Output Template Examples

```bash
# Simple example
youtube-dl -o '%(title)s.%(ext)s' "https://www.youtube.com/watch?v=BaW_jenozKc"

# Download YouTube playlist videos in a separate directory, indexed by video order
youtube-dl -o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' https://www.youtube.com/playlist?list=PLwiyx1dc3P2JR9N8gQaQN_BCvlSlap7re

# Stream video to stdout (for use with a player)
youtube-dl -o - "https://www.youtube.com/watch?v=BaW_jenozKc" | vlc -
```

## Format Selection

The `-f` or `--format` option allows you to select the video and audio quality.

*   **`best`**: Best available quality (single file).
*   **`worst`**: Worst available quality (single file).
*   **`bestvideo`**: Best video-only format.
*   **`bestaudio`**: Best audio-only format.
*   **`<format_code>`**:  Specific format code (use `-F` to list available codes).
*   **`<extension>`**: Best quality format of the given extension served as a single file (e.g., `-f mp4`).
*   **`<format_code1>/<format_code2>`**: Preferred format selection.
*   **`<format_code1>,<format_code2>`**: Download multiple formats.
*   **`bestvideo+bestaudio`**: Merge best video and audio formats.
*   **Filtering**: `-f "best[height=720]"` (or `-f "[filesize>10M]"`).

**Default:**  `bestvideo+bestaudio/best` (when ffmpeg/avconv is present). Otherwise `best`.

### Format Selection Examples

```bash
# Download best mp4 format or fallback to best if no mp4 is available
youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'

# Download best format up to 480p
youtube-dl -f 'bestvideo[height<=480]+bestaudio/best[height<=480]'

# Download best video-only format, no larger than 50 MB
youtube-dl -f 'best[filesize<50M]'

# Download best video format and the best audio format without merging them
youtube-dl -f 'bestvideo,bestaudio' -o '%(title)s.f%(format_id)s.%(ext)s'
```

## Video Selection

You can filter videos based on:

*   `--date DATE`: Download videos uploaded on a specific date (YYYYMMDD).
*   `--datebefore DATE`: Download videos uploaded on or before a date.
*   `--dateafter DATE`: Download videos uploaded on or after a date.

## FAQ

*   **How do I update youtube-dl?**  Run `youtube-dl -U` (or `sudo youtube-dl -U` on Linux).
*   **youtube-dl is slow on Windows:** Add a file exclusion for `youtube-dl.exe` in Windows Defender.
*   **Errors with `Unable to extract OpenGraph title` on YouTube playlists:** Update youtube-dl.
*   **Output template errors:** Ensure you aren't using `-o` with `-t`, `--title`, `--id`, `-A`, or `--auto-number`.
*   **Do I always need `-citw`?**  No, `-i` is often the most useful.
*   **`-b` option is gone:** `youtube-dl` defaults to the highest available quality. Use `-f` if needed.
*   **HTTP error 402:**  Solve the CAPTCHA in your browser, and then use `--cookies`.
*   **Need other programs?**  ffmpeg/avconv for conversion, rtmpdump for RTMP, mplayer/mpv for MMS/RTSP.
*   **Playing downloaded videos:** Use any video player (e.g. mpv, vlc, mplayer).
*   **Video URL errors:** Use `--cookies`, a common user agent, or IPv6.
*   **Older errors like `fmt_url_map`:** Update youtube-dl.
*   **Errors like `ERROR: unable to download video`:** Update youtube-dl.
*   **Ampersand in URL errors:** Enclose the URL in quotes or escape the ampersands.
*   **ExtractorError: Could not find JS function u'OF'**: Update youtube-dl.
*   **HTTP Error 429/402**: May be due to overuse. Try using a proxy ( `--proxy`) or passing cookies.

## Developer Instructions

Developers can build and test youtube-dl.

*   **To run:** `python -m youtube_dl`
*   **To run tests:** `python -m unittest discover` or `python test/test_download.py`
*   **Dependencies:** Python, make (GNU make), pandoc, zip, nosetests

### Adding Support for a New Site

1.  [Fork the repository](https://github.com/ytdl-org/youtube-dl/fork).
2.  Clone your fork.
3.  Create a branch (e.g., `git checkout -b yourextractor`).
4.  Create a new extractor file in `youtube_dl/extractor/yourextractor.py`.
5.  Import the new extractor in [`youtube_dl/extractor/extractors.py`](https://github.com/ytdl-org/youtube-dl/blob/master/youtube_dl/extractor/extractors.py).
6.  Run tests with a command like, `python test/test_download.py TestDownload.test_YourExtractor`.
7.  Make sure your code follows [youtube-dl coding conventions](#youtube-dl-coding-conventions). Use flake8 for code style.
8.  When the tests pass, commit, push, and [create a pull request](https://help.github.com/articles/creating-a-pull-request).

### youtube-dl coding conventions

Follow the guide provided to create reliable, future-proof extractors.
*   Ensure to use `int_or_none`, `float_or_none`, `url_or_none`, `traverse_obj`, and other utility functions.
*   Use methods to get information from multiple sources.
*   Do not extract data that might not be available.
*   Write relaxed and flexible regular expressions.

## Embedding youtube-dl

You can embed youtube-dl into your Python programs:

```python
from __future__ import unicode_literals
import youtube_dl

ydl_opts = {}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=BaW_jenozKc'])
```

## Bugs

Report bugs and suggestions in the [issue tracker](https://github.com/ytdl-org/youtube-dl/issues). Include full output of `youtube-dl -v YOUR_URL_HERE`.  For discussions, join [#youtube-dl](irc://chat.freenode.net/#youtube-dl) on freenode.

## Copyright

youtube-dl is in the public domain.  This README file is also released into the public domain.