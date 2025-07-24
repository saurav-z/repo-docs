[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Your Go-To Tool for Downloading Videos 

**Download videos from YouTube and thousands of other sites with ease using youtube-dl!**

[Visit the youtube-dl GitHub repository for more information.](https://github.com/ytdl-org/youtube-dl)

## Key Features:

*   **Wide Site Support:** Download videos from YouTube, plus thousands of other video platforms. ([See Supported Sites](https://ytdl-org.github.io/youtube-dl/supportedsites.html))
*   **Format Selection:** Choose the video format and quality that best suits your needs.
*   **Playlist Download:** Download entire playlists or specific videos within playlists.
*   **Customizable Output:** Control filenames and output directories using flexible templates.
*   **Metadata Preservation:** Automatically save video descriptions, titles, and other metadata.
*   **Subtitle Support:** Download and embed subtitles in various formats.
*   **Cross-Platform:** Works on Windows, macOS, and Linux.
*   **Active Community:** Benefit from a well-maintained and regularly updated tool.

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
    *   [Opening a bug report or suggestion](#opening-a-bug-report-or-suggestion)
*   [Copyright](#copyright)

## Installation

Choose your operating system for installation instructions:

### UNIX (Linux, macOS, etc.)

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

1.  [Download the `.exe` file](https://yt-dl.org/latest/youtube-dl.exe).
2.  Place the file in a directory included in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) (e.g., `C:\Users\<Your Name>\AppData\Local\Programs\Python\Python3x\Scripts` or similar, or your `C:\bin` if you have that). **Do not** put it in `%SYSTEMROOT%\System32`.

### Using pip

```bash
sudo -H pip install --upgrade youtube-dl
```

### macOS (Homebrew)

```bash
brew install youtube-dl
```

### macOS (MacPorts)

```bash
sudo port install youtube-dl
```

**For advanced options, PGP signatures, or developer installation, refer to the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).**

## Description

`youtube-dl` is a versatile command-line tool designed to download videos from YouTube and many other video-hosting websites. It's written in Python and works on various operating systems. This tool allows you to download videos and, in many cases, customize the format, quality, and output location.

## Options

Run `youtube-dl -h` to see the full list of options. Here are some of the most useful ones:

```bash
youtube-dl [OPTIONS] URL [URL...]
```

*   `-h`, `--help`:  Print help text and exit.
*   `--version`: Print program version and exit.
*   `-U`, `--update`: Update youtube-dl to the latest version.
*   `-i`, `--ignore-errors`:  Continue downloading even if errors occur.
*   `--dump-user-agent`: Display the current browser identification.
*   `--list-extractors`: List all supported extractors.
*   `--extractor-descriptions`: Output descriptions of all supported extractors.
*   `--ignore-config`: Do not read configuration files.
*   `--config-location PATH`: Location of the configuration file.
*   `-q`, `--quiet`: Activate quiet mode.
*   `-s`, `--simulate`: Do not download the video, but simulate the process.
*   `-v`, `--verbose`: Print detailed debugging information.
*   `--no-warnings`: Ignore warnings.
*   `--no-color`: Do not emit color codes in output
*   `-a, --batch-file FILE`: File containing URLs to download ('-' for stdin), one URL per line.
*   `-f, --format FORMAT`: Video format code. (Use `-F` to see available formats).
*   `-o, --output TEMPLATE`: Output filename template, see the "OUTPUT TEMPLATE" for all the info.
*   `-x, --extract-audio`: Convert video files to audio-only files (requires ffmpeg/avconv).
*   `--write-sub`: Write subtitle file.
*   `--sub-lang LANGS`: Languages of the subtitles to download.

### Network Options

*   `--proxy URL`: Use a specified HTTP/HTTPS/SOCKS proxy.
*   `--socket-timeout SECONDS`: Time to wait before giving up, in seconds.
*   `--force-ipv4`: Make all connections via IPv4.
*   `--force-ipv6`: Make all connections via IPv6.

### Geo Restriction

*   `--geo-verification-proxy URL`: Use this proxy to verify the IP address for some geo-restricted sites.
*   `--geo-bypass`: Bypass geographic restriction via faking X-Forwarded-For HTTP header.
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
*   `--no-playlist`: Download only the video if the URL refers to a video and a playlist.
*   `--yes-playlist`: Download the playlist if the URL refers to a video and a playlist.
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
*   `--no-resize-buffer`: Do not automatically adjust the buffer size.
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

*   `--id`: Use only video ID in file name.
*   `-o, --output TEMPLATE`: Output filename template, see the "OUTPUT TEMPLATE" for all the info.
*   `--output-na-placeholder PLACEHOLDER`: Placeholder value for unavailable meta fields in output filename template (default is "NA").
*   `--autonumber-start NUMBER`: Specify the start value for %(autonumber)s (default is 1).
*   `--restrict-filenames`: Restrict filenames to only ASCII characters, and avoid "&" and spaces in filenames.
*   `-w, --no-overwrites`: Do not overwrite files.
*   `-c, --continue`: Force resume of partially downloaded files.
*   `--no-continue`: Do not resume partially downloaded files (restart from beginning).
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

### Thumbnail Options

*   `--write-thumbnail`: Write thumbnail image to disk.
*   `--write-all-thumbnails`: Write all thumbnail image formats to disk.
*   `--list-thumbnails`: Simulate and list all available thumbnail formats.

### Verbosity / Simulation Options

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
*   `--dump-pages`: Print downloaded pages encoded using base64 to debug problems (very verbose).
*   `--write-pages`: Write downloaded intermediary pages to files in the current directory to debug problems.
*   `--print-traffic`: Display sent and read HTTP traffic.
*   `-C, --call-home`: Contact the youtube-dl server for debugging.
*   `--no-call-home`: Do NOT contact the youtube-dl server for debugging.

### Workarounds

*   `--encoding ENCODING`: Force the specified encoding (experimental).
*   `--no-check-certificate`: Suppress HTTPS certificate validation.
*   `--prefer-insecure`: Use an unencrypted connection to retrieve information about the video.
*   `--user-agent UA`: Specify a custom user agent.
*   `--referer URL`: Specify a custom referer, use if the video access is restricted to one domain.
*   `--add-header FIELD:VALUE`: Specify a custom HTTP header and its value.
*   `--bidi-workaround`: Work around terminals that lack bidirectional text support.
*   `--sleep-interval SECONDS`: Number of seconds to sleep before each download.
*   `--max-sleep-interval SECONDS`: Upper bound of a range for randomized sleep before each download.

### Video Format Options

*   `-f, --format FORMAT`: Video format code. (Use `-F` to see available formats).
*   `--all-formats`: Download all available video formats.
*   `--prefer-free-formats`: Prefer free video formats unless a specific one is requested.
*   `-F, --list-formats`: List all available formats of requested videos.
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

*   `-u, --username USERNAME`: Login with this account ID.
*   `-p, --password PASSWORD`: Account password.
*   `-2, --twofactor TWOFACTOR`: Two-factor authentication code.
*   `-n, --netrc`: Use .netrc authentication data.
*   `--video-password PASSWORD`: Video password (vimeo, youku).

### Adobe Pass Options

*   `--ap-mso MSO`: Adobe Pass multiple-system operator (TV provider) identifier.
*   `--ap-username USERNAME`: Multiple-system operator account login.
*   `--ap-password PASSWORD`: Multiple-system operator account password.
*   `--ap-list-mso`: List all supported multiple-system operators.

### Post-processing Options

*   `-x, --extract-audio`: Convert video files to audio-only files (requires ffmpeg/avconv).
*   `--audio-format FORMAT`: Specify audio format: "best", "aac", "flac", "mp3", "m4a", "opus", "vorbis", or "wav".
*   `--audio-quality QUALITY`: Specify ffmpeg/avconv audio quality.
*   `--recode-video FORMAT`: Encode the video to another format.
*   `--postprocessor-args ARGS`: Give these arguments to the postprocessor.
*   `-k, --keep-video`: Keep the video file on disk after the post-processing.
*   `--no-post-overwrites`: Do not overwrite post-processed files.
*   `--embed-subs`: Embed subtitles in the video (only for mp4, webm and mkv videos).
*   `--embed-thumbnail`: Embed thumbnail in the audio as cover art.
*   `--add-metadata`: Write metadata to the video file.
*   `--metadata-from-title FORMAT`: Parse additional metadata like song title / artist from the video title.
*   `--xattrs`: Write metadata to the video file's xattrs.
*   `--fixup POLICY`: Automatically correct known faults of the file.
*   `--prefer-avconv`: Prefer avconv over ffmpeg for running the postprocessors.
*   `--prefer-ffmpeg`: Prefer ffmpeg over avconv for running the postprocessors (default).
*   `--ffmpeg-location PATH`: Location of the ffmpeg/avconv binary.
*   `--exec CMD`: Execute a command on the file after downloading and post-processing.
*   `--convert-subs FORMAT`: Convert the subtitles to other format (currently supported: srt|ass|vtt|lrc).

## Configuration

You can customize `youtube-dl`'s behavior by creating a configuration file. On Linux and macOS, this is located at `/etc/youtube-dl.conf` (system-wide) or `~/.config/youtube-dl/config` (user-specific). On Windows, it's in `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`. Add the options you want to use to the configuration file, one option per line.

For example, to always extract audio and save files to your `Movies` directory:

```
-x
-o ~/Movies/%(title)s.%(ext)s
```

Use `--ignore-config` to disable the configuration file.  Use `--config-location` to specify a custom config file.

### Authentication with .netrc file

You can configure automatic credentials storage for extractors that support authentication (by providing login and password with `--username` and `--password`) in order not to pass credentials as command line arguments on every youtube-dl execution and prevent tracking plain text passwords in the shell command history. You can achieve this using a [`.netrc` file](https://stackoverflow.com/tags/.netrc/info) on a per extractor basis. For that you will need to create a `.netrc` file in your `$HOME` and restrict permissions to read/write by only you:

```bash
touch $HOME/.netrc
chmod a-rwx,u+rw $HOME/.netrc
```

After that you can add credentials for an extractor in the following format, where *extractor* is the name of the extractor in lowercase:

```
machine <extractor> login <login> password <password>
```

For example:

```
machine youtube login myaccount@gmail.com password my_youtube_password
machine twitch login my_twitch_account_name password my_twitch_password
```

To activate authentication with the `.netrc` file you should pass `--netrc` to youtube-dl or place it in the [configuration file](#configuration).

On Windows you may also need to setup the `%HOME%` environment variable manually. For example:

```bash
set HOME=%USERPROFILE%
```

## Output Template

Customize the output filenames with the `-o` or `--output` option. This uses a template system based on [python string formatting operations](https://docs.python.org/2/library/stdtypes.html#string-formatting).

The default template is `%(title)s-%(id)s.%(ext)s`.

**Available Keys:**

*   `id`: Video ID.
*   `title`: Video title.
*   `url`: Video URL.
*   `ext`: Video file extension.
*   `alt_title`: A secondary title of the video.
*   `display_id`: An alternative identifier for the video.
*   `uploader`: Full name of the video uploader.
*   `license`: License name.
*   `creator`: The creator of the video.
*   `release_date`: The date (YYYYMMDD) when the video was released.
*   `timestamp`: UNIX timestamp.
*   `upload_date`: Video upload date (YYYYMMDD).
*   `uploader_id`: Nickname or id of the video uploader.
*   `channel`: Full name of the channel.
*   `channel_id`: Id of the channel.
*   `location`: Physical location where the video was filmed.
*   `duration`: Length of the video in seconds.
*   `view_count`: How many users have watched the video on the platform.
*   `like_count`: Number of positive ratings of the video.
*   `dislike_count`: Number of negative ratings of the video.
*   `repost_count`: Number of reposts of the video.
*   `average_rating`: Average rating given by users, the scale used depends on the webpage.
*   `comment_count`: Number of comments on the video.
*   `age_limit`: Age restriction for the video (years).
*   `is_live`: Whether this video is a live stream.
*   `start_time`: Time in seconds where the reproduction should start.
*   `end_time`: Time in seconds where the reproduction should end.
*   `format`: Human-readable description of the format.
*   `format_id`: Format code.
*   `format_note`: Additional info about the format.
*   `width`: Width of the video.
*   `height`: Height of the video.
*   `resolution`: Textual description of width and height.
*   `tbr`: Average bitrate of audio and video in KBit/s.
*   `abr`: Average audio bitrate in KBit/s.
*   `acodec`: Name of the audio codec in use.
*   `asr`: Audio sampling rate in Hertz.
*   `vbr`: Average video bitrate in KBit/s.
*   `fps`: Frame rate.
*   `vcodec`: Name of the video codec in use.
*   `container`: Name of the container format.
*   `filesize`: The number of bytes, if known in advance.
*   `filesize_approx`: An estimate for the number of bytes.
*   `protocol`: The protocol that will be used for the actual download.
*   `extractor`: Name of the extractor.
*   `extractor_key`: Key name of the extractor.
*   `epoch`: Unix epoch when creating the file.
*   `autonumber`: Number that will be increased with each download.
*   `playlist`: Name or id of the playlist that contains the video.
*   `playlist_index`: Index of the video in the playlist padded with leading zeros according to the total length of the playlist.
*   `playlist_id`: Playlist identifier.
*   `playlist_title`: Playlist title.
*   `playlist_uploader`: Full name of the playlist uploader.
*   `playlist_uploader_id`: Nickname or id of the playlist uploader.
*   `chapter`: Name or title of the chapter the video belongs to.
*   `chapter_number`: Number of the chapter the video belongs to.
*   `chapter_id`: Id of the chapter the video belongs to.
*   `series`: Title of the series or programme the video episode belongs to.
*   `season`: Title of the season the video episode belongs to.
*   `season_number`: Number of the season the video episode belongs to.
*   `season_id`: Id of the season the video episode belongs to.
*   `episode`: Title of the video episode.
*   `episode_number`: Number of the video episode within a season.
*   `episode_id`: Id of the video episode.
*   `track`: Title of the track.
*   `track_number`: Number of the track within an album or a disc.
*   `track_id`: Id of the track.
*   `artist`: Artist(s) of the track.
*   `genre`: Genre(s) of the track.
*   `album`: Title of the album the track belongs to.
*   `album_type`: Type of the album.
*   `album_artist`: List of all artists appeared on the album.
*   `disc_number`: Number of the disc or other physical medium the track belongs to.
*   `release_year`: Year (YYYY) when the album was released.

Use `%%` to represent a literal `%` in the template. Use `-o -` to output to stdout.  To use percent literals in an output template use `%%`. To output to stdout use `-o -`.

### Output Template Examples

```bash
# Basic
youtube-dl -o '%(title)s.%(ext)s' "https://some/video"

# Output to a playlist directory
youtube-dl -o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' https://www.youtube.com/playlist?list=PLwiyx1dc3P2JR9N8gQaQN_BCvlSlap7re

# Downloading YouTube playlists videos in separate directory indexed by video order in a playlist
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

Use the `-f` or `--format` option to specify the video format you want to download.  Use `-F` or `--list-formats` to see available formats.

```bash
youtube-dl -f FORMAT URL
```

*   **Default:**  `youtube-dl` downloads the best quality available by default.
*   **Specific Format:**  Use format codes (e.g., `-f 22`).
*   **File Extension:**  Download the best format for a specific extension (e.g., `-f mp4`).
*   **Special Names:**
    *   `best`: Best quality with both video and audio.
    *   `worst`: Worst quality with both video and audio.
    *   `bestvideo`: Best quality video only.
    *   `worstvideo`: Worst quality video only.
    *   `bestaudio`: Best quality audio only.
    *   `worstaudio`: Worst quality audio only.
*   **Prioritized Formats:** Use `/` to specify preferred formats (e.g., `-f 22/18/17`).
*   **Multiple Formats:** Use `,` to download multiple formats (e.g., `-f 22,17,18`).
*   **Filtering:**  Filter formats based on properties with brackets (e.g., `-f "best[height=720]"`).

    The following numeric meta fields can be used with comparisons `<`, `<=`, `>`, `>=`, `=` (equals), `!=` (not equals):

    *   `filesize`: The number of bytes, if known in advance
    *   `width`: Width of the video, if known
    *   `height`: Height of the video, if known
    *   `tbr`: Average bitrate of audio and video in KBit/s
    *   `abr`: Average audio bitrate in KBit/s
    *   `vbr`: Average video bitrate in KBit/s
    *   `asr`: Audio sampling rate in Hertz
    *   `fps`: Frame rate

    Also filtering work for comparisons `=` (equals), `^=` (starts with), `$=` (ends with), `*=` (contains) and following string meta fields:

    *   `ext`: File extension
    *   `acodec`: Name of the audio codec in use
    *   `vcodec`: Name of the video codec in use
    *   `container`: Name of the container format
    *   `protocol`: The protocol that will be used for the actual download, lower-case (`http`, `https`, `rtsp`, `rtmp`, `rtmpe`, `mms`, `f4m`, `ism`, `http_dash_segments`, `m3u8`, or `m3u8_native`)
    *   `format_id`: A short description of the format
    *   `language`: Language code

    Any string comparison may be prefixed with negation `!` in order to produce an opposite comparison, e.g. `!*=` (does not contain).
*   **Merging:** Use `+` to merge video and audio (requires ffmpeg/avconv), e.g. `-f bestvideo+bestaudio`.

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

*   **Download by Date:**  Use `--date`, `--datebefore`, or `--dateafter` to download videos within a specific date range.
*   **Absolute Dates:**  Use dates in the format `YYYYMMDD`.
*   **Relative Dates:** Use dates like `now-6months` (download videos uploaded in the last 6 months).

## FAQ

### How do I update youtube-dl?

Run `youtube-dl -U`.  If that doesn't work, consult the [manual installation instructions](#installation).

### youtube-dl is extremely slow to start on Windows

Add a file exclusion for `youtube-dl.exe` in Windows Defender settings.

### I'm getting an error `Unable to extract OpenGraph title` on YouTube playlists

Update to the latest version.  See [How do I update youtube-dl?](#how-do-i-update-youtube-dl)

### I'm getting an error when trying to use output template: `error: using output template conflicts with using title, video ID or auto number`

Make sure you are not using `-o` with conflicting options (like `-t`, `--title`, `--id`, `-A`, or `--auto-number`).

### Do I always have to pass `-citw`?

No. youtube-dl defaults to the best options.

### Can you please put the `-b` option back?

The `-b` option is unnecessary because youtube-dl now downloads the highest quality available by default.

### I get HTTP error 402 when trying to download a video. What's this?

YouTube may require a CAPTCHA. Try solving the CAPTCHA in a web browser and then use the `--cookies` option.

### Do I need any other programs?

`youtube-dl` works on its own for most sites. However, `avconv`/`ffmpeg` are needed for converting videos.  `rtmpdump` is needed for RTMP videos. `mplayer` or `mpv` are needed for MMS and RTSP videos.

### I have downloaded a video but how can I play it?

Use a video player like [mpv](https://mpv.io/), [vlc](https://www.videolan.org/) or [mplayer](https://www.mplayerhq.hu/).

### I extracted a video URL with `-g`, but it does not play on another machine / in my web browser.

Make sure your downloader supports the protocols.