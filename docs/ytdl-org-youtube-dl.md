[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# YouTube-dl: The Ultimate Command-Line Video Downloader

**Download videos from YouTube and many other sites with ease using [youtube-dl](https://github.com/ytdl-org/youtube-dl).** This powerful command-line tool supports a wide range of video platforms and offers extensive customization options for your downloading needs.

## Key Features

*   **Wide Site Support:** Download from YouTube, Vimeo, and hundreds of other video sites.
*   **Format Selection:** Choose your preferred video quality, resolution, and format.
*   **Playlist & Channel Downloads:** Easily download entire playlists or all videos from a channel.
*   **Customizable Output:** Control file names, directories, and metadata.
*   **Cross-Platform Compatibility:** Works on Windows, macOS, and Linux.
*   **Active Community:** Benefit from a large and active community of users and developers.

## Installation

Follow these simple steps to get started:

*   **UNIX (Linux, macOS, etc.):**
    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```
    or use `wget`:
    ```bash
    sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```
*   **Windows:**
    Download the executable from [here](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory on your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) *excluding* `%SYSTEMROOT%\System32`.
*   **Using pip:**
    ```bash
    sudo -H pip install --upgrade youtube-dl
    ```
*   **macOS (using Homebrew or MacPorts):**
    ```bash
    brew install youtube-dl
    ```
    or
    ```bash
    sudo port install youtube-dl
    ```

For advanced installation options, including PGP signatures, and to download the latest releases, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## Getting Started

To download a video, simply use the command:
```bash
youtube-dl [OPTIONS] URL [URL...]
```
Replace `[OPTIONS]` with any of the available options and `URL` with the video link.  For example:

```bash
youtube-dl https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

## Key Options

*   `-U, --update`: Update youtube-dl to the latest version.
*   `-f, --format FORMAT`:  Specify the video format (e.g., `-f mp4` for MP4 format or `-f best` for best quality).  Use `-F` to list available formats.
*   `-o, --output TEMPLATE`: Customize the output filename and directory (see the "OUTPUT TEMPLATE" section for details).
*   `--list-formats`: List all available formats for a video.
*   `--help`: Display the help message and available options.

For more detailed information on options, refer to the [OPTIONS](#options) section below.

## Detailed Guide

*   [DESCRIPTION](#description)
*   [OPTIONS](#options)
*   [CONFIGURATION](#configuration)
*   [OUTPUT TEMPLATE](#output-template)
*   [FORMAT SELECTION](#format-selection)
*   [VIDEO SELECTION](#video-selection)
*   [FAQ](#faq)
*   [EMBEDDING YOUTUBE-DL](#embedding-youtube-dl)
*   [BUGS](#bugs)
*   [DEVELOPER INSTRUCTIONS](#developer-instructions)
*   [COPYRIGHT](#copyright)

## DESCRIPTION

**youtube-dl** is a versatile command-line program designed for downloading videos from YouTube.com and other video platforms. It is written in Python and is platform-independent, operating seamlessly on Unix-like systems, Windows, and macOS.  It's released into the public domain, enabling you to freely modify, redistribute, and use it as you wish.

```bash
youtube-dl [OPTIONS] URL [URL...]
```

## OPTIONS

Detailed options for controlling youtube-dl's behavior.

*   `-h, --help`: Print this help text and exit
*   `--version`: Print program version and exit
*   `-U, --update`: Update this program to latest version.
*   `-i, --ignore-errors`: Continue on download errors
*   `--abort-on-error`: Abort downloading of further videos if an error occurs
*   `--dump-user-agent`: Display the current browser identification
*   `--list-extractors`: List all supported extractors
*   `--extractor-descriptions`: Output descriptions of all supported extractors
*   `--force-generic-extractor`: Force extraction to use the generic extractor
*   `--default-search PREFIX`: Use this prefix for unqualified URLs.
*   `--ignore-config`: Do not read configuration files.
*   `--config-location PATH`: Location of the configuration file;
*   `--flat-playlist`: Do not extract the videos of a playlist, only list them.
*   `--mark-watched`: Mark videos watched (YouTube only)
*   `--no-mark-watched`: Do not mark videos watched (YouTube only)
*   `--no-color`: Do not emit color codes in output

### Network Options:

*   `--proxy URL`: Use the specified HTTP/HTTPS/SOCKS proxy.
*   `--socket-timeout SECONDS`: Time to wait before giving up, in seconds
*   `--source-address IP`: Client-side IP address to bind to
*   `-4, --force-ipv4`: Make all connections via IPv4
*   `-6, --force-ipv6`: Make all connections via IPv6

### Geo Restriction:

*   `--geo-verification-proxy URL`: Use this proxy to verify the IP address for some geo-restricted sites.
*   `--geo-bypass`: Bypass geographic restriction via faking X-Forwarded-For HTTP header
*   `--no-geo-bypass`: Do not bypass geographic restriction via faking X-Forwarded-For HTTP header
*   `--geo-bypass-country CODE`: Force bypass geographic restriction with explicitly provided two-letter ISO 3166-2 country code
*   `--geo-bypass-ip-block IP_BLOCK`: Force bypass geographic restriction with explicitly provided IP block in CIDR notation

### Video Selection:

*   `--playlist-start NUMBER`: Playlist video to start at (default is 1)
*   `--playlist-end NUMBER`: Playlist video to end at (default is last)
*   `--playlist-items ITEM_SPEC`: Playlist video items to download.
*   `--match-title REGEX`: Download only matching titles (regex or caseless sub-string)
*   `--reject-title REGEX`: Skip download for matching titles (regex or caseless sub-string)
*   `--max-downloads NUMBER`: Abort after downloading NUMBER files
*   `--min-filesize SIZE`: Do not download any videos smaller than SIZE (e.g. 50k or 44.6m)
*   `--max-filesize SIZE`: Do not download any videos larger than SIZE (e.g. 50k or 44.6m)
*   `--date DATE`: Download only videos uploaded in this date
*   `--datebefore DATE`: Download only videos uploaded on or before this date (i.e. inclusive)
*   `--dateafter DATE`: Download only videos uploaded on or after this date (i.e. inclusive)
*   `--min-views COUNT`: Do not download any videos with less than COUNT views
*   `--max-views COUNT`: Do not download any videos with more than COUNT views
*   `--match-filter FILTER`: Generic video filter.
*   `--no-playlist`: Download only the video, if the URL refers to a video and a playlist.
*   `--yes-playlist`: Download the playlist, if the URL refers to a video and a playlist.
*   `--age-limit YEARS`: Download only videos suitable for the given age
*   `--download-archive FILE`: Download only videos not listed in the archive file.
*   `--include-ads`: Download advertisements as well (experimental)

### Download Options:

*   `-r, --limit-rate RATE`: Maximum download rate in bytes per second (e.g. 50K or 4.2M)
*   `-R, --retries RETRIES`: Number of retries (default is 10), or "infinite".
*   `--fragment-retries RETRIES`: Number of retries for a fragment (default is 10), or "infinite" (DASH, hlsnative and ISM)
*   `--skip-unavailable-fragments`: Skip unavailable fragments (DASH, hlsnative and ISM)
*   `--abort-on-unavailable-fragment`: Abort downloading when some fragment is not available
*   `--keep-fragments`: Keep downloaded fragments on disk after downloading is finished; fragments are erased by default
*   `--buffer-size SIZE`: Size of download buffer (e.g. 1024 or 16K) (default is 1024)
*   `--no-resize-buffer`: Do not automatically adjust the buffer size.
*   `--http-chunk-size SIZE`: Size of a chunk for chunk-based HTTP downloading (e.g. 10485760 or 10M) (default is disabled).
*   `--playlist-reverse`: Download playlist videos in reverse order
*   `--playlist-random`: Download playlist videos in random order
*   `--xattr-set-filesize`: Set file xattribute ytdl.filesize with expected file size
*   `--hls-prefer-native`: Use the native HLS downloader instead of ffmpeg
*   `--hls-prefer-ffmpeg`: Use ffmpeg instead of the native HLS downloader
*   `--hls-use-mpegts`: Use the mpegts container for HLS videos
*   `--external-downloader COMMAND`: Use the specified external downloader.
*   `--external-downloader-args ARGS`: Give these arguments to the external downloader

### Filesystem Options:

*   `-a, --batch-file FILE`: File containing URLs to download ('-' for stdin), one URL per line.
*   `--id`: Use only video ID in file name
*   `-o, --output TEMPLATE`: Output filename template, see the "OUTPUT TEMPLATE" for all the info
*   `--output-na-placeholder PLACEHOLDER`: Placeholder value for unavailable meta fields in output filename template (default is "NA")
*   `--autonumber-start NUMBER`: Specify the start value for %(autonumber)s (default is 1)
*   `--restrict-filenames`: Restrict filenames to only ASCII characters, and avoid "&" and spaces in filenames
*   `-w, --no-overwrites`: Do not overwrite files
*   `-c, --continue`: Force resume of partially downloaded files.
*   `--no-continue`: Do not resume partially downloaded files (restart from beginning)
*   `--no-part`: Do not use .part files - write directly into output file
*   `--no-mtime`: Do not use the Last-modified header to set the file modification time
*   `--write-description`: Write video description to a .description file
*   `--write-info-json`: Write video metadata to a .info.json file
*   `--write-annotations`: Write video annotations to a .annotations.xml file
*   `--load-info-json FILE`: JSON file containing the video information
*   `--cookies FILE`: File to read cookies from and dump cookie jar in
*   `--cache-dir DIR`: Location in the filesystem where youtube-dl can store some downloaded information permanently.
*   `--no-cache-dir`: Disable filesystem caching
*   `--rm-cache-dir`: Delete all filesystem cache files

### Thumbnail Options:

*   `--write-thumbnail`: Write thumbnail image to disk
*   `--write-all-thumbnails`: Write all thumbnail image formats to disk
*   `--list-thumbnails`: Simulate and list all available thumbnail formats

### Verbosity / Simulation Options:

*   `-q, --quiet`: Activate quiet mode
*   `--no-warnings`: Ignore warnings
*   `-s, --simulate`: Do not download the video and do not write anything to disk
*   `--skip-download`: Do not download the video
*   `-g, --get-url`: Simulate, quiet but print URL
*   `-e, --get-title`: Simulate, quiet but print title
*   `--get-id`: Simulate, quiet but print id
*   `--get-thumbnail`: Simulate, quiet but print thumbnail URL
*   `--get-description`: Simulate, quiet but print video description
*   `--get-duration`: Simulate, quiet but print video length
*   `--get-filename`: Simulate, quiet but print output filename
*   `--get-format`: Simulate, quiet but print output format
*   `-j, --dump-json`: Simulate, quiet but print JSON information.
*   `-J, --dump-single-json`: Simulate, quiet but print JSON information for each command-line argument.
*   `--print-json`: Be quiet and print the video information as JSON (video is still being downloaded).
*   `--newline`: Output progress bar as new lines
*   `--no-progress`: Do not print progress bar
*   `--console-title`: Display progress in console titlebar
*   `-v, --verbose`: Print various debugging information
*   `--dump-pages`: Print downloaded pages encoded using base64 to debug problems (very verbose)
*   `--write-pages`: Write downloaded intermediary pages to files in the current directory to debug problems
*   `--print-traffic`: Display sent and read HTTP traffic
*   `-C, --call-home`: Contact the youtube-dl server for debugging
*   `--no-call-home`: Do NOT contact the youtube-dl server for debugging

### Workarounds:

*   `--encoding ENCODING`: Force the specified encoding (experimental)
*   `--no-check-certificate`: Suppress HTTPS certificate validation
*   `--prefer-insecure`: Use an unencrypted connection to retrieve information about the video.
*   `--user-agent UA`: Specify a custom user agent
*   `--referer URL`: Specify a custom referer, use if the video access is restricted to one domain
*   `--add-header FIELD:VALUE`: Specify a custom HTTP header and its value, separated by a colon ':'.
*   `--bidi-workaround`: Work around terminals that lack bidirectional text support.
*   `--sleep-interval SECONDS`: Number of seconds to sleep before each download when used alone or a lower bound of a range for randomized sleep before each download when used along with --max-sleep-interval.
*   `--max-sleep-interval SECONDS`: Upper bound of a range for randomized sleep before each download (maximum possible number of seconds to sleep).

### Video Format Options:

*   `-f, --format FORMAT`: Video format code, see the "FORMAT SELECTION" for all the info
*   `--all-formats`: Download all available video formats
*   `--prefer-free-formats`: Prefer free video formats unless a specific one is requested
*   `-F, --list-formats`: List all available formats of requested videos
*   `--youtube-skip-dash-manifest`: Do not download the DASH manifests and related data on YouTube videos
*   `--merge-output-format FORMAT`: If a merge is required (e.g. bestvideo+bestaudio), output to given container format. One of mkv, mp4, ogg, webm, flv. Ignored if no merge is required

### Subtitle Options:

*   `--write-sub`: Write subtitle file
*   `--write-auto-sub`: Write automatically generated subtitle file (YouTube only)
*   `--all-subs`: Download all the available subtitles of the video
*   `--list-subs`: List all available subtitles for the video
*   `--sub-format FORMAT`: Subtitle format, accepts formats preference, for example: "srt" or "ass/srt/best"
*   `--sub-lang LANGS`: Languages of the subtitles to download (optional) separated by commas, use --list-subs for available language tags

### Authentication Options:

*   `-u, --username USERNAME`: Login with this account ID
*   `-p, --password PASSWORD`: Account password.
*   `-2, --twofactor TWOFACTOR`: Two-factor authentication code
*   `-n, --netrc`: Use .netrc authentication data
*   `--video-password PASSWORD`: Video password (vimeo, youku)

### Adobe Pass Options:

*   `--ap-mso MSO`: Adobe Pass multiple-system operator (TV provider) identifier, use --ap-list-mso for a list of available MSOs
*   `--ap-username USERNAME`: Multiple-system operator account login
*   `--ap-password PASSWORD`: Multiple-system operator account password.
*   `--ap-list-mso`: List all supported multiple-system operators

### Post-processing Options:

*   `-x, --extract-audio`: Convert video files to audio-only files (requires ffmpeg/avconv and ffprobe/avprobe)
*   `--audio-format FORMAT`: Specify audio format: "best", "aac", "flac", "mp3", "m4a", "opus", "vorbis", or "wav"; "best" by default; No effect without -x
*   `--audio-quality QUALITY`: Specify ffmpeg/avconv audio quality, insert a value between 0 (better) and 9 (worse) for VBR or a specific bitrate like 128K (default 5)
*   `--recode-video FORMAT`: Encode the video to another format if necessary (currently supported: mp4|flv|ogg|webm|mkv|avi)
*   `--postprocessor-args ARGS`: Give these arguments to the postprocessor
*   `-k, --keep-video`: Keep the video file on disk after the post-processing; the video is erased by default
*   `--no-post-overwrites`: Do not overwrite post-processed files; the post-processed files are overwritten by default
*   `--embed-subs`: Embed subtitles in the video (only for mp4, webm and mkv videos)
*   `--embed-thumbnail`: Embed thumbnail in the audio as cover art
*   `--add-metadata`: Write metadata to the video file
*   `--metadata-from-title FORMAT`: Parse additional metadata like song title / artist from the video title.
*   `--xattrs`: Write metadata to the video file's xattrs (using dublin core and xdg standards)
*   `--fixup POLICY`: Automatically correct known faults of the file.
*   `--prefer-avconv`: Prefer avconv over ffmpeg for running the postprocessors
*   `--prefer-ffmpeg`: Prefer ffmpeg over avconv for running the postprocessors (default)
*   `--ffmpeg-location PATH`: Location of the ffmpeg/avconv binary; either the path to the binary or its containing directory.
*   `--exec CMD`: Execute a command on the file after downloading and post-processing.
*   `--convert-subs FORMAT`: Convert the subtitles to other format (currently supported: srt|ass|vtt|lrc)

## CONFIGURATION

youtube-dl can be configured using a configuration file. On Linux and macOS, the system-wide configuration file is `/etc/youtube-dl.conf` and the user-specific file is `~/.config/youtube-dl/config`.  On Windows, the user-specific files are `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`. Create the file yourself if it doesn't exist.

Example configuration:
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

You can configure automatic credentials storage for extractors that support authentication (by providing login and password with `--username` and `--password`). You can achieve this using a [`.netrc` file](https://stackoverflow.com/tags/.netrc/info) on a per extractor basis.  For that you will need to create a `.netrc` file in your `$HOME` and restrict permissions to read/write by only you:
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

## OUTPUT TEMPLATE

The `-o` option lets you customize output file names.

Example:
```bash
youtube-dl -o '%(title)s-%(id)s.%(ext)s' "https://www.youtube.com/watch?v=BaW_jenozKcj"
```
For example, with the command above, the downloaded video will be named according to the title and video ID and extension.

**Available template variables:**

*   `id`: Video identifier
*   `title`: Video title
*   `url`: Video URL
*   `ext`: Video filename extension
*   `alt_title`: A secondary title of the video
*   `display_id`: An alternative identifier for the video
*   `uploader`: Full name of the video uploader
*   `license`: License name the video is licensed under
*   `creator`: The creator of the video
*   `release_date`: The date (YYYYMMDD) when the video was released
*   `timestamp`: UNIX timestamp of the moment the video became available
*   `upload_date`: Video upload date (YYYYMMDD)
*   `uploader_id`: Nickname or id of the video uploader
*   `channel`: Full name of the channel the video is uploaded on
*   `channel_id`: Id of the channel
*   `location`: Physical location where the video was filmed
*   `duration`: Length of the video in seconds
*   `view_count`: How many users have watched the video on the platform
*   `like_count`: Number of positive ratings of the video
*   `dislike_count`: Number of negative ratings of the video
*   `repost_count`: Number of reposts of the video
*   `average_rating`: Average rating give by users, the scale used depends on the webpage
*   `comment_count`: Number of comments on the video
*   `age_limit`: Age restriction for the video (years)
*   `is_live`: Whether this video is a live stream or a fixed-length video
*   `start_time`: Time in seconds where the reproduction should start, as specified in the URL
*   `end_time`: Time in seconds where the reproduction should end, as specified in the URL
*   `format`: A human-readable description of the format
*   `format_id`: Format code specified by `--format`
*   `format_note`: Additional info about the format
*   `width`: Width of the video
*   `height`: Height of the video
*   `resolution`: Textual description of width and height
*   `tbr`: Average bitrate of audio and video in KBit/s
*   `abr`: Average audio bitrate in KBit/s
*   `acodec`: Name of the audio codec in use
*   `asr`: Audio sampling rate in Hertz
*   `vbr`: Average video bitrate in KBit/s
*   `fps`: Frame rate
*   `vcodec`: Name of the video codec in use
*   `container`: Name of the container format
*   `filesize`: The number of bytes, if known in advance
*   `filesize_approx`: An estimate for the number of bytes
*   `protocol`: The protocol that will be used for the actual download
*   `extractor`: Name of the extractor
*   `extractor_key`: Key name of the extractor
*   `epoch`: Unix epoch when creating the file
*   `autonumber`: Number that will be increased with each download, starting at `--autonumber-start`
*   `playlist`: Name or id of the playlist that contains the video
*   `playlist_index`: Index of the video in the playlist padded with leading zeros according to the total length of the playlist
*   `playlist_id`: Playlist identifier
*   `playlist_title`: Playlist title
*   `playlist_uploader`: Full name of the playlist uploader
*   `playlist_uploader_id`: Nickname or id of the playlist uploader

Available for the video that belongs to some logical chapter or section:

*   `chapter`: Name or title of the chapter the video belongs to
*   `chapter_number`: Number of the chapter the video belongs to
*   `chapter_id`: Id of the chapter the video belongs to

Available for the video that is an episode of some series or programme:

*   `series`: Title of the series or programme the video episode belongs to
*   `season`: Title of the season the video episode belongs to
*   `season_number`: Number of the season the video episode belongs to
*   `season_id`: Id of the season the video episode belongs to
*   `episode`: Title of the video episode
*   `episode_number`: Number of the video episode within a season
*   `episode_id`: Id of the video episode

Available for the media that is a track or a part of a music album:

*   `track`: Title of the track
*   `track_number`: Number of the track within an album or a disc
*   `track_id`: Id of the track
*   `artist`: Artist(s) of the track
*   `genre`: Genre(s) of the track
*   `album`: Title of the album the track belongs to
*   `album_type`: Type of the album
*   `album_artist`: List of all artists appeared on the album
*   `disc_number`: Number of the disc or other physical medium the track belongs to
*   `release_year`: Year (YYYY) when the album was released

## FORMAT SELECTION

Use the `-f` or `--format` option to specify which format you want to download.

*   `-f 22`: Download the format with code 22.
*   `-f webm`: Download the best quality format with the `webm` extension served as a single file.
*   `-f best`: Selects the best overall quality available. This is often the default.
*   `-f worst`: Selects the worst overall quality available.
*   `-f bestvideo`: Select the best quality video-only format.
*   `-f worstvideo`: Select the worst quality video-only format.
*   `-f bestaudio`: Select the best quality audio only-format.
*   `-f worstaudio`: Select the worst quality audio only-format.

You can specify multiple formats using `/` for precedence or `,` to download multiple formats:
*   `-f 22/17/18`: Download format 22 if available, then 17, then 18.
*   `-f 22,17,18`: Download formats 22, 17, and 18.
*   `-f 136/137/mp4/bestvideo,140/m4a/bestaudio`: Will download both video and audio formats if available.

You can also filter formats by putting a condition in brackets:
*   `-f "best[height=720]"`: Download the best quality video with height equal to 720
*   `-f "[filesize>10M]"`: Download video with size greater than 10 megabytes.

## VIDEO SELECTION

Filter videos by upload date using `--date`, `--datebefore`, or `--dateafter`:

*   `--date 20231027`: Download videos uploaded on October 27, 2023.
*   `--dateafter now-6months`: Download videos uploaded in the last 6 months.

## FAQ

*   **How do I update youtube-dl?** Run `youtube-dl -U`.  If using pip, run `sudo pip install -U youtube-dl`.
*   **How to pass cookies?** Use the `--cookies` option.
*   **How do I stream directly to a media player?**  Use `-o -` to output to stdout and pipe to your player.
*   **What programs do I need?** youtube-dl works standalone for many sites. For conversion, you'll need [avconv](https://libav.org/) or [ffmpeg](https://www.ffmpeg.org/).
*   **I get an HTTP error 429 or 402:** The service may be blocking your IP due to overuse.

For more details, see [FAQ](#faq)

## EMBEDDING YOUTUBE-DL

youtube-dl can be embedded into Python scripts.

```python
from __future__ import unicode_literals
import youtube_dl

ydl_opts = {}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=BaW_jenozKc'])
```

## BUGS

Report bugs and suggestions in the issue tracker: <https://github.com/ytdl-org/youtube-dl/issues>.  **Include the full output of youtube-dl with the `-v` flag when reporting bugs.**

## DEVELOPER INSTRUCTIONS

Instructions for developers, including adding support for new sites, are available in the [DEVELOPER INSTRUCTIONS](#developer-instructions) section of the README.

## COPYRIGHT

youtube-dl is released into the public domain by the copyright holders.