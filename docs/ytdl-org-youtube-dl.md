[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Download Videos from YouTube and Beyond

**Download videos from YouTube and many other sites with ease using `youtube-dl`.** This powerful command-line tool lets you save your favorite videos for offline viewing.

**Key Features:**

*   **Wide Site Support:** Download videos from YouTube, as well as hundreds of other video platforms.
*   **Multiple Format Options:** Choose from various video and audio formats, including high-definition options.
*   **Playlist and Channel Support:** Download entire playlists or all videos from a channel.
*   **Customizable Output:** Control filenames, and output directories.
*   **Subtitle and Metadata Support:** Download subtitles and embed metadata into your downloaded files.
*   **Cross-Platform:** Works on Linux, macOS, Windows, and other operating systems.

[Explore the original repository for full documentation](https://github.com/ytdl-org/youtube-dl).

**Sections**

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
    *   [Output template examples](#output-template-examples)
*   [Format Selection](#format-selection)
    *   [Format selection examples](#format-selection-examples)
*   [Video Selection](#video-selection)
*   [FAQ](#faq)
*   [Developer Instructions](#developer-instructions)
    *   [Adding support for a new site](#adding-support-for-a-new-site)
    *   [youtube-dl coding conventions](#youtube-dl-coding-conventions)
*   [Embedding youtube-dl](#embedding-youtube-dl)
*   [Bugs](#bugs)
*   [Copyright](#copyright)

## INSTALLATION

To install youtube-dl on UNIX-based systems (Linux, macOS), use:

```bash
sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

If `curl` is unavailable, use a recent version of `wget`:

```bash
sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

Windows users can download the [youtube-dl.exe](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory in their [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) except `%SYSTEMROOT%\System32`.

You can also use pip:

```bash
sudo -H pip install --upgrade youtube-dl
```

macOS users can install youtube-dl using [Homebrew](https://brew.sh/):

```bash
brew install youtube-dl
```

Or with [MacPorts](https://www.macports.org/):

```bash
sudo port install youtube-dl
```

Alternatively, refer to the [developer instructions](#developer-instructions) for how to check out and work with the git repository. For further options, including PGP signatures, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## DESCRIPTION

`youtube-dl` is a versatile command-line program designed to download videos from YouTube.com and numerous other websites. It's platform-independent, working on Unix-like systems, Windows, and macOS. The program is in the public domain, allowing for modification, redistribution, and unrestricted use.

## OPTIONS

Use `youtube-dl -h` to see all available options. Here's a summary of key categories:

*   `--help`:  Print help text and exit
*   `--version`: Print program version and exit
*   `-U, --update`: Update this program to latest version.
*   `-i, --ignore-errors`: Continue on download errors, for example to skip unavailable videos in a playlist
*   `--abort-on-error`: Abort downloading of further videos (in the playlist or the command line) if an error occurs
*   `--dump-user-agent`: Display the current browser identification
*   `--list-extractors`: List all supported extractors
*   `--extractor-descriptions`: Output descriptions of all supported extractors
*   `--force-generic-extractor`: Force extraction to use the generic extractor
*   `--default-search PREFIX`:  Use this prefix for unqualified URLs.
*   `--ignore-config`: Do not read configuration files
*   `--config-location PATH`: Location of the configuration file
*   `--flat-playlist`: Do not extract the videos of a playlist, only list them
*   `--mark-watched`: Mark videos watched (YouTube only)
*   `--no-mark-watched`: Do not mark videos watched (YouTube only)
*   `--no-color`: Do not emit color codes in output

### Network Options

*   `--proxy URL`: Use the specified HTTP/HTTPS/SOCKS proxy.
*   `--socket-timeout SECONDS`: Time to wait before giving up, in seconds
*   `--source-address IP`: Client-side IP address to bind to
*   `-4, --force-ipv4`: Make all connections via IPv4
*   `-6, --force-ipv6`: Make all connections via IPv6

### Geo Restriction

*   `--geo-verification-proxy URL`: Use this proxy to verify the IP address for some geo-restricted sites.
*   `--geo-bypass`: Bypass geographic restriction via faking X-Forwarded-For HTTP header
*   `--no-geo-bypass`: Do not bypass geographic restriction via faking X-Forwarded-For HTTP header
*   `--geo-bypass-country CODE`: Force bypass geographic restriction with explicitly provided two-letter ISO 3166-2 country code
*   `--geo-bypass-ip-block IP_BLOCK`: Force bypass geographic restriction with explicitly provided IP block in CIDR notation

### Video Selection

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

### Download Options

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
*   `--hls-use-mpegts`: Use the mpegts container for HLS videos, allowing to play the video while downloading (some players may not be able to play it)
*   `--external-downloader COMMAND`: Use the specified external downloader.
*   `--external-downloader-args ARGS`: Give these arguments to the external downloader

### Filesystem Options

*   `-a, --batch-file FILE`: File containing URLs to download ('-' for stdin), one URL per line.
*   `--id`: Use only video ID in file name
*   `-o, --output TEMPLATE`: Output filename template.
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
*   `--load-info-json FILE`: JSON file containing the video information (created with the "--write-info-json" option)
*   `--cookies FILE`: File to read cookies from and dump cookie jar in
*   `--cache-dir DIR`: Location in the filesystem where youtube-dl can store some downloaded information permanently.
*   `--no-cache-dir`: Disable filesystem caching
*   `--rm-cache-dir`: Delete all filesystem cache files

### Thumbnail Options

*   `--write-thumbnail`: Write thumbnail image to disk
*   `--write-all-thumbnails`: Write all thumbnail image formats to disk
*   `--list-thumbnails`: Simulate and list all available thumbnail formats

### Verbosity / Simulation Options

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

### Workarounds

*   `--encoding ENCODING`: Force the specified encoding (experimental)
*   `--no-check-certificate`: Suppress HTTPS certificate validation
*   `--prefer-insecure`: Use an unencrypted connection to retrieve information about the video. (Currently supported only for YouTube)
*   `--user-agent UA`: Specify a custom user agent
*   `--referer URL`: Specify a custom referer, use if the video access is restricted to one domain
*   `--add-header FIELD:VALUE`: Specify a custom HTTP header and its value, separated by a colon ':'.
*   `--bidi-workaround`: Work around terminals that lack bidirectional text support.
*   `--sleep-interval SECONDS`: Number of seconds to sleep before each download.
*   `--max-sleep-interval SECONDS`: Upper bound of a range for randomized sleep before each download (maximum possible number of seconds to sleep).

### Video Format Options

*   `-f, --format FORMAT`: Video format code.
*   `--all-formats`: Download all available video formats
*   `--prefer-free-formats`: Prefer free video formats unless a specific one is requested
*   `-F, --list-formats`: List all available formats of requested videos
*   `--youtube-skip-dash-manifest`: Do not download the DASH manifests and related data on YouTube videos
*   `--merge-output-format FORMAT`: If a merge is required (e.g. bestvideo+bestaudio), output to given container format. One of mkv, mp4, ogg, webm, flv.

### Subtitle Options

*   `--write-sub`: Write subtitle file
*   `--write-auto-sub`: Write automatically generated subtitle file (YouTube only)
*   `--all-subs`: Download all the available subtitles of the video
*   `--list-subs`: List all available subtitles for the video
*   `--sub-format FORMAT`: Subtitle format, accepts formats preference, for example: "srt" or "ass/srt/best"
*   `--sub-lang LANGS`: Languages of the subtitles to download (optional) separated by commas, use --list-subs for available language tags

### Authentication Options

*   `-u, --username USERNAME`: Login with this account ID
*   `-p, --password PASSWORD`: Account password.
*   `-2, --twofactor TWOFACTOR`: Two-factor authentication code
*   `-n, --netrc`: Use .netrc authentication data
*   `--video-password PASSWORD`: Video password (vimeo, youku)

### Adobe Pass Options

*   `--ap-mso MSO`: Adobe Pass multiple-system operator (TV provider) identifier, use --ap-list-mso for a list of available MSOs
*   `--ap-username USERNAME`: Multiple-system operator account login
*   `--ap-password PASSWORD`: Multiple-system operator account password.
*   `--ap-list-mso`: List all supported multiple-system operators

### Post-processing Options

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
*   `--exec CMD`: Execute a command on the file after downloading and post-processing, similar to find's -exec syntax.
*   `--convert-subs FORMAT`: Convert the subtitles to other format (currently supported: srt|ass|vtt|lrc)

## CONFIGURATION

Configure `youtube-dl` using a configuration file to set default options.
*   On Linux/macOS: `/etc/youtube-dl.conf` (system-wide) and `~/.config/youtube-dl/config` (user-specific).
*   On Windows: `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`.

Use `--ignore-config` to disable config files and `--config-location` to specify a custom file.
For example:

```
# Lines starting with # are comments
-x               # Always extract audio
--no-mtime        # Do not copy the mtime
--proxy 127.0.0.1:3128  # Use this proxy
-o ~/Movies/%(title)s.%(ext)s # Save all videos under Movies
```

### Authentication with .netrc file

Configure authentication for extractors that support login with a `.netrc` file for security. Create `.netrc` in your `$HOME` with restricted permissions:

```bash
touch $HOME/.netrc
chmod a-rwx,u+rw $HOME/.netrc
```

Add credentials for each extractor (e.g., youtube, twitch):

```
machine youtube login myaccount@gmail.com password my_youtube_password
machine twitch login my_twitch_account_name password my_twitch_password
```

Enable by passing `--netrc` or adding it to your configuration file.

## OUTPUT TEMPLATE

Customize output filenames using the `-o` option and a template string. You can use specific sequences to dynamically generate filenames for different videos.

**Examples:**

```bash
youtube-dl -o '%(title)s-%(id)s.%(ext)s'  <video_url>
```

or for a playlist:

```bash
youtube-dl -o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' <playlist_url>
```

See the full output template documentation for all the options available.

### Output template examples

Note that on Windows you may need to use double quotes instead of single.

```bash
# All kinds of weird characters
$ youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc
youtube-dl test video ''_ä↭𝕐.mp4

# A simple file name
$ youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc --restrict-filenames
youtube-dl_test_video_.mp4

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

## FORMAT SELECTION

Use the `-f` or `--format` option to specify the video format(s) you want to download. Use `--list-formats` or `-F` to see available formats for a video.

**Examples:**

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

## VIDEO SELECTION

Filter videos by upload date with `--date`, `--datebefore`, or `--dateafter`.

```bash
youtube-dl --dateafter now-6months  # Videos from the last 6 months
youtube-dl --date 19700101         # Videos uploaded on January 1, 1970
```

## FAQ

A detailed FAQ is available in the original `README`. Includes answers to common questions.

## DEVELOPER INSTRUCTIONS

Detailed information for developers on how to contribute to the project, including setting up a development environment, and how to add support for new sites.

### Adding support for a new site

1.  [Fork this repository](https://github.com/ytdl-org/youtube-dl/fork)
2.  Clone the source code: `git clone git@github.com:YOUR_GITHUB_USERNAME/youtube-dl.git`
3.  Create a new branch: `git checkout -b yourextractor`
4.  Create a new extractor file in `youtube_dl/extractor/yourextractor.py`. Follow the [template provided](README.md#adding-support-for-a-new-site).
5.  Add an import in [`youtube_dl/extractor/extractors.py`](https://github.com/ytdl-org/youtube-dl/blob/master/youtube_dl/extractor/extractors.py).
6.  Test with: `python test/test_download.py TestDownload.test_YourExtractor`.
7.  Add tests and code based on [youtube_dl/extractor/common.py](https://github.com/ytdl-org/youtube-dl/blob/master/youtube_dl/extractor/common.py).
8.  Check code with [flake8](https://flake8.pycqa.org/en/latest/index.html#quickstart).
9.  Ensure your code is compatible with all Python versions supported by youtube-dl.
10. Add new files and commit with a descriptive message, then push to origin.
11. Create a pull request.

### youtube-dl coding conventions

Coding standards and best practices for extractor development.

## EMBEDDING YOUTUBE-DL

Embed `youtube-dl` in your Python programs. Refer to the original README for code examples.

## BUGS

Report bugs and suggest improvements via the issue tracker: <https://github.com/ytdl-org/youtube-dl/issues>

To report a bug, include the full output of `youtube-dl -v <your_command_line>`.  Ensure the bug report is descriptive, the current version is used, and the issue is not already documented.

## COPYRIGHT

`youtube-dl` is released into the public domain.