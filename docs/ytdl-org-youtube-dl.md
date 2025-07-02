[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Download Videos from YouTube and Beyond

**Easily download videos from YouTube and many other sites with this versatile command-line tool.**  [Visit the original repo](https://github.com/ytdl-org/youtube-dl)

**Key Features:**

*   **Wide Site Support:** Download from YouTube, plus hundreds of other video platforms.
*   **Format Selection:** Choose your desired video and audio quality with flexible options.
*   **Playlist Downloads:** Download entire playlists or select specific videos.
*   **Metadata Extraction:** Automatically retrieve video titles, descriptions, and more.
*   **Customization:** Configure output filenames, download settings, and more.
*   **Cross-Platform:** Works on Windows, macOS, and Linux.
*   **Easy to Update:** Keeps itself updated with the latest features and bug fixes.

## Table of Contents

*   [Installation](#installation)
    *   [Unix (Linux, macOS, etc.)](#installation)
    *   [Windows](#installation)
    *   [Using `pip`](#installation)
    *   [macOS with Homebrew/MacPorts](#installation)
    *   [Developer Installation](#developer-instructions)
    *   [Download Page](#installation)
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
    *   [Authentication with `.netrc` file](#authentication-with-.netrc-file)
*   [Output Template](#output-template)
    *   [Output template examples](#output-template-examples)
*   [Format Selection](#format-selection)
    *   [Format selection examples](#format-selection-examples)
*   [Video Selection](#video-selection)
*   [FAQ](#faq)
*   [Developer Instructions](#developer-instructions)
    *   [Adding Support for a New Site](#adding-support-for-a-new-site)
    *   [youtube-dl coding conventions](#youtube-dl-coding-conventions)
*   [Embedding youtube-dl](#embedding-youtube-dl)
*   [Bugs](#bugs)
    *   [Why do I need to go through that much red tape when filing bugs?](#why-do-i-need-to-go-through-that-much-red-tape-when-filing-bugs)
*   [Copyright](#copyright)

## INSTALLATION

### Unix (Linux, macOS, etc.)

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

Download the `.exe` file from [here](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory included in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29), **except** for `%SYSTEMROOT%\System32`.

### Using `pip`

```bash
sudo -H pip install --upgrade youtube-dl
```
This will update `youtube-dl` if already installed. See the [pypi page](https://pypi.python.org/pypi/youtube_dl) for more information.

### macOS with Homebrew/MacPorts

```bash
brew install youtube-dl
```
Or with [MacPorts](https://www.macports.org/):
```bash
sudo port install youtube-dl
```

### Developer Installation

Refer to the [developer instructions](#developer-instructions) for how to check out and work with the git repository.

### Download Page

For further options, including PGP signatures, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## DESCRIPTION

`youtube-dl` is a command-line program that allows you to download videos from YouTube.com and many other sites. It requires the Python interpreter (version 2.6, 2.7, or 3.2+), and is platform-independent, working on Unix, Windows, and macOS. It is released to the public domain.

```bash
youtube-dl [OPTIONS] URL [URL...]
```

## OPTIONS

*   `-h, --help`: Print this help text and exit
*   `--version`: Print program version and exit
*   `-U, --update`: Update this program to the latest version
*   `-i, --ignore-errors`: Continue on download errors
*   `--abort-on-error`: Abort downloading further videos if an error occurs
*   `--dump-user-agent`: Display the current browser identification
*   `--list-extractors`: List all supported extractors
*   `--extractor-descriptions`: Output descriptions of all supported extractors
*   `--force-generic-extractor`: Force extraction to use the generic extractor
*   `--default-search PREFIX`: Use this prefix for unqualified URLs
*   `--ignore-config`: Do not read configuration files
*   `--config-location PATH`: Location of the configuration file
*   `--flat-playlist`: Do not extract the videos of a playlist, only list them
*   `--mark-watched`: Mark videos watched (YouTube only)
*   `--no-mark-watched`: Do not mark videos watched (YouTube only)
*   `--no-color`: Do not emit color codes in output

### Network Options

*   `--proxy URL`: Use the specified HTTP/HTTPS/SOCKS proxy
*   `--socket-timeout SECONDS`: Time to wait before giving up, in seconds
*   `--source-address IP`: Client-side IP address to bind to
*   `-4, --force-ipv4`: Make all connections via IPv4
*   `-6, --force-ipv6`: Make all connections via IPv6

### Geo Restriction

*   `--geo-verification-proxy URL`: Use this proxy to verify the IP address for some geo-restricted sites
*   `--geo-bypass`: Bypass geographic restriction
*   `--no-geo-bypass`: Do not bypass geographic restriction
*   `--geo-bypass-country CODE`: Force bypass geographic restriction with explicitly provided two-letter ISO 3166-2 country code
*   `--geo-bypass-ip-block IP_BLOCK`: Force bypass geographic restriction with explicitly provided IP block in CIDR notation

### Video Selection

*   `--playlist-start NUMBER`: Playlist video to start at
*   `--playlist-end NUMBER`: Playlist video to end at
*   `--playlist-items ITEM_SPEC`: Playlist video items to download.
*   `--match-title REGEX`: Download only matching titles (regex or caseless sub-string)
*   `--reject-title REGEX`: Skip download for matching titles (regex or caseless sub-string)
*   `--max-downloads NUMBER`: Abort after downloading NUMBER files
*   `--min-filesize SIZE`: Do not download any videos smaller than SIZE
*   `--max-filesize SIZE`: Do not download any videos larger than SIZE
*   `--date DATE`: Download only videos uploaded in this date
*   `--datebefore DATE`: Download only videos uploaded on or before this date (i.e. inclusive)
*   `--dateafter DATE`: Download only videos uploaded on or after this date (i.e. inclusive)
*   `--min-views COUNT`: Do not download any videos with less than COUNT views
*   `--max-views COUNT`: Do not download any videos with more than COUNT views
*   `--match-filter FILTER`: Generic video filter
*   `--no-playlist`: Download only the video, if the URL refers to a video and a playlist
*   `--yes-playlist`: Download the playlist, if the URL refers to a video and a playlist
*   `--age-limit YEARS`: Download only videos suitable for the given age
*   `--download-archive FILE`: Download only videos not listed in the archive file. Record the IDs of all downloaded videos in it.
*   `--include-ads`: Download advertisements as well (experimental)

### Download Options

*   `-r, --limit-rate RATE`: Maximum download rate in bytes per second
*   `-R, --retries RETRIES`: Number of retries (default is 10), or "infinite"
*   `--fragment-retries RETRIES`: Number of retries for a fragment (default is 10), or "infinite"
*   `--skip-unavailable-fragments`: Skip unavailable fragments
*   `--abort-on-unavailable-fragment`: Abort downloading when some fragment is not available
*   `--keep-fragments`: Keep downloaded fragments on disk after downloading is finished
*   `--buffer-size SIZE`: Size of download buffer
*   `--no-resize-buffer`: Do not automatically adjust the buffer size
*   `--http-chunk-size SIZE`: Size of a chunk for chunk-based HTTP downloading
*   `--playlist-reverse`: Download playlist videos in reverse order
*   `--playlist-random`: Download playlist videos in random order
*   `--xattr-set-filesize`: Set file xattribute ytdl.filesize with expected file size
*   `--hls-prefer-native`: Use the native HLS downloader instead of ffmpeg
*   `--hls-prefer-ffmpeg`: Use ffmpeg instead of the native HLS downloader
*   `--hls-use-mpegts`: Use the mpegts container for HLS videos, allowing to play the video while downloading
*   `--external-downloader COMMAND`: Use the specified external downloader
*   `--external-downloader-args ARGS`: Give these arguments to the external downloader

### Filesystem Options

*   `-a, --batch-file FILE`: File containing URLs to download, one URL per line
*   `--id`: Use only video ID in file name
*   `-o, --output TEMPLATE`: Output filename template
*   `--output-na-placeholder PLACEHOLDER`: Placeholder value for unavailable meta fields in output filename template
*   `--autonumber-start NUMBER`: Specify the start value for %(autonumber)s
*   `--restrict-filenames`: Restrict filenames to only ASCII characters
*   `-w, --no-overwrites`: Do not overwrite files
*   `-c, --continue`: Force resume of partially downloaded files
*   `--no-continue`: Do not resume partially downloaded files
*   `--no-part`: Do not use .part files
*   `--no-mtime`: Do not use the Last-modified header to set the file modification time
*   `--write-description`: Write video description to a .description file
*   `--write-info-json`: Write video metadata to a .info.json file
*   `--write-annotations`: Write video annotations to a .annotations.xml file
*   `--load-info-json FILE`: JSON file containing the video information
*   `--cookies FILE`: File to read cookies from and dump cookie jar in
*   `--cache-dir DIR`: Location in the filesystem where youtube-dl can store some downloaded information permanently
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
*   `-j, --dump-json`: Simulate, quiet but print JSON information
*   `-J, --dump-single-json`: Simulate, quiet but print JSON information for each command-line argument
*   `--print-json`: Be quiet and print the video information as JSON
*   `--newline`: Output progress bar as new lines
*   `--no-progress`: Do not print progress bar
*   `--console-title`: Display progress in console titlebar
*   `-v, --verbose`: Print various debugging information
*   `--dump-pages`: Print downloaded pages encoded using base64 to debug problems
*   `--write-pages`: Write downloaded intermediary pages to files in the current directory to debug problems
*   `--print-traffic`: Display sent and read HTTP traffic
*   `-C, --call-home`: Contact the youtube-dl server for debugging
*   `--no-call-home`: Do NOT contact the youtube-dl server for debugging

### Workarounds

*   `--encoding ENCODING`: Force the specified encoding (experimental)
*   `--no-check-certificate`: Suppress HTTPS certificate validation
*   `--prefer-insecure`: Use an unencrypted connection to retrieve information about the video.
*   `--user-agent UA`: Specify a custom user agent
*   `--referer URL`: Specify a custom referer
*   `--add-header FIELD:VALUE`: Specify a custom HTTP header and its value
*   `--bidi-workaround`: Work around terminals that lack bidirectional text support
*   `--sleep-interval SECONDS`: Number of seconds to sleep before each download
*   `--max-sleep-interval SECONDS`: Upper bound of a range for randomized sleep before each download

### Video Format Options

*   `-f, --format FORMAT`: Video format code
*   `--all-formats`: Download all available video formats
*   `--prefer-free-formats`: Prefer free video formats unless a specific one is requested
*   `-F, --list-formats`: List all available formats of requested videos
*   `--youtube-skip-dash-manifest`: Do not download the DASH manifests and related data on YouTube videos
*   `--merge-output-format FORMAT`: If a merge is required, output to the given container format. One of mkv, mp4, ogg, webm, flv. Ignored if no merge is required

### Subtitle Options

*   `--write-sub`: Write subtitle file
*   `--write-auto-sub`: Write automatically generated subtitle file (YouTube only)
*   `--all-subs`: Download all the available subtitles of the video
*   `--list-subs`: List all available subtitles for the video
*   `--sub-format FORMAT`: Subtitle format
*   `--sub-lang LANGS`: Languages of the subtitles to download

### Authentication Options

*   `-u, --username USERNAME`: Login with this account ID
*   `-p, --password PASSWORD`: Account password
*   `-2, --twofactor TWOFACTOR`: Two-factor authentication code
*   `-n, --netrc`: Use .netrc authentication data
*   `--video-password PASSWORD`: Video password

### Adobe Pass Options

*   `--ap-mso MSO`: Adobe Pass multiple-system operator identifier
*   `--ap-username USERNAME`: Multiple-system operator account login
*   `--ap-password PASSWORD`: Multiple-system operator account password
*   `--ap-list-mso`: List all supported multiple-system operators

### Post-processing Options

*   `-x, --extract-audio`: Convert video files to audio-only files (requires ffmpeg/avconv)
*   `--audio-format FORMAT`: Specify audio format
*   `--audio-quality QUALITY`: Specify ffmpeg/avconv audio quality
*   `--recode-video FORMAT`: Encode the video to another format if necessary
*   `--postprocessor-args ARGS`: Give these arguments to the postprocessor
*   `-k, --keep-video`: Keep the video file on disk after the post-processing
*   `--no-post-overwrites`: Do not overwrite post-processed files
*   `--embed-subs`: Embed subtitles in the video
*   `--embed-thumbnail`: Embed thumbnail in the audio as cover art
*   `--add-metadata`: Write metadata to the video file
*   `--metadata-from-title FORMAT`: Parse additional metadata from the video title
*   `--xattrs`: Write metadata to the video file's xattrs
*   `--fixup POLICY`: Automatically correct known faults of the file
*   `--prefer-avconv`: Prefer avconv over ffmpeg for running the postprocessors
*   `--prefer-ffmpeg`: Prefer ffmpeg over avconv for running the postprocessors
*   `--ffmpeg-location PATH`: Location of the ffmpeg/avconv binary
*   `--exec CMD`: Execute a command on the file after downloading and post-processing
*   `--convert-subs FORMAT`: Convert the subtitles to other format

## Configuration

You can configure youtube-dl by placing command-line options in a configuration file. On Linux and macOS, the system-wide configuration file is located at `/etc/youtube-dl.conf`, and the user-specific one at `~/.config/youtube-dl/config`. On Windows, user configuration files are `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`.

For example, the following configuration file will always extract audio, disable mtime, use a proxy, and save videos under a `Movies` directory:

```
# Lines starting with # are comments

# Always extract audio
-x

# Do not copy the mtime
--no-mtime

# Use this proxy
--proxy 127.0.0.1:3128

# Save all videos under Movies directory
-o ~/Movies/%(title)s.%(ext)s
```

You can disable the configuration file with `--ignore-config` or use a custom configuration file with `--config-location`.

### Authentication with `.netrc` file

Configure automatic credentials storage for extractors that support authentication to avoid passing credentials on every execution.

Create a `.netrc` file in your `$HOME` and restrict permissions to read/write by only you:
```
touch $HOME/.netrc
chmod a-rwx,u+rw $HOME/.netrc
```
Then add credentials for an extractor in the following format:
```
machine <extractor> login <login> password <password>
```
For example:
```
machine youtube login myaccount@gmail.com password my_youtube_password
machine twitch login my_twitch_account_name password my_twitch_password
```
To activate authentication with the `.netrc` file pass `--netrc` to youtube-dl or place it in the [configuration file](#configuration).

## OUTPUT TEMPLATE

The `-o` option allows users to indicate a template for the output file names.

The template can contain special sequences that will be replaced when downloading each video, formatted according to [python string formatting operations](https://docs.python.org/2/library/stdtypes.html#string-formatting).

Allowed names are:

*   `id` (string): Video identifier
*   `title` (string): Video title
*   `url` (string): Video URL
*   `ext` (string): Video filename extension
*   `alt_title` (string): A secondary title of the video
*   `display_id` (string): An alternative identifier for the video
*   `uploader` (string): Full name of the video uploader
*   `license` (string): License name the video is licensed under
*   `creator` (string): The creator of the video
*   `release_date` (string): The date (YYYYMMDD) when the video was released
*   `timestamp` (numeric): UNIX timestamp of the moment the video became available
*   `upload_date` (string): Video upload date (YYYYMMDD)
*   `uploader_id` (string): Nickname or id of the video uploader
*   `channel` (string): Full name of the channel the video is uploaded on
*   `channel_id` (string): Id of the channel
*   `location` (string): Physical location where the video was filmed
*   `duration` (numeric): Length of the video in seconds
*   `view_count` (numeric): How many users have watched the video on the platform
*   `like_count` (numeric): Number of positive ratings of the video
*   `dislike_count` (numeric): Number of negative ratings of the video
*   `repost_count` (numeric): Number of reposts of the video
*   `average_rating` (numeric): Average rating give by users, the scale used depends on the webpage
*   `comment_count` (numeric): Number of comments on the video
*   `age_limit` (numeric): Age restriction for the video (years)
*   `is_live` (boolean): Whether this video is a live stream or a fixed-length video
*   `start_time` (numeric): Time in seconds where the reproduction should start, as specified in the URL
*   `end_time` (numeric): Time in seconds where the reproduction should end, as specified in the URL
*   `format` (string): A human-readable description of the format
*   `format_id` (string): Format code specified by `--format`
*   `format_note` (string): Additional info about the format
*   `width` (numeric): Width of the video
*   `height` (numeric): Height of the video
*   `resolution` (string): Textual description of width and height
*   `tbr` (numeric): Average bitrate of audio and video in KBit/s
*   `abr` (numeric): Average audio bitrate in KBit/s
*   `acodec` (string): Name of the audio codec in use
*   `asr` (numeric): Audio sampling rate in Hertz
*   `vbr` (numeric): Average video bitrate in KBit/s
*   `fps` (numeric): Frame rate
*   `vcodec` (string): Name of the video codec in use
*   `container` (string): Name of the container format
*   `filesize` (numeric): The number of bytes, if known in advance
*   `filesize_approx` (numeric): An estimate for the number of bytes
*   `protocol` (string): The protocol that will be used for the actual download
*   `extractor` (string): Name of the extractor
*   `extractor_key` (string): Key name of the extractor
*   `epoch` (numeric): Unix epoch when creating the file
*   `autonumber` (numeric): Number that will be increased with each download
*   `playlist` (string): Name or id of the playlist that contains the video
*   `playlist_index` (numeric): Index of the video in the playlist padded with leading zeros according to the total length of the playlist
*   `playlist_id` (string): Playlist identifier
*   `playlist_title` (string): Playlist title
*   `playlist_uploader` (string): Full name of the playlist uploader
*   `playlist_uploader_id` (string): Nickname or id of the playlist uploader

Available for the video that belongs to some logical chapter or section:

*   `chapter` (string): Name or title of the chapter the video belongs to
*   `chapter_number` (numeric): Number of the chapter the video belongs to
*   `chapter_id` (string): Id of the chapter the video belongs to

Available for the video that is an episode of some series or programme:

*   `series` (string): Title of the series or programme the video episode belongs to
*   `season` (string): Title of the season the video episode belongs to
*   `season_number` (numeric): Number of the season the video episode belongs to
*   `season_id` (string): Id of the season the video episode belongs to
*   `episode` (string): Title of the video episode
*   `episode_number` (numeric): Number of the video episode within a season
*   `episode_id` (string): Id of the video episode

Available for the media that is a track or a part of a music album:

*   `track` (string): Title of the track
*   `track_number` (numeric): Number of the track within an album or a disc
*   `track_id` (string): Id of the track
*   `artist` (string): Artist(s) of the track
*   `genre` (string): Genre(s) of the track
*   `album` (string): Title of the album the track belongs to
*   `album_type` (string): Type of the album
*   `album_artist` (string): List of all artists appeared on the album
*   `disc_number` (numeric): Number of the disc or other physical medium the track belongs to
*   `release_year` (numeric): Year (YYYY) when the album was released

Each aforementioned sequence when referenced in an output template will be replaced by the actual value corresponding to the sequence name. Note that some of the sequences are not guaranteed to be present since they depend on the metadata obtained by a particular extractor. Such sequences will be replaced with placeholder value provided with `--output-na-placeholder` (`NA` by default).

To use percent literals in an output template use `%%`. To output to stdout use `-o -`.

The current default template is `%(title)s-%(id)s.%(ext)s`.

To use special characters in an output template use `--restrict-filenames`.

### Output template examples

```bash
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

## FORMAT SELECTION

The `-f` or `--format` option allows you to select the desired video format. By default youtube-dl downloads the best available quality. You can get the list of available format codes for particular video using `--list-formats` or `-F`.

You can use a file extension (currently `3gp`, `aac`, `flv`, `m4a`, `mp3`, `mp4`, `ogg`, `wav`, `webm` are supported) to download the best quality format of a particular file extension served as a single file, e.g. `-f webm` will download the best quality format with the `webm` extension served as a single file.

You can also use special names to select particular edge case formats:

*   `best`: Select the best quality format represented by a single file with video and audio.
*   `worst`: Select the worst quality format represented by a single file with video and audio.
*   `bestvideo`: Select the best quality video-only format. May not be available.
*   `worstvideo`: Select the worst quality video-only format. May not be available.
*   `bestaudio`: Select the best quality audio only-format. May not be available.
*   `worstaudio`: Select the worst quality audio only-format. May not be available.

If you want to download multiple videos and they don't have the same formats available, you can specify the order of preference using slashes. If you want to download several formats of the same video use a comma as a separator.

You can also filter the video formats by putting a condition in brackets, as in `-f "best[height=720]"` or `-f "[filesize>10M]"`.

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
*   `protocol`: The protocol that will be used for the actual download, lower-case
*   `format_id`: A short description of the format
*   `language`: Language code

Formats for which the value is not known are excluded unless you put a question mark (`?`) after the operator. You can merge the video and audio of two formats into a single file using `-f <video-format>+<audio-format>`.

### Format selection examples

```bash
# Download best mp4 format available or any other best if no mp4 available
youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'

# Download best format available but no better than 480p
youtube-dl -f 'bestvideo[height<=480]+bestaudio/best[height<=480]'

# Download best video only format but no bigger than 50 MB
youtube-dl -f 'best[filesize<50M]'

# Download best format available via direct link over HTTP/HTTPS protocol
youtube-dl -f '(bestvideo+bestaudio/best)[protocol^=http]'

# Download the best video format and the best audio format without merging them
youtube-dl -f 'bestvideo,bestaudio' -o '%(title)s.f%(format_id)s.%(ext)s'
```

## VIDEO SELECTION

Videos can be filtered by their upload date using the options `--date`, `--datebefore` or `--dateafter`. They accept dates in two formats:

*   Absolute dates: Dates in the format `YYYYMMDD`.
*   Relative dates: Dates in the format `(now|today)[+-][0-9](day|week|month|year)(s)?`

Examples:

```bash
# Download only the videos uploaded in the last 6 months
youtube-dl --dateafter now-6months

# Download only the videos uploaded on January 1, 1970
youtube-dl --date 19700101

$ # Download only the videos uploaded in the 200x decade
$ youtube-dl --dateafter 20000101 --datebefore 20091231
```

## FAQ

### How do I update youtube-dl?

Run `youtube-dl -U` (or, on Linux, `sudo youtube-dl -U`).

If you have used pip, a simple `sudo pip install -U youtube-dl` is sufficient to update.

If you have installed youtube-dl using a package manager, use the standard system update mechanism to update. Note that distribution packages are often outdated.

If all else fails, uninstall the distribution's package and follow the manual installation instructions.

### youtube-dl is extremely slow to start on Windows

Add a file exclusion for `youtube-dl.exe` in Windows Defender settings.

### I'm getting an error `Unable to extract OpenGraph title` on YouTube playlists

You'll need at least youtube-dl 2014.07.25 to download all YouTube videos.

### I'm getting an error when trying to use output template: `error: using output template conflicts with using title, video ID or auto number`

Make sure you are not using `-o` with any of these options `-t`, `--title`, `--id`, `-A` or `--auto-number`.

### Do I always have to pass `-citw`?

The best options are enabled by default, so often it is unnecessary to copy long option strings from webpages.

### Can you please put the `-b` option back?

youtube-dl now defaults to downloading the highest available quality. If you need a different format, use `-f`.

### I get HTTP error 402 when trying to download a video. What's this?

YouTube may require you to solve a CAPTCHA. Try opening the YouTube URL in a browser, solving the CAPTCHA, and then restart youtube-dl.

### Do I need any other programs?

youtube-dl works fine on its own on most sites.  If you want to convert video/audio, you'll need [avconv](https://libav.org/) or [ffmpeg](https://www.ffmpeg.org/). Videos streamed via RTMP require [rtmpdump](https://rtmpdump.mplayerhq.hu/). Downloading MMS and RTSP videos requires either [mplayer](https://mplayerhq.hu/) or [mpv](https://mpv.io/).

### I have downloaded a video but how can I play it?

Use any video player, such as [mpv](https://mpv.io/), [vlc](https://www.videolan.org/) or [mplayer](https://www.mplayerhq.hu/).

### I extracted a video URL with `-g`, but it does not play on another machine / in my web browser.

You may need to use `--cookies` and/or `--user-agent`. Some sites may only serve the video if it comes from the same IP address and/or has the same cookies/HTTP headers.

###