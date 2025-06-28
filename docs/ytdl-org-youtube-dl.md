[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# YouTube-DL: Download Videos from YouTube and Beyond

Tired of buffering? **Download videos from YouTube and hundreds of other sites with `youtube-dl`!** This powerful command-line tool is your go-to solution for saving videos for offline viewing or archiving your favorite content.  Visit the [original repo](https://github.com/ytdl-org/youtube-dl) for more information.

**Key Features:**

*   **Broad Site Support:** Works with YouTube, Vimeo, Dailymotion, and hundreds of other video platforms ([see supported sites](https://ytdl-org.github.io/youtube-dl/supportedsites.html)).
*   **Format Selection:** Choose from a wide range of video formats and qualities.
*   **Playlist and Channel Downloads:** Easily download entire playlists and channels.
*   **Metadata Extraction:** Automatically extracts video titles, descriptions, and more.
*   **Customization:** Extensive options for output file naming, download speed limiting, and more.
*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux.
*   **Open Source:**  Freely available, modifyable, and redistributable.

## Table of Contents

*   [Installation](#installation)
*   [Description](#description)
*   [Options](#options)
    *   [Network Options:](#network-options)
    *   [Geo Restriction:](#geo-restriction)
    *   [Video Selection:](#video-selection)
    *   [Download Options:](#download-options)
    *   [Filesystem Options:](#filesystem-options)
    *   [Thumbnail Options:](#thumbnail-options)
    *   [Verbosity / Simulation Options:](#verbosity--simulation-options)
    *   [Workarounds:](#workarounds)
    *   [Video Format Options:](#video-format-options)
    *   [Subtitle Options:](#subtitle-options)
    *   [Authentication Options:](#authentication-options)
    *   [Adobe Pass Options:](#adobe-pass-options)
    *   [Post-processing Options:](#post-processing-options)
*   [Configuration](#configuration)
    *   [Authentication with `.netrc` file](#authentication-with-.netrc-file)
*   [Output Template](#output-template)
    *   [Output template examples](#output-template-examples)
*   [Format Selection](#format-selection)
    *   [Format selection examples](#format-selection-examples)
*   [Video Selection](#video-selection)
*   [FAQ](#faq)
*   [Developer Instructions](#developer-instructions)
    *   [Adding support for a new site](#adding-support-for-a-new-site)
    *   [youtube-dl coding conventions](#youtube-dl-coding-conventions)
    *   [Mandatory and optional metafields](#mandatory-and-optional-metafields)
    *   [Provide fallbacks](#provide-fallbacks)
    *   [Regular expressions](#regular-expressions)
    *   [Long lines policy](#long-lines-policy)
    *   [Inline values](#inline-values)
    *   [Collapse fallbacks](#collapse-fallbacks)
    *   [Trailing parentheses](#trailing-parentheses)
    *   [Use convenience conversion and parsing functions](#use-convenience-conversion-and-parsing-functions)
    *   [Safely extract optional description from parsed JSON](#safely-extract-optional-description-from-parsed-json)
    *   [Safely extract more optional metadata](#safely-extract-more-optional-metadata)
    *   [Safely extract nested lists](#safely-extract-nested-lists)
*   [Embedding youtube-dl](#embedding-youtube-dl)
*   [Bugs](#bugs)
    *   [Opening a bug report or suggestion](#opening-a-bug-report-or-suggestion)
    *   [Is the description of the issue itself sufficient?](#is-the-description-of-the-issue-itself-sufficient)
    *   [Is the issue already documented?](#is-the-issue-already-documented)
    *   [Are you using the latest version?](#are-you-using-the-latest-version)
    *   [Why are existing options not enough?](#why-are-existing-options-not-enough)
    *   [Is there enough context in your bug report?](#is-there-enough-context-in-your-bug-report)
    *   [Does the issue involve one problem, and one problem only?](#does-the-issue-involve-one-problem-and-one-problem-only)
    *   [Is anyone going to need the feature?](#is-anyone-going-to-need-the-feature)
    *   [Is your question about youtube-dl?](#is-your-question-about-youtube-dl)
    *   [How can I speed up work on my issue?](#how-can-i-speed-up-work-on-my-issue)
    *   [How can I detect whether a given URL is supported by youtube-dl?](#how-can-i-detect-whether-a-given-url-is-supported-by-youtube-dl)
*   [Why do I need to go through that much red tape when filing bugs?](#why-do-i-need-to-go-through-that-much-red-tape-when-filing-bugs)
*   [Copyright](#copyright)

## Installation

### UNIX (Linux, macOS, etc.)

To install for all users, run the following commands:

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

Download the [`.exe file`](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29), *except* for `%SYSTEMROOT%\System32`.

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

For advanced installation options and PGP signatures, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## Description

**youtube-dl** is a versatile command-line tool designed for downloading videos from YouTube.com and numerous other websites. It requires Python (version 2.6, 2.7, or 3.2+) and works on various operating systems including Unix-like systems, Windows, and macOS. It's released into the public domain, allowing for modification, redistribution, and unrestricted use.

To get started, simply use the following command structure:

```bash
youtube-dl [OPTIONS] URL [URL...]
```

## Options

For detailed usage, run `youtube-dl -h`.

### Network Options:

*   `--proxy URL`: Use a specified HTTP/HTTPS/SOCKS proxy.
*   `--socket-timeout SECONDS`: Set the socket timeout in seconds.
*   `--source-address IP`: Bind to a specific client-side IP address.
*   `-4, --force-ipv4`: Force IPv4 connections.
*   `-6, --force-ipv6`: Force IPv6 connections.

### Geo Restriction:

*   `--geo-verification-proxy URL`: Use a proxy to verify the IP address.
*   `--geo-bypass`: Bypass geographic restrictions.
*   `--no-geo-bypass`: Do not bypass geographic restrictions.
*   `--geo-bypass-country CODE`: Force bypass with a country code (ISO 3166-2).
*   `--geo-bypass-ip-block IP_BLOCK`: Force bypass with an IP block in CIDR notation.

### Video Selection:

*   `--playlist-start NUMBER`: Start playlist download at this number (default: 1).
*   `--playlist-end NUMBER`: End playlist download at this number (default: last).
*   `--playlist-items ITEM_SPEC`: Download specific playlist items (e.g., `1,2,5,8` or `1-3,7,10-13`).
*   `--match-title REGEX`: Download only videos with matching titles (regex or substring).
*   `--reject-title REGEX`: Skip videos with matching titles.
*   `--max-downloads NUMBER`: Limit the number of downloads.
*   `--min-filesize SIZE`: Do not download videos smaller than a certain size (e.g., `50k` or `44.6m`).
*   `--max-filesize SIZE`: Do not download videos larger than a certain size.
*   `--date DATE`: Download videos uploaded on a specific date.
*   `--datebefore DATE`: Download videos uploaded on or before a specific date.
*   `--dateafter DATE`: Download videos uploaded on or after a specific date.
*   `--min-views COUNT`: Do not download videos with fewer views.
*   `--max-views COUNT`: Do not download videos with more views.
*   `--match-filter FILTER`: Generic video filter.
*   `--no-playlist`: Download only the video if the URL refers to both a video and a playlist.
*   `--yes-playlist`: Download the playlist if the URL refers to both a video and a playlist.
*   `--age-limit YEARS`: Download only videos suitable for the given age.
*   `--download-archive FILE`: Download only videos not listed in the archive file.
*   `--include-ads`: Download advertisements as well (experimental).

### Download Options:

*   `-r, --limit-rate RATE`: Limit download rate (e.g., `50K` or `4.2M`).
*   `-R, --retries RETRIES`: Number of retries (default: 10) or "infinite".
*   `--fragment-retries RETRIES`: Number of retries for a fragment (default: 10).
*   `--skip-unavailable-fragments`: Skip unavailable fragments.
*   `--abort-on-unavailable-fragment`: Abort if a fragment is unavailable.
*   `--keep-fragments`: Keep downloaded fragments.
*   `--buffer-size SIZE`: Set download buffer size (default: 1024).
*   `--no-resize-buffer`: Do not adjust the buffer size.
*   `--http-chunk-size SIZE`: Size of chunk for chunk-based HTTP downloading.
*   `--playlist-reverse`: Download playlist videos in reverse order.
*   `--playlist-random`: Download playlist videos in random order.
*   `--xattr-set-filesize`: Set file xattribute `ytdl.filesize`.
*   `--hls-prefer-native`: Use the native HLS downloader.
*   `--hls-prefer-ffmpeg`: Use ffmpeg for HLS downloads.
*   `--hls-use-mpegts`: Use the mpegts container for HLS videos.
*   `--external-downloader COMMAND`: Use an external downloader (e.g., `aria2c`, `wget`).
*   `--external-downloader-args ARGS`: Arguments for the external downloader.

### Filesystem Options:

*   `-a, --batch-file FILE`: Download from a file with URLs (one per line).
*   `--id`: Use only the video ID in the filename.
*   `-o, --output TEMPLATE`: Output filename template.
*   `--output-na-placeholder PLACEHOLDER`: Placeholder for unavailable metadata (default: "NA").
*   `--autonumber-start NUMBER`: Start autonumbering at this value (default: 1).
*   `--restrict-filenames`: Restrict filenames to ASCII and avoid special characters.
*   `-w, --no-overwrites`: Do not overwrite existing files.
*   `-c, --continue`: Resume partially downloaded files.
*   `--no-continue`: Do not resume downloads.
*   `--no-part`: Do not use `.part` files.
*   `--no-mtime`: Do not set the file modification time.
*   `--write-description`: Write video description to a `.description` file.
*   `--write-info-json`: Write video metadata to a `.info.json` file.
*   `--write-annotations`: Write video annotations to a `.annotations.xml` file.
*   `--load-info-json FILE`: Load video information from a `.info.json` file.
*   `--cookies FILE`: Read cookies from a file.
*   `--cache-dir DIR`: Location for youtube-dl's cache.
*   `--no-cache-dir`: Disable caching.
*   `--rm-cache-dir`: Delete cache files.

### Thumbnail Options:

*   `--write-thumbnail`: Write thumbnail image to disk.
*   `--write-all-thumbnails`: Write all available thumbnail formats to disk.
*   `--list-thumbnails`: List available thumbnail formats.

### Verbosity / Simulation Options:

*   `-q, --quiet`: Activate quiet mode.
*   `--no-warnings`: Ignore warnings.
*   `-s, --simulate`: Do not download; just simulate.
*   `--skip-download`: Do not download the video.
*   `-g, --get-url`: Simulate, print URL.
*   `-e, --get-title`: Simulate, print title.
*   `--get-id`: Simulate, print ID.
*   `--get-thumbnail`: Simulate, print thumbnail URL.
*   `--get-description`: Simulate, print video description.
*   `--get-duration`: Simulate, print video length.
*   `--get-filename`: Simulate, print output filename.
*   `--get-format`: Simulate, print output format.
*   `-j, --dump-json`: Simulate, print JSON info.
*   `-J, --dump-single-json`: Simulate, print JSON for each argument.
*   `--print-json`: Print video info as JSON (while downloading).
*   `--newline`: Output progress bar on new lines.
*   `--no-progress`: Do not print progress bar.
*   `--console-title`: Display progress in the console titlebar.
*   `-v, --verbose`: Print debugging information.
*   `--dump-pages`: Print downloaded pages (encoded with base64).
*   `--write-pages`: Write downloaded intermediary pages to files.
*   `--print-traffic`: Display HTTP traffic.
*   `-C, --call-home`: Contact the youtube-dl server for debugging.
*   `--no-call-home`: Do NOT contact the server.

### Workarounds:

*   `--encoding ENCODING`: Force a specific encoding.
*   `--no-check-certificate`: Suppress HTTPS certificate validation.
*   `--prefer-insecure`: Use an unencrypted connection (YouTube only).
*   `--user-agent UA`: Specify a custom user agent.
*   `--referer URL`: Specify a custom referer.
*   `--add-header FIELD:VALUE`: Specify custom HTTP headers.
*   `--bidi-workaround`: Work around bidirectional text issues.
*   `--sleep-interval SECONDS`: Sleep before each download.
*   `--max-sleep-interval SECONDS`: Maximum sleep interval.

### Video Format Options:

*   `-f, --format FORMAT`: Video format code (see FORMAT SELECTION).
*   `--all-formats`: Download all available video formats.
*   `--prefer-free-formats`: Prefer free video formats.
*   `-F, --list-formats`: List all available formats.
*   `--youtube-skip-dash-manifest`: Do not download DASH manifests (YouTube).
*   `--merge-output-format FORMAT`: Output to the given container format.

### Subtitle Options:

*   `--write-sub`: Write subtitle file.
*   `--write-auto-sub`: Write automatically generated subtitles (YouTube only).
*   `--all-subs`: Download all available subtitles.
*   `--list-subs`: List available subtitles.
*   `--sub-format FORMAT`: Subtitle format (e.g., `srt` or `ass/srt/best`).
*   `--sub-lang LANGS`: Languages of the subtitles (separated by commas).

### Authentication Options:

*   `-u, --username USERNAME`: Login with a username.
*   `-p, --password PASSWORD`: Account password.
*   `-2, --twofactor TWOFACTOR`: Two-factor authentication code.
*   `-n, --netrc`: Use `.netrc` authentication data.
*   `--video-password PASSWORD`: Video password (Vimeo, Youku).

### Adobe Pass Options:

*   `--ap-mso MSO`: Adobe Pass MSO identifier.
*   `--ap-username USERNAME`: MSO account login.
*   `--ap-password PASSWORD`: MSO account password.
*   `--ap-list-mso`: List supported MSOs.

### Post-processing Options:

*   `-x, --extract-audio`: Convert video to audio (requires ffmpeg/avconv).
*   `--audio-format FORMAT`: Audio format (e.g., `mp3`, `aac`).
*   `--audio-quality QUALITY`: Audio quality (0-9 for VBR, bitrate like 128K).
*   `--recode-video FORMAT`: Encode the video to another format.
*   `--postprocessor-args ARGS`: Give arguments to the postprocessor.
*   `-k, --keep-video`: Keep the video after post-processing.
*   `--no-post-overwrites`: Do not overwrite post-processed files.
*   `--embed-subs`: Embed subtitles in the video.
*   `--embed-thumbnail`: Embed a thumbnail.
*   `--add-metadata`: Write metadata to the video file.
*   `--metadata-from-title FORMAT`: Parse metadata from the video title.
*   `--xattrs`: Write metadata to video file's xattrs.
*   `--fixup POLICY`: Automatically correct file faults.
*   `--prefer-avconv`: Prefer avconv over ffmpeg.
*   `--prefer-ffmpeg`: Prefer ffmpeg over avconv (default).
*   `--ffmpeg-location PATH`: Location of the ffmpeg/avconv binary.
*   `--exec CMD`: Execute a command after downloading and post-processing.
*   `--convert-subs FORMAT`: Convert subtitles to another format.

## Configuration

You can configure youtube-dl by placing command-line options in a configuration file. On Linux and macOS, the system-wide configuration file is at `/etc/youtube-dl.conf` and the user-specific one is at `~/.config/youtube-dl/config`. On Windows, user-specific configuration file locations are `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`.  Note that by default, configuration file may not exist so you may need to create it yourself.

For example, to always extract audio, disable mtime, use a proxy, and save videos in a `Movies` directory, you could use the following configuration file:

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

Use `--ignore-config` to disable the configuration file for a specific run or `--config-location` to use a custom config file.

### Authentication with `.netrc` file

For extractors that support authentication, you can configure automatic credentials storage (login and password provided with `--username` and `--password`) using a `.netrc` file.
1.  Create a `.netrc` file in your `$HOME` directory:
    ```bash
    touch $HOME/.netrc
    chmod a-rwx,u+rw $HOME/.netrc
    ```
2.  Add credentials for an extractor (lowercase extractor name):
    ```
    machine <extractor> login <login> password <password>
    ```
    For example:
    ```
    machine youtube login myaccount@gmail.com password my_youtube_password
    machine twitch login my_twitch_account_name password my_twitch_password
    ```
    Pass `--netrc` to youtube-dl or include in your configuration.  On Windows, you may also need to set the `%HOME%` environment variable:
    ```bash
    set HOME=%USERPROFILE%
    ```

## Output Template

The `-o` or `--output` option lets you customize the output file names.  See below for examples.

The basic usage is to set the desired name, such as: `youtube-dl -o funny_video.flv "https://some/video"`.

Output templates can also contain special sequences that will be replaced when downloading each video, which may be formatted according to [python string formatting operations](https://docs.python.org/2/library/stdtypes.html#string-formatting). For example, `%(NAME)s` or `%(NAME)05d`. Allowed names along with sequence type are:

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
*   `autonumber` (numeric): Number that will be increased with each download, starting at `--autonumber-start`
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

Each sequence will be replaced by the corresponding video metadata. Unavailable metadata will be replaced by a placeholder value, provided with `--output-na-placeholder` (default: `NA`).

For example, `-o %(title)s-%(id)s.%(ext)s` will create a file named `youtube-dl test video-BaW_jenozKcj.mp4`.

For numeric sequences, you can use formatting, e.g., `%(view_count)05d` results in a zero-padded view count (`00042`).

Output templates support hierarchical paths, like `-o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s'`. Missing directories will be automatically created.

To use percent literals in an output template use `%%`. To output to stdout use `-o -`.

The current default template is `%(title)s-%(id)s.%(ext)s`.

To avoid special characters like spaces or ampersands, add the `--restrict-filenames` flag.

#### Output template examples

Note that on Windows, you might need to use double quotes instead of single quotes.

```bash
# Get the filename of a video
$ youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc

# Get the filename of a video with restricted filenames
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

The `-f` or `--format` option allows selecting the desired video format.

By default youtube-dl tries to download the best available quality, i.e. if you want the best quality you **don't need** to pass any special options, youtube-dl will guess it for you by **default**.

The general syntax for format selection is `--format FORMAT` or shorter `-f FORMAT` where `FORMAT` is a *selector expression*, i.e. an expression that describes format or formats you would like to download.

The simplest case is requesting a specific format, for example with `-f 22` you can download the format with format code equal to 22. You can get the list of available format codes for particular video using `--list-formats` or `-F`. Note that these format codes are extractor specific.

You can also use a file extension (currently `3gp`, `aac`, `flv`, `m4a`, `mp3`, `mp4`, `ogg`, `wav`, `webm` are supported) to download the best quality format of a particular file extension served as a single file, e.g. `-f webm` will download the best quality format with the `webm` extension served as a single file.

You can also use special names to select particular edge case formats:

*   `best`: Select the best quality format represented by a single file with video and audio.
*   `worst`: Select the worst quality format represented by a single file with video and audio.
*   `bestvideo`: Select the best quality video-only format (e.g. DASH video). May not be available.
*   `worstvideo`: Select the worst quality video-only format. May not be available.
*   `bestaudio`: Select the best quality audio only-format. May not be available.
*   `worstaudio`: Select the worst quality audio only-format. May not be available.

For example, to download the worst quality video-only format you can use `-f worstvideo`.

If you want to download multiple videos and they don't have the same formats available, you can specify the order of preference using slashes. Note that slash is left-associative, i.e. formats on the left hand side are preferred, for example `-f 22/17/18` will download format 22 if it's available, otherwise it will download format 17 if it's available, otherwise it will download format 18 if it's available, otherwise it will complain that no suitable formats are available for download.

If you want to download several formats of the same video use a comma as a separator, e.g. `-f 22,17,18` will download all these three formats, of course if they are available. Or a more sophisticated example combined with the precedence feature: `-f 136/137/mp4/bestvideo,140/m4a/bestaudio`.

You can also filter the video formats by putting a condition in brackets, as in `-f "best[height=720]"` (or `-f "[filesize>10M]"`).

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

Note that none of the aforementioned meta fields are guaranteed to be present since this solely depends on the metadata obtained by particular extractor, i.e. the metadata offered by the video hoster.

Formats for which the value is not known are excluded unless you put a question mark (`?`) after the operator. You can combine format filters, so `-f "[height <=? 720][tbr>500]"` selects up to 720p videos (or videos where