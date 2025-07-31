[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Download Videos from YouTube and Beyond 

**youtube-dl is a versatile command-line tool that lets you download videos from YouTube and thousands of other sites.** 

[Visit the Official Repository](https://github.com/ytdl-org/youtube-dl)

**Key Features:**

*   **Wide Site Support:** Download from YouTube, Vimeo, and hundreds of other video platforms.
*   **Format Selection:** Choose from a variety of video and audio formats, or let youtube-dl choose the best one.
*   **Playlist and Channel Downloads:** Download entire playlists, channels, or user uploads with ease.
*   **Customization:**  Control output filenames, video quality, and more with extensive command-line options.
*   **Metadata:** Automatically include video descriptions, titles, and other metadata.
*   **Subtitle Support:** Download subtitles in multiple languages.
*   **Post-Processing:** Convert videos to audio-only formats (MP3, etc.) and embed metadata.
*   **Cross-Platform:** Works on Windows, macOS, and Linux systems.

---

## Table of Contents

*   [Installation](#installation)
    *   [UNIX (Linux, macOS, etc.)](#unix-linux-macos-etc)
    *   [Windows](#windows)
    *   [Using pip](#using-pip)
    *   [macOS using Homebrew](#macos-using-homebrew)
    *   [macOS using MacPorts](#macos-using-macports)
    *   [Developer Instructions](#developer-instructions)
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
    *   [Output template examples](#output-template-examples)
    *   [Output template and Windows batch files](#output-template-and-windows-batch-files)
*   [Format Selection](#format-selection)
    *   [Format selection examples](#format-selection-examples)
*   [Video Selection](#video-selection)
*   [FAQ](#faq)
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
    *   [ExtractorError: Could not find JS function u'OF'](#extractoreerror-could-not-find-js-function-uof)
    *   [HTTP Error 429: Too Many Requests or 402: Payment Required](#http-error-429-too-many-requests-or-402-payment-required)
    *   [SyntaxError: Non-ASCII character](#syntaxerror-non-ascii-character)
    *   [What is this binary file? Where has the code gone?](#what-is-this-binary-file-where-has-the-code-gone)
    *   [The exe throws an error due to missing `MSVCR100.dll`](#the-exe-throws-an-error-due-to-missing-msvcr100dll)
    *   [On Windows, how should I set up ffmpeg and youtube-dl? Where should I put the exe files?](#on-windows-how-should-i-set-up-ffmpeg-and-youtube-dl-where-should-i-put-the-exe-files)
    *   [How do I put downloads into a specific folder?](#how-do-i-put-downloads-into-a-specific-folder)
    *   [How do I download a video starting with a `-`?](#how-do-i-download-a-video-starting-with-a--)
    *   [How do I pass cookies to youtube-dl?](#how-do-i-pass-cookies-to-youtube-dl)
    *   [How do I stream directly to media player?](#how-do-i-stream-directly-to-media-player)
    *   [How do I download only new videos from a playlist?](#how-do-i-download-only-new-videos-from-a-playlist)
    *   [Should I add `--hls-prefer-native` into my config?](#should-i-add--hls-prefer-native-into-my-config)
    *   [Can you add support for this anime video site, or site which shows current movies for free?](#can-you-add-support-for-this-anime-video-site-or-site-which-shows-current-movies-for-free)
    *   [How can I speed up work on my issue?](#how-can-i-speed-up-work-on-my-issue)
    *   [How can I detect whether a given URL is supported by youtube-dl?](#how-can-i-detect-whether-a-given-url-is-supported-by-youtube-dl)
*   [Why do I need to go through that much red tape when filing bugs?](#why-do-i-need-to-go-through-that-much-red-tape-when-filing-bugs)
*   [Developer Instructions](#developer-instructions)
    *   [youtube-dl coding conventions](#youtube-dl-coding-conventions)
        *   [Mandatory and optional metafields](#mandatory-and-optional-metafields)
            *   [Example](#example)
        *   [Provide fallbacks](#provide-fallbacks)
            *   [Example](#example-1)
        *   [Regular expressions](#regular-expressions)
            *   [Don't capture groups you don't use](#dont-capture-groups-you-dont-use)
                *   [Example](#example-2)
            *   [Make regular expressions relaxed and flexible](#make-regular-expressions-relaxed-and-flexible)
                *   [Example](#example-3)
        *   [Long lines policy](#long-lines-policy)
        *   [Inline values](#inline-values)
            *   [Example](#example-4)
        *   [Collapse fallbacks](#collapse-fallbacks)
            *   [Example](#example-5)
        *   [Trailing parentheses](#trailing-parentheses)
            *   [Example](#example-6)
        *   [Use convenience conversion and parsing functions](#use-convenience-conversion-and-parsing-functions)
            *   [More examples](#more-examples)
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
*   [Copyright](#copyright)

---

## Installation

### UNIX (Linux, macOS, etc.)

To install quickly on all UNIX systems, type:

```bash
sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

If you do not have curl, use wget:

```bash
sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

### Windows

Download the `.exe` file from [here](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory that is in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29), except for `%SYSTEMROOT%\System32`.

### Using pip

```bash
sudo -H pip install --upgrade youtube-dl
```

This will update youtube-dl if you already installed it. See the [pypi page](https://pypi.python.org/pypi/youtube_dl) for more information.

### macOS using Homebrew

```bash
brew install youtube-dl
```

### macOS using MacPorts

```bash
sudo port install youtube-dl
```

### Developer Instructions

Refer to the [developer instructions](#developer-instructions) for instructions on working with the git repository.

### YouTube-dl Download Page

For further options, including PGP signatures, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

---

## Description

youtube-dl is a command-line program designed to download videos from YouTube.com and many other video platforms. It is written in Python and is compatible with various operating systems including Unix, Windows, and macOS.  It is released into the public domain.

```bash
youtube-dl [OPTIONS] URL [URL...]
```

---

## Options

### Network Options

*   `--proxy URL`: Use the specified HTTP/HTTPS/SOCKS proxy.
*   `--socket-timeout SECONDS`: Time to wait before giving up, in seconds.
*   `--source-address IP`: Client-side IP address to bind to.
*   `-4, --force-ipv4`: Make all connections via IPv4.
*   `-6, --force-ipv6`: Make all connections via IPv6.

### Geo Restriction

*   `--geo-verification-proxy URL`: Use this proxy to verify the IP address for geo-restricted sites.
*   `--geo-bypass`: Bypass geographic restriction via faking X-Forwarded-For HTTP header.
*   `--no-geo-bypass`: Do not bypass geographic restriction via faking X-Forwarded-For HTTP header.
*   `--geo-bypass-country CODE`: Force bypass geographic restriction with explicitly provided two-letter ISO 3166-2 country code.
*   `--geo-bypass-ip-block IP_BLOCK`: Force bypass geographic restriction with explicitly provided IP block in CIDR notation.

### Video Selection

*   `--playlist-start NUMBER`: Playlist video to start at (default is 1).
*   `--playlist-end NUMBER`: Playlist video to end at (default is last).
*   `--playlist-items ITEM_SPEC`: Playlist video items to download (e.g., `--playlist-items 1,2,5,8` or `--playlist-items 1-3,7,10-13`).
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
*   `--match-filter FILTER`: Generic video filter.  Specify any key (see the "OUTPUT TEMPLATE" for a list of available keys) to match if the key is present, !key to check if the key is not present, key > NUMBER (like "comment_count > 12", also works with >=, <, <=, !=, =) to compare against a number, key = 'LITERAL' (like "uploader = 'Mike Smith'", also works with !=) to match against a string literal and & to require multiple matches. Values which are not known are excluded unless you put a question mark (?) after the operator. For example, to only match videos that have been liked more than 100 times and disliked less than 50 times (or the dislike functionality is not available at the given service), but who also have a description, use --match-filter "like_count > 100 & dislike_count <? 50 & description" .
*   `--no-playlist`: Download only the video if the URL refers to a video and a playlist.
*   `--yes-playlist`: Download the playlist if the URL refers to a video and a playlist.
*   `--age-limit YEARS`: Download only videos suitable for the given age.
*   `--download-archive FILE`: Download only videos not listed in the archive file. Record the IDs of all downloaded videos in it.
*   `--include-ads`: Download advertisements as well (experimental).

### Download Options

*   `-r, --limit-rate RATE`: Maximum download rate in bytes per second (e.g., 50K or 4.2M).
*   `-R, --retries RETRIES`: Number of retries (default is 10), or "infinite".
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
*   `--external-downloader COMMAND`: Use the specified external downloader.  Currently supports aria2c,avconv,axel,c url,ffmpeg,httpie,wget.
*   `--external-downloader-args ARGS`: Give these arguments to the external downloader.

### Filesystem Options

*   `-a, --batch-file FILE`: File containing URLs to download ('-' for stdin), one URL per line.
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
*   `--add-header FIELD:VALUE`: Specify a custom HTTP header and its value, separated by a colon ':'.
*   `--bidi-workaround`: Work around terminals that lack bidirectional text support.
*   `--sleep-interval SECONDS`: Number of seconds to sleep before each download when used alone or a lower bound of a range for randomized sleep before each download (minimum possible number of seconds to sleep) when used along with --max-sleep-interval.
*   `--max-sleep-interval SECONDS`: Upper bound of a range for randomized sleep before each download (maximum possible number of seconds to sleep). Must only be used along with --min-sleep-interval.

### Video Format Options

*   `-f, --format FORMAT`: Video format code, see the "FORMAT SELECTION" for all the info.
*   `--all-formats`: Download all available video formats.
*   `--prefer-free-formats`: Prefer free video formats unless a specific one is requested.
*   `-F, --list-formats`: List all available formats of requested videos.
*   `--youtube-skip-dash-manifest`: Do not download the DASH manifests and related data on YouTube videos.
*   `--merge-output-format FORMAT`: If a merge is required (e.g. bestvideo+bestaudio), output to given container format. One of mkv, mp4, ogg, webm, flv. Ignored if no merge is required.

### Subtitle Options

*   `--write-sub`: Write subtitle file.
*   `--write-auto-sub`: Write automatically generated subtitle file (YouTube only).
*   `--all-subs`: Download all the available subtitles of the video.
*   `--list-subs`: List all available subtitles for the video.
*   `--sub-format FORMAT`: Subtitle format (e.g., "srt" or "ass/srt/best").
*   `--sub-lang LANGS`: Languages of the subtitles to download (optional), separated by commas.

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

*   `-x, --extract-audio`: Convert video files to audio-only files (requires ffmpeg/avconv and ffprobe/avprobe).
*   `--audio-format FORMAT`: Specify audio format: "best", "aac", "flac", "mp3", "m4a", "opus", "vorbis", or "wav".
*   `--audio-quality QUALITY`: Specify ffmpeg/avconv audio quality (0-9 for VBR or a specific bitrate like 128K).
*   `--recode-video FORMAT`: Encode the video to another format if necessary (currently supported: mp4|flv|ogg|webm|mkv|avi).
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
*   `--ffmpeg-location PATH`: Location of the ffmpeg/avconv binary; either the path to the binary or its containing directory.
*   `--exec CMD`: Execute a command on the file after downloading and post-processing.
*   `--convert-subs FORMAT`: Convert the subtitles to other format (currently supported: srt|ass|vtt|lrc).

---

## Configuration

You can configure youtube-dl by placing any supported command line option in a configuration file. System-wide configuration files are in `/etc/youtube-dl.conf` (Linux/macOS) and user-specific files are located at `~/.config/youtube-dl/config` (Linux/macOS) and `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf` (Windows). Use `--ignore-config` to disable the configuration file, and `--config-location` to specify a custom configuration file.

```bash
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

Note that options in the configuration file do **not** include whitespace.

### Authentication with .netrc file

You may also want to configure automatic credentials storage for extractors that support authentication (by providing login and password with `--username` and `--password`) in order not to pass credentials as command line arguments on every youtube-dl execution and prevent tracking plain text passwords in the shell command history. You can achieve this using a [`.netrc` file](https://stackoverflow.com/tags/.netrc/info) on a per extractor basis. For that you will need to create a `.netrc` file in your `$HOME` and restrict permissions to read/write by only you:
```
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
```
set HOME=%USERPROFILE%
```

---

## Output Template

The `-o` option allows users to indicate a template for the output file names.

The basic usage is not to set any template arguments when downloading a single file, like in `youtube-dl -o funny_video.flv "https://some/video"`. However, it may contain special sequences that will be replaced when downloading each video. The special sequences may be formatted according to [python string formatting operations](https://docs.python.org/2/library/stdtypes.html#string-formatting). For example, `%(NAME)s` or `%(NAME)05d`. To clarify, that is a percent symbol followed by a name in parentheses, followed by formatting operations. Allowed names along with sequence type are:

 - `id` (string): Video identifier
 - `title` (string): Video title
 - `url` (string): Video URL
 - `ext` (string): Video filename extension
 - `alt_title` (string): A secondary title of the video
 - `display_id` (string): An alternative identifier for the video
 - `uploader` (string): Full name of the video uploader
 - `license` (string): License name the video is licensed under
 - `creator` (string): The creator of the video
 - `release_date` (string): The date (YYYYMMDD) when the video was released
 - `timestamp` (numeric): UNIX timestamp of the moment the video became available
 - `upload_date` (string): Video upload date (YYYYMMDD)
 - `uploader_id` (string): Nickname or id of the video uploader
 - `channel` (string): Full name of the channel the video is uploaded on
 - `channel_id` (string): Id of the channel
 - `location` (string): Physical location where the video was filmed
 - `duration` (numeric): Length of the video in seconds
 - `view_count` (numeric): How many users have watched the video on the platform
 - `like_count` (numeric): Number of positive ratings of the video
 - `dislike_count` (numeric): Number of negative ratings of the video
 - `repost_count` (numeric): Number of reposts of the video
 - `average_rating` (numeric): Average rating give by users, the scale used depends on the webpage
 - `comment_count` (numeric): Number of comments on the video
 - `age_limit` (numeric): Age restriction for the video (years)
 - `is_live` (boolean): Whether this video is a live stream or a fixed-length video
 - `start_time` (numeric): Time in seconds where the reproduction should start, as specified in the URL
 - `end_time` (numeric): Time in seconds where the reproduction should end, as specified in the URL
 - `format` (string): A human-readable description of the format
 - `format_id` (string): Format code specified by `--format`
 - `format_note` (string): Additional info about the format
 - `width` (numeric): Width of the video
 - `height` (numeric): Height of the video
 - `resolution` (string): Textual description of width and height
 - `tbr` (numeric): Average bitrate of audio and video in KBit/s
 - `abr` (numeric): Average audio bitrate in KBit/s
 - `acodec` (string): Name of the audio codec in use
 - `asr` (numeric): Audio sampling rate in Hertz
 - `vbr` (numeric): Average video bitrate in KBit/s
 - `fps` (numeric): Frame rate
 - `vcodec` (string): Name of the video codec in use
 - `container` (string): Name of the container format
 - `filesize` (numeric): The number of bytes, if known in advance
 - `filesize_approx` (numeric): An estimate for the number of bytes
 - `protocol` (string): The protocol that will be used for the actual download
 - `extractor` (string): Name of the extractor
 - `extractor_key` (string): Key name of the extractor
 - `epoch` (numeric): Unix epoch when creating the file
 - `autonumber` (numeric): Number that will be increased with each download, starting at `--autonumber-start`
 - `playlist` (string): Name or id of the playlist that contains the video
 - `playlist_index` (numeric): Index of the video in the playlist padded with leading zeros according to the total length of the playlist
 - `playlist_id` (string): Playlist identifier
 - `playlist_title` (string): Playlist title
 - `playlist_uploader` (string): Full name of the playlist uploader
 - `playlist_uploader_id` (string): Nickname or id