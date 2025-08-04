[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Your Go-To Tool for Downloading Videos from the Web

**Download videos from YouTube and thousands of other sites with ease using [youtube-dl](https://github.com/ytdl-org/youtube-dl) — the versatile command-line video downloader.**

## Key Features

*   **Wide Site Support:** Downloads videos from YouTube, Vimeo, and thousands of other video-sharing sites. See the [list of supported sites](https://ytdl-org.github.io/youtube-dl/supportedsites.html) for details.
*   **Format Selection:** Choose your preferred video and audio formats for downloads.
*   **Playlist and Channel Downloads:** Download entire playlists or all videos from a channel.
*   **Customizable Output:** Control filenames, output directories, and metadata.
*   **Subtitle Support:** Download and convert subtitles in various formats.
*   **Authentication:** Login to sites requiring authentication (e.g., YouTube) using your credentials.
*   **Post-Processing:** Convert videos to audio-only files and apply other post-processing options.
*   **Cross-Platform Compatibility:** Works seamlessly on Windows, macOS, and Linux.
*   **Free and Open Source:** Released into the public domain; modify, redistribute, and use as you wish.

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
    *   [Output template and Windows batch files](#output-template-and-windows-batch-files)
    *   [Output template examples](#output-template-examples)
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
    *   [ExtractorError: Could not find JS function u'OF'](#extractorerror-could-not-find-js-function-uof)
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
    *   [Adding support for a new site](#adding-support-for-a-new-site)
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
                *   [Example](#example-4)
            *   [Inline values](#inline-values)
                *   [Example](#example-5)
            *   [Collapse fallbacks](#collapse-fallbacks)
                *   [Example](#example-6)
            *   [Trailing parentheses](#trailing-parentheses)
                *   [Example](#example-7)
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

## Installation

Install youtube-dl on your system:

*   **UNIX (Linux, macOS, etc.):**
    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```
    If you don't have curl, use wget instead:
    ```bash
    sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```

*   **Windows:** [Download the .exe file](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) (except `%SYSTEMROOT%\System32`).

*   **Using pip:**
    ```bash
    sudo -H pip install --upgrade youtube-dl
    ```

*   **macOS (Homebrew):**
    ```bash
    brew install youtube-dl
    ```

*   **macOS (MacPorts):**
    ```bash
    sudo port install youtube-dl
    ```

For more installation options, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## Description

**youtube-dl** is a versatile command-line program designed for downloading videos from YouTube.com and numerous other video platforms. It's written in Python and is platform-independent, working on Unix-like systems, Windows, and macOS.  youtube-dl is released into the public domain, allowing you to freely modify, redistribute, and use it.

```bash
youtube-dl [OPTIONS] URL [URL...]
```

## Options

Run `youtube-dl --help` to see all available options. Some key options are listed below, grouped for clarity.

### Network Options
    --proxy URL                          Use the specified HTTP/HTTPS/SOCKS proxy.
    --socket-timeout SECONDS             Time to wait before giving up, in seconds
    --source-address IP                  Client-side IP address to bind to
    -4, --force-ipv4                     Make all connections via IPv4
    -6, --force-ipv6                     Make all connections via IPv6

### Geo Restriction
    --geo-verification-proxy URL         Use this proxy to verify the IP address
    --geo-bypass                         Bypass geographic restriction via faking X-Forwarded-For HTTP header
    --no-geo-bypass                      Do not bypass geographic restriction
    --geo-bypass-country CODE            Force bypass geographic restriction
    --geo-bypass-ip-block IP_BLOCK       Force bypass geographic restriction

### Video Selection
    --playlist-start NUMBER              Playlist video to start at (default is 1)
    --playlist-end NUMBER                Playlist video to end at (default is last)
    --playlist-items ITEM_SPEC           Playlist video items to download.
    --match-title REGEX                  Download only matching titles (regex or caseless sub-string)
    --reject-title REGEX                 Skip download for matching titles
    --max-downloads NUMBER               Abort after downloading NUMBER files
    --min-filesize SIZE                  Do not download any videos smaller than SIZE (e.g. 50k or 44.6m)
    --max-filesize SIZE                  Do not download any videos larger than SIZE (e.g. 50k or 44.6m)
    --date DATE                          Download only videos uploaded in this date
    --datebefore DATE                    Download only videos uploaded on or before this date (i.e. inclusive)
    --dateafter DATE                     Download only videos uploaded on or after this date (i.e. inclusive)
    --min-views COUNT                    Do not download any videos with less than COUNT views
    --max-views COUNT                    Do not download any videos with more than COUNT views
    --match-filter FILTER                Generic video filter.
    --no-playlist                        Download only the video, if the URL refers to a video and a playlist.
    --yes-playlist                       Download the playlist, if the URL refers to a video and a playlist.
    --age-limit YEARS                    Download only videos suitable for the given age
    --download-archive FILE              Download only videos not listed in the archive file. Record the IDs of all downloaded videos in it.
    --include-ads                        Download advertisements as well (experimental)

### Download Options
    -r, --limit-rate RATE                Maximum download rate in bytes per second (e.g. 50K or 4.2M)
    -R, --retries RETRIES                Number of retries (default is 10), or "infinite".
    --fragment-retries RETRIES           Number of retries for a fragment (default is 10), or "infinite" (DASH, hlsnative and ISM)
    --skip-unavailable-fragments         Skip unavailable fragments (DASH, hlsnative and ISM)
    --abort-on-unavailable-fragment      Abort downloading when some fragment is not available
    --keep-fragments                     Keep downloaded fragments on disk after downloading is finished; fragments are erased by default
    --buffer-size SIZE                   Size of download buffer (e.g. 1024 or 16K) (default is 1024)
    --no-resize-buffer                   Do not automatically adjust the buffer size.
    --http-chunk-size SIZE               Size of a chunk for chunk-based HTTP downloading (e.g. 10485760 or 10M)
    --playlist-reverse                   Download playlist videos in reverse order
    --playlist-random                    Download playlist videos in random order
    --xattr-set-filesize                 Set file xattribute ytdl.filesize with expected file size
    --hls-prefer-native                  Use the native HLS downloader instead of ffmpeg
    --hls-prefer-ffmpeg                  Use ffmpeg instead of the native HLS downloader
    --hls-use-mpegts                     Use the mpegts container for HLS videos, allowing to play the video while downloading (some players may not be able to play it)
    --external-downloader COMMAND        Use the specified external downloader. Currently supports aria2c,avconv,axel,curl,ffmpeg,httpie,wget
    --external-downloader-args ARGS      Give these arguments to the external downloader

### Filesystem Options
    -a, --batch-file FILE                File containing URLs to download ('-' for stdin), one URL per line.
    --id                                 Use only video ID in file name
    -o, --output TEMPLATE                Output filename template, see the "OUTPUT TEMPLATE" for all the info
    --output-na-placeholder PLACEHOLDER  Placeholder value for unavailable meta fields in output filename template (default is "NA")
    --autonumber-start NUMBER            Specify the start value for %(autonumber)s (default is 1)
    --restrict-filenames                 Restrict filenames to only ASCII characters, and avoid "&" and spaces in filenames
    -w, --no-overwrites                  Do not overwrite files
    -c, --continue                       Force resume of partially downloaded files.
    --no-continue                        Do not resume partially downloaded files (restart from beginning)
    --no-part                            Do not use .part files - write directly into output file
    --no-mtime                           Do not use the Last-modified header to set the file modification time
    --write-description                  Write video description to a .description file
    --write-info-json                    Write video metadata to a .info.json file
    --write-annotations                  Write video annotations to a .annotations.xml file
    --load-info-json FILE                JSON file containing the video information (created with the "--write- info-json" option)
    --cookies FILE                       File to read cookies from and dump cookie jar in
    --cache-dir DIR                      Location in the filesystem where youtube-dl can store some downloaded information permanently. By default $XDG_CACHE_HOME/youtube-dl or ~/.cache/youtube-dl . At the moment, only YouTube player files (for videos with obfuscated signatures) are cached, but that may change.
    --no-cache-dir                       Disable filesystem caching
    --rm-cache-dir                       Delete all filesystem cache files

### Thumbnail Options
    --write-thumbnail                    Write thumbnail image to disk
    --write-all-thumbnails               Write all thumbnail image formats to disk
    --list-thumbnails                    Simulate and list all available thumbnail formats

### Verbosity / Simulation Options
    -q, --quiet                          Activate quiet mode
    --no-warnings                        Ignore warnings
    -s, --simulate                       Do not download the video and do not write anything to disk
    --skip-download                      Do not download the video
    -g, --get-url                        Simulate, quiet but print URL
    -e, --get-title                      Simulate, quiet but print title
    --get-id                             Simulate, quiet but print id
    --get-thumbnail                      Simulate, quiet but print thumbnail URL
    --get-description                    Simulate, quiet but print video description
    --get-duration                       Simulate, quiet but print video length
    --get-filename                       Simulate, quiet but print output filename
    --get-format                         Simulate, quiet but print output format
    -j, --dump-json                      Simulate, quiet but print JSON information.
    -J, --dump-single-json               Simulate, quiet but print JSON information for each command-line argument.
    --print-json                         Be quiet and print the video information as JSON (video is still being downloaded).
    --newline                            Output progress bar as new lines
    --no-progress                        Do not print progress bar
    --console-title                      Display progress in console titlebar
    -v, --verbose                        Print various debugging information
    --dump-pages                         Print downloaded pages encoded using base64 to debug problems (very verbose)
    --write-pages                        Write downloaded intermediary pages to files in the current directory to debug problems
    --print-traffic                      Display sent and read HTTP traffic
    -C, --call-home                      Contact the youtube-dl server for debugging
    --no-call-home                       Do NOT contact the youtube-dl server for debugging

### Workarounds
    --encoding ENCODING                  Force the specified encoding (experimental)
    --no-check-certificate               Suppress HTTPS certificate validation
    --prefer-insecure                    Use an unencrypted connection to retrieve information about the video.
    --user-agent UA                      Specify a custom user agent
    --referer URL                        Specify a custom referer, use if the video access is restricted to one domain
    --add-header FIELD:VALUE             Specify a custom HTTP header and its value, separated by a colon ':'. You can use this option multiple times
    --bidi-workaround                    Work around terminals that lack bidirectional text support. Requires bidiv or fribidi executable in PATH
    --sleep-interval SECONDS             Number of seconds to sleep before each download when used alone or a lower bound of a range for randomized sleep before each download (minimum possible number of seconds to sleep) when used along with --max-sleep-interval.
    --max-sleep-interval SECONDS         Upper bound of a range for randomized sleep before each download (maximum possible number of seconds to sleep). Must only be used along with --min-sleep-interval.

### Video Format Options
    -f, --format FORMAT                  Video format code, see the "FORMAT SELECTION" for all the info
    --all-formats                        Download all available video formats
    --prefer-free-formats                Prefer free video formats unless a specific one is requested
    -F, --list-formats                   List all available formats of requested videos
    --youtube-skip-dash-manifest         Do not download the DASH manifests and related data on YouTube videos
    --merge-output-format FORMAT         If a merge is required (e.g. bestvideo+bestaudio), output to given container format. One of mkv, mp4, ogg, webm, flv. Ignored if no merge is required

### Subtitle Options
    --write-sub                          Write subtitle file
    --write-auto-sub                     Write automatically generated subtitle file (YouTube only)
    --all-subs                           Download all the available subtitles of the video
    --list-subs                          List all available subtitles for the video
    --sub-format FORMAT                  Subtitle format, accepts formats preference, for example: "srt" or "ass/srt/best"
    --sub-lang LANGS                     Languages of the subtitles to download (optional) separated by commas, use --list-subs for available language tags

### Authentication Options
    -u, --username USERNAME              Login with this account ID
    -p, --password PASSWORD              Account password. If this option is left out, youtube-dl will ask interactively.
    -2, --twofactor TWOFACTOR            Two-factor authentication code
    -n, --netrc                          Use .netrc authentication data
    --video-password PASSWORD            Video password (vimeo, youku)

### Adobe Pass Options
    --ap-mso MSO                         Adobe Pass multiple-system operator (TV provider) identifier, use --ap-list-mso for a list of available MSOs
    --ap-username USERNAME               Multiple-system operator account login
    --ap-password PASSWORD               Multiple-system operator account password.
    --ap-list-mso                        List all supported multiple-system operators

### Post-processing Options
    -x, --extract-audio                  Convert video files to audio-only files (requires ffmpeg/avconv and ffprobe/avprobe)
    --audio-format FORMAT                Specify audio format: "best", "aac", "flac", "mp3", "m4a", "opus", "vorbis", or "wav"; "best" by default; No effect without -x
    --audio-quality QUALITY              Specify ffmpeg/avconv audio quality, insert a value between 0 (better) and 9 (worse) for VBR or a specific bitrate like 128K (default 5)
    --recode-video FORMAT                Encode the video to another format if necessary (currently supported: mp4|flv|ogg|webm|mkv|avi)
    --postprocessor-args ARGS            Give these arguments to the postprocessor
    -k, --keep-video                     Keep the video file on disk after the post-processing; the video is erased by default
    --no-post-overwrites                 Do not overwrite post-processed files; the post-processed files are overwritten by default
    --embed-subs                         Embed subtitles in the video (only for mp4, webm and mkv videos)
    --embed-thumbnail                    Embed thumbnail in the audio as cover art
    --add-metadata                       Write metadata to the video file
    --metadata-from-title FORMAT         Parse additional metadata like song title / artist from the video title.
    --xattrs                             Write metadata to the video file's xattrs (using dublin core and xdg standards)
    --fixup POLICY                       Automatically correct known faults of the file. One of never (do nothing), warn (only emit a warning), detect_or_warn (the default; fix file if we can, warn otherwise)
    --prefer-avconv                      Prefer avconv over ffmpeg for running the postprocessors
    --prefer-ffmpeg                      Prefer ffmpeg over avconv for running the postprocessors (default)
    --ffmpeg-location PATH               Location of the ffmpeg/avconv binary; either the path to the binary or its containing directory.
    --exec CMD                           Execute a command on the file after downloading and post-processing, similar to find's -exec syntax.
    --convert-subs FORMAT                Convert the subtitles to other format (currently supported: srt|ass|vtt|lrc)

## Configuration

Customize youtube-dl's behavior by using a configuration file. The system-wide configuration file is located at `/etc/youtube-dl.conf` (Linux/macOS) and the user configuration file at `~/.config/youtube-dl/config` (Linux/macOS), `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf` (Windows).
For example:
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

Use `--ignore-config` to disable the configuration file for a specific run, or `--config-location` to specify a custom configuration file.

### Authentication with .netrc file

For extractors that support authentication (by providing login and password with `--username` and `--password`) you can automatically store credentials by using a [`.netrc` file](https://stackoverflow.com/tags/.netrc/info). Create a `.netrc` file in your `$HOME` directory and restrict permissions:

```bash
touch $HOME/.netrc
chmod a-rwx,u+rw $HOME/.netrc
```
Then add credentials in the following format, where *extractor* is the name of the extractor in lowercase:

```
machine <extractor> login <login> password <password>
```
For example:

```
machine youtube login myaccount@gmail.com password my_youtube_password
machine twitch login my_twitch_account_name password my_twitch_password
```
To activate authentication with the `.netrc` file you should pass `--netrc` to youtube-dl or place it in the [configuration file](#configuration).

On Windows you may also need to setup the `%HOME%` environment variable manually:
```bash
set HOME=%USERPROFILE%
```

## Output Template

Use the `-o` option to specify a template for the output filenames.  This allows for great flexibility in naming your downloaded files.

```bash
$ youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc
youtube-dl test video ''_ä↭𝕐.mp4

$ youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc --restrict-filenames
youtube-dl_test_video_.mp4
```

The template uses special sequences replaced with video metadata. Some available sequences are:

*   `id`: Video identifier
*   `title`: Video title
*   `url`: Video URL
*   `ext`: Video filename extension
*   `uploader`: Uploader's name
*   `upload_date`:  Video upload date (YYYYMMDD)
*   `width`, `height`: Video dimensions
*   `format`:  Human-readable format description

For example, `-o '%(title)s-%(id)s.%(ext)s'` creates a file named `youtube-dl test video-BaW_jenozKcj.mp4`.

### Output template and Windows batch files

If you are using an output template inside a Windows batch file then you must escape plain percent characters (`%`) by doubling, so that `-o "%(title)s-%(id)s.%(ext)s"` should become `-o "%%(title)s-%%(id)s.%%(ext)s"`.

### Output template examples

```bash
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

By default, youtube-dl downloads the best available quality.  Use the `-f` or `--format` option for more control.  Use `--list-formats` (`-F`) to see available formats for a video.

```bash
$ youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
```

You can select formats by:

*   Format code (e.g., `-f 22`)
*   File extension (e.g., `-f webm`)
*   Special names: `best`, `worst`, `bestvideo`, `bestaudio`, `worstvideo`, `worstaudio`
*   Filters (e.g., `-f "best[height=720]"` for 720p)
*   Merging video and audio (`-f <video-format>+<audio-format>`)

### Format selection examples

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

Filter videos based on upload date, title, and other criteria.

```bash
# Download only the videos uploaded in the last 6 months
$ youtube-dl --dateafter now-6months

# Download only the videos uploaded on January 1, 1970
$ youtube-dl --date 19700101

$ # Download only the videos uploaded in the 200x decade
$ youtube-dl --dateafter 20000101 --datebefore 20091231
```

## FAQ

### How do I update youtube-dl?

Run `youtube-dl -U` (or `sudo youtube-dl -U` on Linux). If you used pip, use `sudo pip install -U youtube-dl`. Update with your package manager if youtube-dl was installed through your distribution.  See [the download instructions](https://ytdl-org.github.io/youtube-dl/download.html) if you are having issues.

### youtube-dl is extremely slow to start on Windows

Add a file exclusion for `youtube-dl.exe` in Windows Defender settings.

### I'm getting an error `Unable to extract OpenGraph title` on YouTube playlists

Update to at least youtube-dl 2014.07.25 or later.

### I'm getting an error when trying to use output template: `error: using output template conflicts with using title, video ID or auto number`

Make sure you are not using `-o` with any of these options `-t`, `--title`, `--id`, `-A` or `--auto-number` set in command line or in a configuration file.

### Do I always have to pass `-citw`?

These flags are not typically needed. Use `-i` if you want to ignore download errors.

### Can you please put the `-b` option back?

youtube-dl now defaults to downloading the highest available quality (1080p or 720p in some cases),