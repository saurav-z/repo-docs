[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# YouTube-dl: Your Go-To Command-Line Tool for Video Downloads

Tired of struggling to save videos from the web? **youtube-dl is a versatile, open-source command-line tool that lets you download videos from YouTube and hundreds of other sites.**  This powerful program gives you full control over your video downloads, offering advanced features and extensive customization options.

[Visit the official repository for more details.](https://github.com/ytdl-org/youtube-dl)

## Key Features

*   **Wide Platform Support:** Works on Linux, macOS, Windows, and Unix-like systems.
*   **Broad Site Compatibility:** Downloads videos from YouTube and numerous other video hosting sites.  [See the full list of supported sites](https://ytdl-org.github.io/youtube-dl/supportedsites.html).
*   **Format Selection:** Choose your preferred video format and quality.
*   **Playlist Downloading:** Download entire playlists with ease.
*   **Subtitle Support:** Download subtitles in various languages.
*   **Customization Options:** Fine-tune downloads with a vast array of options including:
    *   Output filename templates.
    *   Proxy support.
    *   Download rate limiting.
    *   Metadata embedding.
    *   And much more!

## Installation

Get started with youtube-dl quickly on any system. Choose the method that best suits your needs:

**For all UNIX users (Linux, macOS, etc.):**

```bash
sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

**Alternative (if you don't have curl):**

```bash
sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

**For Windows users:**

[Download the .exe file](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory included in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29), avoiding `%SYSTEMROOT%\System32`.

**Using pip:**

```bash
sudo -H pip install --upgrade youtube-dl
```

**For macOS users (Homebrew):**

```bash
brew install youtube-dl
```

**For macOS users (MacPorts):**

```bash
sudo port install youtube-dl
```

For further options, including PGP signatures, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## Basic Usage

```bash
youtube-dl [OPTIONS] URL [URL...]
```

Replace `[OPTIONS]` with desired flags and `URL` with the video's web address.  For example:

```bash
youtube-dl https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

## Key Options & Functionality

*   **Update:** `-U` or `--update`: Update youtube-dl to the latest version.
*   **Help:** `-h` or `--help`: Displays all available options.
*   **Format Selection:** `-f FORMAT` or `--format FORMAT`:  Specify video format (e.g., `-f mp4`, `-f best`, `-f 22`). See detailed information in the [FORMAT SELECTION](#format-selection) section.
*   **List Formats:** `-F` or `--list-formats`: List all available formats for a video.
*   **Output Template:** `-o TEMPLATE` or `--output TEMPLATE`: Set a custom filename template. See the [OUTPUT TEMPLATE](#output-template) section.

### Network Options
*   **Proxy:** `--proxy URL`
*   **Geo Restriction Bypass:** `--geo-bypass`

### Video Selection
*   **Playlist Selection:** `--playlist-start NUMBER`, `--playlist-end NUMBER`
*   **Filters:** `--match-title REGEX`, `--reject-title REGEX`

### Download Options
*   **Rate Limit:** `-r RATE` or `--limit-rate RATE`: Specify a maximum download rate (e.g. 50k or 4.2M).

### Filesystem Options
*   **Batch File:** `-a FILE` or `--batch-file FILE`: Load URLs from a text file.
*   **Output Template:** `-o TEMPLATE` or `--output TEMPLATE`: Define how the downloaded video should be saved.

### Subtitle Options
*   **Download Subtitles:** `--write-sub`, `--write-auto-sub`, `--all-subs`

### Authentication Options
*   **Login Credentials:** `-u USERNAME`, `-p PASSWORD`

For a comprehensive list of options, see the [OPTIONS](#options) section below.

## Detailed Sections

Dive deeper into these important topics:

*   [OPTIONS](#options)
*   [CONFIGURATION](#configuration)
*   [OUTPUT TEMPLATE](#output-template)
*   [FORMAT SELECTION](#format-selection)
*   [VIDEO SELECTION](#video-selection)
*   [FAQ](#faq)
*   [EMBEDDING YOUTUBE-DL](#embedding-youtube-dl)

## OPTIONS

```
    -h, --help                           Print this help text and exit
    --version                            Print program version and exit
    -U, --update                         Update this program to latest version.
                                         Make sure that you have sufficient
                                         permissions (run with sudo if needed)
    -i, --ignore-errors                  Continue on download errors, for
                                         example to skip unavailable videos in a
                                         playlist
    --abort-on-error                     Abort downloading of further videos (in
                                         the playlist or the command line) if an
                                         error occurs
    --dump-user-agent                    Display the current browser
                                         identification
    --list-extractors                    List all supported extractors
    --extractor-descriptions             Output descriptions of all supported
                                         extractors
    --force-generic-extractor            Force extraction to use the generic
                                         extractor
    --default-search PREFIX              Use this prefix for unqualified URLs.
                                         For example "gvsearch2:" downloads two
                                         videos from google videos for youtube-
                                         dl "large apple". Use the value "auto"
                                         to let youtube-dl guess ("auto_warning"
                                         to emit a warning when guessing).
                                         "error" just throws an error. The
                                         default value "fixup_error" repairs
                                         broken URLs, but emits an error if this
                                         is not possible instead of searching.
    --ignore-config                      Do not read configuration files. When
                                         given in the global configuration file
                                         /etc/youtube-dl.conf: Do not read the
                                         user configuration in
                                         ~/.config/youtube-dl/config
                                         (%APPDATA%/youtube-dl/config.txt on
                                         Windows)
    --config-location PATH               Location of the configuration file;
                                         either the path to the config or its
                                         containing directory.
    --flat-playlist                      Do not extract the videos of a
                                         playlist, only list them.
    --mark-watched                       Mark videos watched (YouTube only)
    --no-mark-watched                    Do not mark videos watched (YouTube
                                         only)
    --no-color                           Do not emit color codes in output

## Network Options:
    --proxy URL                          Use the specified HTTP/HTTPS/SOCKS
                                         proxy. To enable SOCKS proxy, specify a
                                         proper scheme. For example
                                         socks5://127.0.0.1:1080/. Pass in an
                                         empty string (--proxy "") for direct
                                         connection
    --socket-timeout SECONDS             Time to wait before giving up, in
                                         seconds
    --source-address IP                  Client-side IP address to bind to
    -4, --force-ipv4                     Make all connections via IPv4
    -6, --force-ipv6                     Make all connections via IPv6

## Geo Restriction:
    --geo-verification-proxy URL         Use this proxy to verify the IP address
                                         for some geo-restricted sites. The
                                         default proxy specified by --proxy (or
                                         none, if the option is not present) is
                                         used for the actual downloading.
    --geo-bypass                         Bypass geographic restriction via
                                         faking X-Forwarded-For HTTP header
    --no-geo-bypass                      Do not bypass geographic restriction
                                         via faking X-Forwarded-For HTTP header
    --geo-bypass-country CODE            Force bypass geographic restriction
                                         with explicitly provided two-letter ISO
                                         3166-2 country code
    --geo-bypass-ip-block IP_BLOCK       Force bypass geographic restriction
                                         with explicitly provided IP block in
                                         CIDR notation

## Video Selection:
    --playlist-start NUMBER              Playlist video to start at (default is
                                         1)
    --playlist-end NUMBER                Playlist video to end at (default is
                                         last)
    --playlist-items ITEM_SPEC           Playlist video items to download.
                                         Specify indices of the videos in the
                                         playlist separated by commas like: "--
                                         playlist-items 1,2,5,8" if you want to
                                         download videos indexed 1, 2, 5, 8 in
                                         the playlist. You can specify range: "
                                         --playlist-items 1-3,7,10-13", it will
                                         download the videos at index 1, 2, 3,
                                         7, 10, 11, 12 and 13.
    --match-title REGEX                  Download only matching titles (regex or
                                         caseless sub-string)
    --reject-title REGEX                 Skip download for matching titles
                                         (regex or caseless sub-string)
    --max-downloads NUMBER               Abort after downloading NUMBER files
    --min-filesize SIZE                  Do not download any videos smaller than
                                         SIZE (e.g. 50k or 44.6m)
    --max-filesize SIZE                  Do not download any videos larger than
                                         SIZE (e.g. 50k or 44.6m)
    --date DATE                          Download only videos uploaded in this
                                         date
    --datebefore DATE                    Download only videos uploaded on or
                                         before this date (i.e. inclusive)
    --dateafter DATE                     Download only videos uploaded on or
                                         after this date (i.e. inclusive)
    --min-views COUNT                    Do not download any videos with less
                                         than COUNT views
    --max-views COUNT                    Do not download any videos with more
                                         than COUNT views
    --match-filter FILTER                Generic video filter. Specify any key
                                         (see the "OUTPUT TEMPLATE" for a list
                                         of available keys) to match if the key
                                         is present, !key to check if the key is
                                         not present, key > NUMBER (like
                                         "comment_count > 12", also works with
                                         >=, <, <=, !=, =) to compare against a
                                         number, key = 'LITERAL' (like "uploader
                                         = 'Mike Smith'", also works with !=) to
                                         match against a string literal and & to
                                         require multiple matches. Values which
                                         are not known are excluded unless you
                                         put a question mark (?) after the
                                         operator. For example, to only match
                                         videos that have been liked more than
                                         100 times and disliked less than 50
                                         times (or the dislike functionality is
                                         not available at the given service),
                                         but who also have a description, use
                                         --match-filter "like_count > 100 &
                                         dislike_count <? 50 & description" .
    --no-playlist                        Download only the video, if the URL
                                         refers to a video and a playlist.
    --yes-playlist                       Download the playlist, if the URL
                                         refers to a video and a playlist.
    --age-limit YEARS                    Download only videos suitable for the
                                         given age
    --download-archive FILE              Download only videos not listed in the
                                         archive file. Record the IDs of all
                                         downloaded videos in it.
    --include-ads                        Download advertisements as well
                                         (experimental)

## Download Options:
    -r, --limit-rate RATE                Maximum download rate in bytes per
                                         second (e.g. 50K or 4.2M)
    -R, --retries RETRIES                Number of retries (default is 10), or
                                         "infinite".
    --fragment-retries RETRIES           Number of retries for a fragment
                                         (default is 10), or "infinite" (DASH,
                                         hlsnative and ISM)
    --skip-unavailable-fragments         Skip unavailable fragments (DASH,
                                         hlsnative and ISM)
    --abort-on-unavailable-fragment      Abort downloading when some fragment is
                                         not available
    --keep-fragments                     Keep downloaded fragments on disk after
                                         downloading is finished; fragments are
                                         erased by default
    --buffer-size SIZE                   Size of download buffer (e.g. 1024 or
                                         16K) (default is 1024)
    --no-resize-buffer                   Do not automatically adjust the buffer
                                         size. By default, the buffer size is
                                         automatically resized from an initial
                                         value of SIZE.
    --http-chunk-size SIZE               Size of a chunk for chunk-based HTTP
                                         downloading (e.g. 10485760 or 10M)
                                         (default is disabled). May be useful
                                         for bypassing bandwidth throttling
                                         imposed by a webserver (experimental)
    --playlist-reverse                   Download playlist videos in reverse
                                         order
    --playlist-random                    Download playlist videos in random
                                         order
    --xattr-set-filesize                 Set file xattribute ytdl.filesize with
                                         expected file size
    --hls-prefer-native                  Use the native HLS downloader instead
                                         of ffmpeg
    --hls-prefer-ffmpeg                  Use ffmpeg instead of the native HLS
                                         downloader
    --hls-use-mpegts                     Use the mpegts container for HLS
                                         videos, allowing to play the video
                                         while downloading (some players may not
                                         be able to play it)
    --external-downloader COMMAND        Use the specified external downloader.
                                         Currently supports aria2c,avconv,axel,c
                                         url,ffmpeg,httpie,wget
    --external-downloader-args ARGS      Give these arguments to the external
                                         downloader

## Filesystem Options:
    -a, --batch-file FILE                File containing URLs to download ('-'
                                         for stdin), one URL per line. Lines
                                         starting with '#', ';' or ']' are
                                         considered as comments and ignored.
    --id                                 Use only video ID in file name
    -o, --output TEMPLATE                Output filename template, see the
                                         "OUTPUT TEMPLATE" for all the info
    --output-na-placeholder PLACEHOLDER  Placeholder value for unavailable meta
                                         fields in output filename template
                                         (default is "NA")
    --autonumber-start NUMBER            Specify the start value for
                                         %(autonumber)s (default is 1)
    --restrict-filenames                 Restrict filenames to only ASCII
                                         characters, and avoid "&" and spaces in
                                         filenames
    -w, --no-overwrites                  Do not overwrite files
    -c, --continue                       Force resume of partially downloaded
                                         files. By default, youtube-dl will
                                         resume downloads if possible.
    --no-continue                        Do not resume partially downloaded
                                         files (restart from beginning)
    --no-part                            Do not use .part files - write directly
                                         into output file
    --no-mtime                           Do not use the Last-modified header to
                                         set the file modification time
    --write-description                  Write video description to a
                                         .description file
    --write-info-json                    Write video metadata to a .info.json
                                         file
    --write-annotations                  Write video annotations to a
                                         .annotations.xml file
    --load-info-json FILE                JSON file containing the video
                                         information (created with the "--write-
                                         info-json" option)
    --cookies FILE                       File to read cookies from and dump
                                         cookie jar in
    --cache-dir DIR                      Location in the filesystem where
                                         youtube-dl can store some downloaded
                                         information permanently. By default
                                         $XDG_CACHE_HOME/youtube-dl or
                                         ~/.cache/youtube-dl . At the moment,
                                         only YouTube player files (for videos
                                         with obfuscated signatures) are cached,
                                         but that may change.
    --no-cache-dir                       Disable filesystem caching
    --rm-cache-dir                       Delete all filesystem cache files

## Thumbnail Options:
    --write-thumbnail                    Write thumbnail image to disk
    --write-all-thumbnails               Write all thumbnail image formats to
                                         disk
    --list-thumbnails                    Simulate and list all available
                                         thumbnail formats

## Verbosity / Simulation Options:
    -q, --quiet                          Activate quiet mode
    --no-warnings                        Ignore warnings
    -s, --simulate                       Do not download the video and do not
                                         write anything to disk
    --skip-download                      Do not download the video
    -g, --get-url                        Simulate, quiet but print URL
    -e, --get-title                      Simulate, quiet but print title
    --get-id                             Simulate, quiet but print id
    --get-thumbnail                      Simulate, quiet but print thumbnail URL
    --get-description                    Simulate, quiet but print video
                                         description
    --get-duration                       Simulate, quiet but print video length
    --get-filename                       Simulate, quiet but print output
                                         filename
    --get-format                         Simulate, quiet but print output format
    -j, --dump-json                      Simulate, quiet but print JSON
                                         information. See the "OUTPUT TEMPLATE"
                                         for a description of available keys.
    -J, --dump-single-json               Simulate, quiet but print JSON
                                         information for each command-line
                                         argument. If the URL refers to a
                                         playlist, dump the whole playlist
                                         information in a single line.
    --print-json                         Be quiet and print the video
                                         information as JSON (video is still
                                         being downloaded).
    --newline                            Output progress bar as new lines
    --no-progress                        Do not print progress bar
    --console-title                      Display progress in console titlebar
    -v, --verbose                        Print various debugging information
    --dump-pages                         Print downloaded pages encoded using
                                         base64 to debug problems (very verbose)
    --write-pages                        Write downloaded intermediary pages to
                                         files in the current directory to debug
                                         problems
    --print-traffic                      Display sent and read HTTP traffic
    -C, --call-home                      Contact the youtube-dl server for
                                         debugging
    --no-call-home                       Do NOT contact the youtube-dl server
                                         for debugging

## Workarounds:
    --encoding ENCODING                  Force the specified encoding
                                         (experimental)
    --no-check-certificate               Suppress HTTPS certificate validation
    --prefer-insecure                    Use an unencrypted connection to
                                         retrieve information about the video.
                                         (Currently supported only for YouTube)
    --user-agent UA                      Specify a custom user agent
    --referer URL                        Specify a custom referer, use if the
                                         video access is restricted to one
                                         domain
    --add-header FIELD:VALUE             Specify a custom HTTP header and its
                                         value, separated by a colon ':'. You
                                         can use this option multiple times
    --bidi-workaround                    Work around terminals that lack
                                         bidirectional text support. Requires
                                         bidiv or fribidi executable in PATH
    --sleep-interval SECONDS             Number of seconds to sleep before each
                                         download when used alone or a lower
                                         bound of a range for randomized sleep
                                         before each download (minimum possible
                                         number of seconds to sleep) when used
                                         along with --max-sleep-interval.
    --max-sleep-interval SECONDS         Upper bound of a range for randomized
                                         sleep before each download (maximum
                                         possible number of seconds to sleep).
                                         Must only be used along with --min-
                                         sleep-interval.

## Video Format Options:
    -f, --format FORMAT                  Video format code, see the "FORMAT
                                         SELECTION" for all the info
    --all-formats                        Download all available video formats
    --prefer-free-formats                Prefer free video formats unless a
                                         specific one is requested
    -F, --list-formats                   List all available formats of requested
                                         videos
    --youtube-skip-dash-manifest         Do not download the DASH manifests and
                                         related data on YouTube videos
    --merge-output-format FORMAT         If a merge is required (e.g.
                                         bestvideo+bestaudio), output to given
                                         container format. One of mkv, mp4, ogg,
                                         webm, flv. Ignored if no merge is
                                         required

## Subtitle Options:
    --write-sub                          Write subtitle file
    --write-auto-sub                     Write automatically generated subtitle
                                         file (YouTube only)
    --all-subs                           Download all the available subtitles of
                                         the video
    --list-subs                          List all available subtitles for the
                                         video
    --sub-format FORMAT                  Subtitle format, accepts formats
                                         preference, for example: "srt" or
                                         "ass/srt/best"
    --sub-lang LANGS                     Languages of the subtitles to download
                                         (optional) separated by commas, use
                                         --list-subs for available language tags

## Authentication Options:
    -u, --username USERNAME              Login with this account ID
    -p, --password PASSWORD              Account password. If this option is
                                         left out, youtube-dl will ask
                                         interactively.
    -2, --twofactor TWOFACTOR            Two-factor authentication code
    -n, --netrc                          Use .netrc authentication data
    --video-password PASSWORD            Video password (vimeo, youku)

## Adobe Pass Options:
    --ap-mso MSO                         Adobe Pass multiple-system operator (TV
                                         provider) identifier, use --ap-list-mso
                                         for a list of available MSOs
    --ap-username USERNAME               Multiple-system operator account login
    --ap-password PASSWORD               Multiple-system operator account
                                         password. If this option is left out,
                                         youtube-dl will ask interactively.
    --ap-list-mso                        List all supported multiple-system
                                         operators

## Post-processing Options:
    -x, --extract-audio                  Convert video files to audio-only files
                                         (requires ffmpeg/avconv and
                                         ffprobe/avprobe)
    --audio-format FORMAT                Specify audio format: "best", "aac",
                                         "flac", "mp3", "m4a", "opus", "vorbis",
                                         or "wav"; "best" by default; No effect
                                         without -x
    --audio-quality QUALITY              Specify ffmpeg/avconv audio quality,
                                         insert a value between 0 (better) and 9
                                         (worse) for VBR or a specific bitrate
                                         like 128K (default 5)
    --recode-video FORMAT                Encode the video to another format if
                                         necessary (currently supported:
                                         mp4|flv|ogg|webm|mkv|avi)
    --postprocessor-args ARGS            Give these arguments to the
                                         postprocessor
    -k, --keep-video                     Keep the video file on disk after the
                                         post-processing; the video is erased by
                                         default
    --no-post-overwrites                 Do not overwrite post-processed files;
                                         the post-processed files are
                                         overwritten by default
    --embed-subs                         Embed subtitles in the video (only for
                                         mp4, webm and mkv videos)
    --embed-thumbnail                    Embed thumbnail in the audio as cover
                                         art
    --add-metadata                       Write metadata to the video file
    --metadata-from-title FORMAT         Parse additional metadata like song
                                         title / artist from the video title.
                                         The format syntax is the same as
                                         --output. Regular expression with named
                                         capture groups may also be used. The
                                         parsed parameters replace existing
                                         values. Example: --metadata-from-title
                                         "%(artist)s - %(title)s" matches a
                                         title like "Coldplay - Paradise".
                                         Example (regex): --metadata-from-title
                                         "(?P<artist>.+?) - (?P<title>.+)"
    --xattrs                             Write metadata to the video file's
                                         xattrs (using dublin core and xdg
                                         standards)
    --fixup POLICY                       Automatically correct known faults of
                                         the file. One of never (do nothing),
                                         warn (only emit a warning),
                                         detect_or_warn (the default; fix file
                                         if we can, warn otherwise)
    --prefer-avconv                      Prefer avconv over ffmpeg for running
                                         the postprocessors
    --prefer-ffmpeg                      Prefer ffmpeg over avconv for running
                                         the postprocessors (default)
    --ffmpeg-location PATH               Location of the ffmpeg/avconv binary;
                                         either the path to the binary or its
                                         containing directory.
    --exec CMD                           Execute a command on the file after
                                         downloading and post-processing,
                                         similar to find's -exec syntax.
                                         Example: --exec 'adb push {}
                                         /sdcard/Music/ && rm {}'
    --convert-subs FORMAT                Convert the subtitles to other format
                                         (currently supported: srt|ass|vtt|lrc)
```

## CONFIGURATION

You can configure youtube-dl by placing any supported command line option to a configuration file. On Linux and macOS, the system wide configuration file is located at `/etc/youtube-dl.conf` and the user wide configuration file at `~/.config/youtube-dl/config`. On Windows, the user wide configuration file locations are `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`. Note that by default configuration file may not exist so you may need to create it yourself.

For example, with the following configuration file youtube-dl will always extract the audio, not copy the mtime, use a proxy and save all videos under `Movies` directory in your home directory:
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

Note that options in configuration file are just the same options aka switches used in regular command line calls thus there **must be no whitespace** after `-` or `--`, e.g. `-o` or `--proxy` but not `- o` or `-- proxy`.

You can use `--ignore-config` if you want to disable the configuration file for a particular youtube-dl run.

You can also use `--config-location` if you want to use custom configuration file for a particular youtube-dl run.

### Authentication with `.netrc` file

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

## OUTPUT TEMPLATE

The `-o` option allows users to indicate a template for the output file names.

**tl;dr:** [navigate me to examples](#output-template-examples).

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
 - `playlist_uploader_id` (string): Nickname or id of the playlist uploader

Available for the video that belongs to some logical chapter or section:

 - `chapter` (string): Name or title of the chapter the video belongs to
 - `chapter_number` (numeric): Number of the chapter the video belongs to
 - `chapter_id` (string): Id of the chapter the video belongs to

Available for the video that is an episode of some series or programme:

 - `series` (string): Title of the series or programme the video episode belongs to
 - `season` (string): Title of the season the video episode belongs to
 - `season_number` (numeric): Number of the season the video episode belongs to
 - `season_id` (string): Id of the season the video episode belongs to
 - `episode` (string): Title of the video episode
 - `episode_number` (numeric): Number of the video episode within a season
 - `episode_id` (string): Id of the video episode

Available for the media that is a track or a part of a music album:

 - `track` (string): Title of the track
 - `track_number` (numeric): Number of the track within an album or a disc
 - `track_id` (string): Id of the track
 - `artist` (string): Artist(s) of the track
 - `genre` (string): Genre(s) of the track
 - `album` (string): Title of the album the track belongs to
 - `album_type` (string): Type of the album
 - `album_artist` (string): List