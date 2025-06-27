[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Download Videos from the Web with Ease

**youtube-dl is a versatile command-line tool that allows you to download videos from YouTube and thousands of other websites.** This powerful open-source program gives you complete control over your video downloads.  [Visit the original repository](https://github.com/ytdl-org/youtube-dl) to learn more.

## Key Features

*   **Wide Site Support:** Download videos from YouTube and a vast array of other video platforms.
*   **Format Selection:**  Choose from a variety of video formats and qualities to suit your needs.
*   **Playlist and Channel Downloads:** Download entire playlists, channels, or specific videos within them.
*   **Customization:** Configure output filenames, download locations, and more using command-line options and configuration files.
*   **Metadata Extraction:**  Automatically extract video titles, descriptions, and other metadata.
*   **Post-Processing:** Convert downloaded videos to audio files, embed subtitles, and add metadata.
*   **Extensive Options:**  Fine-tune your downloads with options for network settings, geo-restrictions, and more.

## Table of Contents

*   [Installation](#installation)
    *   [UNIX (Linux, macOS, etc.)](#installation)
    *   [Windows](#installation)
    *   [Using pip](#installation)
    *   [macOS with Homebrew or MacPorts](#installation)
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
    *   [Output template and Windows batch files](#output-template-and-windows-batch-files)
*   [Format Selection](#format-selection)
    *   [Format selection examples](#format-selection-examples)
*   [Video Selection](#video-selection)
*   [FAQ](#faq)
*   [Developer Instructions](#developer-instructions)
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
*   [Copyright](#copyright)

## Installation

### UNIX (Linux, macOS, etc.)

To install for all UNIX users, run the following commands in your terminal:

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

Download the executable file:  [youtube-dl.exe](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory within your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) environment variable, except for `%SYSTEMROOT%\System32` (e.g. **do not** put it in `C:\Windows\System32`).

### Using pip

You can also install youtube-dl using pip:

```bash
sudo -H pip install --upgrade youtube-dl
```

### macOS with Homebrew or MacPorts

Install youtube-dl with [Homebrew](https://brew.sh/):

```bash
brew install youtube-dl
```

Or with [MacPorts](https://www.macports.org/):

```bash
sudo port install youtube-dl
```

For further options, including PGP signatures, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## Description

**youtube-dl** is a command-line program designed to download videos from YouTube.com and a variety of other websites. It's platform-independent, working on Unix-like systems, Windows, and macOS.  The source code is released into the public domain, permitting you to modify and redistribute it as desired.

```bash
youtube-dl [OPTIONS] URL [URL...]
```

## Options

The following sections explain the options available when using youtube-dl.

*   `-h, --help`: Prints help text and exits.
*   `--version`: Prints the program version and exits.
*   `-U, --update`: Updates youtube-dl to the latest version. Requires sufficient permissions (use `sudo` if needed).
*   `-i, --ignore-errors`: Continues downloading even if errors occur, such as skipping unavailable videos in a playlist.
*   `--abort-on-error`: Halts downloading of further videos (in a playlist or from the command line) if an error is encountered.
*   `--dump-user-agent`: Displays the current browser identification.
*   `--list-extractors`: Lists all supported extractors.
*   `--extractor-descriptions`: Outputs descriptions of all supported extractors.
*   `--force-generic-extractor`: Forces extraction to use the generic extractor.
*   `--default-search PREFIX`:  Sets a prefix for unqualified URLs.
    *   For example, `"gvsearch2:"` downloads two videos from Google Videos for "youtube-dl large apple."  Use `"auto"` to let youtube-dl guess (`"auto_warning"` will show a warning when guessing).  `"error"` causes an error. The default value "fixup\_error" repairs broken URLs, but emits an error if not possible instead of searching.
*   `--ignore-config`:  Disables reading of configuration files.  When used in the global configuration file (`/etc/youtube-dl.conf`), it prevents reading the user configuration in `~/.config/youtube-dl/config` (`%APPDATA%/youtube-dl/config.txt` on Windows).
*   `--config-location PATH`: Specifies the location of the configuration file, either the file path or its containing directory.
*   `--flat-playlist`:  Lists videos in a playlist without extracting them.
*   `--mark-watched`:  Marks videos as watched (YouTube only).
*   `--no-mark-watched`:  Does not mark videos as watched (YouTube only).
*   `--no-color`:  Disables color codes in the output.

### Network Options

*   `--proxy URL`: Uses the specified HTTP/HTTPS/SOCKS proxy.  Enable SOCKS by specifying a proper scheme (e.g., `socks5://127.0.0.1:1080/`). An empty string (`--proxy ""`) enables a direct connection.
*   `--socket-timeout SECONDS`: Sets the timeout (in seconds) before giving up.
*   `--source-address IP`:  Specifies the client-side IP address to bind to.
*   `-4, --force-ipv4`: Forces all connections to use IPv4.
*   `-6, --force-ipv6`: Forces all connections to use IPv6.

### Geo Restriction

*   `--geo-verification-proxy URL`: Uses this proxy for verifying the IP address for some geo-restricted sites.  The default proxy (set by `--proxy` or none) is used for downloading.
*   `--geo-bypass`:  Bypasses geographic restrictions using a fake X-Forwarded-For HTTP header.
*   `--no-geo-bypass`:  Disables bypassing geographic restrictions.
*   `--geo-bypass-country CODE`: Forces bypass of geographic restrictions with a two-letter ISO 3166-2 country code.
*   `--geo-bypass-ip-block IP_BLOCK`: Forces bypass of geographic restrictions with an IP block in CIDR notation.

### Video Selection

*   `--playlist-start NUMBER`: Starts downloading a playlist at the specified video number (default: 1).
*   `--playlist-end NUMBER`: Ends downloading a playlist at the specified video number (default: last).
*   `--playlist-items ITEM_SPEC`: Downloads specific playlist items.  Specify indices, separated by commas (e.g., `--playlist-items 1,2,5,8` to download videos indexed 1, 2, 5, and 8). Ranges can also be used (`--playlist-items 1-3,7,10-13`).
*   `--match-title REGEX`: Downloads only videos with matching titles (regex or caseless sub-string).
*   `--reject-title REGEX`: Skips downloads for videos with matching titles.
*   `--max-downloads NUMBER`: Aborts after downloading a specified number of files.
*   `--min-filesize SIZE`:  Skips videos smaller than the specified size (e.g., `50k` or `44.6m`).
*   `--max-filesize SIZE`:  Skips videos larger than the specified size.
*   `--date DATE`: Downloads videos uploaded on a specific date.
*   `--datebefore DATE`: Downloads videos uploaded on or before a date.
*   `--dateafter DATE`: Downloads videos uploaded on or after a date.
*   `--min-views COUNT`:  Skips videos with fewer than a specified number of views.
*   `--max-views COUNT`:  Skips videos with more than a specified number of views.
*   `--match-filter FILTER`:  Filters videos based on generic criteria.
    *   Specify any key (see "OUTPUT TEMPLATE" for available keys) to match if present, `!key` if not present, `key > NUMBER` (also works with `>=`, `<`, `<=`, `!=`, `=`) to compare to a number, and `key = 'LITERAL'` (also works with `!=`) to match a string literal.  Use `&` to require multiple matches.  Values which are not known are excluded unless a question mark (`?`) follows the operator.  (e.g., `--match-filter "like_count > 100 & dislike_count <? 50 & description"`)
*   `--no-playlist`: Downloads only the video if the URL refers to both a video and a playlist.
*   `--yes-playlist`: Downloads the playlist if the URL refers to both a video and a playlist.
*   `--age-limit YEARS`: Downloads only videos suitable for the given age.
*   `--download-archive FILE`: Downloads only videos not listed in the archive file. Records the IDs of downloaded videos.
*   `--include-ads`: Downloads advertisements (experimental).

### Download Options

*   `-r, --limit-rate RATE`: Limits the maximum download rate in bytes per second (e.g., `50K` or `4.2M`).
*   `-R, --retries RETRIES`: Sets the number of retries (default is 10) or use "infinite."
*   `--fragment-retries RETRIES`:  Sets the number of retries for a fragment (default is 10) or use "infinite" (DASH, hlsnative, and ISM).
*   `--skip-unavailable-fragments`: Skips unavailable fragments (DASH, hlsnative, and ISM).
*   `--abort-on-unavailable-fragment`: Aborts download if a fragment is unavailable.
*   `--keep-fragments`: Keeps downloaded fragments on disk after downloading is complete (fragments are normally erased).
*   `--buffer-size SIZE`: Sets the download buffer size (e.g., `1024` or `16K`; default: 1024).
*   `--no-resize-buffer`:  Disables automatic buffer size adjustment.
*   `--http-chunk-size SIZE`: Sets the chunk size for chunk-based HTTP downloading (e.g., `10485760` or `10M`; disabled by default). May help bypass bandwidth throttling.
*   `--playlist-reverse`: Downloads playlist videos in reverse order.
*   `--playlist-random`: Downloads playlist videos in random order.
*   `--xattr-set-filesize`:  Sets the `ytdl.filesize` file xattribute with the expected file size.
*   `--hls-prefer-native`:  Uses the native HLS downloader instead of ffmpeg.
*   `--hls-prefer-ffmpeg`:  Uses ffmpeg for HLS downloading.
*   `--hls-use-mpegts`: Uses the mpegts container for HLS videos, allowing playback while downloading.
*   `--external-downloader COMMAND`:  Uses an external downloader (e.g., `aria2c`, `avconv`, `axel`, `curl`, `ffmpeg`, `httpie`, `wget`).
*   `--external-downloader-args ARGS`:  Passes arguments to the external downloader.

### Filesystem Options

*   `-a, --batch-file FILE`: Specifies a file containing URLs to download (one URL per line; use `-` for stdin). Lines starting with `#`, `;`, or `]` are treated as comments and ignored.
*   `--id`: Uses only the video ID in the filename.
*   `-o, --output TEMPLATE`:  Sets the output filename template (see the "OUTPUT TEMPLATE" section).
*   `--output-na-placeholder PLACEHOLDER`:  Sets a placeholder value for unavailable metadata fields in the output filename (default: "NA").
*   `--autonumber-start NUMBER`:  Sets the starting value for `%(autonumber)s` (default: 1).
*   `--restrict-filenames`: Restricts filenames to ASCII characters and avoids spaces and "&".
*   `-w, --no-overwrites`: Does not overwrite existing files.
*   `-c, --continue`:  Forces the resume of partially downloaded files (youtube-dl resumes downloads if possible by default).
*   `--no-continue`: Disables resuming partially downloaded files (restarts from the beginning).
*   `--no-part`:  Disables the use of `.part` files (writes directly to the output file).
*   `--no-mtime`:  Disables using the Last-modified header to set the file modification time.
*   `--write-description`: Writes the video description to a `.description` file.
*   `--write-info-json`: Writes video metadata to a `.info.json` file.
*   `--write-annotations`:  Writes video annotations to a `.annotations.xml` file.
*   `--load-info-json FILE`: Loads video information from a `.info.json` file.
*   `--cookies FILE`: Specifies a file to read cookies from and dump a cookie jar in.
*   `--cache-dir DIR`:  Specifies a directory for youtube-dl to store downloaded information (defaults: `$XDG_CACHE_HOME/youtube-dl` or `~/.cache/youtube-dl`).
*   `--no-cache-dir`: Disables filesystem caching.
*   `--rm-cache-dir`: Deletes all filesystem cache files.

### Thumbnail Options

*   `--write-thumbnail`:  Writes the thumbnail image to disk.
*   `--write-all-thumbnails`:  Writes all available thumbnail image formats to disk.
*   `--list-thumbnails`:  Lists all available thumbnail formats (simulation).

### Verbosity / Simulation Options

*   `-q, --quiet`: Activates quiet mode.
*   `--no-warnings`:  Ignores warnings.
*   `-s, --simulate`:  Simulates the download (does not download the video).
*   `--skip-download`:  Does not download the video.
*   `-g, --get-url`:  Simulates, but prints the URL.
*   `-e, --get-title`:  Simulates, but prints the title.
*   `--get-id`: Simulates, but prints the ID.
*   `--get-thumbnail`: Simulates, but prints the thumbnail URL.
*   `--get-description`:  Simulates, but prints the video description.
*   `--get-duration`: Simulates, but prints the video length.
*   `--get-filename`: Simulates, but prints the output filename.
*   `--get-format`: Simulates, but prints the output format.
*   `-j, --dump-json`: Simulates, but prints JSON information (see "OUTPUT TEMPLATE" for keys).
*   `-J, --dump-single-json`:  Simulates, but prints JSON information for each command-line argument.  If the URL is a playlist, it dumps all playlist information in one line.
*   `--print-json`: Prints video information as JSON (while the video is still being downloaded).
*   `--newline`:  Outputs the progress bar as new lines.
*   `--no-progress`:  Disables the progress bar.
*   `--console-title`: Displays the progress in the console title bar.
*   `-v, --verbose`:  Prints debugging information.
*   `--dump-pages`: Prints downloaded pages (base64-encoded) for debugging.
*   `--write-pages`: Writes downloaded intermediary pages to files in the current directory for debugging.
*   `--print-traffic`: Displays sent and read HTTP traffic.
*   `-C, --call-home`: Contacts the youtube-dl server for debugging.
*   `--no-call-home`:  Does NOT contact the youtube-dl server for debugging.

### Workarounds

*   `--encoding ENCODING`: Forces the specified encoding (experimental).
*   `--no-check-certificate`:  Suppresses HTTPS certificate validation.
*   `--prefer-insecure`:  Uses an unencrypted connection to retrieve information about the video (currently supported only for YouTube).
*   `--user-agent UA`: Specifies a custom user agent.
*   `--referer URL`:  Specifies a custom referer; use if the video access is restricted to a specific domain.
*   `--add-header FIELD:VALUE`:  Specifies a custom HTTP header and its value, separated by a colon. You can use this multiple times.
*   `--bidi-workaround`:  Works around terminals that lack bidirectional text support (requires `bidiv` or `fribidi`).
*   `--sleep-interval SECONDS`:  Sets the sleep interval before each download (or a lower bound for randomized sleep when used with `--max-sleep-interval`).
*   `--max-sleep-interval SECONDS`:  Sets the upper bound for randomized sleep. Must be used with `--min-sleep-interval`.

### Video Format Options

*   `-f, --format FORMAT`:  Specifies the video format code (see "FORMAT SELECTION" section).
*   `--all-formats`: Downloads all available video formats.
*   `--prefer-free-formats`: Prefers free video formats unless a specific one is requested.
*   `-F, --list-formats`: Lists all available formats of requested videos.
*   `--youtube-skip-dash-manifest`:  Disables downloading DASH manifests and related data on YouTube videos.
*   `--merge-output-format FORMAT`:  If a merge is required, outputs to the specified container format (e.g., `mkv`, `mp4`, `ogg`, `webm`, `flv`). Ignored if no merge is required.

### Subtitle Options

*   `--write-sub`:  Writes a subtitle file.
*   `--write-auto-sub`:  Writes automatically generated subtitle file (YouTube only).
*   `--all-subs`:  Downloads all available subtitles.
*   `--list-subs`:  Lists all available subtitles.
*   `--sub-format FORMAT`: Sets the subtitle format (e.g., `srt` or `ass/srt/best`).
*   `--sub-lang LANGS`: Specifies the languages of the subtitles (optional, separated by commas; use `--list-subs` for available language tags).

### Authentication Options

*   `-u, --username USERNAME`:  Logs in with this account ID.
*   `-p, --password PASSWORD`: Sets the account password (youtube-dl will prompt if omitted).
*   `-2, --twofactor TWOFACTOR`: Sets the two-factor authentication code.
*   `-n, --netrc`: Uses .netrc authentication data.
*   `--video-password PASSWORD`: Sets the video password (Vimeo, Youku).

### Adobe Pass Options

*   `--ap-mso MSO`:  Specifies the Adobe Pass multiple-system operator (TV provider) identifier; use `--ap-list-mso` for a list.
*   `--ap-username USERNAME`: Sets the multiple-system operator account login.
*   `--ap-password PASSWORD`: Sets the multiple-system operator account password (youtube-dl will prompt if omitted).
*   `--ap-list-mso`:  Lists all supported multiple-system operators.

### Post-processing Options

*   `-x, --extract-audio`: Converts video files to audio-only files (requires ffmpeg/avconv and ffprobe/avprobe).
*   `--audio-format FORMAT`: Specifies the audio format (e.g., `best`, `aac`, `flac`, `mp3`, `m4a`, `opus`, `vorbis`, `wav`; "best" is the default).
*   `--audio-quality QUALITY`: Sets the ffmpeg/avconv audio quality (0-9 for VBR or specific bitrate like `128K`; default: 5).
*   `--recode-video FORMAT`: Encodes the video to another format if necessary (currently: `mp4`, `flv`, `ogg`, `webm`, `mkv`, `avi`).
*   `--postprocessor-args ARGS`: Passes arguments to the postprocessor.
*   `-k, --keep-video`:  Keeps the video file on disk after post-processing (video is erased by default).
*   `--no-post-overwrites`:  Does not overwrite post-processed files (they are overwritten by default).
*   `--embed-subs`: Embeds subtitles in the video (only for mp4, webm, and mkv videos).
*   `--embed-thumbnail`: Embeds the thumbnail in the audio as cover art.
*   `--add-metadata`: Writes metadata to the video file.
*   `--metadata-from-title FORMAT`:  Parses metadata (title, artist) from the video title, using the same format syntax as `--output`. Regular expressions with named capture groups can also be used.
*   `--xattrs`: Writes metadata to the video file's xattrs (using Dublin Core and XDG standards).
*   `--fixup POLICY`: Automatically corrects known file faults (e.g., `never`, `warn`, `detect_or_warn` - default).
*   `--prefer-avconv`:  Prefers avconv over ffmpeg for post-processing.
*   `--prefer-ffmpeg`:  Prefers ffmpeg over avconv for post-processing (default).
*   `--ffmpeg-location PATH`:  Specifies the location of the ffmpeg/avconv binary (path or containing directory).
*   `--exec CMD`: Executes a command after download and post-processing (similar to find's `-exec`).
*   `--convert-subs FORMAT`: Converts subtitles to another format (e.g., `srt`, `ass`, `vtt`, `lrc`).

## Configuration

youtube-dl can be configured via a configuration file. On Linux and macOS, the system-wide configuration file is `/etc/youtube-dl.conf`, and the user-specific file is `~/.config/youtube-dl/config`. On Windows, the user-specific files are `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`. The configuration file may not exist by default, so you may need to create it.

For example, the following configuration file will always extract audio, disable copying the file modification time, use a proxy, and save videos under a "Movies" directory:

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

**Important:**  Options in the configuration file use the same syntax as command-line options (no whitespace after `-` or `--`, e.g., `-o` or `--proxy`, but not `- o` or `-- proxy`).

Use `--ignore-config` to disable the configuration file for a particular run.

Use `--config-location` to specify a custom configuration file.

### Authentication with `.netrc` file

You can configure automatic credentials storage for extractors that support authentication (login/password via `--username` and `--password`) to avoid command-line credential passing, which can expose passwords in shell history. This can be done using a [`.netrc` file](https://stackoverflow.com/tags/.netrc/info) on a per-extractor basis. Create a `.netrc` file in your `$HOME` and restrict permissions to read/write only by you:

```bash
touch $HOME/.netrc
chmod a-rwx,u+rw $HOME/.netrc
```

Then add credentials for an extractor in the following format, where *extractor* is the name of the extractor in lowercase:

```
machine <extractor> login <login> password <password>
```

For example:

```
machine youtube login myaccount@gmail.com password my_youtube_password
machine twitch login my_twitch_account_name password my_twitch_password
```

To activate authentication with the `.netrc` file, pass `--netrc` to youtube-dl or include it in the [configuration file](#configuration).

On Windows, you may also need to set the `%HOME%` environment variable:

```bash
set HOME=%USERPROFILE%
```

## Output Template

The `-o` option uses a template for output file names.

The basic usage is not to set any template arguments when downloading a single file, like in `youtube-dl -o funny_video.flv "https://some/video"`. However, the output template may contain special sequences that will be replaced during each video's download. These sequences may be formatted according to [python string formatting operations](https://docs.python.org/2/library/stdtypes.html#string-formatting) (e.g., `%(NAME)s` or `%(NAME)05d`). Allowed names along with sequence type are:

 -   `id` (string): Video identifier
 -   `title` (string): Video title
 -   `url` (string): Video URL
 -   `ext` (string): Video filename extension
 -   `alt_title` (string): A secondary title of the video
 -   `display_id` (string): An alternative identifier for the video
 -   `uploader` (string): Full name of the video uploader
 -   `license` (string): License name the video is licensed under
 -   `creator` (string): The creator of the video
 -   `release_date` (string): The date (YYYYMMDD) when the video was released
 -   `timestamp` (numeric): UNIX timestamp of the moment the video became available
 -   `upload_date` (string): Video upload date (YYYYMMDD)
 -   `uploader_id` (string): Nickname or id of the video uploader
 -   `channel` (string): Full name of the channel the video is uploaded on
 -   `channel_id` (string): Id of the channel
 -   `location` (string): Physical location where the video was filmed
 -   `duration` (numeric): Length of the video in seconds
 -   `view_count` (numeric): How many users have watched the video on the platform
 -   `like_count` (numeric): Number of positive ratings of the video
 -   `dislike_count` (numeric): Number of negative ratings of the video
 -   `repost_count` (numeric): Number of reposts of the video
 -   `average_rating` (numeric): Average rating give by users, the scale used depends on the webpage
 -   `comment_count` (numeric): Number of comments on the video
 -   `age_limit` (numeric): Age restriction for the video (years)
 -   `is_live` (boolean): Whether this video is a live stream or a fixed-length video
 -   `start_time` (numeric): Time in seconds where the reproduction should start, as specified in the URL
 -   `end_time` (numeric): Time in seconds where the reproduction should end, as specified in the URL
 -   `format` (string): A human-readable description of the format
 -   `format_id` (string): Format code specified by `--format`
 -   `format_note` (string): Additional info about the format
 -   `width` (numeric): Width of the video
 -   `height` (numeric): Height of the video
 -   `resolution` (string): Textual description of width and height
 -   `tbr` (numeric): Average bitrate of audio and video in KBit/s
 -   `abr` (numeric): Average audio bitrate in KBit/s
 -   `acodec` (string): Name of the audio codec in use
 -   `asr` (numeric): Audio sampling rate in Hertz
 -   `vbr` (numeric): Average video bitrate in KBit/s
 -   `fps` (numeric): Frame rate
 -   `vcodec` (string): Name of the video codec in use
 -   `container` (string): Name of the container format
 -   `filesize` (numeric): The number of bytes, if known in advance
 -   `filesize_approx` (numeric): An estimate for the number of bytes
 -   `protocol` (string): The protocol that will be used for the actual download
 -   `extractor` (string): Name of the extractor
 -   `extractor_key` (string): Key name of the extractor
 -   `epoch` (numeric): Unix epoch when creating the file
 -   `autonumber` (numeric): Number that will be increased with each download, starting at `--autonumber-start`
 -   `playlist` (string): Name or id of the playlist that contains the video
 -   `playlist_index` (numeric): Index of the video in the playlist padded with leading zeros according to the total length of the playlist
 -   `playlist_id` (string): Playlist identifier
 -   `playlist_title` (string): Playlist title
 -   `playlist_uploader` (string): Full name of the playlist uploader
 -   `playlist_uploader_id` (string): Nickname or id of the playlist uploader

Available for the video that belongs to some logical chapter or section:

 -   `chapter` (string): Name or title of the chapter the video belongs to
 -   `chapter_number` (numeric): Number of the chapter the video belongs to
 -   `chapter_id` (string): Id of the chapter the video belongs to

Available for the video that is an episode of some series or programme:

 -   `series` (string): Title of the series or programme the video episode belongs to
 -   `season` (string): Title of the season the video episode belongs to
 -   `season_number` (numeric): Number of the season the video episode belongs to
 -   `season_id` (string): Id of the season the video episode belongs to
 -   `episode` (string): Title of the video episode
 -   `episode_number` (numeric): Number of the video episode within a season
 -   `episode_id` (string): Id of the video episode

Available for the media that is a track or a part of a music album:

 -   `track` (string): Title of the track
 -   `track_number` (numeric): Number of the track within an album or a disc
 -   `track_id` (string): Id of the track
 -   `artist` (string): Artist(s) of the track
 -   `genre` (string): Genre(s) of the track
 -   `album` (string): Title of the album the track belongs to
 -   `album_type` (string): Type of the album
 -   `album_artist` (string): List of all artists appeared on the album
 -   `disc_number` (numeric): Number of the disc or other physical medium the track belongs to
 -   `release_year` (numeric): Year (YYYY) when the album was released

Each sequence will be replaced with its corresponding value.  Sequences not present in the metadata (due to a particular extractor not providing them) will be replaced with the placeholder value provided with `--output-na-placeholder` (default: `NA`).

For example,  `-o %(title)s-%(id)s.%(ext)s` and an mp4 video with the title  `youtube-dl test video` and ID `BaW_jenozKcj`, this will result in a `youtube-dl test video-BaW_jenozKcj.mp4` file created in the current directory.

For numeric sequences, you can use numeric formatting (e.g., `%(view_count)05d` will pad with zeros).

Output templates can also contain arbitrary hierarchical paths (e.g., `-o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s'`), which will create directories automatically.

To use percent literals in an output template use `%%`.  To output to stdout, use `-o -`.

The current default template is `%(title)s-%(id)s.%(ext)s`.

Add the `--restrict-filenames` flag to get a shorter title in some cases.

#### Output template and Windows batch files

If using an output template in a Windows batch file, escape `%` characters by doubling them (`%%`).

#### Output template examples

Note that on Windows, you may need to use double quotes instead of single quotes.

```bash
$ youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc
youtube-dl test video ''_ä↭𝕐.mp4    # All kinds of weird characters

$ youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc --restrict-filenames
youtube-