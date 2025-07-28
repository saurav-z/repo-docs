[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Your Ultimate Video Download Solution

**Download videos from YouTube and thousands of other sites with ease!** [Explore the original repository](https://github.com/ytdl-org/youtube-dl).

**Key Features:**

*   **Wide Compatibility:** Supports thousands of video and audio websites.
*   **Format Selection:** Download videos in your preferred formats and qualities.
*   **Playlist Download:** Effortlessly download entire playlists.
*   **Customization:** Configure output filenames, download locations, and more.
*   **Authentication Support:** Log in to download restricted content.
*   **Post-Processing:** Convert videos to audio, embed subtitles, and add metadata.
*   **Cross-Platform:** Works on Linux, macOS, and Windows.

**Table of Contents**

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
    *   [Authentication with `.netrc` file](#authentication-with-.netrc-file)
*   [Output Template](#output-template)
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
    *   [Adding support for a new site](#adding-support-for-a-new-site)
    *   [youtube-dl coding conventions](#youtube-dl-coding-conventions)
        *   [Mandatory and optional metafields](#mandatory-and-optional-metafields)
        *   [Provide fallbacks](#provide-fallbacks)
        *   [Regular expressions](#regular-expressions)
        *   [Don't capture groups you don't use](#dont-capture-groups-you-dont-use)
        *   [Make regular expressions relaxed and flexible](#make-regular-expressions-relaxed-and-flexible)
        *   [Long lines policy](#long-lines-policy)
        *   [Inline values](#inline-values)
        *   [Collapse fallbacks](#collapse-fallbacks)
        *   [Trailing parentheses](#trailing-parentheses)
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

Choose your operating system for installation instructions:

To install it right away for all UNIX users (Linux, macOS, etc.), type:

    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl

If you do not have curl, you can alternatively use a recent wget:

    sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl

Windows users can [download an .exe file](https://yt-dl.org/latest/youtube-dl.exe) and place it in any location on their [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) except for `%SYSTEMROOT%\System32` (e.g. **do not** put in `C:\Windows\System32`).

You can also use pip:

    sudo -H pip install --upgrade youtube-dl

This command will update youtube-dl if you have already installed it. See the [pypi page](https://pypi.python.org/pypi/youtube_dl) for more information.

macOS users can install youtube-dl with [Homebrew](https://brew.sh/):

    brew install youtube-dl

Or with [MacPorts](https://www.macports.org/):

    sudo port install youtube-dl

Alternatively, refer to the [developer instructions](#developer-instructions) for how to check out and work with the git repository. For further options, including PGP signatures, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## Description

**youtube-dl** is a command-line program designed for downloading videos from YouTube.com and numerous other video platforms.  It requires the Python interpreter (version 2.6, 2.7, or 3.2+), and operates independently of the underlying operating system. It's compatible with Unix systems, Windows, and macOS. It is released to the public domain, which means you're free to modify, redistribute, and use it as you wish.

To use youtube-dl, use the command format:

    youtube-dl [OPTIONS] URL [URL...]

## Options

Options are used to configure the video download process.

*   `-h, --help` Print help and exit.
*   `--version` Print the program version and exit.
*   `-U, --update` Update youtube-dl to the latest version.
*   `-i, --ignore-errors` Continue on download errors.
*   `--abort-on-error` Abort downloading if an error occurs.
*   `--dump-user-agent` Display the current browser identification.
*   `--list-extractors` List all supported extractors.
*   `--extractor-descriptions` Output descriptions of all supported extractors.
*   `--force-generic-extractor` Force extraction to use the generic extractor.
*   `--default-search PREFIX` Use a prefix for unqualified URLs.
*   `--ignore-config` Do not read configuration files.
*   `--config-location PATH` Location of the configuration file.
*   `--flat-playlist` Only list playlist videos.
*   `--mark-watched` Mark videos watched (YouTube only).
*   `--no-mark-watched` Do not mark videos watched (YouTube only).
*   `--no-color` Do not emit color codes in output.

### Network Options

These options configure network settings.

*   `--proxy URL` Use the specified HTTP/HTTPS/SOCKS proxy.
*   `--socket-timeout SECONDS` Time to wait before giving up, in seconds.
*   `--source-address IP` Client-side IP address to bind to.
*   `-4, --force-ipv4` Force IPv4 connections.
*   `-6, --force-ipv6` Force IPv6 connections.

### Geo Restriction

Options to bypass geographical restrictions.

*   `--geo-verification-proxy URL` Use this proxy to verify the IP address.
*   `--geo-bypass` Bypass geographic restriction via faking X-Forwarded-For HTTP header.
*   `--no-geo-bypass` Do not bypass geographic restriction.
*   `--geo-bypass-country CODE` Force bypass geographic restriction with a two-letter ISO 3166-2 country code.
*   `--geo-bypass-ip-block IP_BLOCK` Force bypass geographic restriction with an IP block in CIDR notation.

### Video Selection

Options to select videos for download.

*   `--playlist-start NUMBER` Playlist start index.
*   `--playlist-end NUMBER` Playlist end index.
*   `--playlist-items ITEM_SPEC` Specify playlist video items to download.
*   `--match-title REGEX` Download only matching titles.
*   `--reject-title REGEX` Skip download for matching titles.
*   `--max-downloads NUMBER` Abort after downloading this many files.
*   `--min-filesize SIZE` Do not download videos smaller than this size.
*   `--max-filesize SIZE` Do not download videos larger than this size.
*   `--date DATE` Download videos uploaded on this date.
*   `--datebefore DATE` Download videos uploaded on or before this date.
*   `--dateafter DATE` Download videos uploaded on or after this date.
*   `--min-views COUNT` Do not download videos with less than this number of views.
*   `--max-views COUNT` Do not download videos with more than this number of views.
*   `--match-filter FILTER` Generic video filter.
*   `--no-playlist` Download only the video if the URL refers to a video and playlist.
*   `--yes-playlist` Download the playlist.
*   `--age-limit YEARS` Download only videos suitable for the given age.
*   `--download-archive FILE` Download only videos not listed in the archive file.
*   `--include-ads` Download advertisements as well (experimental).

### Download Options

Options to control the download process.

*   `-r, --limit-rate RATE` Limit download rate in bytes per second.
*   `-R, --retries RETRIES` Number of retries.
*   `--fragment-retries RETRIES` Number of fragment retries.
*   `--skip-unavailable-fragments` Skip unavailable fragments.
*   `--abort-on-unavailable-fragment` Abort when a fragment is unavailable.
*   `--keep-fragments` Keep downloaded fragments on disk.
*   `--buffer-size SIZE` Set download buffer size.
*   `--no-resize-buffer` Do not automatically adjust buffer size.
*   `--http-chunk-size SIZE` Size of a chunk for chunk-based HTTP downloading.
*   `--playlist-reverse` Download playlist videos in reverse order.
*   `--playlist-random` Download playlist videos in random order.
*   `--xattr-set-filesize` Set file xattribute ytdl.filesize with expected file size.
*   `--hls-prefer-native` Use the native HLS downloader instead of ffmpeg.
*   `--hls-prefer-ffmpeg` Use ffmpeg instead of the native HLS downloader.
*   `--hls-use-mpegts` Use the mpegts container for HLS videos.
*   `--external-downloader COMMAND` Use the specified external downloader.
*   `--external-downloader-args ARGS` Give these arguments to the external downloader.

### Filesystem Options

Options related to the file system.

*   `-a, --batch-file FILE` File containing URLs to download.
*   `--id` Use only video ID in the file name.
*   `-o, --output TEMPLATE` Output filename template.
*   `--output-na-placeholder PLACEHOLDER` Placeholder value for unavailable metadata fields.
*   `--autonumber-start NUMBER` Start value for autonumber.
*   `--restrict-filenames` Restrict filenames to ASCII characters.
*   `-w, --no-overwrites` Do not overwrite files.
*   `-c, --continue` Force resume of partially downloaded files.
*   `--no-continue` Do not resume partially downloaded files.
*   `--no-part` Do not use .part files.
*   `--no-mtime` Do not set the file modification time.
*   `--write-description` Write video description to a .description file.
*   `--write-info-json` Write video metadata to a .info.json file.
*   `--write-annotations` Write video annotations to a .annotations.xml file.
*   `--load-info-json FILE` Load video information from a .info.json file.
*   `--cookies FILE` Read cookies from this file.
*   `--cache-dir DIR` Location for youtube-dl to store downloaded information.
*   `--no-cache-dir` Disable filesystem caching.
*   `--rm-cache-dir` Delete all filesystem cache files.

### Thumbnail Options

Options for handling thumbnails.

*   `--write-thumbnail` Write thumbnail image to disk.
*   `--write-all-thumbnails` Write all thumbnail image formats to disk.
*   `--list-thumbnails` List available thumbnail formats.

### Verbosity / Simulation Options

Options for controlling verbosity and simulation.

*   `-q, --quiet` Activate quiet mode.
*   `--no-warnings` Ignore warnings.
*   `-s, --simulate` Do not download the video.
*   `--skip-download` Do not download the video.
*   `-g, --get-url` Simulate, but print URL.
*   `-e, --get-title` Simulate, but print title.
*   `--get-id` Simulate, but print id.
*   `--get-thumbnail` Simulate, but print thumbnail URL.
*   `--get-description` Simulate, but print video description.
*   `--get-duration` Simulate, but print video length.
*   `--get-filename` Simulate, but print output filename.
*   `--get-format` Simulate, but print output format.
*   `-j, --dump-json` Simulate, but print JSON information.
*   `-J, --dump-single-json` Simulate, but print JSON information for each command-line argument.
*   `--print-json` Print video information as JSON.
*   `--newline` Output progress bar as new lines.
*   `--no-progress` Do not print progress bar.
*   `--console-title` Display progress in the console titlebar.
*   `-v, --verbose` Print debugging information.
*   `--dump-pages` Print downloaded pages encoded using base64.
*   `--write-pages` Write downloaded intermediary pages to files.
*   `--print-traffic` Display HTTP traffic.
*   `-C, --call-home` Contact the youtube-dl server for debugging.
*   `--no-call-home` Do NOT contact the youtube-dl server.

### Workarounds

Options for dealing with potential issues.

*   `--encoding ENCODING` Force the specified encoding.
*   `--no-check-certificate` Suppress HTTPS certificate validation.
*   `--prefer-insecure` Use an unencrypted connection to retrieve information about the video.
*   `--user-agent UA` Specify a custom user agent.
*   `--referer URL` Specify a custom referer.
*   `--add-header FIELD:VALUE` Specify a custom HTTP header.
*   `--bidi-workaround` Work around terminals that lack bidirectional text support.
*   `--sleep-interval SECONDS` Sleep interval before each download.
*   `--max-sleep-interval SECONDS` Upper bound of a range for randomized sleep.

### Video Format Options

Options for selecting video formats.

*   `-f, --format FORMAT` Video format code.
*   `--all-formats` Download all available video formats.
*   `--prefer-free-formats` Prefer free video formats.
*   `-F, --list-formats` List all available formats.
*   `--youtube-skip-dash-manifest` Do not download the DASH manifests.
*   `--merge-output-format FORMAT` Output to the given container format.

### Subtitle Options

Options for handling subtitles.

*   `--write-sub` Write subtitle file.
*   `--write-auto-sub` Write automatically generated subtitle file (YouTube only).
*   `--all-subs` Download all available subtitles.
*   `--list-subs` List all available subtitles.
*   `--sub-format FORMAT` Subtitle format.
*   `--sub-lang LANGS` Languages of the subtitles to download.

### Authentication Options

Options for authentication.

*   `-u, --username USERNAME` Login with this account.
*   `-p, --password PASSWORD` Account password.
*   `-2, --twofactor TWOFACTOR` Two-factor authentication code.
*   `-n, --netrc` Use .netrc authentication data.
*   `--video-password PASSWORD` Video password (vimeo, youku).

### Adobe Pass Options

Options for Adobe Pass authentication.

*   `--ap-mso MSO` Adobe Pass multiple-system operator identifier.
*   `--ap-username USERNAME` Multiple-system operator account login.
*   `--ap-password PASSWORD` Multiple-system operator account password.
*   `--ap-list-mso` List all supported multiple-system operators.

### Post-processing Options

Options for post-processing the downloaded video.

*   `-x, --extract-audio` Convert video files to audio-only files.
*   `--audio-format FORMAT` Specify audio format.
*   `--audio-quality QUALITY` Specify audio quality.
*   `--recode-video FORMAT` Encode the video to another format.
*   `--postprocessor-args ARGS` Give these arguments to the postprocessor.
*   `-k, --keep-video` Keep the video file after post-processing.
*   `--no-post-overwrites` Do not overwrite post-processed files.
*   `--embed-subs` Embed subtitles in the video.
*   `--embed-thumbnail` Embed thumbnail in the audio as cover art.
*   `--add-metadata` Write metadata to the video file.
*   `--metadata-from-title FORMAT` Parse metadata from the video title.
*   `--xattrs` Write metadata to the video file's xattrs.
*   `--fixup POLICY` Automatically correct known faults of the file.
*   `--prefer-avconv` Prefer avconv over ffmpeg.
*   `--prefer-ffmpeg` Prefer ffmpeg over avconv.
*   `--ffmpeg-location PATH` Location of the ffmpeg/avconv binary.
*   `--exec CMD` Execute a command on the file after downloading and post-processing.
*   `--convert-subs FORMAT` Convert the subtitles to other format.

## Configuration

Configure youtube-dl by adding command-line options to a configuration file. System-wide configuration is at `/etc/youtube-dl.conf` on Linux/macOS and user-specific configuration at `~/.config/youtube-dl/config` (Linux/macOS) or `%APPDATA%\youtube-dl\config.txt` (Windows).

Example Configuration:

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

Use `--ignore-config` to disable the configuration file, or `--config-location` to specify a custom configuration file for a single run.

### Authentication with `.netrc` file

You can configure credentials for extractors that support authentication. For that, create a `.netrc` file in your `$HOME` with permissions set to read/write only by you:

```
touch $HOME/.netrc
chmod a-rwx,u+rw $HOME/.netrc
```

Then add credentials in the format:

```
machine <extractor> login <login> password <password>
```

For example:

```
machine youtube login myaccount@gmail.com password my_youtube_password
machine twitch login my_twitch_account_name password my_twitch_password
```

To activate authentication, pass `--netrc` or include it in your configuration file.

## Output Template

The `-o` option enables customization of the output filename.

*   `id`: Video identifier
*   `title`: Video title
*   `url`: Video URL
*   `ext`: Video extension
*   `alt_title`: Secondary title
*   `display_id`: Alternative identifier
*   `uploader`: Uploader name
*   `license`: Video license
*   `creator`: Video creator
*   `release_date`: Release date (YYYYMMDD)
*   `timestamp`: UNIX timestamp
*   `upload_date`: Upload date (YYYYMMDD)
*   `uploader_id`: Uploader ID
*   `channel`: Channel name
*   `channel_id`: Channel ID
*   `location`: Filming location
*   `duration`: Video length in seconds
*   `view_count`: Number of views
*   `like_count`: Number of likes
*   `dislike_count`: Number of dislikes
*   `repost_count`: Number of reposts
*   `average_rating`: Average rating
*   `comment_count`: Number of comments
*   `age_limit`: Age restriction
*   `is_live`: Is it a live stream?
*   `start_time`: Start time in seconds
*   `end_time`: End time in seconds
*   `format`: Format description
*   `format_id`: Format code
*   `format_note`: Format info
*   `width`: Video width
*   `height`: Video height
*   `resolution`: Resolution description
*   `tbr`: Bitrate (audio+video) in KBit/s
*   `abr`: Audio bitrate in KBit/s
*   `acodec`: Audio codec
*   `asr`: Audio sampling rate
*   `vbr`: Video bitrate in KBit/s
*   `fps`: Frame rate
*   `vcodec`: Video codec
*   `container`: Container format
*   `filesize`: File size in bytes
*   `filesize_approx`: Approximate file size
*   `protocol`: Download protocol
*   `extractor`: Extractor name
*   `extractor_key`: Extractor key
*   `epoch`: Unix epoch of file creation
*   `autonumber`: Autonumber
*   `playlist`: Playlist name
*   `playlist_index`: Playlist index (with leading zeros)
*   `playlist_id`: Playlist ID
*   `playlist_title`: Playlist title
*   `playlist_uploader`: Playlist uploader
*   `playlist_uploader_id`: Playlist uploader ID
*   `chapter`: Chapter title
*   `chapter_number`: Chapter number
*   `chapter_id`: Chapter ID
*   `series`: Series title
*   `season`: Season title
*   `season_number`: Season number
*   `season_id`: Season ID
*   `episode`: Episode title
*   `episode_number`: Episode number
*   `episode_id`: Episode ID
*   `track`: Track title
*   `track_number`: Track number
*   `track_id`: Track ID
*   `artist`: Artist(s) of the track
*   `genre`: Genre(s) of the track
*   `album`: Album title
*   `album_type`: Album type
*   `album_artist`: Album artist(s)
*   `disc_number`: Disc number
*   `release_year`: Release year

Unspecified values use `--output-na-placeholder` ("NA" by default).

**Example:**  `-o "%(title)s-%(id)s.%(ext)s"`

*Note:* Windows batch files need to escape `%` characters, e.g., `-o "%%(title)s-%%(id)s.%%(ext)s"`.

### Output template examples

*   `youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc`: Example file name using the title and file extension.
*   `youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc --restrict-filenames`: Creates simpler file names.
*   `youtube-dl -o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' https://www.youtube.com/playlist?list=PLwiyx1dc3P2JR9N8gQaQN_BCvlSlap7re`: Downloads YouTube playlist videos into separate directories.
*   `youtube-dl -o '%(uploader)s/%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' https://www.youtube.com/user/TheLinuxFoundation/playlists`: Downloads all playlists of a YouTube channel into separate directories.
*   `youtube-dl -u user -p password -o '~/MyVideos/%(playlist)s/%(chapter_number)s - %(chapter)s/%(title)s.%(ext)s' https://www.udemy.com/java-tutorial/`: Downloads Udemy courses.
*   `youtube-dl -o "C:/MyVideos/%(series)s/%(season_number)s - %(season)s/%(episode_number)s - %(episode)s.%(ext)s" https://videomore.ru/kino_v_detalayah/5_sezon/367617`: Downloads entire series episodes.
*   `youtube-dl -o - BaW_jenozKc`: Streams to stdout.

## Format Selection

Use `-f FORMAT` (or `--format FORMAT`) to specify download formats.  `--list-formats` or `-F` lists available formats.

*   `best`: Best single-file quality.
*   `worst`: Worst single-file quality.
*   `bestvideo`: Best video-only format.
*   `worstvideo`: Worst video-only format.
*   `bestaudio`: Best audio-only format.
*   `worstaudio`: Worst audio-only format.

You can use specific format codes, for example: `-f 22`. Use slashes for preference: `-f 22/17/18`. Use commas to download multiple formats: `-f 22,17,18`. Use brackets for filtering: `-f "best[height=720]"`.

Numeric comparisons include `<, <=, >, >=, =, !=`. String comparisons include `ext, acodec, vcodec, container, protocol, format_id, language` with `=, ^=, $=, *=, !`.  Use a question mark `?` after the operator to exclude unknown values.
You can merge video and audio: `-f <video-format>+<audio-format>` (requires ffmpeg/avconv).

Since April 2015 and version 2015.04.26, youtube-dl uses `-f bestvideo+bestaudio/best` as the default format selection.
If ffmpeg or avconv are installed this results in downloading `bestvideo` and `bestaudio` separately and muxing them together into a single file giving the best overall quality available. Otherwise it falls back to `best` and results in downloading the best available quality served as a single file.

### Format selection examples

*   `youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'`: Downloads best quality with mp4 file extension.
*   `youtube-dl -f 'bestvideo[height<=480]+bestaudio/best[height<=480]'`: Download best video format, but no bigger than 480p.
*   `youtube-dl -f 'best[filesize<50M]'`: Downloads the best format, but no bigger than 50MB.
*   `youtube-dl -f '(bestvideo+bestaudio/best)[protocol^=http]'`: Downloads the best via direct link over HTTP/HTTPS protocol.
*   `youtube-dl -f 'bestvideo,bestaudio' -o '%(title)s.f%(format_id)s.%(ext)s'`: Downloads separate video and audio formats.

## Video Selection

Videos can be selected based on their upload date: `--date`, `--datebefore`, `--dateafter`. Use absolute dates `YYYYMMDD` or relative dates: `(now|today)[+-][0-9](day|week|month|year)(s)?`

Examples:

*   `youtube-dl --dateafter now-6months`: Downloads videos uploaded in the last 6 months.
*   `youtube-dl --date 19700101`: Downloads videos uploaded on January 1, 1970.
*   `youtube-dl --dateafter 20000101 --datebefore