[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Your Go-To Command-Line Tool for Video and Audio Downloads

Tired of buffering? **Download videos and audio from YouTube and thousands of other sites with ease using youtube-dl!** This versatile command-line tool empowers you to save your favorite content for offline viewing, format conversion, and more. For advanced users, youtube-dl supports many options including downloading entire playlists, extracting audio, and post-processing options to convert video files.

[**Get Started with youtube-dl**](https://github.com/ytdl-org/youtube-dl)

**Key Features:**

*   **Wide Site Support:** Download from YouTube, plus thousands of other video and audio platforms (see [Supported Sites](https://ytdl-org.github.io/youtube-dl/supportedsites.html) for the complete list).
*   **Multiple Format Options:** Download videos in various resolutions, formats, and qualities, or extract audio only.
*   **Playlist and Channel Downloads:** Download entire playlists, user channels, or individual videos with ease.
*   **Customizable Output:** Control file names, formats, and output directories using flexible template options.
*   **Subtitle Support:** Download subtitles and closed captions in multiple languages.
*   **Post-Processing:** Convert videos to audio, embed subtitles, and add metadata.
*   **Configuration:** Easily configure youtube-dl using configuration files for consistent settings.
*   **Cross-Platform Compatibility:** Works seamlessly on Linux, macOS, and Windows.

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
    *   [How do I download a video starting with a `-`?](#how-do-i-download-a-video-starting-with--)
    *   [How do I pass cookies to youtube-dl?](#how-do-i-pass-cookies-to-youtube-dl)
    *   [How do I stream directly to media player?](#how-do-i-stream-directly-to-media-player)
    *   [How do I download only new videos from a playlist?](#how-do-i-download-only-new-videos-from-a-playlist)
    *   [Should I add `--hls-prefer-native` into my config?](#should-i-add--hls-prefer-native-into-my-config)
    *   [Can you add support for this anime video site, or site which shows current movies for free?](#can-you-add-support-for-this-anime-video-site-or-site-which-shows-current-movies-for-free)
*   [How can I speed up work on my issue?](#how-can-i-speed-up-work-on-my-issue)
*   [How can I detect whether a given URL is supported by youtube-dl?](#how-can-i-detect-whether-a-given-url-is-supported-by-youtube-dl)
*   [Why do I need to go through that much red tape when filing bugs?](#why-do-i-need-to-go-through-that-much-red-tape-when-filing-bugs)
*   [DEVELOPER INSTRUCTIONS](#developer-instructions)
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
*   [EMBEDDING YOUTUBE-DL](#embedding-youtube-dl)
*   [BUGS](#bugs)
    *   [Opening a bug report or suggestion](#opening-a-bug-report-or-suggestion)
        *   [Is the description of the issue itself sufficient?](#is-the-description-of-the-issue-itself-sufficient)
        *   [Is the issue already documented?](#is-the-issue-already-documented)
        *   [Are you using the latest version?](#are-you-using-the-latest-version)
        *   [Why are existing options not enough?](#why-are-existing-options-not-enough)
        *   [Is there enough context in your bug report?](#is-there-enough-context-in-your-bug-report)
        *   [Does the issue involve one problem, and one problem only?](#does-the-issue-involve-one-problem-and-one-problem-only)
        *   [Is anyone going to need the feature?](#is-anyone-going-to-need-the-feature)
        *   [Is your question about youtube-dl?](#is-your-question-about-youtube-dl)
*   [COPYRIGHT](#copyright)

## INSTALLATION

Install youtube-dl on Linux, macOS, and Windows with the following command:

```bash
sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```
or on Windows, download the .exe file from here: [Download youtube-dl.exe](https://yt-dl.org/latest/youtube-dl.exe).

**Other Installation Options:**

*   **pip:** `sudo -H pip install --upgrade youtube-dl`
*   **Homebrew (macOS):** `brew install youtube-dl`
*   **MacPorts (macOS):** `sudo port install youtube-dl`

For more detailed installation instructions and alternative methods, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## DESCRIPTION

**youtube-dl** is a versatile command-line tool designed to download videos from YouTube.com and a wide variety of other video platforms. It's platform-independent, written in Python, and works on Unix-like systems, Windows, and macOS. This open-source program allows users to save videos locally for offline viewing, modification, redistribution, and personal use.

## OPTIONS

Use `youtube-dl --help` to see a full list of the available options. Here are some of the most commonly used options, organized by category:

*   `-h, --help`: Print help text and exit
*   `--version`: Print program version and exit
*   `-U, --update`: Update to the latest version
*   `-i, --ignore-errors`: Skip errors and continue downloading

### Network Options

*   `--proxy URL`: Specify an HTTP/HTTPS/SOCKS proxy.
*   `--socket-timeout SECONDS`: Set the socket timeout in seconds.
*   `-4, --force-ipv4`: Force IPv4 connections.
*   `-6, --force-ipv6`: Force IPv6 connections.

### Geo Restriction

*   `--geo-bypass`: Bypass geographic restrictions.
*   `--geo-bypass-country CODE`: Force a specific country code for geo-bypassing.

### Video Selection

*   `--playlist-start NUMBER`: Playlist video to start at.
*   `--playlist-end NUMBER`: Playlist video to end at.
*   `--playlist-items ITEM_SPEC`: Download specific playlist items.
*   `--match-title REGEX`: Download only videos with matching titles.
*   `--reject-title REGEX`: Skip downloads for matching titles.
*   `--date DATE`: Download videos uploaded on a specific date.
*   `--datebefore DATE`: Download videos uploaded on or before a date.
*   `--dateafter DATE`: Download videos uploaded on or after a date.
*   `--min-views COUNT`: Minimum view count.
*   `--max-views COUNT`: Maximum view count.
*   `--no-playlist`: Download only the video if the URL refers to both a video and a playlist.
*   `--age-limit YEARS`: Download videos suitable for a given age.
*   `--download-archive FILE`: Only download videos not listed in the archive file.

### Download Options

*   `-r, --limit-rate RATE`: Limit download rate in bytes per second.
*   `-R, --retries RETRIES`: Number of retries.
*   `--fragment-retries RETRIES`: Number of retries for a fragment (DASH, hlsnative, and ISM).
*   `--keep-fragments`: Keep downloaded fragments after download.
*   `--buffer-size SIZE`: Set the download buffer size.
*   `--playlist-reverse`: Download playlist videos in reverse order.
*   `--playlist-random`: Download playlist videos in random order.

### Filesystem Options

*   `-a, --batch-file FILE`: Download from a file with URLs.
*   `-o, --output TEMPLATE`: Set the output filename template.
*   `--restrict-filenames`: Restrict filenames to ASCII characters.
*   `-w, --no-overwrites`: Don't overwrite files.
*   `-c, --continue`: Resume partially downloaded files.
*   `--no-part`: Don't write to .part files.
*   `--write-description`: Write video description to a .description file.
*   `--write-info-json`: Write video metadata to a .info.json file.
*   `--cookies FILE`: Read cookies from a file.
*   `--cache-dir DIR`: Set the cache directory.

### Thumbnail Options

*   `--write-thumbnail`: Write thumbnail image to disk.
*   `--list-thumbnails`: List available thumbnail formats.

### Verbosity / Simulation Options

*   `-q, --quiet`: Activate quiet mode.
*   `-s, --simulate`: Simulate a download (no actual download).
*   `-g, --get-url`: Simulate and print the URL.
*   `-e, --get-title`: Simulate and print the title.
*   `-j, --dump-json`: Simulate and print JSON information.
*   `-v, --verbose`: Print debugging information.

### Workarounds

*   `--encoding ENCODING`: Force the specified encoding.
*   `--no-check-certificate`: Suppress HTTPS certificate validation.
*   `--user-agent UA`: Specify a custom user agent.
*   `--referer URL`: Specify a custom referer.

### Video Format Options

*   `-f, --format FORMAT`: Select video format.
*   `--all-formats`: Download all available video formats.
*   `-F, --list-formats`: List available formats.
*   `--merge-output-format FORMAT`: Merge video and audio into a specific container.

### Subtitle Options

*   `--write-sub`: Write subtitle file.
*   `--write-auto-sub`: Write automatically generated subtitle file (YouTube only).
*   `--all-subs`: Download all available subtitles.
*   `--list-subs`: List available subtitles.
*   `--sub-format FORMAT`: Specify subtitle format.
*   `--sub-lang LANGS`: Download subtitles in specific languages.

### Authentication Options

*   `-u, --username USERNAME`: Login with a username.
*   `-p, --password PASSWORD`: Login with a password.
*   `-n, --netrc`: Use .netrc authentication data.

### Adobe Pass Options

*   `--ap-mso MSO`: Adobe Pass MSO identifier.
*   `--ap-username USERNAME`: Adobe Pass username.
*   `--ap-password PASSWORD`: Adobe Pass password.
*   `--ap-list-mso`: List supported multiple-system operators.

### Post-processing Options

*   `-x, --extract-audio`: Extract audio from video.
*   `--audio-format FORMAT`: Specify audio format.
*   `--audio-quality QUALITY`: Specify audio quality.
*   `--recode-video FORMAT`: Encode the video to another format.
*   `-k, --keep-video`: Keep the video file after post-processing.
*   `--embed-subs`: Embed subtitles in the video.
*   `--add-metadata`: Write metadata to the video file.
*   `--fixup POLICY`: Automatically correct known faults of the file.

## CONFIGURATION

You can configure youtube-dl by placing command-line options into a configuration file.
On Linux and macOS, the system-wide configuration file is located at `/etc/youtube-dl.conf`,
and the user configuration file is at `~/.config/youtube-dl/config`.
On Windows, it's `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`.

For example, to always extract audio, disable mtime, use a proxy, and save videos in your home "Movies" directory, you can create a config file with content like:

```
-x
--no-mtime
--proxy 127.0.0.1:3128
-o ~/Movies/%(title)s.%(ext)s
```

Use `--ignore-config` to disable the configuration file. You can also use `--config-location` to specify a custom configuration file for a particular youtube-dl run.

### Authentication with .netrc file

You may also configure automatic credentials storage for extractors that support authentication.
For this you will need to create a `.netrc` file in your `$HOME` and restrict permissions to read/write by only you:

```bash
touch $HOME/.netrc
chmod a-rwx,u+rw $HOME/.netrc
```
After that you can add credentials for an extractor in the following format, where `extractor` is the name of the extractor in lowercase:

```
machine <extractor> login <login> password <password>
```

To activate authentication with the `.netrc` file you should pass `--netrc` to youtube-dl or place it in the [configuration file](#configuration).

On Windows you may also need to setup the `%HOME%` environment variable manually. For example:
```bash
set HOME=%USERPROFILE%
```
## OUTPUT TEMPLATE

The `-o` option allows users to specify a template for output filenames.

**For examples, see the section below:** [Output template examples](#output-template-examples)

The basic usage is: `youtube-dl -o funny_video.flv "https://some/video"`.
However, output templates can contain special sequences that are replaced when downloading each video, formatted according to [Python string formatting operations](https://docs.python.org/2/library/stdtypes.html#string-formatting), such as `%(NAME)s`.
Example: `-o "%(title)s-%(id)s.%(ext)s"`.

Allowed names include:

*   `id`: Video identifier
*   `title`: Video title
*   `url`: Video URL
*   `ext`: Video filename extension
*   `alt_title`: Secondary title
*   `display_id`: Alternative identifier
*   `uploader`: Video uploader
*   `license`: License name
*   `creator`: Video creator
*   `release_date`: Release date (YYYYMMDD)
*   `timestamp`: UNIX timestamp
*   `upload_date`: Upload date (YYYYMMDD)
*   `uploader_id`: Uploader nickname/ID
*   `channel`: Channel name
*   `channel_id`: Channel ID
*   `location`: Filming location
*   `duration`: Video length in seconds
*   `view_count`: View count
*   `like_count`: Positive ratings
*   `dislike_count`: Negative ratings
*   `repost_count`: Repost count
*   `average_rating`: Average user rating
*   `comment_count`: Comment count
*   `age_limit`: Age restriction
*   `is_live`: Live stream indicator
*   `start_time`: Start time in the URL
*   `end_time`: End time in the URL
*   `format`: Human-readable format description
*   `format_id`: Format code
*   `format_note`: Format information
*   `width`: Video width
*   `height`: Video height
*   `resolution`: Textual resolution description
*   `tbr`: Average bitrate (KBit/s)
*   `abr`: Average audio bitrate (KBit/s)
*   `acodec`: Audio codec
*   `asr`: Audio sampling rate (Hz)
*   `vbr`: Average video bitrate (KBit/s)
*   `fps`: Frame rate
*   `vcodec`: Video codec
*   `container`: Container format
*   `filesize`: File size in bytes
*   `filesize_approx`: Approximate file size
*   `protocol`: Download protocol
*   `extractor`: Extractor name
*   `extractor_key`: Extractor key name
*   `epoch`: Unix epoch of file creation
*   `autonumber`: Auto-numbering value
*   `playlist`: Playlist name/ID
*   `playlist_index`: Playlist index
*   `playlist_id`: Playlist ID
*   `playlist_title`: Playlist title
*   `playlist_uploader`: Playlist uploader
*   `playlist_uploader_id`: Playlist uploader ID

**For videos belonging to a chapter or section:**

*   `chapter`: Chapter name/title
*   `chapter_number`: Chapter number
*   `chapter_id`: Chapter ID

**For videos that are episodes of a series:**

*   `series`: Series title
*   `season`: Season title
*   `season_number`: Season number
*   `season_id`: Season ID
*   `episode`: Episode title
*   `episode_number`: Episode number
*   `episode_id`: Episode ID

**For music tracks or albums:**

*   `track`: Track title
*   `track_number`: Track number
*   `track_id`: Track ID
*   `artist`: Track artist(s)
*   `genre`: Track genre(s)
*   `album`: Album title
*   `album_type`: Album type
*   `album_artist`: Album artist list
*   `disc_number`: Disc number
*   `release_year`: Album release year

Unrecognized sequences are replaced by `--output-na-placeholder`, which defaults to "NA".

Use `%%` to represent a literal percent sign.

To output to stdout, use `-o -`.

The default template is `%(title)s-%(id)s.%(ext)s`.

To avoid issues with special characters or Windows file systems, use the `--restrict-filenames` flag.

#### Output template and Windows batch files

When using output templates in Windows batch files, escape percent characters (`%`) by doubling them: `-o "%%(title)s-%%(id)s.%%(ext)s"`. However, do *not* escape `%` characters used for environment variables, such as `-o "C:\%HOMEPATH%\Desktop\%%(title)s.%%(ext)s"`.

#### Output template examples

```bash
$ youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc
youtube-dl test video ''_ä↭𝕐.mp4    # All kinds of weird characters

$ youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc --restrict-filenames
youtube-dl_test_video_.mp4          # A simple file name

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

By default, youtube-dl downloads the best available quality.

**For examples, see the section below:** [Format selection examples](#format-selection-examples)

To specify a different format, use the `--format FORMAT` or `-f FORMAT` option, where `FORMAT` is a *selector expression*.

The simplest case is to request a specific format by its code (e.g., `-f 22`). You can get a list of available format codes using `--list-formats` or `-F`. These codes are extractor-specific.

You can also use file extensions (e.g., `-f webm`) to get the best quality format of a specific extension.

Special names for specific formats include:

*   `best`: Best quality video and audio in a single file.
*   `worst`: Worst quality video and audio in a single file.
*   `bestvideo`: Best quality video only.
*   `worstvideo`: Worst quality video only.
*   `bestaudio`: Best quality audio only.
*   `worstaudio`: Worst quality audio only.

You can specify format precedence using slashes, such as `-f 22/17/18`.

To download multiple formats, separate them with commas, such as `-f 22,17,18`.
You can also combine this with precedence (e.g., `-f 136/137/mp4/bestvideo,140/m4a/bestaudio`).

You can filter video formats using conditions in brackets, as in `-f "best[height=720]"`.
You can compare the following numeric meta fields:

*   `filesize`: File size in bytes
*   `width`: Video width
*   `height`: Video height
*   `tbr`: Average bitrate (KBit/s)
*   `abr`: Average audio bitrate (KBit/s)
*   `vbr`: Average video bitrate (KBit/s)
*   `asr`: Audio sampling rate (Hz)
*   `fps`: Frame rate

You can also filter string meta fields with the comparisons `=`, `^=`, `$=`, `*=`, and negation `!`:

*   `ext`: File extension
*   `acodec`: Audio codec
*   `vcodec`: Video codec
*   `container`: Container format
*   `protocol`: Download protocol
*   `format_id`: Format description
*   `language`: Language code

Formats with unknown values are excluded unless you use a question mark (`?`) after the operator. Combine format filters: `-f "[height <=? 720][tbr>500]"`.

You can merge video and audio formats using `-f <video-format>+<audio-format>` (requires ffmpeg or avconv).

Format selectors can be grouped using parentheses, such as `-f '(mp4,webm)[height<480]'`.

The default format selection is `-f bestvideo+bestaudio/best`.

To preserve the pre-2015.04.26 behavior of downloading the best quality as a single file, explicitly specify `-f best`.

#### Format selection examples

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

You can filter videos by upload date using `--date`, `--datebefore`, or `--dateafter`.  Dates can be specified in:

*   **Absolute dates:** `YYYYMMDD`.
*   **Relative dates:** `(now|today)[+-][0-9](day|week|month|year)(s)?`

Examples:

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

Run `youtube-dl -U`. If that doesn't work, try `sudo youtube-dl -U` on Linux.
If you have used pip, `sudo pip install -U youtube-dl`.
Otherwise, use your system's package manager.
If using a package manager's version, be aware those versions may be outdated.

### youtube-dl is extremely slow to start on Windows

Add a file exclusion for `youtube-dl.exe` in Windows Defender settings.

### I'm getting an error `Unable to extract OpenGraph title` on YouTube playlists

Update to the latest version of youtube-dl as described above.

### I'm getting an error when trying to use output template: `error: using output template conflicts with using title, video ID or auto number`

Remove any conflicting options like `-t`, `--title`, `--id`, `-A` or `--auto-number` from your command line or configuration file.

### Do I always have to pass `-citw`?

No. youtube-dl intends to have the best options by default.  `-i` is generally the most useful option.

### Can you please put the `-b` option back?

youtube-dl now defaults to downloading the highest available quality, making `-b` unnecessary.

### I get HTTP error 402 when trying to download a video. What's this?

YouTube may require a CAPTCHA test if you download too much. Try opening the URL in a browser, solving the CAPTCHA, and then restarting youtube-dl or [pass cookies](#how-do-i-pass-cookies-to-youtube-dl).

### Do I need any other programs?

youtube-dl works on its own on most sites. However, for video/audio conversion, you'll need [avconv](https://libav.org/) or [ffmpeg](https://www.ffmpeg.org/). RTMP protocol support requires [rtmpdump](https://rtmpdump.mplayerhq.hu/). MMS and RTSP videos require [mplayer](https://mplayerhq.hu/) or [mpv](https://mpv.io/).

### I have downloaded a video but how can I play it?

Use any video player such as [mpv](https://mpv.io/), [vlc](https://www.videolan.org/), or [mplayer](https://www.mplayerhq.hu/).

### I extracted a video URL with `-g`, but it does not play on another machine / in my web browser.

Use the `--cookies` option. Some sites require the same IP address, cookies, and/or HTTP headers.
Use `--dump-user-agent`. See `--cookies`, and `--dump-json` for more details.

### ERROR: no fmt_url_map or conn information found in video info

Update to the latest version of youtube-dl as described above.

### ERROR: unable to download video

Update to the latest version of youtube-dl as described above.

### Video URL contains an ampersand and I'm getting some strange output `[1] 2839` or `'v' is not recognized as an internal or external command`

Wrap the URL in quotes (single or double) or escape the ampersands.

### ExtractorError: Could not find JS function u'OF'

Update to the latest version of youtube-dl as described above.

### HTTP Error 429: Too Many Requests or 402: Payment Required

The service is blocking your IP address due to overuse.  Solve a CAPTCHA and pass cookies to youtube-dl as described [here](#how-do-i-pass-cookies-to-youtube-dl).

If this is not the case, use `--proxy` or `--source-address`.

### SyntaxError: Non-ASCII character

Update to Python 2.6, 2.7, or 3.2+.

### What is this binary file? Where has the code gone?

youtube-dl is packaged as an executable zipfile since June 2012. Unzip it or clone the git repository.

### The exe throws an error due to missing `MSVCR100.dll`

Install the [Microsoft Visual C++ 2010 Service Pack 1 Redistributable Package (x86)](https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A4