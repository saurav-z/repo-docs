[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# Download Videos From 1,000+ Sites with youtube-dl

**Tired of manually downloading videos?** [youtube-dl](https://github.com/ytdl-org/youtube-dl) is a powerful, command-line tool that lets you download videos from YouTube and thousands of other video and streaming platforms!

## Key Features:

*   **Wide Site Support:** Download videos from YouTube, Vimeo, Facebook, and 1,000+ other sites.
*   **Format Selection:** Choose your preferred video and audio formats for optimal quality and size.
*   **Playlist & Channel Downloads:** Easily download entire playlists or all videos from a channel.
*   **Customization:**  Control output filenames, resolution, subtitles, and more with extensive options.
*   **Cross-Platform:** Works seamlessly on Linux, macOS, and Windows.
*   **Easy Updates:** Keep youtube-dl up-to-date with a simple command.

## Table of Contents

*   [Installation](#installation)
    *   [Unix (Linux, macOS)](#installation)
    *   [Windows](#installation)
    *   [Using pip](#installation)
    *   [macOS with Homebrew](#installation)
    *   [macOS with MacPorts](#installation)
    *   [Developer Installation](#developer-instructions)
    *   [Download Page](#download-page)
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
    *   [Output Template Examples](#output-template-examples)
    *   [Output template and Windows batch files](#output-template-and-windows-batch-files)
*   [Format Selection](#format-selection)
    *   [Format Selection Examples](#format-selection-examples)
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
    *   [How do I download a video starting with a `-`?](#how-do-i-download-a-video-starting-with--)
    *   [How do I pass cookies to youtube-dl?](#how-do-i-pass-cookies-to-youtube-dl)
    *   [How do I stream directly to media player?](#how-do-i-stream-directly-to-media-player)
    *   [How do I download only new videos from a playlist?](#how-do-i-download-only-new-videos-from-a-playlist)
    *   [Should I add `--hls-prefer-native` into my config?](#should-i-add---hls-prefer-native-into-my-config)
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
            *   [Safely extract optional description from parsed JSON](#safely-extract-optional-description-from-parsed-json)
            *   [Safely extract more optional metadata](#safely-extract-more-optional-metadata)
            *   [Safely extract nested lists](#safely-extract-nested-lists)
*   [Embedding Youtube-dl](#embedding-youtube-dl)
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

### Unix (Linux, macOS)

To install the latest version on Unix-like systems, run these commands in your terminal:

```bash
sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

### Windows

Download the `youtube-dl.exe` file from [the official download page](https://ytdl-org.github.io/youtube-dl/download.html) and place it in a directory included in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) (e.g., `C:\Windows` or `C:\Users\<YourUsername>\bin`).  **Do not** place it in `%SYSTEMROOT%\System32`.

### Using pip

You can also install or update youtube-dl using pip:

```bash
sudo -H pip install --upgrade youtube-dl
```

### macOS with Homebrew

```bash
brew install youtube-dl
```

### macOS with MacPorts

```bash
sudo port install youtube-dl
```

For other installation options, including PGP signatures, please see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## Description

**youtube-dl** is a versatile command-line program designed for downloading videos from various online platforms. It supports a vast number of websites and offers a wide range of features for customization and control over the download process.

```bash
youtube-dl [OPTIONS] URL [URL...]
```

## Options

For detailed information about all the options, please consult the [original repository](https://github.com/ytdl-org/youtube-dl).  Here's a summary:

### Network Options

*   `--proxy URL`: Use the specified HTTP/HTTPS/SOCKS proxy.
*   `--socket-timeout SECONDS`: Set a timeout for network connections.
*   `-4, --force-ipv4`: Force IPv4 connections.
*   `-6, --force-ipv6`: Force IPv6 connections.

### Geo Restriction

*   `--geo-verification-proxy URL`: Use a proxy to verify the IP address.
*   `--geo-bypass`: Bypass geographic restrictions.
*   `--geo-bypass-country CODE`: Force bypass with a specific country code.
*   `--geo-bypass-ip-block IP_BLOCK`: Force bypass with an IP block in CIDR notation.

### Video Selection

*   `--playlist-start NUMBER`: Start downloading from a specific playlist item.
*   `--playlist-end NUMBER`: End downloading at a specific playlist item.
*   `--playlist-items ITEM_SPEC`: Download specific playlist items.
*   `--match-title REGEX`: Download only videos with matching titles.
*   `--reject-title REGEX`: Skip videos with matching titles.
*   `--max-downloads NUMBER`: Limit the number of downloads.
*   `--min-filesize SIZE`: Don't download files smaller than this size.
*   `--max-filesize SIZE`: Don't download files larger than this size.
*   `--date DATE`: Download videos uploaded on a specific date.
*   `--datebefore DATE`: Download videos uploaded before this date.
*   `--dateafter DATE`: Download videos uploaded after this date.
*   `--min-views COUNT`: Don't download videos with fewer views.
*   `--max-views COUNT`: Don't download videos with more views.
*   `--match-filter FILTER`: Generic video filter.
*   `--no-playlist`: Download only the video if the URL refers to both a video and a playlist.
*   `--yes-playlist`: Download the playlist if the URL refers to both a video and a playlist.
*   `--age-limit YEARS`: Limit downloads to videos suitable for the given age.
*   `--download-archive FILE`: Only download videos not listed in the archive file.
*   `--include-ads`: Download advertisements (experimental).

### Download Options

*   `-r, --limit-rate RATE`: Limit the download rate.
*   `-R, --retries RETRIES`: Set the number of retries.
*   `--fragment-retries RETRIES`: Set the number of retries for a fragment.
*   `--skip-unavailable-fragments`: Skip unavailable fragments.
*   `--abort-on-unavailable-fragment`: Abort if a fragment is unavailable.
*   `--keep-fragments`: Keep downloaded fragments.
*   `--buffer-size SIZE`: Set the download buffer size.
*   `--no-resize-buffer`: Don't adjust the buffer size.
*   `--http-chunk-size SIZE`: Set the chunk size for HTTP downloads.
*   `--playlist-reverse`: Download playlists in reverse order.
*   `--playlist-random`: Download playlists in random order.
*   `--xattr-set-filesize`: Set file xattribute ytdl.filesize.
*   `--hls-prefer-native`: Use the native HLS downloader.
*   `--hls-prefer-ffmpeg`: Use ffmpeg for HLS downloads.
*   `--hls-use-mpegts`: Use the mpegts container for HLS videos.
*   `--external-downloader COMMAND`: Use an external downloader.
*   `--external-downloader-args ARGS`: Give arguments to the external downloader.

### Filesystem Options

*   `-a, --batch-file FILE`: Read URLs from a file.
*   `--id`: Use only the video ID in the filename.
*   `-o, --output TEMPLATE`: Set the output filename template.
*   `--output-na-placeholder PLACEHOLDER`: Set a placeholder for unavailable meta fields.
*   `--autonumber-start NUMBER`: Specify the starting value for autonumber.
*   `--restrict-filenames`: Restrict filenames to ASCII and avoid special characters.
*   `-w, --no-overwrites`: Don't overwrite files.
*   `-c, --continue`: Resume partially downloaded files.
*   `--no-continue`: Don't resume partially downloaded files.
*   `--no-part`: Don't use .part files.
*   `--no-mtime`: Don't set the file modification time.
*   `--write-description`: Write video description to a file.
*   `--write-info-json`: Write video metadata to a .info.json file.
*   `--write-annotations`: Write video annotations to a .annotations.xml file.
*   `--load-info-json FILE`: Load video information from a JSON file.
*   `--cookies FILE`: Load cookies from a file.
*   `--cache-dir DIR`: Set the cache directory.
*   `--no-cache-dir`: Disable filesystem caching.
*   `--rm-cache-dir`: Delete the cache directory.

### Thumbnail Options

*   `--write-thumbnail`: Write thumbnail image to disk.
*   `--write-all-thumbnails`: Write all available thumbnail formats to disk.
*   `--list-thumbnails`: List all available thumbnail formats.

### Verbosity / Simulation Options

*   `-q, --quiet`: Activate quiet mode.
*   `--no-warnings`: Ignore warnings.
*   `-s, --simulate`: Simulate, don't download.
*   `--skip-download`: Don't download the video.
*   `-g, --get-url`: Simulate, print URL.
*   `-e, --get-title`: Simulate, print title.
*   `--get-id`: Simulate, print ID.
*   `--get-thumbnail`: Simulate, print thumbnail URL.
*   `--get-description`: Simulate, print video description.
*   `--get-duration`: Simulate, print video length.
*   `--get-filename`: Simulate, print output filename.
*   `--get-format`: Simulate, print output format.
*   `-j, --dump-json`: Simulate, print JSON information.
*   `-J, --dump-single-json`: Simulate, print JSON information for each argument.
*   `--print-json`: Print video information as JSON.
*   `--newline`: Output progress bar on new lines.
*   `--no-progress`: Don't print the progress bar.
*   `--console-title`: Display progress in the console titlebar.
*   `-v, --verbose`: Print debugging information.
*   `--dump-pages`: Print downloaded pages.
*   `--write-pages`: Write downloaded intermediary pages to files.
*   `--print-traffic`: Display HTTP traffic.
*   `-C, --call-home`: Contact the youtube-dl server for debugging.
*   `--no-call-home`: Don't contact the youtube-dl server.

### Workarounds

*   `--encoding ENCODING`: Force the specified encoding.
*   `--no-check-certificate`: Suppress HTTPS certificate validation.
*   `--prefer-insecure`: Use an unencrypted connection.
*   `--user-agent UA`: Specify a custom user agent.
*   `--referer URL`: Specify a custom referer.
*   `--add-header FIELD:VALUE`: Add a custom HTTP header.
*   `--bidi-workaround`: Work around terminals that lack bidirectional text support.
*   `--sleep-interval SECONDS`: Sleep before each download.
*   `--max-sleep-interval SECONDS`: Upper bound for randomized sleep.

### Video Format Options

*   `-f, --format FORMAT`: Select video format.
*   `--all-formats`: Download all available video formats.
*   `--prefer-free-formats`: Prefer free video formats.
*   `-F, --list-formats`: List available formats.
*   `--youtube-skip-dash-manifest`: Do not download DASH manifests on YouTube.
*   `--merge-output-format FORMAT`: Merge to the specified container format.

### Subtitle Options

*   `--write-sub`: Write subtitle file.
*   `--write-auto-sub`: Write automatically generated subtitles (YouTube only).
*   `--all-subs`: Download all available subtitles.
*   `--list-subs`: List available subtitles.
*   `--sub-format FORMAT`: Set subtitle format.
*   `--sub-lang LANGS`: Select subtitle languages.

### Authentication Options

*   `-u, --username USERNAME`: Login with this username.
*   `-p, --password PASSWORD`: Login with this password.
*   `-2, --twofactor TWOFACTOR`: Two-factor authentication code.
*   `-n, --netrc`: Use .netrc authentication data.
*   `--video-password PASSWORD`: Video password.

### Adobe Pass Options

*   `--ap-mso MSO`: Adobe Pass MSO identifier.
*   `--ap-username USERNAME`: Adobe Pass username.
*   `--ap-password PASSWORD`: Adobe Pass password.
*   `--ap-list-mso`: List supported MSOs.

### Post-processing Options

*   `-x, --extract-audio`: Convert video to audio.
*   `--audio-format FORMAT`: Specify audio format.
*   `--audio-quality QUALITY`: Specify audio quality.
*   `--recode-video FORMAT`: Encode the video to another format.
*   `--postprocessor-args ARGS`: Pass arguments to the postprocessor.
*   `-k, --keep-video`: Keep the video file after post-processing.
*   `--no-post-overwrites`: Don't overwrite post-processed files.
*   `--embed-subs`: Embed subtitles in the video.
*   `--embed-thumbnail`: Embed thumbnail as cover art.
*   `--add-metadata`: Write metadata to the video file.
*   `--metadata-from-title FORMAT`: Parse metadata from the video title.
*   `--xattrs`: Write metadata to the video file's xattrs.
*   `--fixup POLICY`: Automatically correct file faults.
*   `--prefer-avconv`: Prefer avconv over ffmpeg.
*   `--prefer-ffmpeg`: Prefer ffmpeg over avconv.
*   `--ffmpeg-location PATH`: Set the ffmpeg/avconv binary location.
*   `--exec CMD`: Execute a command after downloading and post-processing.
*   `--convert-subs FORMAT`: Convert subtitles to another format.

## Configuration

youtube-dl can be configured using a configuration file. On Linux and macOS, this file is located at `/etc/youtube-dl.conf` (system-wide) and `~/.config/youtube-dl/config` (user-specific). On Windows, the user-specific configuration file is `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`.

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

### Authentication with .netrc file

You can also store credentials for extractors that support authentication in a `.netrc` file. Create a `.netrc` file in your `$HOME` and restrict permissions:
```
touch $HOME/.netrc
chmod a-rwx,u+rw $HOME/.netrc
```
Add credentials in the format:
```
machine <extractor> login <login> password <password>
```
Activate authentication with the `.netrc` file by passing `--netrc` to youtube-dl or place it in the configuration file.

## Output Template

The `-o` option allows you to customize the output filenames.  You can use special sequences (e.g., `%(title)s`, `%(id)s`, `%(ext)s`) to represent video metadata.

### Output Template Examples

```bash
# Basic template
youtube-dl -o '%(title)s.%(ext)s' "https://www.example.com/video"

# Playlist with index
youtube-dl -o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' "https://www.youtube.com/playlist?list=PL..."

# Restrict Filenames
youtube-dl --get-filename -o '%(title)s.%(ext)s' "BaW_jenozKc" --restrict-filenames
```

### Output template and Windows batch files

If you are using an output template inside a Windows batch file then you must escape plain percent characters (`%`) by doubling, so that `-o "%(title)s-%(id)s.%(ext)s"` should become `-o "%%(title)s-%%(id)s.%%(ext)s"`.

## Format Selection

Use the `-f` or `--format` option to select the desired video format. You can use format codes, extensions, and special names.

### Format Selection Examples

```bash
# Download best mp4 format available or any other best if no mp4 available
youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'

# Download best format available but no better than 480p
youtube-dl -f 'bestvideo[height<=480]+bestaudio/best[height<=480]'
```

## Video Selection

Use the options like `--date`, `--datebefore`, and `--dateafter` to filter videos by upload date.

```bash
# Download only the videos uploaded in the last 6 months
$ youtube-dl --dateafter now-6months

# Download only the videos uploaded on January 1, 1970
$ youtube-dl --date 19700101
```

## FAQ

### How do I update youtube-dl?

Run `youtube-dl -U` (or, on Linux, `sudo youtube-dl -U`).

### youtube-dl is extremely slow to start on Windows

Add a file exclusion for `youtube-dl.exe` in Windows Defender settings.

### I'm getting an error `Unable to extract OpenGraph title` on YouTube playlists

Update youtube-dl to at least version 2014.07.25.

### I'm getting an error when trying to use output template: `error: using output template conflicts with using title, video ID or auto number`

Make sure you are not using `-o` with `-t`, `--title`, `--id`, `-A` or `--auto-number`.

### Do I always have to pass `-citw`?

No, youtube-dl is intended to have the best options by default.

### Can you please put the `-b` option back?

youtube-dl defaults to downloading the highest available quality as reported by YouTube, so the `-b` option is often unnecessary.

### I get HTTP error 402 when trying to download a video. What's this?

YouTube requires you to pass a CAPTCHA test if you download too much. Solve the CAPTCHA in a browser, then restart youtube-dl.

### Do I need any other programs?

You may need [avconv](https://libav.org/) or [ffmpeg](https://www.ffmpeg.org/) for converting video/audio.  For RTMP streams, you'll need [rtmpdump](https://rtmpdump.mplayerhq.hu/).

### I have downloaded a video but how can I play it?

Use any video player, such as [mpv](https://mpv.io/), [vlc](https://www.videolan.org/) or [mplayer](https://www.mplayerhq.hu/).

### I extracted a video URL with `-g`, but it does not play on another machine / in my web browser.

You may need to pass cookies and/or HTTP headers. Use `--cookies` option.

### ERROR: no fmt_url_map or conn information found in video info

Update youtube-dl.

### ERROR: unable to download video

Update youtube-dl.

### Video URL contains an ampersand and I'm getting some strange output `[1] 2839` or `'v' is not recognized as an internal or external command`

Enclose the URL in quotes or escape the ampersands (`&`).

### ExtractorError: Could not find JS function u'OF'

Update youtube-dl.

### HTTP Error 429: Too Many Requests or 402: Payment Required

The service is blocking your IP address.  Solve CAPTCHA, pass cookies, and/or use a proxy.

### SyntaxError: Non-ASCII character

Update to Python 2.6 or 2.7.

### What is this binary file? Where has the code gone?

youtube-dl is packed as an executable zipfile.

### The exe throws an error due to missing `MSVCR100.dll`

Install the [Microsoft Visual C++ 2010 Service Pack 1 Redistributable Package (x86)](https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x86.exe).

### On Windows, how should I set up ffmpeg and youtube-dl? Where should I put the exe files?

Put the executables in a directory and add that directory to your PATH environment variable.

### How do I put downloads into a specific folder?

Use the `-o` option with an [output template](#output-template).

### How do I download a video starting with a `-`?

Use `--` to separate the ID from the options:
```bash
youtube-dl -- -wNyEUrxzFU
```

### How do I pass cookies to youtube-dl?

Use the `--cookies` option.

### How do I stream directly to media player?

Use `-o -` and pipe the output to your media player.
```bash
youtube-dl -o - "https://www.youtube.com/watch?v=BaW_jenozKcj" | vlc -
```

### How do I download only new videos from a playlist?

Use the download-archive feature with `--download-archive /path/to/download/archive/file.txt`.

### Should I add `--hls-prefer-native` into my config?

Only if you know one downloader works better for a specific website.

### Can you add support for this anime video site, or site which shows current movies for free?

youtube-dl does not support services that specialize in infringing copyright.

### How can I speed up work on my issue?

Provide the full output of `youtube-dl -v YOUR_URL_HERE`.

### How can I detect whether a given URL is supported by youtube-dl?

Simply call youtube-dl with the URL.

## Why do I need to go through that much red tape when filing bugs?

The issue template ensures that reports include the necessary information for developers to understand and fix the issue.

## Developer Instructions

To contribute, [fork the repository](https://github.com/ytdl-org/youtube-dl/fork), clone the code, and create a new branch for your changes.  Follow the instructions below and in the [original repository](https://github.com/ytdl-org/youtube-dl) to add support for a new site.

### Adding support for a new site

1.  Ensure the site distributes content legally.
2.  Follow the steps in the repository's [developer instructions](#developer-instructions).

#### youtube-dl coding conventions

Extractors are fragile since they depend on the source data.

##### Mandatory and optional metafields

Extractors rely on metadata like `id`, `title`, and `url` in the *info dict*.

##### Provide fallbacks

Extract from multiple sources.

##### Regular expressions

Use relaxed, flexible, and non-capturing regular expressions to ensure the widest applicability.

##### Don't capture groups you don't use

Capturing group must be an indication that it's used somewhere in the code. Any group that is not used must be non capturing.

##### Make regular expressions relaxed and flexible

##### Long lines policy

Keep lines under 80 characters if possible.

##### Inline values

Avoid extracting variables used only once.

##### Collapse fallbacks

Use a list of patterns.

##### Trailing parentheses

Move trailing parentheses after the last argument.

##### Use convenience conversion and parsing functions

Use `int_or_none`, `float_or_none`, `url_or_none`, and other utils.

##### Safely extract optional description from parsed JSON

Use `traverse_obj`.

##### Safely extract more optional metadata

Use `traverse_obj`.

##### Safely extract nested lists

Use `traverse_obj`.

## Embedding Youtube-dl

youtube-dl can be embedded in Python programs:

```python
from __future__ import unicode_literals
import youtube_dl

ydl_opts = {}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=BaW_jenozKc'])
```

## Bugs

Report bugs and suggestions in the issue tracker: <https://github.com/ytdl-org/youtube-dl/issues>.

### Opening a bug report or suggestion

Follow instructions in the issue tracker and **include the full output of `youtube-dl -v`**.

### Is the description of the issue itself sufficient?

Provide a clear explanation of the problem, how to fix it, and your proposed solution.

### Is the issue already documented?

Search existing issues.

### Are you using the latest version?

Update youtube-dl with `youtube-dl -U` before reporting.

### Why are existing options not enough?

Describe how existing options don't solve your problem.

### Is there enough context in your bug report?

Provide the greater context and a use case scenario.

### Does the issue involve one problem, and one problem only?

Report one issue per ticket.

### Is anyone going to need the feature?

Only post features you or an incapacitated friend personally require.

### Is your question about youtube-dl?

Make sure you are actually using youtube-dl.

## Copyright

youtube-dl is released into the public domain.

This README file was written by [Daniel Bolton](https://github.com/dbbolton) and is likewise released into the public domain.