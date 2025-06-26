[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Download Videos from YouTube and Beyond

**youtube-dl is a versatile command-line tool for downloading videos from YouTube and hundreds of other video platforms.** [Visit the original repository](https://github.com/ytdl-org/youtube-dl) for the source code and latest updates.

## Key Features

*   **Broad Platform Support:** Download videos from YouTube, Vimeo, Dailymotion, and many other popular and obscure video hosting sites.
*   **Format Selection:** Choose the video format and quality you want, including options for audio-only downloads.
*   **Playlist and Channel Downloads:** Download entire playlists and channels with a single command.
*   **Metadata Handling:** Automatically embed video metadata and download subtitles.
*   **Customization Options:** Extensive options for controlling download behavior, including proxy support, file naming, and more.

## Installation

Choose the installation method that best suits your operating system:

*   **UNIX (Linux, macOS, etc.):**
    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```
    or
    ```bash
    sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```
*   **Windows:**  Download the executable and place it in a directory within your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) variable. [Download .exe file](https://yt-dl.org/latest/youtube-dl.exe)
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

For advanced installation options, including PGP signatures, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## Usage

```bash
youtube-dl [OPTIONS] URL [URL...]
```

For a full list of options, refer to the [OPTIONS](#options) section below or run `youtube-dl -h`.

## Documentation

*   [Installation](#installation)
*   [Description](#description)
*   [Options](#options)
*   [Configuration](#configuration)
*   [Output Template](#output-template)
*   [Format Selection](#format-selection)
*   [Video Selection](#video-selection)
*   [FAQ](#faq)
*   [Developer Instructions](#developer-instructions)
*   [Embedding youtube-dl](#embedding-youtube-dl)
*   [Bugs](#bugs)
*   [Copyright](#copyright)

## OPTIONS

Provides a comprehensive list of available command-line options.

*   `-h, --help`: Print help text and exit
*   `--version`: Print program version and exit
*   `-U, --update`: Update this program to latest version.
*   `-i, --ignore-errors`: Continue on download errors, for example to skip unavailable videos in a playlist
*   `--abort-on-error`: Abort downloading of further videos (in the playlist or the command line) if an error occurs
*   `--dump-user-agent`: Display the current browser identification
*   `--list-extractors`: List all supported extractors
*   `--extractor-descriptions`: Output descriptions of all supported extractors
*   `--force-generic-extractor`: Force extraction to use the generic extractor
*   `--default-search PREFIX`: Use this prefix for unqualified URLs.
*   `--ignore-config`: Do not read configuration files.
*   `--config-location PATH`: Location of the configuration file; either the path to the config or its containing directory.
*   `--flat-playlist`: Do not extract the videos of a playlist, only list them.
*   `--mark-watched`: Mark videos watched (YouTube only)
*   `--no-mark-watched`: Do not mark videos watched (YouTube only)
*   `--no-color`: Do not emit color codes in output

### Network Options:
*   `--proxy URL`: Use the specified HTTP/HTTPS/SOCKS proxy.
*   `--socket-timeout SECONDS`: Time to wait before giving up, in seconds
*   `--source-address IP`: Client-side IP address to bind to
*   `-4, --force-ipv4`: Make all connections via IPv4
*   `-6, --force-ipv6`: Make all connections via IPv6

### Geo Restriction:
*   `--geo-verification-proxy URL`: Use this proxy to verify the IP address for some geo-restricted sites.
*   `--geo-bypass`: Bypass geographic restriction via faking X-Forwarded-For HTTP header
*   `--no-geo-bypass`: Do not bypass geographic restriction via faking X-Forwarded-For HTTP header
*   `--geo-bypass-country CODE`: Force bypass geographic restriction with explicitly provided two-letter ISO 3166-2 country code
*   `--geo-bypass-ip-block IP_BLOCK`: Force bypass geographic restriction with explicitly provided IP block in CIDR notation

### Video Selection:
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

### Download Options:
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

### Filesystem Options:
*   `-a, --batch-file FILE`: File containing URLs to download ('-' for stdin), one URL per line.
*   `--id`: Use only video ID in file name
*   `-o, --output TEMPLATE`: Output filename template, see the "OUTPUT TEMPLATE" for all the info
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

### Thumbnail Options:
*   `--write-thumbnail`: Write thumbnail image to disk
*   `--write-all-thumbnails`: Write all thumbnail image formats to disk
*   `--list-thumbnails`: Simulate and list all available thumbnail formats

### Verbosity / Simulation Options:
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

### Workarounds:
*   `--encoding ENCODING`: Force the specified encoding (experimental)
*   `--no-check-certificate`: Suppress HTTPS certificate validation
*   `--prefer-insecure`: Use an unencrypted connection to retrieve information about the video.
*   `--user-agent UA`: Specify a custom user agent
*   `--referer URL`: Specify a custom referer, use if the video access is restricted to one domain
*   `--add-header FIELD:VALUE`: Specify a custom HTTP header and its value, separated by a colon ':'.
*   `--bidi-workaround`: Work around terminals that lack bidirectional text support.
*   `--sleep-interval SECONDS`: Number of seconds to sleep before each download when used alone or a lower bound of a range for randomized sleep before each download when used along with --max-sleep-interval.
*   `--max-sleep-interval SECONDS`: Upper bound of a range for randomized sleep before each download (maximum possible number of seconds to sleep).

### Video Format Options:
*   `-f, --format FORMAT`: Video format code, see the "FORMAT SELECTION" for all the info
*   `--all-formats`: Download all available video formats
*   `--prefer-free-formats`: Prefer free video formats unless a specific one is requested
*   `-F, --list-formats`: List all available formats of requested videos
*   `--youtube-skip-dash-manifest`: Do not download the DASH manifests and related data on YouTube videos
*   `--merge-output-format FORMAT`: If a merge is required (e.g. bestvideo+bestaudio), output to given container format.

### Subtitle Options:
*   `--write-sub`: Write subtitle file
*   `--write-auto-sub`: Write automatically generated subtitle file (YouTube only)
*   `--all-subs`: Download all the available subtitles of the video
*   `--list-subs`: List all available subtitles for the video
*   `--sub-format FORMAT`: Subtitle format, accepts formats preference, for example: "srt" or "ass/srt/best"
*   `--sub-lang LANGS`: Languages of the subtitles to download (optional) separated by commas, use --list-subs for available language tags

### Authentication Options:
*   `-u, --username USERNAME`: Login with this account ID
*   `-p, --password PASSWORD`: Account password.
*   `-2, --twofactor TWOFACTOR`: Two-factor authentication code
*   `-n, --netrc`: Use .netrc authentication data
*   `--video-password PASSWORD`: Video password (vimeo, youku)

### Adobe Pass Options:
*   `--ap-mso MSO`: Adobe Pass multiple-system operator (TV provider) identifier, use --ap-list-mso for a list of available MSOs
*   `--ap-username USERNAME`: Multiple-system operator account login
*   `--ap-password PASSWORD`: Multiple-system operator account password.
*   `--ap-list-mso`: List all supported multiple-system operators

### Post-processing Options:
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

## Configuration

Customize youtube-dl's behavior by placing command-line options in a configuration file.

*   **System-wide:** `/etc/youtube-dl.conf` (Linux/macOS)
*   **User-specific:** `~/.config/youtube-dl/config` (Linux/macOS), `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf` (Windows)
*   You can use `--ignore-config` to disable the configuration file for a particular youtube-dl run.
*   Use `--config-location` if you want to use custom configuration file for a particular youtube-dl run.

## OUTPUT TEMPLATE

Use the `-o` or `--output` option to customize the output filenames.

**Examples:**

*   `-o '%(title)s-%(id)s.%(ext)s'` - Downloads as "title-id.mp4".
*   `-o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s'` - Downloads playlist videos into separate directories.
*   See the "OUTPUT TEMPLATE" for more information about the available special sequences.

## FORMAT SELECTION

Use `-f` or `--format` to specify the video format you want.

**Examples:**

*   `-f 22`: Download format code 22 (usually 720p MP4).
*   `-f best`: Download the best available quality.
*   `-f "bestvideo[height<=720]+bestaudio"` - Download the best video format with a height of 720p or lower and combine it with best audio.
*   `-f 'bestvideo,bestaudio' -o '%(title)s.f%(format_id)s.%(ext)s'` - Download best video and audio formats without merging them.

## VIDEO SELECTION

Use options like `--date`, `--datebefore`, and `--dateafter` to filter videos by their upload date.

**Examples:**

*   `--dateafter now-6months` - Download videos uploaded in the last 6 months.
*   `--date 19700101` - Download videos uploaded on January 1, 1970.

## FAQ

A collection of frequently asked questions to help with common issues.

### How do I update youtube-dl?

Run `youtube-dl -U` (or `sudo youtube-dl -U` on some systems). If you have used pip, a simple `sudo pip install -U youtube-dl` is sufficient to update.

### youtube-dl is extremely slow to start on Windows

Add a file exclusion for `youtube-dl.exe` in Windows Defender settings.

### I'm getting an error `Unable to extract OpenGraph title` on YouTube playlists

Update to the latest version of youtube-dl,  `youtube-dl -U`.

### I'm getting an error when trying to use output template: `error: using output template conflicts with using title, video ID or auto number`

Make sure you are not using `-o` with any of these options `-t`, `--title`, `--id`, `-A` or `--auto-number`. Remove the latter if any.

### Do I always have to pass `-citw`?

By default, youtube-dl intends to have the best options (incidentally, if you have a convincing case that these should be different, [please file an issue where you explain that](https://yt-dl.org/bug)). Therefore, it is unnecessary and sometimes harmful to copy long option strings from webpages. In particular, the only option out of `-citw` that is regularly useful is `-i`.

### Can you please put the `-b` option back?

Most people asking this question are not aware that youtube-dl now defaults to downloading the highest available quality as reported by YouTube, which will be 1080p or 720p in some cases, so you no longer need the `-b` option. For some specific videos, maybe YouTube does not report them to be available in a specific high quality format you're interested in. In that case, simply request it with the `-f` option and youtube-dl will try to download it.

### I get HTTP error 402 when trying to download a video. What's this?

Apparently YouTube requires you to pass a CAPTCHA test if you download too much. We're [considering to provide a way to let you solve the CAPTCHA](https://github.com/ytdl-org/youtube-dl/issues/154), but at the moment, your best course of action is pointing a web browser to the youtube URL, solving the CAPTCHA, and restart youtube-dl.

### Do I need any other programs?

youtube-dl works fine on its own on most sites. However, if you want to convert video/audio, you'll need [avconv](https://libav.org/) or [ffmpeg](https://www.ffmpeg.org/). On some sites - most notably YouTube - videos can be retrieved in a higher quality format without sound. youtube-dl will detect whether avconv/ffmpeg is present and automatically pick the best option.

Videos or video formats streamed via RTMP protocol can only be downloaded when [rtmpdump](https://rtmpdump.mplayerhq.hu/) is installed. Downloading MMS and RTSP videos requires either [mplayer](https://mplayerhq.hu/) or [mpv](https://mpv.io/) to be installed.

### I have downloaded a video but how can I play it?

Once the video is fully downloaded, use any video player, such as [mpv](https://mpv.io/), [vlc](https://www.videolan.org/) or [mplayer](https://www.mplayerhq.hu/).

### I extracted a video URL with `-g`, but it does not play on another machine / in my web browser.

It depends a lot on the service. In many cases, requests for the video (to download/play it) must come from the same IP address and with the same cookies and/or HTTP headers. Use the `--cookies` option to write the required cookies into a file, and advise your downloader to read cookies from that file. Some sites also require a common user agent to be used, use `--dump-user-agent` to see the one in use by youtube-dl. You can also get necessary cookies and HTTP headers from JSON output obtained with `--dump-json`.

It may be beneficial to use IPv6; in some cases, the restrictions are only applied to IPv4. Some services (sometimes only for a subset of videos) do not restrict the video URL by IP address, cookie, or user-agent, but these are the exception rather than the rule.

Please bear in mind that some URL protocols are **not** supported by browsers out of the box, including RTMP. If you are using `-g`, your own downloader must support these as well.

If you want to play the video on a machine that is not running youtube-dl, you can relay the video content from the machine that runs youtube-dl. You can use `-o -` to let youtube-dl stream a video to stdout, or simply allow the player to download the files written by youtube-dl in turn.

### ERROR: no fmt_url_map or conn information found in video info

YouTube has switched to a new video info format in July 2011 which is not supported by old versions of youtube-dl. See [above](#how-do-i-update-youtube-dl) for how to update youtube-dl.

### ERROR: unable to download video

YouTube requires an additional signature since September 2012 which is not supported by old versions of youtube-dl. See [above](#how-do-i-update-youtube-dl) for how to update youtube-dl.

### Video URL contains an ampersand and I'm getting some strange output `[1] 2839` or `'v' is not recognized as an internal or external command`

That's actually the output from your shell. Since ampersand is one of the special shell characters it's interpreted by the shell preventing you from passing the whole URL to youtube-dl. To disable your shell from interpreting the ampersands (or any other special characters) you have to either put the whole URL in quotes or escape them with a backslash (which approach will work depends on your shell).

For example if your URL is https://www.youtube.com/watch?t=4&v=BaW_jenozKc you should end up with following command:

```youtube-dl 'https://www.youtube.com/watch?t=4&v=BaW_jenozKc'```

or

```youtube-dl https://www.youtube.com/watch?t=4\&v=BaW_jenozKc```

For Windows you have to use the double quotes:

```youtube-dl "https://www.youtube.com/watch?t=4&v=BaW_jenozKc"```

### ExtractorError: Could not find JS function u'OF'

In February 2015, the new YouTube player contained a character sequence in a string that was misinterpreted by old versions of youtube-dl. See [above](#how-do-i-update-youtube-dl) for how to update youtube-dl.

### HTTP Error 429: Too Many Requests or 402: Payment Required

These two error codes indicate that the service is blocking your IP address because of overuse. Usually this is a soft block meaning that you can gain access again after solving CAPTCHA. Just open a browser and solve a CAPTCHA the service suggests you and after that [pass cookies](#how-do-i-pass-cookies-to-youtube-dl) to youtube-dl. Note that if your machine has multiple external IPs then you should also pass exactly the same IP you've used for solving CAPTCHA with [`--source-address`](#network-options). Also you may need to pass a `User-Agent` HTTP header of your browser with [`--user-agent`](#workarounds).

If this is not the case (no CAPTCHA suggested to solve by the service) then you can contact the service and ask them to unblock your IP address, or - if you have acquired a whitelisted IP address already - use the [`--proxy` or `--source-address` options](#network-options) to select another IP address.

### SyntaxError: Non-ASCII character

The error

    File "youtube-dl", line 2
    SyntaxError: Non-ASCII character '\x93' ...

means you're using an outdated version of Python. Please update to Python 2.6 or 2.7.

### What is this binary file? Where has the code gone?

Since June 2012 ([#342](https://github.com/ytdl-org/youtube-dl/issues/342)) youtube-dl is packed as an executable zipfile, simply unzip it (might need renaming to `youtube-dl.zip` first on some systems) or clone the git repository, as laid out above. If you modify the code, you can run it by executing the `__main__.py` file. To recompile the executable, run `make youtube-dl`.

### The exe throws an error due to missing `MSVCR100.dll`

To run the exe you need to install first the [Microsoft Visual C++ 2010 Service Pack 1 Redistributable Package (x86)](https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x86.exe).

### On Windows, how should I set up ffmpeg and youtube-dl? Where should I put the exe files?

If you put youtube-dl and ffmpeg in the same directory that you're running the command from, it will work, but that's rather cumbersome.

To make a different directory work - either for ffmpeg, or for youtube-dl, or for both - simply create the directory (say, `C:\bin`, or `C:\Users\<User name>\bin`), put all the executables directly in there, and then [set your PATH environment variable](https://www.java.com/en/download/help/path.xml) to include that directory.

From then on, after restarting your shell, you will be able to access both youtube-dl and ffmpeg (and youtube-dl will be able to find ffmpeg) by simply typing `youtube-dl` or `ffmpeg`, no matter what directory you're in.

### How do I put downloads into a specific folder?

Use the `-o` to specify an [output template](#output-template), for example `-o "/home/user/videos/%(title)s-%(id)s.%(ext)s"`. If you want this for all of your downloads, put the option into your [configuration file](#configuration).

### How do I download a video starting with a `-`?

Either prepend `https://www.youtube.com/watch?v=` or separate the ID from the options with `--`:

    youtube-dl -- -wNyEUrxzFU
    youtube-dl "https://www.youtube.com/watch?v=-wNyEUrxzFU"

### How do I pass cookies to youtube-dl?

Use the `--cookies` option, for example `--cookies /path/to/cookies/file.txt`.

In order to extract cookies from browser use any conforming browser extension for exporting cookies. For example, [Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) (for Chrome) or [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/) (for Firefox).

Note that the cookies file must be in Mozilla/Netscape format and the first line of the cookies file must be either `# HTTP Cookie File` or `# Netscape HTTP Cookie File`. Make sure you have correct [newline format](https://en.wikipedia.org/wiki/Newline) in the cookies file and convert newlines if necessary to correspond with your OS, namely `CRLF` (`\r\n`) for Windows and `LF` (`\n`) for Unix and Unix-like systems (Linux, macOS, etc.). `HTTP Error 400: Bad Request` when using `--cookies` is a good sign of invalid newline format.

Passing cookies to youtube-dl is a good way to workaround login when a particular extractor does not implement it explicitly. Another use case is working around [CAPTCHA](https://en.wikipedia.org/wiki/CAPTCHA) some websites require you to solve in particular cases in order to get access (e.g. YouTube, CloudFlare).

### How do I stream directly to media player?

You will first need to tell youtube-dl to stream media to stdout with `-o -`, and also tell your media player to read from stdin (it must be capable of this for streaming) and then pipe former to latter. For example, streaming to [vlc](https://www.videolan.org/) can be achieved with:

    youtube-dl -o - "https://www.youtube.com/watch?v=BaW_jenozKcj" | vlc -

### How do I download only new videos from a playlist?

Use download-archive feature. With this feature you should initially download the complete playlist with `--download-archive /path/to/download/archive/file.txt` that will record identifiers of all the videos in a special file. Each subsequent run with the same `--download-archive` will download only new videos and skip all videos that have been downloaded before. Note that only successful downloads are recorded in the file.

For example, at first,

    youtube-dl --download-archive archive.txt "https://www.youtube.com/playlist?list=PLwiyx1dc3P2JR9N8gQaQN_BCvlSlap7re"

will download the complete `PLwiyx1dc3P2JR9N8gQaQN_BCvlSlap7re` playlist and create a file `archive.txt`. Each subsequent run will only download new videos if any:

    youtube-dl --download-archive archive.txt "https://www.youtube.com/playlist?list=PLwiyx1dc3P2JR9N8gQaQN_BCvlSlap7re"

### Should I add `--hls-prefer-native` into my config?

When youtube-dl detects an HLS video, it can download it either with the built-in downloader or ffmpeg. Since many HLS streams are slightly invalid and ffmpeg/youtube-dl each handle some invalid cases better than the other, there is an option to switch the downloader if needed.

When youtube-dl knows that one particular downloader works better for a given website, that downloader will be picked. Otherwise, youtube-dl will pick the best downloader for general compatibility, which at the moment happens to be ffmpeg. This choice may change in future versions of youtube-dl, with improvements of the built-in downloader and/or ffmpeg.

In particular, the generic extractor (used when your website is not in the [list of supported sites by youtube-dl](https://ytdl-org.github.io/youtube-dl/supportedsites.html) cannot mandate one specific downloader.

If you put either `--hls-prefer-native` or `--hls-prefer-ffmpeg` into your configuration, a different subset of videos will fail to download correctly. Instead, it is much better to [file an issue](https://