[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# YouTube-DL: Download Videos from YouTube and Beyond

Tired of streaming? **YouTube-DL is your go-to command-line tool for downloading videos from YouTube.com and thousands of other sites.** 

[Visit the original repository for more information](https://github.com/ytdl-org/youtube-dl).

## Key Features

*   **Broad Site Support:** Works with YouTube and a vast array of other video platforms.
*   **Format Selection:** Download videos in your preferred quality and format.
*   **Playlist and Channel Downloads:** Easily download entire playlists or all videos from a channel.
*   **Metadata Extraction:** Get video titles, descriptions, and more.
*   **Customization:** Tailor downloads with a wide range of options, including file naming, download speed limits, and more.
*   **Cross-Platform:** Runs on Windows, macOS, and Linux.
*   **Active Community & Regular Updates:** Benefit from constant improvements and updates.

## Getting Started

### Installation

Choose your platform to install YouTube-DL:

*   **UNIX (Linux, macOS, etc.):**

    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```

    If you don't have `curl`, use `wget`:

    ```bash
    sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```
*   **Windows:**
    *   [Download the .exe file](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory included in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) (e.g., *not* `C:\Windows\System32`).
*   **Pip:**
    ```bash
    sudo -H pip install --upgrade youtube-dl
    ```
    This command will update youtube-dl if you have already installed it. See the [pypi page](https://pypi.python.org/pypi/youtube_dl) for more information.
*   **Homebrew (macOS):**
    ```bash
    brew install youtube-dl
    ```
*   **MacPorts (macOS):**
    ```bash
    sudo port install youtube-dl
    ```
*   **Developer Installation** refer to the [developer instructions](#developer-instructions) for how to check out and work with the git repository.

For other options, including PGP signatures, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

### Usage

To download a video, simply run the command:

```bash
youtube-dl [OPTIONS] [URL]
```

Replace `[URL]` with the video's web address. For example:

```bash
youtube-dl https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

## Core Concepts & Common Commands

*   **Options:** Modify download behavior (e.g., format, output file).
*   **Formats:** Select video and audio qualities using the `-f` or `--format` flag. Use `-F` or `--list-formats` to view available formats.
*   **Output Template:** Customize filenames with the `-o` or `--output` flag (e.g., `-o '%(title)s-%(id)s.%(ext)s'`).
*   **Playlists and Channels:** Download entire playlists or channels by providing their URLs.

## Advanced Options

Explore the comprehensive list of options below to fine-tune your downloads.

### General Options

*   `-h`, `--help`: Print help.
*   `--version`: Show version.
*   `-U`, `--update`: Update youtube-dl.
*   `-i`, `--ignore-errors`: Skip errors and continue.
*   `--abort-on-error`: Stop on errors.
*   `--dump-user-agent`: Show browser identifier.
*   `--list-extractors`: List supported extractors.
*   `--extractor-descriptions`: Display descriptions of extractors.
*   `--force-generic-extractor`: Force the generic extractor.
*   `--default-search PREFIX`: Add a prefix to unqualified URLs.
*   `--ignore-config`: Ignore config files.
*   `--config-location PATH`: Location of the config file.
*   `--flat-playlist`: List videos only.
*   `--mark-watched`: Mark videos as watched (YouTube only).
*   `--no-mark-watched`: Don't mark videos as watched.
*   `--no-color`: Disable colored output.

### Network Options

*   `--proxy URL`: Use a proxy.
*   `--socket-timeout SECONDS`: Set socket timeout.
*   `--source-address IP`: Bind to a specific IP.
*   `-4`, `--force-ipv4`: Force IPv4.
*   `-6`, `--force-ipv6`: Force IPv6.

### Geo Restriction Options

*   `--geo-verification-proxy URL`: Proxy for geo-restricted sites.
*   `--geo-bypass`: Bypass geo-restrictions.
*   `--no-geo-bypass`: Disable geo-bypass.
*   `--geo-bypass-country CODE`: Bypass geo-restrictions with a country code.
*   `--geo-bypass-ip-block IP_BLOCK`: Bypass with an IP block.

### Video Selection Options

*   `--playlist-start NUMBER`: Start at playlist video.
*   `--playlist-end NUMBER`: End at playlist video.
*   `--playlist-items ITEM_SPEC`: Download specific items (e.g., `--playlist-items 1,2,5-8`).
*   `--match-title REGEX`: Download titles that match (regex).
*   `--reject-title REGEX`: Skip titles that match (regex).
*   `--max-downloads NUMBER`: Limit downloads.
*   `--min-filesize SIZE`: Minimum file size.
*   `--max-filesize SIZE`: Maximum file size.
*   `--date DATE`: Download videos uploaded on a date.
*   `--datebefore DATE`: Download before a date.
*   `--dateafter DATE`: Download after a date.
*   `--min-views COUNT`: Minimum views.
*   `--max-views COUNT`: Maximum views.
*   `--match-filter FILTER`: Filter based on video properties.
*   `--no-playlist`: Download the video only, if the URL is a video and playlist.
*   `--yes-playlist`: Download the playlist, if the URL is a video and playlist.
*   `--age-limit YEARS`: Limit age for videos.
*   `--download-archive FILE`: Archive downloaded videos.
*   `--include-ads`: Download ads (experimental).

### Download Options

*   `-r`, `--limit-rate RATE`: Limit download rate (e.g., `50K` or `4.2M`).
*   `-R`, `--retries RETRIES`: Retries on failure.
*   `--fragment-retries RETRIES`: Retries for fragments.
*   `--skip-unavailable-fragments`: Skip unavailable fragments.
*   `--abort-on-unavailable-fragment`: Abort on unavailable fragment.
*   `--keep-fragments`: Keep fragments on disk.
*   `--buffer-size SIZE`: Download buffer size.
*   `--no-resize-buffer`: Disable automatic buffer resizing.
*   `--http-chunk-size SIZE`: HTTP chunk size.
*   `--playlist-reverse`: Reverse playlist download.
*   `--playlist-random`: Random playlist download.
*   `--xattr-set-filesize`: Set file size xattribute.
*   `--hls-prefer-native`: Use native HLS downloader.
*   `--hls-prefer-ffmpeg`: Use ffmpeg for HLS.
*   `--hls-use-mpegts`: Use mpegts container for HLS.
*   `--external-downloader COMMAND`: Use an external downloader. (aria2c, avconv, axel, curl, ffmpeg, httpie, wget)
*   `--external-downloader-args ARGS`: Arguments for the external downloader.

### Filesystem Options

*   `-a`, `--batch-file FILE`: Download URLs from a file.
*   `--id`: Use only video ID for filenames.
*   `-o`, `--output TEMPLATE`: Output filename template (see examples in [output template section](#output-template)).
*   `--output-na-placeholder PLACEHOLDER`: Placeholder for missing metadata.
*   `--autonumber-start NUMBER`: Autonumber start value.
*   `--restrict-filenames`: Restrict filenames to ASCII.
*   `-w`, `--no-overwrites`: Don't overwrite files.
*   `-c`, `--continue`: Resume downloads.
*   `--no-continue`: Don't resume downloads.
*   `--no-part`: Don't use .part files.
*   `--no-mtime`: Don't set file modification time.
*   `--write-description`: Write description to a file.
*   `--write-info-json`: Write metadata to a .info.json file.
*   `--write-annotations`: Write annotations to a .annotations.xml file.
*   `--load-info-json FILE`: Load metadata from a .info.json file.
*   `--cookies FILE`: Read cookies from a file.
*   `--cache-dir DIR`: Cache directory.
*   `--no-cache-dir`: Disable caching.
*   `--rm-cache-dir`: Delete cache.

### Thumbnail Options

*   `--write-thumbnail`: Write thumbnail to disk.
*   `--write-all-thumbnails`: Write all thumbnails.
*   `--list-thumbnails`: List available thumbnails.

### Verbosity / Simulation Options

*   `-q`, `--quiet`: Quiet mode.
*   `--no-warnings`: Ignore warnings.
*   `-s`, `--simulate`: Simulate download.
*   `--skip-download`: Skip download.
*   `-g`, `--get-url`: Get URL only.
*   `-e`, `--get-title`: Get title only.
*   `--get-id`: Get ID only.
*   `--get-thumbnail`: Get thumbnail URL.
*   `--get-description`: Get description.
*   `--get-duration`: Get duration.
*   `--get-filename`: Get filename.
*   `--get-format`: Get format.
*   `-j`, `--dump-json`: Dump JSON information.
*   `-J`, `--dump-single-json`: Dump JSON for a single argument.
*   `--print-json`: Print video information as JSON.
*   `--newline`: Output progress bar as new lines.
*   `--no-progress`: Disable progress bar.
*   `--console-title`: Display progress in console title.
*   `-v`, `--verbose`: Verbose mode.
*   `--dump-pages`: Dump downloaded pages.
*   `--write-pages`: Write downloaded pages.
*   `--print-traffic`: Display HTTP traffic.
*   `-C`, `--call-home`: Contact youtube-dl server for debugging.
*   `--no-call-home`: Do NOT contact youtube-dl server.

### Workarounds

*   `--encoding ENCODING`: Force an encoding.
*   `--no-check-certificate`: Suppress certificate validation.
*   `--prefer-insecure`: Use an unencrypted connection.
*   `--user-agent UA`: Specify a custom user agent.
*   `--referer URL`: Specify a custom referer.
*   `--add-header FIELD:VALUE`: Add a custom HTTP header.
*   `--bidi-workaround`: Work around bidirectional text issues.
*   `--sleep-interval SECONDS`: Sleep before each download.
*   `--max-sleep-interval SECONDS`: Sleep range upper bound.

### Video Format Options

*   `-f`, `--format FORMAT`: Video format code. See [format selection examples](#format-selection-examples).
*   `--all-formats`: Download all formats.
*   `--prefer-free-formats`: Prefer free formats.
*   `-F`, `--list-formats`: List available formats.
*   `--youtube-skip-dash-manifest`: Don't download DASH manifests.
*   `--merge-output-format FORMAT`: Container format for merge (mkv, mp4, ogg, webm, flv).

### Subtitle Options

*   `--write-sub`: Write subtitles.
*   `--write-auto-sub`: Write auto-generated subtitles.
*   `--all-subs`: Download all subtitles.
*   `--list-subs`: List subtitles.
*   `--sub-format FORMAT`: Subtitle format.
*   `--sub-lang LANGS`: Subtitle languages.

### Authentication Options

*   `-u`, `--username USERNAME`: Username.
*   `-p`, `--password PASSWORD`: Password.
*   `-2`, `--twofactor TWOFACTOR`: Two-factor authentication code.
*   `-n`, `--netrc`: Use .netrc authentication data.
*   `--video-password PASSWORD`: Video password.

### Adobe Pass Options

*   `--ap-mso MSO`: Adobe Pass MSO.
*   `--ap-username USERNAME`: Adobe Pass username.
*   `--ap-password PASSWORD`: Adobe Pass password.
*   `--ap-list-mso`: List available MSOS.

### Post-processing Options

*   `-x`, `--extract-audio`: Extract audio.
*   `--audio-format FORMAT`: Audio format. (best, aac, flac, mp3, m4a, opus, vorbis, wav)
*   `--audio-quality QUALITY`: Audio quality.
*   `--recode-video FORMAT`: Recode video.
*   `--postprocessor-args ARGS`: Post-processor arguments.
*   `-k`, `--keep-video`: Keep video after processing.
*   `--no-post-overwrites`: Don't overwrite processed files.
*   `--embed-subs`: Embed subtitles.
*   `--embed-thumbnail`: Embed thumbnail.
*   `--add-metadata`: Add metadata.
*   `--metadata-from-title FORMAT`: Parse metadata from title.
*   `--xattrs`: Write metadata to xattrs.
*   `--fixup POLICY`: Fix file issues. (never, warn, detect_or_warn)
*   `--prefer-avconv`: Prefer avconv.
*   `--prefer-ffmpeg`: Prefer ffmpeg (default).
*   `--ffmpeg-location PATH`: FFmpeg location.
*   `--exec CMD`: Execute a command after processing.
*   `--convert-subs FORMAT`: Convert subtitles.

## Configuration

Customize youtube-dl behavior using a configuration file:

*   **System-wide:** `/etc/youtube-dl.conf` (Linux/macOS).
*   **User-specific:** `~/.config/youtube-dl/config` (Linux/macOS), `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf` (Windows).

Example Config file:

```ini
# Comments start with #
-x  # Always extract audio
--no-mtime  # Don't set modification time
--proxy 127.0.0.1:3128  # Use this proxy
-o ~/Movies/%(title)s.%(ext)s  # Save to Movies directory
```
*  `--ignore-config` and `--config-location` can also be used to disable or specify custom config files.

### Authentication with `.netrc` file
Configure automatic credentials storage for extractors that support authentication using a [`.netrc` file](https://stackoverflow.com/tags/.netrc/info):
1.  Create a `.netrc` file in your `$HOME` and restrict permissions:
    ```bash
    touch $HOME/.netrc
    chmod a-rwx,u+rw $HOME/.netrc
    ```
2.  Add credentials for an extractor in the following format, where *extractor* is the name of the extractor in lowercase:
    ```
    machine <extractor> login <login> password <password>
    ```
    For example:
    ```
    machine youtube login myaccount@gmail.com password my_youtube_password
    machine twitch login my_twitch_account_name password my_twitch_password
    ```
3.  Activate authentication with the `.netrc` file by passing `--netrc` to youtube-dl or placing it in the configuration file.
*   On Windows, setup the `%HOME%` environment variable manually (e.g. `set HOME=%USERPROFILE%`).

## Output Template Examples
Specify output file names using the `-o` flag and [formatting codes](#output-template).
```bash
# Basic filename
youtube-dl --get-filename -o '%(title)s.%(ext)s' "video_url"

# Download to separate directory
youtube-dl -o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' "playlist_url"

# Restrict filenames
youtube-dl --get-filename -o '%(title)s.%(ext)s' "video_url" --restrict-filenames
```

## Format Selection Examples
Choose specific formats with the `-f` flag.
```bash
# Best mp4 format or any other best if no mp4 available
youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' "video_url"

# Best format, no larger than 480p
youtube-dl -f 'bestvideo[height<=480]+bestaudio/best[height<=480]' "video_url"

# Best video no bigger than 50 MB
youtube-dl -f 'best[filesize<50M]' "video_url"

# Best format over http/https
youtube-dl -f '(bestvideo+bestaudio/best)[protocol^=http]' "video_url"

# Best video and best audio separately, without merging them
youtube-dl -f 'bestvideo,bestaudio' -o '%(title)s.f%(format_id)s.%(ext)s' "video_url"
```

## Video Selection Examples
Filter videos with date constraints.

```bash
# Videos uploaded in the last 6 months
youtube-dl --dateafter now-6months "playlist_url"

# Videos uploaded on January 1, 1970
youtube-dl --date 19700101 "playlist_url"

# Videos uploaded in the 200x decade
youtube-dl --dateafter 20000101 --datebefore 20091231 "playlist_url"
```
## FAQ (Frequently Asked Questions)

*   **How do I update youtube-dl?**
    Run `youtube-dl -U` (or `sudo youtube-dl -U` on Linux).  If using pip, `sudo pip install -U youtube-dl`. If using a package manager, use the system update tools.
*   **Slow start on Windows?**
    Add an exclusion for `youtube-dl.exe` in Windows Defender settings.
*   **Unable to extract OpenGraph title?**
    Update youtube-dl.  Ensure you have at least version 2014.07.25 for YouTube playlist errors.
*   **Conflicting output template and other options?**
    Don't use `-o` with `-t`, `--title`, `--id`, `-A` or `--auto-number`.
*   **Do I always need `-citw`?**
    No, the default settings are designed to be the best options in most cases.
*   **Why was the `-b` option removed?**
    youtube-dl defaults to the best quality, and `-f` is used to select alternatives if the highest quality is not desired.
*   **HTTP error 402?**
    YouTube CAPTCHA is triggered by excessive downloading. Solve the CAPTCHA in a browser, then restart youtube-dl, also consider using a `--proxy` or `--source-address` option.
*   **Do I need other programs?**
    You may need `avconv` or `ffmpeg` for conversion, `rtmpdump` for RTMP, and `mplayer` or `mpv` for MMS and RTSP.
*   **How do I play downloaded videos?**
    Use a video player like `mpv`, `vlc`, or `mplayer`.
*   **Video URL does not play?**
    Use the `--cookies` option, make sure your downloader supports the URL protocol, and consider using IPv6.
*   **Error: no fmt\_url\_map or conn information found in video info/ERROR: unable to download video?**
    Update youtube-dl.
*   **Amperands and URL problems?**
    Wrap the URL in single quotes or escape the ampersands (e.g., `youtube-dl 'url?v=1&q=2'`).
*   **ExtractorError: Could not find JS function u'OF'?**
    Update youtube-dl.
*   **HTTP Error 429: Too Many Requests or 402: Payment Required?**
    The service is blocking your IP address due to overuse, possibly because of CAPTCHA. Open a browser and solve a CAPTCHA, then [pass cookies](#how-do-i-pass-cookies-to-youtube-dl) to youtube-dl.
*   **SyntaxError: Non-ASCII character?**
    Use Python 2.6 or 2.7.
*   **What is this binary file? Where has the code gone?**
    Since June 2012 youtube-dl is packed as an executable zipfile, simply unzip it (might need renaming to `youtube-dl.zip` first on some systems) or clone the git repository.
*   **The exe throws an error due to missing `MSVCR100.dll`?**
    Install the [Microsoft Visual C++ 2010 Service Pack 1 Redistributable Package (x86)](https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x86.exe).
*   **How do I set up ffmpeg and youtube-dl on Windows? Where should I put the exe files?**
    Put youtube-dl and ffmpeg in the same directory or add their directories to your PATH environment variable.
*   **How do I put downloads into a specific folder?**
    Use the `-o` option with an [output template](#output-template).
*   **How do I download a video starting with a `-`?**
    Prepend `https://www.youtube.com/watch?v=` or use `--` to separate options and the ID (e.g., `youtube-dl -- -wNyEUrxzFU`).
*   **How do I pass cookies to youtube-dl?**
    Use the `--cookies` option. The cookies file must be in Mozilla/Netscape format.
*   **How do I stream directly to media player?**
    Use `-o -` to output to stdout, and pipe that to your media player (e.g., `youtube-dl -o - "url" | vlc -`).
*   **How do I download only new videos from a playlist?**
    Use the download-archive feature with `--download-archive /path/to/download/archive/file.txt`.
*   **Should I add `--hls-prefer-native` into my config?**
    File an issue or a pull request if a different HLS downloader should be preferred for your use case.
*   **Can you add support for this anime video site, or site which shows current movies for free?**
    youtube-dl does not include support for services that specialize in infringing copyright.
*   **How can I speed up work on my issue?**
    Provide full output from `youtube-dl -v YOUR_URL_HERE`.
*   **How can I detect whether a given URL is supported by youtube-dl?**
    Call youtube-dl with the URL; if it doesn't work, it's either unsupported or invalid.

## Developer Instructions

See the [developer instructions](#developer-instructions) if you want to contribute to this project.

## Copyright

youtube-dl is released into the public domain.