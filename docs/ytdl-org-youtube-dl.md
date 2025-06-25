```markdown
[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Download Videos from YouTube and Beyond

Tired of buffering?  **Download your favorite videos from YouTube and hundreds of other sites with the powerful and versatile youtube-dl!**  ([Original Repo](https://github.com/ytdl-org/youtube-dl))

## Key Features:

*   **Wide Site Support:** Download from YouTube, Vimeo, Facebook, and **hundreds** of other video platforms ([Supported Sites](https://ytdl-org.github.io/youtube-dl/supportedsites.html)).
*   **Format Flexibility:** Choose from various video and audio formats, including best quality or specific resolutions.
*   **Playlist & Channel Downloads:** Easily download entire playlists and channels.
*   **Customization:** Extensive options for file naming, metadata, and more.
*   **Cross-Platform:** Works on Windows, macOS, Linux, and Unix-like systems.
*   **Active Community:** Benefit from ongoing updates and community support.

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
    *   [Output Template Examples](#output-template-examples)
*   [Format Selection](#format-selection)
    *   [Format Selection Examples](#format-selection-examples)
*   [Video Selection](#video-selection)
*   [FAQ](#faq)
*   [Developer Instructions](#developer-instructions)
    *   [Adding Support for a New Site](#adding-support-for-a-new-site)
    *   [youtube-dl coding conventions](#youtube-dl-coding-conventions)
*   [Embedding youtube-dl](#embedding-youtube-dl)
*   [Bugs](#bugs)
    *   [Opening a bug report or suggestion](#opening-a-bug-report-or-suggestion)
*   [Copyright](#copyright)

## INSTALLATION

### UNIX (Linux, macOS, etc.)

1.  **Install directly:**

    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```

2.  **Alternative using wget:**

    ```bash
    sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```
    
3.  **Update the program:**
    ```bash
    sudo youtube-dl -U
    ```

### Windows

1.  **Download:** [Download .exe](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) (not `%SYSTEMROOT%\System32`).
    
2.  **Update the program:**  Run `youtube-dl -U` in the command line.
    
### Other Installation Methods

*   **pip:** `sudo -H pip install --upgrade youtube-dl` ([pypi page](https://pypi.python.org/pypi/youtube_dl))
*   **macOS (Homebrew):** `brew install youtube-dl`
*   **macOS (MacPorts):** `sudo port install youtube-dl`
*   **Developer Installation:** Refer to the [developer instructions](#developer-instructions).
*   **Download Page:** For advanced options and PGP signatures, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## DESCRIPTION

**youtube-dl** is a versatile command-line tool to download videos from various video platforms.  It's platform-independent, running on Unix, Windows, and macOS, and requires Python 2.6, 2.7, or 3.2+.

Basic Usage:

```bash
youtube-dl [OPTIONS] URL [URL...]
```

## OPTIONS

Use `youtube-dl -h` for the most up-to-date option descriptions.  Here's a summary:

### Network Options

*   `--proxy URL`: Use a proxy for connections.
*   `--socket-timeout SECONDS`: Set the socket timeout.
*   `--source-address IP`: Bind to a specific client-side IP.
*   `-4, --force-ipv4`: Force IPv4 connections.
*   `-6, --force-ipv6`: Force IPv6 connections.

### Geo Restriction

*   `--geo-verification-proxy URL`: Use a proxy for geo-restricted sites.
*   `--geo-bypass`: Bypass geo-restrictions.
*   `--geo-bypass-country CODE`: Force a specific country code bypass.
*   `--geo-bypass-ip-block IP_BLOCK`: Force a specific IP block bypass.

### Video Selection

*   `--playlist-start NUMBER`: Start at a specific playlist video.
*   `--playlist-end NUMBER`: End at a specific playlist video.
*   `--playlist-items ITEM_SPEC`: Download specific playlist items (e.g., `1,2,5-8`).
*   `--match-title REGEX`: Download videos with matching titles.
*   `--reject-title REGEX`: Skip videos with matching titles.
*   `--max-downloads NUMBER`: Limit the number of downloads.
*   `--min-filesize SIZE`: Minimum video file size.
*   `--max-filesize SIZE`: Maximum video file size.
*   `--date DATE`: Download videos uploaded on a specific date.
*   `--datebefore DATE`: Download videos uploaded on or before a date.
*   `--dateafter DATE`: Download videos uploaded on or after a date.
*   `--min-views COUNT`: Minimum video views.
*   `--max-views COUNT`: Maximum video views.
*   `--match-filter FILTER`: Generic video filtering (see "OUTPUT TEMPLATE").
*   `--no-playlist`: Download only the video if the URL is for a video and a playlist.
*   `--yes-playlist`: Download the playlist if the URL refers to both a video and a playlist.
*   `--age-limit YEARS`: Download videos suitable for a certain age.
*   `--download-archive FILE`: Download only videos not in an archive file.
*   `--include-ads`: Download advertisements (experimental).

### Download Options

*   `-r, --limit-rate RATE`: Limit download rate (e.g., `50K` or `4.2M`).
*   `-R, --retries RETRIES`: Number of retries (default 10).
*   `--fragment-retries RETRIES`: Number of retries for fragments (DASH, HLS, ISM).
*   `--skip-unavailable-fragments`: Skip unavailable fragments.
*   `--abort-on-unavailable-fragment`: Abort if a fragment is unavailable.
*   `--keep-fragments`: Keep downloaded fragments on disk.
*   `--buffer-size SIZE`: Set the download buffer size.
*   `--no-resize-buffer`: Disable automatic buffer resizing.
*   `--http-chunk-size SIZE`: Chunk size for HTTP downloads (experimental).
*   `--playlist-reverse`: Download playlist videos in reverse order.
*   `--playlist-random`: Download playlist videos in random order.
*   `--xattr-set-filesize`: Set file xattribute ytdl.filesize.
*   `--hls-prefer-native`: Use native HLS downloader.
*   `--hls-prefer-ffmpeg`: Use ffmpeg for HLS downloading.
*   `--hls-use-mpegts`: Use MPEGTS container for HLS videos.
*   `--external-downloader COMMAND`: Use an external downloader (e.g., aria2c, wget).
*   `--external-downloader-args ARGS`: Arguments for the external downloader.

### Filesystem Options

*   `-a, --batch-file FILE`: File with URLs, one per line.
*   `--id`: Use only the video ID in the filename.
*   `-o, --output TEMPLATE`: Output filename template (see "OUTPUT TEMPLATE").
*   `--output-na-placeholder PLACEHOLDER`: Placeholder for unavailable metadata.
*   `--autonumber-start NUMBER`: Starting value for autonumber.
*   `--restrict-filenames`: Restrict filenames to ASCII and avoid special characters.
*   `-w, --no-overwrites`: Do not overwrite files.
*   `-c, --continue`: Resume partially downloaded files.
*   `--no-continue`: Do not resume partially downloaded files.
*   `--no-part`: Write directly to the output file (no .part files).
*   `--no-mtime`: Do not use the Last-modified header to set the file modification time.
*   `--write-description`: Write video description to a `.description` file.
*   `--write-info-json`: Write video metadata to a `.info.json` file.
*   `--write-annotations`: Write video annotations to a `.annotations.xml` file.
*   `--load-info-json FILE`: Load video information from a JSON file.
*   `--cookies FILE`: Read cookies from a file.
*   `--cache-dir DIR`: Cache downloaded information (default: `~/.cache/youtube-dl`).
*   `--no-cache-dir`: Disable filesystem caching.
*   `--rm-cache-dir`: Delete the filesystem cache.

### Thumbnail Options

*   `--write-thumbnail`: Write thumbnail image to disk.
*   `--write-all-thumbnails`: Write all thumbnail image formats to disk.
*   `--list-thumbnails`: List available thumbnail formats.

### Verbosity / Simulation Options

*   `-q, --quiet`: Quiet mode.
*   `--no-warnings`: Ignore warnings.
*   `-s, --simulate`: Simulate (do not download).
*   `--skip-download`: Skip the download.
*   `-g, --get-url`: Simulate, print URL.
*   `-e, --get-title`: Simulate, print title.
*   `--get-id`: Simulate, print ID.
*   `--get-thumbnail`: Simulate, print thumbnail URL.
*   `--get-description`: Simulate, print description.
*   `--get-duration`: Simulate, print duration.
*   `--get-filename`: Simulate, print filename.
*   `--get-format`: Simulate, print format.
*   `-j, --dump-json`: Simulate, print JSON information.
*   `-J, --dump-single-json`: Simulate, print single-line JSON for each argument.
*   `--print-json`: Print video information as JSON (while downloading).
*   `--newline`: Output progress bar on new lines.
*   `--no-progress`: Do not print progress bar.
*   `--console-title`: Display progress in the console title bar.
*   `-v, --verbose`: Print debugging information.
*   `--dump-pages`: Print downloaded pages (for debugging).
*   `--write-pages`: Write downloaded intermediary pages to files (for debugging).
*   `--print-traffic`: Display HTTP traffic.
*   `-C, --call-home`: Contact the youtube-dl server for debugging.
*   `--no-call-home`: Do NOT contact the youtube-dl server.

### Workarounds

*   `--encoding ENCODING`: Force encoding.
*   `--no-check-certificate`: Suppress HTTPS certificate validation.
*   `--prefer-insecure`: Use an unencrypted connection.
*   `--user-agent UA`: Specify a custom user agent.
*   `--referer URL`: Specify a custom referer.
*   `--add-header FIELD:VALUE`: Add a custom HTTP header.
*   `--bidi-workaround`: Work around bidirectional text issues.
*   `--sleep-interval SECONDS`: Sleep interval before each download.
*   `--max-sleep-interval SECONDS`: Upper bound for randomized sleep.

### Video Format Options

*   `-f, --format FORMAT`: Video format code (see "FORMAT SELECTION").
*   `--all-formats`: Download all available formats.
*   `--prefer-free-formats`: Prefer free video formats.
*   `-F, --list-formats`: List all available formats.
*   `--youtube-skip-dash-manifest`: Skip DASH manifest download (YouTube).
*   `--merge-output-format FORMAT`: Merge video and audio into a specific format.

### Subtitle Options

*   `--write-sub`: Write subtitle file.
*   `--write-auto-sub`: Write automatically generated subtitle file (YouTube).
*   `--all-subs`: Download all available subtitles.
*   `--list-subs`: List available subtitles.
*   `--sub-format FORMAT`: Subtitle format (e.g., `srt`, `ass/srt/best`).
*   `--sub-lang LANGS`: Languages of subtitles (e.g., `en,fr`).

### Authentication Options

*   `-u, --username USERNAME`: Login with a username.
*   `-p, --password PASSWORD`: Login with a password.
*   `-2, --twofactor TWOFACTOR`: Two-factor authentication code.
*   `-n, --netrc`: Use .netrc authentication data.
*   `--video-password PASSWORD`: Video password (Vimeo, Youku).

### Adobe Pass Options

*   `--ap-mso MSO`: Adobe Pass MSO identifier.
*   `--ap-username USERNAME`: MSO account login.
*   `--ap-password PASSWORD`: MSO account password.
*   `--ap-list-mso`: List supported MSOS.

### Post-processing Options

*   `-x, --extract-audio`: Convert video to audio.
*   `--audio-format FORMAT`: Audio format (e.g., `mp3`, `wav`).
*   `--audio-quality QUALITY`: Audio quality (0-9 for VBR or bitrate).
*   `--recode-video FORMAT`: Encode video to another format.
*   `--postprocessor-args ARGS`: Arguments for the post-processor.
*   `-k, --keep-video`: Keep the video after post-processing.
*   `--no-post-overwrites`: Do not overwrite post-processed files.
*   `--embed-subs`: Embed subtitles in the video.
*   `--embed-thumbnail`: Embed thumbnail in audio as cover art.
*   `--add-metadata`: Write metadata to the video file.
*   `--metadata-from-title FORMAT`: Parse metadata from the video title.
*   `--xattrs`: Write metadata to the video file's xattrs.
*   `--fixup POLICY`: Automatically correct file faults.
*   `--prefer-avconv`: Prefer avconv for post-processing.
*   `--prefer-ffmpeg`: Prefer ffmpeg for post-processing (default).
*   `--ffmpeg-location PATH`: Location of ffmpeg/avconv binary.
*   `--exec CMD`: Execute a command after download and post-processing.
*   `--convert-subs FORMAT`: Convert subtitles to another format.

## CONFIGURATION

Configure youtube-dl by creating a configuration file.

*   **System-wide:** `/etc/youtube-dl.conf` (Linux/macOS)
*   **User-specific:** `~/.config/youtube-dl/config` (Linux/macOS), `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf` (Windows)

Example:

```
# Comments start with '#'

# Always extract audio
-x

# Do not copy the mtime
--no-mtime

# Use this proxy
--proxy 127.0.0.1:3128

# Save all videos under Movies directory in your home directory
-o ~/Movies/%(title)s.%(ext)s
```

Use `--ignore-config` to disable the configuration file and `--config-location` to specify a custom config file.

### Authentication with .netrc file

Configure automatic credentials storage for extractors using a [.netrc file](https://stackoverflow.com/tags/.netrc/info).

1.  Create `.netrc` file and restrict permissions:

    ```bash
    touch $HOME/.netrc
    chmod a-rwx,u+rw $HOME/.netrc
    ```

2.  Add credentials (replace `<extractor>`, `<login>`, `<password>`):

    ```
    machine <extractor> login <login> password <password>
    ```

    Example:

    ```
    machine youtube login myaccount@gmail.com password my_youtube_password
    machine twitch login my_twitch_account_name password my_twitch_password
    ```

3.  Activate by passing `--netrc` to youtube-dl or by adding it to the configuration file.
    On Windows, setup `%HOME%` environment variable manually.

## OUTPUT TEMPLATE

Customize filenames with the `-o` option.

**Key:**

*   `id`: Video identifier
*   `title`: Video title
*   `url`: Video URL
*   `ext`: Video filename extension
*   `alt_title`: A secondary title of the video
*   `display_id`: An alternative identifier for the video
*   `uploader`: Full name of the video uploader
*   `license`: License name the video is licensed under
*   `creator`: The creator of the video
*   `release_date`: The date (YYYYMMDD) when the video was released
*   `timestamp`: UNIX timestamp of the moment the video became available
*   `upload_date`: Video upload date (YYYYMMDD)
*   `uploader_id`: Nickname or id of the video uploader
*   `channel`: Full name of the channel the video is uploaded on
*   `channel_id`: Id of the channel
*   `location`: Physical location where the video was filmed
*   `duration`: Length of the video in seconds
*   `view_count`: How many users have watched the video on the platform
*   `like_count`: Number of positive ratings of the video
*   `dislike_count`: Number of negative ratings of the video
*   `repost_count`: Number of reposts of the video
*   `average_rating`: Average rating give by users, the scale used depends on the webpage
*   `comment_count`: Number of comments on the video
*   `age_limit`: Age restriction for the video (years)
*   `is_live`: Whether this video is a live stream or a fixed-length video
*   `start_time`: Time in seconds where the reproduction should start, as specified in the URL
*   `end_time`: Time in seconds where the reproduction should end, as specified in the URL
*   `format`: A human-readable description of the format
*   `format_id`: Format code specified by `--format`
*   `format_note`: Additional info about the format
*   `width`: Width of the video
*   `height`: Height of the video
*   `resolution`: Textual description of width and height
*   `tbr`: Average bitrate of audio and video in KBit/s
*   `abr`: Average audio bitrate in KBit/s
*   `acodec`: Name of the audio codec in use
*   `asr`: Audio sampling rate in Hertz
*   `vbr`: Average video bitrate in KBit/s
*   `fps`: Frame rate
*   `vcodec`: Name of the video codec in use
*   `container`: Name of the container format
*   `filesize`: The number of bytes, if known in advance
*   `filesize_approx`: An estimate for the number of bytes
*   `protocol`: The protocol that will be used for the actual download
*   `extractor`: Name of the extractor
*   `extractor_key`: Key name of the extractor
*   `epoch`: Unix epoch when creating the file
*   `autonumber`: Number that will be increased with each download, starting at `--autonumber-start`
*   `playlist`: Name or id of the playlist that contains the video
*   `playlist_index`: Index of the video in the playlist padded with leading zeros according to the total length of the playlist
*   `playlist_id`: Playlist identifier
*   `playlist_title`: Playlist title
*   `playlist_uploader`: Full name of the playlist uploader
*   `playlist_uploader_id`: Nickname or id of the playlist uploader

Available for the video that belongs to some logical chapter or section:

*   `chapter`: Name or title of the chapter the video belongs to
*   `chapter_number`: Number of the chapter the video belongs to
*   `chapter_id`: Id of the chapter the video belongs to

Available for the video that is an episode of some series or programme:

*   `series`: Title of the series or programme the video episode belongs to
*   `season`: Title of the season the video episode belongs to
*   `season_number`: Number of the season the video episode belongs to
*   `season_id`: Id of the season the video episode belongs to
*   `episode`: Title of the video episode
*   `episode_number`: Number of the video episode within a season
*   `episode_id`: Id of the video episode

Available for the media that is a track or a part of a music album:

*   `track`: Title of the track
*   `track_number`: Number of the track within an album or a disc
*   `track_id`: Id of the track
*   `artist`: Artist(s) of the track
*   `genre`: Genre(s) of the track
*   `album`: Title of the album the track belongs to
*   `album_type`: Type of the album
*   `album_artist`: List of all artists appeared on the album
*   `disc_number`: Number of the disc or other physical medium the track belongs to
*   `release_year`: Year (YYYY) when the album was released

Use `%%` for literal percent signs.  Use `-o -` to output to stdout.

### Output Template Examples

```bash
# Simple
youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc

# Restrict filenames
youtube-dl --get-filename -o '%(title)s.%(ext)s' BaW_jenozKc --restrict-filenames

# Playlist videos in separate directories
youtube-dl -o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' https://www.youtube.com/playlist?list=PLwiyx1dc3P2JR9N8gQaQN_BCvlSlap7re

# YouTube playlists of channel/user
youtube-dl -o '%(uploader)s/%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' https://www.youtube.com/user/TheLinuxFoundation/playlists

# Download Udemy course
youtube-dl -u user -p password -o '~/MyVideos/%(playlist)s/%(chapter_number)s - %(chapter)s/%(title)s.%(ext)s' https://www.udemy.com/java-tutorial/

# Download entire series season
youtube-dl -o "C:/MyVideos/%(series)s/%(season_number)s - %(season)s/%(episode_number)s - %(episode)s.%(ext)s" https://videomore.ru/kino_v_detalayah/5_sezon/367617

# Stream the video being downloaded to stdout
youtube-dl -o - BaW_jenozKc
```

## FORMAT SELECTION

Choose video formats with the `-f` or `--format` option.

**Key:**

*   `best`: Best quality (video + audio)
*   `worst`: Worst quality (video + audio)
*   `bestvideo`: Best video only
*   `worstvideo`: Worst video only
*   `bestaudio`: Best audio only
*   `worstaudio`: Worst audio only

Use format codes obtained with `-F` or `--list-formats`.

Use `/` for preference order, e.g., `-f 22/17/18`.

Use `,` to download multiple formats, e.g., `-f 22,17,18`.

Use brackets `[]` for filtering (numeric and string fields).

Merge video and audio with `+`, e.g., `-f bestvideo+bestaudio`. Requires ffmpeg/avconv.

### Format Selection Examples

```bash
# Best mp4 or best other if not available
$ youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'

# Best format but not higher than 480p
$ youtube-dl -f 'bestvideo[height<=480]+bestaudio/best[height<=480]'

# Best video, no bigger than 50 MB
$ youtube-dl -f 'best[filesize<50M]'

# Best via HTTP/HTTPS
$ youtube-dl -f '(bestvideo+bestaudio/best)[protocol^=http]'

# Best video + audio, but don't merge
$ youtube-dl -f 'bestvideo,bestaudio' -o '%(title)s.f%(format_id)s.%(ext)s'
```

## VIDEO SELECTION

Filter videos by upload date using `--date`, `--datebefore`, or `--dateafter`. Dates are in the format `YYYYMMDD` or relative (e.g., `now-6months`).

```bash
# Last 6 months
$ youtube-dl --dateafter now-6months

# January 1, 1970
$ youtube-dl --date 19700101

# 200x decade
$ youtube-dl --dateafter 20000101 --datebefore 20091231
```

## FAQ

Answers to common questions can be found in the original repository's README.

## DEVELOPER INSTRUCTIONS

Instructions for developers are also included in the original repository's README.

### Adding Support for a New Site

Follow the [adding support for a new site tutorial](#adding-support-for-a-new-site) in the original README.

### youtube-dl coding conventions

Follow the [youtube-dl coding conventions](#youtube-dl-coding-conventions) in the original README.

## EMBEDDING YOUTUBE-DL

Embed youtube-dl in Python programs for advanced functionality.  See [embedding youtube-dl](#embedding-youtube-dl) in the original README for details.

## BUGS

Report bugs and suggestions in the [issue tracker](https://github.com/ytdl-org/youtube-dl/issues).

### Opening a bug report or suggestion

*   Provide the full output of `youtube-dl -v YOUR_URL_HERE`.
*   Be detailed about the problem and how to replicate it.
*   Search for existing issues.
*   Use the latest version (`youtube-dl -U`).
*   Explain why existing options are insufficient.
*   Provide context and use cases.
*   Limit each issue to a single problem.
*   Make sure it's a question about youtube-dl and not a related application.

## COPYRIGHT

Released into the public domain by the copyright holders.  See the [original repository](https://github.com/ytdl-org/youtube-dl) for details.
```
