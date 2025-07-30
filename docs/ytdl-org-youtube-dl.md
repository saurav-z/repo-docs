[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: The Ultimate Video Downloader

**Easily download videos from YouTube and thousands of other sites with youtube-dl, the powerful command-line tool that puts you in control.**  [Visit the original repository on GitHub](https://github.com/ytdl-org/youtube-dl).

*   **Key Features:**
    *   **Wide Site Support:** Download from YouTube, plus thousands of other video platforms.
    *   **Format Flexibility:** Select your preferred video and audio formats, including best quality or specific resolutions.
    *   **Playlist & Channel Support:** Download entire playlists or all videos from a channel.
    *   **Customization:**  Extensive options for file naming, output templates, and post-processing.
    *   **Subtitle Management:** Download and manage subtitles in various formats.
    *   **Authentication:** Supports login for premium content and restricted sites.
    *   **Cross-Platform:** Works seamlessly on Linux, macOS, and Windows.

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

## Installation

### Unix (Linux, macOS, etc.)

For a quick install:

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

Download the `.exe` from [here](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory on your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29), *except* for `%SYSTEMROOT%\System32`.

### Other Options

*   **pip:**  `sudo -H pip install --upgrade youtube-dl`
*   **macOS (Homebrew):** `brew install youtube-dl`
*   **macOS (MacPorts):** `sudo port install youtube-dl`
*   **Developer Installation:** Refer to [Developer Instructions](#developer-instructions) for git repository setup.
*   **Advanced Download:** Visit the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html) for further options, including PGP signatures.

## Description

**youtube-dl** is a versatile command-line tool designed to download videos from YouTube.com and a multitude of other video hosting sites. It leverages the Python interpreter (version 2.6, 2.7, or 3.2+) for its functionality and is platform-agnostic, ensuring compatibility across various operating systems, including Unix, Windows, and macOS.  youtube-dl is released into the public domain, granting users the freedom to modify, redistribute, and utilize it according to their preferences.

Usage:

```bash
youtube-dl [OPTIONS] URL [URL...]
```

## Options

Comprehensive options are available, categorized below for better understanding:

*   **General Options:** Basic program control (help, version, update, etc.).
*   **Network Options:** Proxy configuration, connection timeouts, and IP settings.
*   **Geo Restriction:**  Bypass geographic restrictions using proxies or IP manipulation.
*   **Video Selection:**  Filter downloads by playlist position, title, file size, upload date, view count, and more.
*   **Download Options:**  Control download speed, retries, buffer size, and more.
*   **Filesystem Options:**  Specify output file names, locations, and handle overwrites.
*   **Thumbnail Options:**  Write thumbnail images to disk.
*   **Verbosity / Simulation Options:**  Control output, display debugging information, and simulate downloads.
*   **Workarounds:**  Address issues like encoding, certificate verification, and user agent.
*   **Video Format Options:**  Choose your preferred video format and quality.
*   **Subtitle Options:**  Download and manage subtitles.
*   **Authentication Options:**  Log in using usernames and passwords, or with `.netrc` files.
*   **Adobe Pass Options:**  Support for Adobe Pass authentication.
*   **Post-processing Options:**  Convert videos to audio, add metadata, and more.

[See the original README for a complete list of options.](https://github.com/ytdl-org/youtube-dl#options)

## Configuration

Customize youtube-dl's behavior using a configuration file.

*   **System-wide (Linux/macOS):** `/etc/youtube-dl.conf`
*   **User-specific (Linux/macOS):** `~/.config/youtube-dl/config`
*   **User-specific (Windows):** `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`

Example configuration:

```
# Always extract audio
-x

# Do not copy the mtime
--no-mtime

# Use this proxy
--proxy 127.0.0.1:3128

# Save all videos under Movies directory in your home directory
-o ~/Movies/%(title)s.%(ext)s
```

Use `--ignore-config` to disable configuration for a specific run.  Use `--config-location` to specify a custom configuration file.

### Authentication with `.netrc` file

You can configure automatic credentials storage for extractors that support authentication in order not to pass credentials as command line arguments on every youtube-dl execution.  For that you will need to create a `.netrc` file in your `$HOME` and restrict permissions to read/write by only you:
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

## Output Template

The `-o` option controls the output filename using a template system.  Template arguments can be formatted using Python string formatting.

**Key Templates:**
*   `%(id)s`: Video identifier
*   `%(title)s`: Video title
*   `%(ext)s`: Video file extension
*   `%(playlist)s`: Playlist name
*   `%(playlist_index)s`: Index in the playlist (with leading zeros)
*   **(and many more)** - see original README for a full list.

Example: `-o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s'` (downloading playlists into directories)

### Output template and Windows batch files

If you are using an output template inside a Windows batch file then you must escape plain percent characters (`%`) by doubling, so that `-o "%(title)s-%(id)s.%(ext)s"` should become `-o "%%(title)s-%%(id)s.%%(ext)s"`. However you should not touch `%`'s that are not plain characters, e.g. environment variables for expansion should stay intact: `-o "C:\%HOMEPATH%\Desktop\%%(title)s.%%(ext)s"`.

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

## Format Selection

Use `--format FORMAT` or `-f FORMAT` to choose the video format.  Get a list of available formats with `--list-formats` or `-F`.  `FORMAT` is a *selector expression*.

*   **Specific Format:**  `-f 22` (where 22 is the format code)
*   **File Extension:**  `-f webm` (best quality WebM format)
*   **Special Names:** `best`, `worst`, `bestvideo`, `bestaudio`, `worstvideo`, `worstaudio`
*   **Precedence:**  `-f 22/17/18` (try 22, then 17, then 18)
*   **Multiple Formats:**  `-f 22,17,18` (download all three)
*   **Filtering:** `-f "best[height=720]"` (720p videos) or `-f "[filesize>10M]"`
*   **Merge:** `-f bestvideo+bestaudio` (requires ffmpeg/avconv)

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
Note that in the last example, an output template is recommended as bestvideo and bestaudio may have the same file name.

## Video Selection

Use these options to filter videos:

*   `--date`:  Download videos uploaded on a specific date (YYYYMMDD).
*   `--datebefore`: Download videos uploaded on or before a date.
*   `--dateafter`:  Download videos uploaded on or after a date.
    *   Relative dates: `now-6months` (last 6 months), `today` (current date).

```bash
# Download only the videos uploaded in the last 6 months
$ youtube-dl --dateafter now-6months

# Download only the videos uploaded on January 1, 1970
$ youtube-dl --date 19700101

$ # Download only the videos uploaded in the 200x decade
$ youtube-dl --dateafter 20000101 --datebefore 20091231
```

## FAQ

Common questions and answers are provided for users to troubleshoot their experience.

### How do I update youtube-dl?

Run `youtube-dl -U` (or, on Linux, `sudo youtube-dl -U`).

### youtube-dl is extremely slow to start on Windows

Add a file exclusion for `youtube-dl.exe` in Windows Defender settings.

### I'm getting an error `Unable to extract OpenGraph title` on YouTube playlists

Ensure your version of youtube-dl is up to date.

### Do I always have to pass `-citw`?

No.  youtube-dl defaults to good settings; `-i` (ignore errors) is often useful.

### Can you please put the `-b` option back?

No longer needed.  youtube-dl downloads the highest quality video by default.

### I get HTTP error 402 when trying to download a video. What's this?

Solve the CAPTCHA on YouTube, and restart youtube-dl.

### Do I need any other programs?

Yes.  avconv/ffmpeg for video/audio conversion. rtmpdump for RTMP streams.  mplayer/mpv for MMS/RTSP videos.

### I have downloaded a video but how can I play it?

Use a video player (mpv, vlc, mplayer).

### I extracted a video URL with `-g`, but it does not play on another machine / in my web browser.

You may need to copy cookies and HTTP headers.

### ERROR: no fmt_url_map or conn information found in video info

Update youtube-dl.

### ERROR: unable to download video

Update youtube-dl.

### Video URL contains an ampersand and I'm getting some strange output `[1] 2839` or `'v' is not recognized as an internal or external command`

Put the URL in quotes or escape the ampersands.

### ExtractorError: Could not find JS function u'OF'

Update youtube-dl.

### HTTP Error 429: Too Many Requests or 402: Payment Required

Service is blocking your IP; solve the CAPTCHA, use cookies, and/or use a proxy.

### SyntaxError: Non-ASCII character

Use Python 2.6 or 2.7.

### What is this binary file? Where has the code gone?

youtube-dl is now an executable zipfile.

### The exe throws an error due to missing `MSVCR100.dll`

Install the [Microsoft Visual C++ 2010 Service Pack 1 Redistributable Package (x86)](https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x86.exe).

### On Windows, how should I set up ffmpeg and youtube-dl? Where should I put the exe files?

Add the directory containing the executables to your PATH environment variable.

### How do I put downloads into a specific folder?

Use the `-o` option with an output template.

### How do I download a video starting with a `-`?

Prepend `https://www.youtube.com/watch?v=` or use `--`.

### How do I pass cookies to youtube-dl?

Use the `--cookies` option.

### How do I stream directly to media player?

Use `-o -` and pipe the output to the player.

### How do I download only new videos from a playlist?

Use the download-archive feature.

### Should I add `--hls-prefer-native` into my config?

No.

### Can you add support for this anime video site, or site which shows current movies for free?

youtube-dl does not support infringing sites.

### How can I speed up work on my issue?

Provide the full output of `youtube-dl -v YOUR_URL_HERE`.

### How can I detect whether a given URL is supported by youtube-dl?

Call youtube-dl with it;  examine the output.

## Developer Instructions

For developers, youtube-dl can be executed directly with `python -m youtube_dl` (no build is needed).

To run the tests, use `python -m unittest discover` or `python test/test_download.py` or `nosetests`.

### Adding support for a new site

Follow the detailed instructions in the original README for adding support for a new site.

## Embedding youtube-dl

youtube-dl can be embedded in Python programs:

```python
from __future__ import unicode_literals
import youtube_dl

ydl_opts = {}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=BaW_jenozKc'])
```

See [`youtube_dl/YoutubeDL.py`](https://github.com/ytdl-org/youtube-dl/blob/3e4cedf9e8cd3157df2457df7274d0c842421945/youtube_dl/YoutubeDL.py#L137-L312) for available options and how to customize output.

## Bugs

Report bugs and suggestions in the issue tracker: <https://github.com/ytdl-org/youtube-dl/issues>

Include the full output of `youtube-dl -v YOUR_URL_HERE` in your bug reports.  Detailed instructions are in the original README.

## COPYRIGHT

youtube-dl is released into the public domain.

This README file was originally written by [Daniel Bolton](https://github.com/dbbolton) and is likewise released into the public domain.