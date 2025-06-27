[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Your Go-To Command-Line Tool for Downloading Videos

**Download videos from YouTube and many other sites with ease!** youtube-dl is a versatile command-line program that allows you to download videos from YouTube.com and a wide array of other video platforms. This open-source tool, released into the public domain, is platform-independent, working seamlessly on Unix-like systems, Windows, and macOS.

## Key Features

*   **Wide Site Support:** Download from YouTube and numerous other video platforms.
*   **Format Selection:** Choose your preferred video format and quality.
*   **Playlist and Channel Downloads:** Download entire playlists or all videos from a channel.
*   **Customization:** Configure output filenames, download speed, and more.
*   **Subtitle Support:** Download and manage subtitles in various formats.
*   **Post-processing Options:** Convert videos to audio, embed metadata, and more.
*   **Cross-Platform Compatibility:** Works on Linux, macOS, and Windows.

## Core Functionality

*   **Installation:**  Quickly set up youtube-dl on your system:
    *   [Installation instructions](https://github.com/ytdl-org/youtube-dl#installation)
    *   Download executable or install via pip, Homebrew, or MacPorts.
*   **Usage:**  Download videos using the command line:
    ```bash
    youtube-dl [OPTIONS] URL [URL...]
    ```
    *   Detailed [options](https://github.com/ytdl-org/youtube-dl#options) available.
*   **Configuration:** Customize your youtube-dl experience:
    *   [Configuration](https://github.com/ytdl-org/youtube-dl#configuration) with config files.
    *   Set default options to streamline your workflow.

## Main Sections

*   [INSTALLATION](#installation)
*   [DESCRIPTION](#description)
*   [OPTIONS](#options)
*   [CONFIGURATION](#configuration)
*   [OUTPUT TEMPLATE](#output-template)
*   [FORMAT SELECTION](#format-selection)
*   [VIDEO SELECTION](#video-selection)
*   [FAQ](#faq)
*   [DEVELOPER INSTRUCTIONS](#developer-instructions)
*   [EMBEDDING YOUTUBE-DL](#embedding-youtube-dl)
*   [BUGS](#bugs)
*   [COPYRIGHT](#copyright)

## Output Template Examples
*   ```bash
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

## Format Selection Examples

*   ```bash
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
*   Note that in the last example, an output template is recommended as bestvideo and bestaudio may have the same file name.

## Advanced Usage
*   Detailed documentation on [options](https://github.com/ytdl-org/youtube-dl#options) for advanced use cases, including:
    *   Network and Geo Restriction options
    *   Video Selection options
    *   Download options
    *   Filesystem Options
    *   Thumbnail options
    *   Verbosity options
    *   Workarounds
    *   Video Format options
    *   Subtitle options
    *   Authentication options
    *   Adobe Pass options
    *   Post-processing options

## Bugs

Report any issues in the [Issue Tracker](https://github.com/ytdl-org/youtube-dl/issues).  Be sure to include the full output of youtube-dl with the `-v` flag for effective debugging.

## Development

Contributions are welcome! See the [Developer Instructions](https://github.com/ytdl-org/youtube-dl#developer-instructions) for details on contributing to the project.

## License and Copyright

youtube-dl is released into the public domain.

*Original repository: [https://github.com/ytdl-org/youtube-dl](https://github.com/ytdl-org/youtube-dl)*