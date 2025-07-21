[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# YouTube-DL: Your Go-To Video Downloader

**Download videos from YouTube and numerous other sites with ease using the versatile and open-source YouTube-DL!** ([See the original repo](https://github.com/ytdl-org/youtube-dl))

*   **Cross-Platform Compatibility:** Works seamlessly on Linux, macOS, and Windows.
*   **Broad Site Support:** Downloads videos from YouTube and many other video platforms.
*   **Format Flexibility:** Choose from a variety of video and audio formats.
*   **Playlist Download:** Easily download entire playlists.
*   **Customizable Output:** Control file names and organization.
*   **Subtitle Support:** Download subtitles in multiple formats.
*   **Authentication:** Supports login for sites requiring credentials.
*   **Post-Processing:** Convert videos to audio and other formats.

## Key Features

*   **Versatile Downloading:** Download videos and audio from a vast number of supported websites, with frequent updates to add new sites and features.
*   **Format and Quality Selection:** Choose the best available quality or specify your preferred format (e.g., MP4, WebM, MP3) and resolution.
*   **Playlist and Channel Downloads:** Download entire playlists or all videos from a channel with ease.
*   **Subtitle Support:** Download subtitles in various languages and formats.
*   **Customizable Output:** Configure file names and organization with flexible output templates.
*   **Download Resumption and File Management:** Resume interrupted downloads and manage downloaded files efficiently.
*   **Command-Line Interface:** Powerful command-line interface for automation and scripting.
*   **No Configuration Required by Default:** youtube-dl by default will automatically select the best available format for you.

## Installation

**For Linux/macOS:**

1.  **Using `curl` (Recommended):**

    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```

2.  **Using `wget` (Alternative):**

    ```bash
    sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```

3.  **Homebrew (macOS):**

    ```bash
    brew install youtube-dl
    ```

4.  **MacPorts (macOS):**

    ```bash
    sudo port install youtube-dl
    ```

**For Windows:**

1.  **Download the `.exe`:** Get the latest `youtube-dl.exe` file from [https://yt-dl.org/latest/youtube-dl.exe](https://yt-dl.org/latest/youtube-dl.exe).
2.  **Place in PATH:** Place the `.exe` file in a directory included in your system's [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) (e.g., `C:\Users\<YourUsername>\bin`). **Do not** place it in `C:\Windows\System32`.
3.  **Alternative: Using `pip`:**

    ```bash
    pip install --upgrade youtube-dl
    ```

For more installation methods, including advanced options, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## Usage

To download a video, simply use the following command:

```bash
youtube-dl [OPTIONS] URL [URL...]
```

Replace `URL` with the video's web address.

## Quick Start: Examples

*   **Download a single video:**

    ```bash
    youtube-dl https://www.youtube.com/watch?v=dQw4w9WgXcQ
    ```

*   **Download a playlist:**

    ```bash
    youtube-dl https://www.youtube.com/playlist?list=PLx1...
    ```

*   **Specify output filename:**

    ```bash
    youtube-dl -o "MyVideo.%(ext)s" https://www.youtube.com/watch?v=dQw4w9WgXcQ
    ```

*   **List available formats:**

    ```bash
    youtube-dl -F https://www.youtube.com/watch?v=dQw4w9WgXcQ
    ```

*   **Download a specific format:**

    ```bash
    youtube-dl -f 22 https://www.youtube.com/watch?v=dQw4w9WgXcQ
    ```

## Available Options

Refer to the [OPTIONS](#options) section in the original README (linked at the top) for the full list of available options.  Here's a summarized view to get you started:

### Core Functionality
*   `-U, --update`: Update youtube-dl to the latest version.
*   `-h, --help`: Displays the help message.
*   `-v, --verbose`: Print detailed debugging information.
*   `-i, --ignore-errors`: Skip download errors and continue with other videos.

### Network
*   `--proxy URL`: Use a specified proxy server.
*   `--socket-timeout SECONDS`: Set a socket timeout in seconds.

### Video Selection
*   `--playlist-start NUMBER`: Start playlist download from video number.
*   `--playlist-end NUMBER`: End playlist download at video number.
*   `--match-title REGEX`: Download videos matching the title regex.
*   `--reject-title REGEX`: Skip videos matching the title regex.
*   `--max-downloads NUMBER`: Limit the number of videos to download.

### Download Options
*   `-r, --limit-rate RATE`: Limit download rate (e.g., `50K` or `4.2M`).
*   `-R, --retries RETRIES`: Number of retries for download errors.
*   `-c, --continue`: Resume partially downloaded files.

### Filesystem Options
*   `-a, --batch-file FILE`: Download videos from a list in a file.
*   `-o, --output TEMPLATE`: Set the output filename template.
*   `-w, --no-overwrites`: Do not overwrite existing files.

### Video Format Options
*   `-f, --format FORMAT`: Select video format (see [FORMAT SELECTION](#format-selection)).
*   `-F, --list-formats`: List all available formats for a video.
*   `--all-formats`: Download all available formats.

### Subtitle Options
*   `--write-sub`: Download subtitles.
*   `--write-auto-sub`: Download automatically generated subtitles.
*   `--sub-lang LANGS`: Specify subtitle languages (e.g., `en,fr`).

### Authentication Options
*   `-u, --username USERNAME`: Login with a username.
*   `-p, --password PASSWORD`: Provide a password.

### Post-processing Options
*   `-x, --extract-audio`: Extract audio from video.
*   `--audio-format FORMAT`: Specify audio format (e.g., `mp3`, `wav`).

## Output Template
The `-o` option lets you set the output filename using a template.  You can use placeholders like `%(title)s`, `%(id)s`, `%(ext)s`, and many more.  See the original [OUTPUT TEMPLATE](#output-template) section for details.

## Format Selection
Use the `-f` or `--format` options for specific video formats.  You can use format codes (found with `-F`), file extensions (e.g., `-f mp4`), or special names like `best` or `bestvideo`. See the original [FORMAT SELECTION](#format-selection) section for more information, including how to merge video and audio formats.

## Troubleshooting and FAQ

*   **Updating:** Use `youtube-dl -U` to update.  If using a package manager, use your system's update tools.
*   **Slow Start on Windows:**  Add a file exclusion for `youtube-dl.exe` in your Windows Defender settings.
*   **"Unable to extract OpenGraph title" on YouTube playlists:** Update youtube-dl to the latest version.
*   **Common errors, HTTP 429, and other issues:** See the original [FAQ](#faq) section of the README for fixes and explanations.

## Bugs and Support

Report bugs and suggestions in the issue tracker:  [https://github.com/ytdl-org/youtube-dl/issues](https://github.com/ytdl-org/youtube-dl/issues).  Include the full output of youtube-dl run with the `-v` option. Read [BUGS](#bugs) section in the original README.

## Developer Instructions

See the original [DEVELOPER INSTRUCTIONS](#developer-instructions) section for information on contributing to the project.

## Copyright
youtube-dl is released into the public domain by the copyright holders.