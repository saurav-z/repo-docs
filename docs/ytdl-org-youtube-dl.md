[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Download Videos from YouTube and Beyond

**Effortlessly download videos from YouTube and hundreds of other sites with youtube-dl, a powerful command-line tool.** Visit the [original repository](https://github.com/ytdl-org/youtube-dl) for the most up-to-date information.

## Key Features:

*   **Wide Site Support:** Download videos from YouTube, Vimeo, Facebook, and hundreds of other video and audio sharing sites.
*   **Format Selection:** Choose your preferred video and audio quality, resolution, and format.
*   **Playlist & Channel Downloads:** Download entire playlists or channels with ease.
*   **Metadata Extraction:** Automatically extract and store video titles, descriptions, and other metadata.
*   **Customizable Output:** Customize filenames and output directories using flexible templates.
*   **Subtitle Support:** Download and embed subtitles in various formats.
*   **Cross-Platform:** Works on Windows, macOS, Linux, and other Unix-like systems.
*   **Active Community:** Benefit from a constantly updated tool with an active community.

## Installation

Choose your operating system and desired method:

*   **UNIX (Linux, macOS, etc.):**

    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```
    *(If `curl` is unavailable, use `wget` instead)*

*   **Windows:**
    *   Download the [.exe file](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29), but not `%SYSTEMROOT%\System32`.
*   **Pip:**

    ```bash
    sudo -H pip install --upgrade youtube-dl
    ```

*   **Homebrew (macOS):**

    ```bash
    brew install youtube-dl
    ```

*   **MacPorts (macOS):**

    ```bash
    sudo port install youtube-dl
    ```

*   **Other Methods:**
    *   Explore the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html) for more installation options, including PGP signatures.
    *   Consider developer instructions for working with the Git repository.

## Usage

Basic syntax for downloading a video:

```bash
youtube-dl [OPTIONS] URL [URL...]
```

Replace `URL` with the video's web address.  Use `youtube-dl -h` for a full list of options.

## Core Functionality

*   **Options:**  Comprehensive list of options available to customize your downloads.
*   **Configuration:**  Configure youtube-dl settings using configuration files.
*   **Output Template:** Control output filenames with flexible output templates.
*   **Format Selection:** Select specific video and audio formats.
*   **Video Selection:** Filter downloads based on various criteria.
*   **FAQ:** See the FAQ for common questions and troubleshooting steps.

## Detailed Sections

### Installation

*   Detailed instructions for installing on various operating systems using different methods.
*   Links to download specific files and documentation for installation.
*   Instructions for using `pip`, `Homebrew`, and `MacPorts`.

### Description

*   Overview of what youtube-dl is and its functionality.
*   Information about the Python interpreter requirement and platform compatibility.
*   License information (Public Domain).

### Options

*   Provides detailed explanation of all available options to customize how youtube-dl downloads content.
*   Organized into categories for easier navigation.
    *   Network Options
    *   Geo Restriction
    *   Video Selection
    *   Download Options
    *   Filesystem Options
    *   Thumbnail Options
    *   Verbosity / Simulation Options
    *   Workarounds
    *   Video Format Options
    *   Subtitle Options
    *   Authentication Options
    *   Adobe Pass Options
    *   Post-processing Options
    *   Configuration Options

### Output Template

*   Explanation of the `-o` option and how to use it to customize filenames.
*   List of available special sequences (e.g., `%(title)s`, `%(id)s`, `%(ext)s`) and their functionalities.
*   Formatting examples for various use cases.

### Format Selection

*   Detailed information on the `-f` or `--format` option and how to select desired video and audio formats.
*   Description of available format codes and special names (e.g., `best`, `worst`, `bestvideo`).
*   Examples of format selection with precedence and filtering.
*   Explanation of merging video and audio streams.

### Video Selection

*   Guide to options that let users to filter and select videos for download.
*   Includes options that allow users to filter videos by upload date, title matching, and more.

### FAQ

*   Answers to frequently asked questions:
    *   How to update youtube-dl.
    *   Troubleshooting common errors, including Windows Defender, `Unable to extract OpenGraph title`, and `ERROR: no fmt_url_map or conn information found in video info`.
    *   Answers to questions about dependencies (ffmpeg/avconv, rtmpdump, mplayer/mpv).
    *   How to play downloaded videos.
    *   Solutions for issues with video URLs containing ampersands.
    *   How to deal with common errors.
    *   How to stream directly to media players.
    *   How to download only new videos from a playlist.
    *   How to download new videos from a playlist.
    *   Discusses the `--hls-prefer-native` settings.
    *   Addresses copyright infringement and site support policies.

### Embedding youtube-dl

*   Instructions for embedding youtube-dl within Python code.
*   Example code snippets to get started.

### Bugs

*   Guidance on how to report bugs and suggestions.
*   Instructions to include the full verbose output (`-v`) and how to format it.
*   Checklist for filing issues, including using the latest version, describing the issue clearly, and providing a URL.

### Developer Instructions

*   Instructions for developers who wish to add support for new sites or contribute to the project.
*   Step-by-step guide for setting up the development environment and adding a new extractor.
*   Coding conventions for creating extractors.

### Copyright

*   Information regarding the licensing of youtube-dl (public domain) and the original README file.