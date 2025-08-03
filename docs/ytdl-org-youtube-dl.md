[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Download Videos from YouTube and Beyond

Tired of limited online video access? **youtube-dl** is your go-to command-line tool for downloading videos from YouTube and many other popular video platforms. 

[Visit the original repository for youtube-dl](https://github.com/ytdl-org/youtube-dl)

**Key Features:**

*   **Wide Platform Support:** Download videos from hundreds of sites, including YouTube, Vimeo, and more.
*   **Format Selection:** Choose your preferred video quality and format.
*   **Playlist and Channel Downloads:** Download entire playlists or all videos from a channel.
*   **Metadata Handling:** Automatically save video titles, descriptions, and other metadata.
*   **Customizable Output:** Control file naming, directory structure, and more.
*   **Cross-Platform Compatibility:** Works seamlessly on Linux, macOS, Windows, and other operating systems.
*   **Active Development:** Benefit from continuous updates and improvements to support new sites and features.

## Installation

Easily install youtube-dl on your system using the following methods:

*   **Unix (Linux, macOS):**
    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```
    (or use `wget` if you don't have `curl`)
*   **Windows:** Download the [youtube-dl.exe](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29).
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

For more detailed installation instructions, including options like PGP signatures, refer to the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## Usage

Download a video:

```bash
youtube-dl [OPTIONS] "VIDEO_URL"
```

Download a playlist:

```bash
youtube-dl [OPTIONS] "PLAYLIST_URL"
```

For example:

```bash
youtube-dl https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

View available options:

```bash
youtube-dl -h
```

## Core Functionality & Important Options

*   **Updating:** `youtube-dl -U` (or `sudo youtube-dl -U` on Linux) to update to the latest version.
*   **Format Selection:** `-f FORMAT` to select video quality.  Use `-F` or `--list-formats` to see available formats.
*   **Output Template:** `-o TEMPLATE` for custom file names and organization.
*   **Playlist Handling:** Download an entire playlist with a single command.
*   **Authentication:** Use `-u USERNAME` and `-p PASSWORD` for sites requiring login.

### Detailed Documentation

*   **Options:** Explore the comprehensive [OPTIONS](#options) section in the original README for a full list of available options and their uses.
*   **Output Template:** Customize the names and locations of downloaded files with the [OUTPUT TEMPLATE](#output-template) section.
*   **Format Selection:** Refine your downloads by video quality and resolution using the [FORMAT SELECTION](#format-selection) section.
*   **Video Selection:** Use options like `--date`, `--datebefore` and `--dateafter` to control which videos are downloaded.
*   **Configuration:** Customize the program behavior using configuration files - see [CONFIGURATION](#configuration)
*   **FAQ:** Visit the [FAQ](#faq) section to address common questions.
*   **Developer Guide:** Learn how to contribute by reviewing the [DEVELOPER INSTRUCTIONS](#developer-instructions).

## Troubleshooting and Support

*   **Update youtube-dl:**  Always update to the latest version (`youtube-dl -U`) before reporting an issue.
*   **Check the FAQ:** Consult the [FAQ](#faq) section for common issues and solutions.
*   **Report Bugs:**  Report bugs and suggestions on the [issue tracker](https://github.com/ytdl-org/youtube-dl/issues), providing the full output with the `-v` flag. See the [BUGS](#bugs) for more details.
*   **Join the Community:** Connect with other users and developers on the IRC channel [#youtube-dl](irc://chat.freenode.net/#youtube-dl) on freenode.

## Additional Resources

*   **Supported Sites:** [See the list of supported sites](https://ytdl-org.github.io/youtube-dl/supportedsites.html)
*   **Embedding youtube-dl:** Explore ways to embed youtube-dl into other applications - see [EMBEDDING YOUTUBE-DL](#embedding-youtube-dl)
*   **Copyright:** Learn about the copyright and terms of use in the [COPYRIGHT](#copyright) section

This README is an overview of the youtube-dl tool.  For detailed information and advanced features, refer to the original repository.