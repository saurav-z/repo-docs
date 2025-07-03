[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Your Go-To Tool for Downloading Videos from the Web

Tired of buffering or wanting to save videos for offline viewing?  [**youtube-dl**](https://github.com/ytdl-org/youtube-dl) is a powerful, command-line utility that lets you download videos from YouTube and thousands of other sites!

## Key Features

*   **Wide Site Support:** Download from YouTube, Facebook, Instagram, Vimeo, and many more (check the [supported sites](https://ytdl-org.github.io/youtube-dl/supportedsites.html)!).
*   **Format Selection:** Choose your preferred video quality, resolution, and format with flexible options.
*   **Playlist Downloads:** Download entire playlists with ease.
*   **Metadata Handling:** Automatically includes video titles, descriptions, and other metadata.
*   **Customization:** Customize output filenames, download speed limits, and more.
*   **Cross-Platform:** Works on Windows, macOS, and Linux.
*   **Active Community:** Benefit from a project with a large and active community.

## Installation

Get started quickly with these installation guides:

*   **UNIX (Linux, macOS):**  
    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```
    If `curl` is unavailable, use `wget`:
    ```bash
    sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```
*   **Windows:**  
    [Download the `.exe` file](https://yt-dl.org/latest/youtube-dl.exe) and place it in a folder within your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) (avoid `%SYSTEMROOT%\System32`).
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

For more installation options, including PGP signatures, see the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html).

## Usage

Basic usage:

```bash
youtube-dl [OPTIONS] URL [URL...]
```

Example:

```bash
youtube-dl https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

View all of the available options:

```bash
youtube-dl -h
```

## Key Options (Simplified)

*   `-h, --help`: Show help.
*   `-U, --update`: Update youtube-dl to the latest version.
*   `-f, --format FORMAT`: Select video format (see [Format Selection](#format-selection) below).
*   `-o, --output TEMPLATE`: Specify the output filename template (see [Output Template](#output-template) below).
*   `--proxy URL`: Use a proxy server.
*   `-i, --ignore-errors`: Continue on download errors.
*   `--write-sub`: Write subtitle file.
*   `-x, --extract-audio`: Extract audio from video.

## Format Selection

Control your downloads with the `--format` option:

*   `-f best`: Selects the best available quality. (default)
*   `-f 22`: Downloads the format with code 22 (e.g., 720p MP4).  Use `-F` to list available formats.
*   `-f "bestvideo[height<=?1080]+bestaudio/best"`:  Select the best video up to 1080p resolution and combine it with the best audio.
*   `-F, --list-formats`: List all available formats of requested videos

See the full [Format Selection](#format-selection) section in the original README for detailed explanations.

## Output Template

Customize your file names with the `-o` option and templates:

*   `-o '%(title)s-%(id)s.%(ext)s'`: Creates filenames like "Video Title-VideoID.mp4".
*   `-o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s'`:  Downloads playlist videos into separate directories, named by the playlist.

Available template variables: `id`, `title`, `url`, `ext`, `alt_title`, `display_id`, `uploader`, `license`, `creator`, `release_date`, `timestamp`, `upload_date`, `uploader_id`, `channel`, `channel_id`, `location`, `duration`, `view_count`, `like_count`, `dislike_count`, `repost_count`, `average_rating`, `comment_count`, `age_limit`, `is_live`, `start_time`, `end_time`, `format`, `format_id`, `format_note`, `width`, `height`, `resolution`, `tbr`, `abr`, `acodec`, `asr`, `vbr`, `fps`, `vcodec`, `container`, `filesize`, `filesize_approx`, `protocol`, `extractor`, `extractor_key`, `epoch`, `autonumber`, `playlist`, `playlist_index`, `playlist_id`, `playlist_title`, `playlist_uploader`, `playlist_uploader_id`.  Also, `chapter`, `chapter_number`, `chapter_id`, `series`, `season`, `season_number`, `season_id`, `episode`, `episode_number`, `episode_id`, `track`, `track_number`, `track_id`, `artist`, `genre`, `album`, `album_type`, `album_artist`, `disc_number`, `release_year`.

See the full [Output Template](#output-template) section in the original README for detailed explanations.

## Configuration

Customize youtube-dl's behavior by creating a configuration file:

*   **Linux/macOS:**  `/etc/youtube-dl.conf` (system-wide) or `~/.config/youtube-dl/config` (user-specific).
*   **Windows:** `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`.

Example:

```
# Always extract audio
-x
# Do not copy the mtime
--no-mtime
# Use this proxy
--proxy 127.0.0.1:3128
# Save all videos under Movies directory
-o ~/Movies/%(title)s.%(ext)s
```

## More Information

*   **Frequently Asked Questions (FAQ):** See the [FAQ](#faq) section in the original README for solutions to common problems.
*   **Bugs and Suggestions:** Report issues on the [issue tracker](https://github.com/ytdl-org/youtube-dl/issues).  Be sure to include the output of `youtube-dl -v [your URL]`
*   **Developer Instructions:**  For contributors, see the [Developer Instructions](#developer-instructions) section in the original README for information about adding new site support.
*   **Complete Documentation:**  Refer to the original [README](https://github.com/ytdl-org/youtube-dl) for a comprehensive guide to all options and features.

**Download videos from the web, simplified. Get started with youtube-dl today!**