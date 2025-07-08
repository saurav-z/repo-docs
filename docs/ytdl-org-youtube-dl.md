[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Download Videos from YouTube and Beyond

Tired of buffering?  **Download videos from YouTube and hundreds of other sites with ease using youtube-dl!** [Visit the official GitHub repository for more information](https://github.com/ytdl-org/youtube-dl).

**Key Features:**

*   **Broad Site Support:** Download videos from a vast array of video platforms, including YouTube, Vimeo, and many more.
*   **Format Selection:** Choose your preferred video and audio formats, or let youtube-dl select the best available.
*   **Playlist and Channel Downloads:** Easily download entire playlists or all videos from a channel.
*   **Customization:** Tailor downloads with options for output filenames, video quality, subtitles, and more.
*   **Cross-Platform Compatibility:** Works seamlessly on Linux, macOS, and Windows.
*   **Open Source & Free:** Benefit from a community-driven project released into the public domain.

---
## Installation

### For Unix-like Systems (Linux, macOS, etc.)

To install directly:

```bash
sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

If you don't have `curl`, use `wget`:

```bash
sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
```

### For Windows

1.  Download the latest `.exe` file:  [youtube-dl.exe](https://yt-dl.org/latest/youtube-dl.exe)
2.  Place the `.exe` in a directory included in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) (e.g., `C:\Users\<YourUsername>\AppData\Local\Programs\Python\Python311`). **Avoid** placing it in `%SYSTEMROOT%\System32`.

### Using pip (Python Package Manager)

```bash
sudo -H pip install --upgrade youtube-dl
```

### For macOS (using package managers)

*   **Homebrew:**  `brew install youtube-dl`
*   **MacPorts:**  `sudo port install youtube-dl`

---
## Core Functionality

**youtube-dl** is a command-line tool to download videos. It's designed to be flexible and powerful. To download a video, simply use the basic command:

```bash
youtube-dl [OPTIONS] URL [URL...]
```

Replace `[OPTIONS]` with any of the many available options and `URL` with the video's address.

**Example:**

```bash
youtube-dl https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

---
## Key Options

*   **`-U, --update`**: Update youtube-dl to the latest version.
*   **`-o, --output TEMPLATE`**: Specify the output filename template (see [Output Template](#output-template) section).
*   **`-f, --format FORMAT`**: Choose a specific video format (see [Format Selection](#format-selection) section).
*   **`--list-formats`**:  List all available formats for a video.
*   **`--write-sub`, `--write-auto-sub`**: Download subtitles.
*   **`-x, --extract-audio`**: Extract audio from video files (requires ffmpeg/avconv).
*   **`-i, --ignore-errors`**:  Continue downloading even if errors occur.

---
## Output Template

The `-o` option lets you customize output filenames using special sequences.

**Example:**

```bash
youtube-dl -o '%(title)s-%(id)s.%(ext)s' https://www.youtube.com/watch?v=dQw4w9WgXcQ
```
Would save the file in the current directory as `youtube-dl test video-BaW_jenozKcj.mp4`.
For more template options, see the [Output Template](#output-template) Section.

---
## Format Selection

Use the `-f` or `--format` option to select the video format.
youtube-dl automatically selects the best format by default.

**Examples:**

*   `-f 22`: Download the format with code 22.
*   `-f best`:  Download the best available format (video and audio).
*   `-f webm`: Download the best WebM format.
*   `-f bestvideo+bestaudio`: Download the best video and audio and merge them.

For advanced filtering, use format selectors like `-f "best[height=720]"`.
Read more in the [Format Selection](#format-selection) Section.

---
## Video Selection

Use these options to filter which videos you download.
*   `--playlist-start NUMBER` : Start at a specific position in a playlist.
*   `--playlist-end NUMBER` : End at a specific position in a playlist.
*   `--match-title REGEX` : Only downloads videos with matching titles (REGEX or caseless sub-string).
*   `--date DATE` : Download only videos uploaded on this date.
*   `--datebefore DATE` : Download only videos uploaded before this date.
*   `--dateafter DATE` : Download only videos uploaded after this date.
*   `--min-views COUNT` : Do not download any videos with less than COUNT views.
*   `--max-views COUNT` : Do not download any videos with more than COUNT views.
*   `--download-archive FILE` : Download only videos not listed in the archive file.

---
## Frequently Asked Questions (FAQ)

*   **How do I update youtube-dl?** Run `youtube-dl -U`.  If that doesn't work, follow the [installation instructions](#installation) again.
*   **Why is youtube-dl slow to start on Windows?** Add an exclusion for `youtube-dl.exe` in Windows Defender settings.
*   **How can I stream directly to media player?** Use `-o -` with your command, and pipe to your media player.
*   For answers to these, and other frequently asked questions, please see the [FAQ](#faq) section

---
For more detailed information and advanced options, refer to the [original README](https://github.com/ytdl-org/youtube-dl) and the extensive documentation within.

---

**Copyright:** youtube-dl is released into the public domain.

**Contribute:** For bugs, suggestions, or help, visit the [issue tracker](https://github.com/ytdl-org/youtube-dl/issues).