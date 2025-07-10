[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Your Go-To Tool for Downloading Videos from the Web

**youtube-dl is a powerful command-line tool that lets you download videos from YouTube and thousands of other video sites, offering unparalleled flexibility and control.**

**Key Features:**

*   **Wide Site Support:** Download videos from a vast array of platforms, including YouTube, Vimeo, and many more. ([See supported sites](https://ytdl-org.github.io/youtube-dl/supportedsites.html))
*   **Format Selection:** Choose your preferred video and audio formats, including options for the best available quality or specific formats.
*   **Playlist & Channel Support:** Download entire playlists, channels, or specific videos from within them.
*   **Customization Options:** Tailor your downloads with features like output templates, proxy settings, subtitle downloads, and more.
*   **Cross-Platform Compatibility:** Works seamlessly on Linux, macOS, Windows, and other operating systems.
*   **Active Community & Updates:** Benefit from a large and active community, with frequent updates to support new sites and features.

**[Click here to visit the original repository.](https://github.com/ytdl-org/youtube-dl)**

**Installation**

*   **Unix (Linux, macOS, etc.):**

    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```

    *If `curl` is unavailable, use `wget` instead:*

    ```bash
    sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```

*   **Windows:** Download the [.exe file](https://yt-dl.org/latest/youtube-dl.exe) and place it in a directory included in your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) (e.g., `C:\Windows\System32`).

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

**[See the download page](https://ytdl-org.github.io/youtube-dl/download.html) for further options and PGP signatures.**

**Description**

youtube-dl is a versatile command-line program designed to download videos from YouTube.com and a wide range of other video-hosting websites. Written in Python and compatible with Python 2.6, 2.7, and 3.2+, it offers a platform-independent solution for video acquisition. The project is in the public domain, allowing for modification, redistribution, and usage as desired.

```bash
youtube-dl [OPTIONS] URL [URL...]
```

**Key Options**

*   **`-U, --update`:** Update youtube-dl to the latest version.
*   **`-i, --ignore-errors`:** Continue downloading even if errors occur.
*   **`-o, --output TEMPLATE`:** Specify the output filename template.
*   **`-f, --format FORMAT`:** Choose the video format.
*   **`-x, --extract-audio`:** Extract audio from video files.
*   **`--write-sub` / `--write-auto-sub`:** Download subtitles.
*   **`--cookies FILE`:** Use a cookies file for authentication.
*   **`--proxy URL`:** Use a proxy server.
*   **`-h, --help`:** Display the help message.

**Detailed Options**

*   **Network Options:** Customize network settings such as proxy usage, socket timeout, and IP address binding.
*   **Geo Restriction:** Bypass geographic restrictions using proxy servers or by spoofing the X-Forwarded-For HTTP header.
*   **Video Selection:** Download specific videos based on playlist position, title matching, file size limits, upload dates, view counts, and more.
*   **Download Options:** Control download rates, retries, buffer size, and external downloader integration.
*   **Filesystem Options:** Specify output file names, restrict filenames, resume downloads, and manage caching.
*   **Thumbnail Options:** Write thumbnail images to disk.
*   **Verbosity / Simulation Options:** Control output verbosity, simulate downloads, and print debugging information.
*   **Workarounds:** Override encoding, ignore certificate validation, specify user agents, and use custom HTTP headers.
*   **Video Format Options:** Select video formats, download all formats, and merge audio/video streams.
*   **Subtitle Options:** Download and format subtitles.
*   **Authentication Options:** Log in with usernames, passwords, and two-factor authentication.
*   **Adobe Pass Options:** Access Adobe Pass-protected content.
*   **Post-processing Options:** Convert videos to audio, embed subtitles, and add metadata.

**Configuration**

Configure youtube-dl settings through a configuration file. On Linux and macOS, the system-wide configuration file is `/etc/youtube-dl.conf`, and the user-specific file is `~/.config/youtube-dl/config`. On Windows, the user-specific files are `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`. Use `--ignore-config` to disable configuration files or `--config-location` to specify a custom file.
A configuration example is:
```
# Lines starting with # are comments

# Always extract audio
-x

# Do not copy the mtime
--no-mtime

# Use this proxy
--proxy 127.0.0.1:3128

# Save all videos under Movies directory in your home directory
-o ~/Movies/%(title)s.%(ext)s
```

**Output Template**

Use the `-o` option to define the output filename using a template. The template can contain special sequences (e.g., `%(title)s`, `%(id)s`, `%(ext)s`) that are replaced with video metadata.

**Format Selection**

Use the `-f` or `--format` option to specify the video format. You can choose formats based on format code, file extension, or specific criteria (e.g., `bestvideo`, `bestaudio`). Slashes and commas separate preferred and alternative formats.  You can also use filters to download only videos that meet certain criteria like resolution or bitrates, using expressions like `-f "best[height<=720][tbr>500]"`.
You can download a video and an audio format and mux them by using -f `<video-format>+<audio-format>`

**Video Selection**

Filter videos based on upload date, playlist position, title matching, and other criteria.

**FAQ**

*   **How do I update youtube-dl?** Run `youtube-dl -U` (or `sudo youtube-dl -U` on Linux).
*   **youtube-dl is extremely slow to start on Windows:** Add a file exclusion for `youtube-dl.exe` in Windows Defender settings.
*   **I'm getting an error `Unable to extract OpenGraph title` on YouTube playlists:** Update youtube-dl.
*   **I'm getting an error when trying to use output template: `error: using output template conflicts with using title, video ID or auto number`**: Make sure you are not using `-o` with any of these options `-t`, `--title`, `--id`, `-A` or `--auto-number`.
*   **Do I always have to pass `-citw`?** It is unnecessary and sometimes harmful to copy long option strings from webpages
*   **Can you please put the `-b` option back?**  youtube-dl now defaults to downloading the highest available quality
*   **I get HTTP error 402 when trying to download a video. What's this?** You've likely reached a CAPTCHA. Open the URL in a browser, solve the CAPTCHA, and then retry with youtube-dl.
*   **Do I need any other programs?** avconv/ffmpeg is needed for video/audio conversion. rtmpdump is needed for RTMP streams.
*   **I have downloaded a video but how can I play it?** Use any video player, such as [mpv](https://mpv.io/), [vlc](https://www.videolan.org/) or [mplayer](https://www.mplayerhq.hu/).
*   **I extracted a video URL with `-g`, but it does not play on another machine / in my web browser:** Use the `--cookies` option to write the required cookies into a file, and advise your downloader to read cookies from that file.
*   **ERROR: no fmt_url_map or conn information found in video info:** Update youtube-dl.
*   **ERROR: unable to download video:** Update youtube-dl.
*   **Video URL contains an ampersand and I'm getting some strange output `[1] 2839` or `'v' is not recognized as an internal or external command`**: Enclose the URL in quotes or escape ampersands.
*   **ExtractorError: Could not find JS function u'OF'**: Update youtube-dl.
*   **HTTP Error 429: Too Many Requests or 402: Payment Required**: The service is blocking your IP, try solving a CAPTCHA.
*   **SyntaxError: Non-ASCII character**: Update your Python installation.
*   **What is this binary file? Where has the code gone?**  youtube-dl is packaged as an executable zipfile.  Unzip it or clone the git repository to access the code.
*   **The exe throws an error due to missing `MSVCR100.dll`**: Install the [Microsoft Visual C++ 2010 Service Pack 1 Redistributable Package (x86)](https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x86.exe).
*   **On Windows, how should I set up ffmpeg and youtube-dl? Where should I put the exe files?** Put youtube-dl and ffmpeg in a directory included in your [PATH](https://www.java.com/en/download/help/path.xml).
*   **How do I put downloads into a specific folder?**  Use `-o /path/to/folder/%(title)s-%(id)s.%(ext)s`.
*   **How do I download a video starting with a `-`?** Prepend `https://www.youtube.com/watch?v=` or use `--` to separate the ID from the options
*   **How do I pass cookies to youtube-dl?** Use the `--cookies` option.
*   **How do I stream directly to media player?** Use `-o -` with the media player capable of reading from stdin.
*   **How do I download only new videos from a playlist?** Use the `--download-archive` option.

**Bugs & Suggestions**

Report bugs and suggestions in the issue tracker: <https://github.com/ytdl-org/youtube-dl/issues>. Provide the full output of `youtube-dl -v YOUR_URL_HERE`.

**Developer Instructions**

*   Run `python -m youtube_dl` to execute as a developer.
*   Tests can be run using `python -m unittest discover` or `nosetests`.
*   Add support for new sites following the steps in the Developer Instructions section.

**Copyright**

youtube-dl is released into the public domain.