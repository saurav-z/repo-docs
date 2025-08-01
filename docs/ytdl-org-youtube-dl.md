[![Build Status](https://github.com/ytdl-org/youtube-dl/workflows/CI/badge.svg)](https://github.com/ytdl-org/youtube-dl/actions?query=workflow%3ACI)

# youtube-dl: Download Videos from YouTube and More!

Tired of buffering? **youtube-dl is your go-to command-line tool for downloading videos from YouTube.com and thousands of other video platforms.**

[**View the original repository**](https://github.com/ytdl-org/youtube-dl)

## Key Features

*   **Wide Platform Support:** Download videos from YouTube, Vimeo, Dailymotion, and thousands of other sites.
*   **Multiple Format Options:** Choose your preferred video quality, resolution, and format.
*   **Playlist Downloads:** Easily download entire playlists with a single command.
*   **Subtitle Support:** Download and embed subtitles in various languages.
*   **Customizable Output:** Control file names, directory structure, and metadata.
*   **Audio Extraction:** Convert videos to audio-only files (MP3, AAC, FLAC, etc.).
*   **Cross-Platform Compatibility:** Works seamlessly on Linux, macOS, and Windows.
*   **Active Development:** Regularly updated to support new sites and features.

## Installation

Choose your preferred method:

### Unix (Linux, macOS, etc.)

1.  **Using `curl`:**

    ```bash
    sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```

2.  **Using `wget` (if `curl` is unavailable):**

    ```bash
    sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
    sudo chmod a+rx /usr/local/bin/youtube-dl
    ```

### Windows

1.  **Download:** [Download the .exe file](https://yt-dl.org/latest/youtube-dl.exe).
2.  **Placement:** Place it in a directory within your [PATH](https://en.wikipedia.org/wiki/PATH_%28variable%29) (e.g., `C:\Users\<User name>\bin`).  **Do NOT** place it in `%SYSTEMROOT%\System32`.

### Using pip

```bash
sudo -H pip install --upgrade youtube-dl
```

### macOS (using package managers)

*   **Homebrew:** `brew install youtube-dl`
*   **MacPorts:** `sudo port install youtube-dl`

### Alternative

Refer to the [youtube-dl Download Page](https://ytdl-org.github.io/youtube-dl/download.html) for more installation options, including PGP signatures and developer instructions.

## Usage

Basic Usage:

```bash
youtube-dl [OPTIONS] URL [URL...]
```

Where `URL` is the video or playlist URL.  See the [OPTIONS](#options) section for more detailed usage instructions.

## Core Functionality

*   **DESCRIPTION**
*   **OPTIONS**
*   **CONFIGURATION**
*   **OUTPUT TEMPLATE**
*   **FORMAT SELECTION**
*   **VIDEO SELECTION**
*   **FAQ**
*   **DEVELOPER INSTRUCTIONS**
*   **EMBEDDING YOUTUBE-DL**
*   **BUGS**
*   **COPYRIGHT**

## Options

*   **General Options:**  Help, version, update, ignore errors, configuration, and more.
*   **Network Options:** Proxy configuration, socket timeout, IP address settings, and more.
*   **Geo Restriction:** Bypass geo-restrictions, country code, and IP block options.
*   **Video Selection:** Playlist start/end, title filtering, max downloads, date range, and more.
*   **Download Options:** Rate limits, retries, fragmentation, buffer size, and more.
*   **Filesystem Options:** Batch file, output template, file naming, and more.
*   **Thumbnail Options:** Write thumbnail, list formats.
*   **Verbosity / Simulation Options:** Quiet mode, simulation, get information, and more.
*   **Workarounds:** Encoding, certificate validation, user agent, referer, and more.
*   **Video Format Options:** Specific format selection, all formats, and more.
*   **Subtitle Options:** Download and format subtitles.
*   **Authentication Options:** Login with a username and password.
*   **Adobe Pass Options:** Options for authentication.
*   **Post-processing Options:** Convert video to audio, specify audio format, add metadata.

For more information on each option, see the original [README](https://github.com/ytdl-org/youtube-dl).

## Output Template Examples

*   **Basic Filename:** `youtube-dl -o '%(title)s.%(ext)s' BaW_jenozKc`
*   **Playlist Download in Subdirectories:** `youtube-dl -o '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s' https://www.youtube.com/playlist?list=PLwiyx1dc3P2JR9N8gQaQN_BCvlSlap7re`

## Format Selection Examples

*   **Best mp4:** `youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'`
*   **Best but no better than 480p:** `youtube-dl -f 'bestvideo[height<=480]+bestaudio/best[height<=480]'`
*   **Best Video, no merge:** `youtube-dl -f 'bestvideo,bestaudio' -o '%(title)s.f%(format_id)s.%(ext)s'`

## Video Selection Examples

*   **Download only videos uploaded in the last 6 months:** `youtube-dl --dateafter now-6months`

## Configuration

Customize youtube-dl using a configuration file. On Linux and macOS, the system-wide configuration file is located at `/etc/youtube-dl.conf`, and the user-wide configuration file is at `~/.config/youtube-dl/config`. On Windows, the user-wide configuration files are `%APPDATA%\youtube-dl\config.txt` or `C:\Users\<user name>\youtube-dl.conf`.  See the [README](https://github.com/ytdl-org/youtube-dl) for example configuration files.

## FAQ

Answers to common questions, including updating, error resolution, and more.

### How do I update youtube-dl?

Run `youtube-dl -U` (or, on Linux, `sudo youtube-dl -U`)

If you have used pip, a simple `sudo pip install -U youtube-dl` is sufficient to update.

If you have installed youtube-dl using a package manager like *apt-get* or *yum*, use the standard system update mechanism to update. Note that distribution packages are often outdated. As a rule of thumb, youtube-dl releases at least once a month, and often weekly or even daily. Simply go to https://yt-dl.org to find out the current version. Unfortunately, there is nothing we youtube-dl developers can do if your distribution serves a really outdated version. You can (and should) complain to your distribution in their bugtracker or support forum.

As a last resort, you can also uninstall the version installed by your package manager and follow our manual installation instructions. For that, remove the distribution's package, with a line like

    sudo apt-get remove -y youtube-dl

Afterwards, simply follow [our manual installation instructions](https://ytdl-org.github.io/youtube-dl/download.html):

```
sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl
hash -r
```

Again, from then on you'll be able to update with `sudo youtube-dl -U`.

### I'm getting an error `Unable to extract OpenGraph title` on YouTube playlists

YouTube changed their playlist format in March 2014 and later on, so you'll need at least youtube-dl 2014.07.25 to download all YouTube videos.

If you have installed youtube-dl with a package manager, pip, setup.py or a tarball, please use that to update. Note that Ubuntu packages do not seem to get updated anymore. Since we are not affiliated with Ubuntu, there is little we can do. Feel free to [report bugs](https://bugs.launchpad.net/ubuntu/+source/youtube-dl/+filebug) to the [Ubuntu packaging people](mailto:ubuntu-motu@lists.ubuntu.com?subject=outdated%20version%20of%20youtube-dl) - all they have to do is update the package to a somewhat recent version. See above for a way to update.

### I'm getting an error when trying to use output template: `error: using output template conflicts with using title, video ID or auto number`

Make sure you are not using `-o` with any of these options `-t`, `--title`, `--id`, `-A` or `--auto-number` set in command line or in a configuration file. Remove the latter if any.

### Do I always have to pass `-citw`?

By default, youtube-dl intends to have the best options (incidentally, if you have a convincing case that these should be different, [please file an issue where you explain that](https://yt-dl.org/bug)). Therefore, it is unnecessary and sometimes harmful to copy long option strings from webpages. In particular, the only option out of `-citw` that is regularly useful is `-i`.

### Can you please put the `-b` option back?

Most people asking this question are not aware that youtube-dl now defaults to downloading the highest available quality as reported by YouTube, which will be 1080p or 720p in some cases, so you no longer need the `-b` option. For some specific videos, maybe YouTube does not report them to be available in a specific high quality format you're interested in. In that case, simply request it with the `-f` option and youtube-dl will try to download it.

### I get HTTP error 402 when trying to download a video. What's this?

Apparently YouTube requires you to pass a CAPTCHA test if you download too much. We're [considering to provide a way to let you solve the CAPTCHA](https://github.com/ytdl-org/youtube-dl/issues/154), but at the moment, your best course of action is pointing a web browser to the youtube URL, solving the CAPTCHA, and restart youtube-dl.

### Do I need any other programs?

youtube-dl works fine on its own on most sites. However, if you want to convert video/audio, you'll need [avconv](https://libav.org/) or [ffmpeg](https://www.ffmpeg.org/). On some sites - most notably YouTube - videos can be retrieved in a higher quality format without sound. youtube-dl will detect whether avconv/ffmpeg is present and automatically pick the best option.

Videos or video formats streamed via RTMP protocol can only be downloaded when [rtmpdump](https://rtmpdump.mplayerhq.hu/) is installed. Downloading MMS and RTSP videos requires either [mplayer](https://mplayerhq.hu/) or [mpv](https://mpv.io/) to be installed.

### I have downloaded a video but how can I play it?

Once the video is fully downloaded, use any video player, such as [mpv](https://mpv.io/), [vlc](https://www.videolan.org/) or [mplayer](https://www.mplayerhq.hu/).

### I extracted a video URL with `-g`, but it does not play on another machine / in my web browser.

It depends a lot on the service. In many cases, requests for the video (to download/play it) must come from the same IP address and with the same cookies and/or HTTP headers. Use the `--cookies` option to write the required cookies into a file, and advise your downloader to read cookies from that file. Some sites also require a common user agent to be used, use `--dump-user-agent` to see the one in use by youtube-dl. You can also get necessary cookies and HTTP headers from JSON output obtained with `--dump-json`.

It may be beneficial to use IPv6; in some cases, the restrictions are only applied to IPv4. Some services (sometimes only for a subset of videos) do not restrict the video URL by IP address, cookie, or user-agent, but these are the exception rather than the rule.

Please bear in mind that some URL protocols are **not** supported by browsers out of the box, including RTMP. If you are using `-g`, your own downloader must support these as well.

If you want to play the video on a machine that is not running youtube-dl, you can relay the video content from the machine that runs youtube-dl. You can use `-o -` to let youtube-dl stream a video to stdout, or simply allow the player to download the files written by youtube-dl in turn.

### ERROR: no fmt_url_map or conn information found in video info

YouTube has switched to a new video info format in July 2011 which is not supported by old versions of youtube-dl. See [above](#how-do-i-update-youtube-dl) for how to update youtube-dl.

### ERROR: unable to download video

YouTube requires an additional signature since September 2012 which is not supported by old versions of youtube-dl. See [above](#how-do-i-update-youtube-dl) for how to update youtube-dl.

### Video URL contains an ampersand and I'm getting some strange output `[1] 2839` or `'v' is not recognized as an internal or external command`

That's actually the output from your shell. Since ampersand is one of the special shell characters it's interpreted by the shell preventing you from passing the whole URL to youtube-dl. To disable your shell from interpreting the ampersands (or any other special characters) you have to either put the whole URL in quotes or escape them with a backslash (which approach will work depends on your shell).

For example if your URL is https://www.youtube.com/watch?t=4&v=BaW_jenozKc you should end up with following command:

```youtube-dl 'https://www.youtube.com/watch?t=4&v=BaW_jenozKc'```

or

```youtube-dl https://www.youtube.com/watch?t=4\&v=BaW_jenozKc```

For Windows you have to use the double quotes:

```youtube-dl "https://www.youtube.com/watch?t=4&v=BaW_jenozKc"```

### ExtractorError: Could not find JS function u'OF'

In February 2015, the new YouTube player contained a character sequence in a string that was misinterpreted by old versions of youtube-dl. See [above](#how-do-i-update-youtube-dl) for how to update youtube-dl.

### HTTP Error 429: Too Many Requests or 402: Payment Required

These two error codes indicate that the service is blocking your IP address because of overuse. Usually this is a soft block meaning that you can gain access again after solving CAPTCHA. Just open a browser and solve a CAPTCHA the service suggests you and after that [pass cookies](#how-do-i-pass-cookies-to-youtube-dl) to youtube-dl. Note that if your machine has multiple external IPs then you should also pass exactly the same IP you've used for solving CAPTCHA with [`--source-address`](#network-options). Also you may need to pass a `User-Agent` HTTP header of your browser with [`--user-agent`](#workarounds).

If this is not the case (no CAPTCHA suggested to solve by the service) then you can contact the service and ask them to unblock your IP address, or - if you have acquired a whitelisted IP address already - use the [`--proxy` or `--source-address` options](#network-options) to select another IP address.

### SyntaxError: Non-ASCII character

The error

    File "youtube-dl", line 2
    SyntaxError: Non-ASCII character '\x93' ...

means you're using an outdated version of Python. Please update to Python 2.6 or 2.7.

### What is this binary file? Where has the code gone?

Since June 2012 ([#342](https://github.com/ytdl-org/youtube-dl/issues/342)) youtube-dl is packed as an executable zipfile, simply unzip it (might need renaming to `youtube-dl.zip` first on some systems) or clone the git repository, as laid out above. If you modify the code, you can run it by executing the `__main__.py` file. To recompile the executable, run `make youtube-dl`.

### The exe throws an error due to missing `MSVCR100.dll`

To run the exe you need to install first the [Microsoft Visual C++ 2010 Service Pack 1 Redistributable Package (x86)](https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x86.exe).

### On Windows, how should I set up ffmpeg and youtube-dl? Where should I put the exe files?

If you put youtube-dl and ffmpeg in the same directory that you're running the command from, it will work, but that's rather cumbersome.

To make a different directory work - either for ffmpeg, or for youtube-dl, or for both - simply create the directory (say, `C:\bin`, or `C:\Users\<User name>\bin`), put all the executables directly in there, and then [set your PATH environment variable](https://www.java.com/en/download/help/path.xml) to include that directory.

From then on, after restarting your shell, you will be able to access both youtube-dl and ffmpeg (and youtube-dl will be able to find ffmpeg) by simply typing `youtube-dl` or `ffmpeg`, no matter what directory you're in.

### How do I put downloads into a specific folder?

Use the `-o` to specify an [output template](#output-template), for example `-o "/home/user/videos/%(title)s-%(id)s.%(ext)s"`. If you want this for all of your downloads, put the option into your [configuration file](#configuration).

### How do I download a video starting with a `-`?

Either prepend `https://www.youtube.com/watch?v=` or separate the ID from the options with `--`:

    youtube-dl -- -wNyEUrxzFU
    youtube-dl "https://www.youtube.com/watch?v=-wNyEUrxzFU"

### How do I pass cookies to youtube-dl?

Use the `--cookies` option, for example `--cookies /path/to/cookies/file.txt`.

In order to extract cookies from browser use any conforming browser extension for exporting cookies. For example, [Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) (for Chrome) or [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/) (for Firefox).

Note that the cookies file must be in Mozilla/Netscape format and the first line of the cookies file must be either `# HTTP Cookie File` or `# Netscape HTTP Cookie File`. Make sure you have correct [newline format](https://en.wikipedia.org/wiki/Newline) in the cookies file and convert newlines if necessary to correspond with your OS, namely `CRLF` (`\r\n`) for Windows and `LF` (`\n`) for Unix and Unix-like systems (Linux, macOS, etc.). `HTTP Error 400: Bad Request` when using `--cookies` is a good sign of invalid newline format.

Passing cookies to youtube-dl is a good way to workaround login when a particular extractor does not implement it explicitly. Another use case is working around [CAPTCHA](https://en.wikipedia.org/wiki/CAPTCHA) some websites require you to solve in particular cases in order to get access (e.g. YouTube, CloudFlare).

### How do I stream directly to media player?

You will first need to tell youtube-dl to stream media to stdout with `-o -`, and also tell your media player to read from stdin (it must be capable of this for streaming) and then pipe former to latter. For example, streaming to [vlc](https://www.videolan.org/) can be achieved with:

    youtube-dl -o - "https://www.youtube.com/watch?v=BaW_jenozKcj" | vlc -

### How do I download only new videos from a playlist?

Use download-archive feature. With this feature you should initially download the complete playlist with `--download-archive /path/to/download/archive/file.txt` that will record identifiers of all the videos in a special file. Each subsequent run with the same `--download-archive` will download only new videos and skip all videos that have been downloaded before. Note that only successful downloads are recorded in the file.

For example, at first,

    youtube-dl --download-archive archive.txt "https://www.youtube.com/playlist?list=PLwiyx1dc3P2JR9N8gQaQN_BCvlSlap7re"

will download the complete `PLwiyx1dc3P2JR9N8gQaQN_BCvlSlap7re` playlist and create a file `archive.txt`. Each subsequent run will only download new videos if any:

    youtube-dl --download-archive archive.txt "https://www.youtube.com/playlist?list=PLwiyx1dc3P2JR9N8gQaQN_BCvlSlap7re"

### Should I add `--hls-prefer-native` into my config?

When youtube-dl detects an HLS video, it can download it either with the built-in downloader or ffmpeg. Since many HLS streams are slightly invalid and ffmpeg/youtube-dl each handle some invalid cases better than the other, there is an option to switch the downloader if needed.

When youtube-dl knows that one particular downloader works better for a given website, that downloader will be picked. Otherwise, youtube-dl will pick the best downloader for general compatibility, which at the moment happens to be ffmpeg. This choice may change in future versions of youtube-dl, with improvements of the built-in downloader and/or ffmpeg.

In particular, the generic extractor (used when your website is not in the [list of supported sites by youtube-dl](https://ytdl-org.github.io/youtube-dl/supportedsites.html) cannot mandate one specific downloader.

If you put either `--hls-prefer-native` or `--hls-prefer-ffmpeg` into your configuration, a different subset of videos will fail to download correctly. Instead, it is much better to [file an issue](https://yt-dl.org/bug) or a pull request which details why the native or the ffmpeg HLS downloader is a better choice for your use case.

### Can you add support for this anime video site, or site which shows current movies for free?

As a matter of policy (as well as legality), youtube-dl does not include support for services that specialize in infringing copyright. As a rule of thumb, if you cannot easily find a video that the service is quite obviously allowed to distribute (i.e. that has been uploaded by the creator, the creator's distributor, or is published under a free license), the service is probably unfit for inclusion to youtube-dl.

A note on the service that they don't host the infringing content, but just link to those who do, is evidence that the service should **not** be included into youtube-dl. The same goes for any DMCA note when the whole front page of the service is filled with videos they are not allowed to distribute. A "fair use" note is equally unconvincing if the service shows copyright-protected videos in full without authorization.

Support requests for services that **do** purchase the rights to distribute their content are perfectly fine though. If in doubt, you can simply include a source that mentions the legitimate purchase of content.

### How can I speed up work on my issue?

(Also known as: Help, my important issue not being solved!) The youtube-dl core developer team is quite small. While we do our best to solve as many issues as possible, sometimes that can take quite a while. To speed up your issue, here's what you can do:

First of all, please do report the issue [at our issue tracker](https://yt-dl.org/bugs). That allows us to coordinate all efforts by users and developers, and serves as a unified point. Unfortunately, the youtube-dl project has grown too large to use personal email as an effective communication channel.

Please read the [bug reporting instructions](#bugs) below. A lot of bugs lack all the necessary information. If you can, offer proxy, VPN, or shell access to the youtube-dl developers. If you are able to, test the issue from multiple computers in multiple countries to exclude local censorship or misconfiguration issues.

If nobody is interested in solving your issue, you are welcome to take matters into your own hands and submit a pull request (or coerce/pay somebody else to do so).

Feel free to bump the issue from time to time by writing a small comment ("Issue is still present in youtube-dl version ...from France, but fixed from Belgium"), but please not more than once a month. Please do not declare your issue as `important` or `urgent`.

### How can I detect whether a given URL is supported by youtube-dl?

For one, have a look at the [list of supported sites](docs/supportedsites.md). Note that it can sometimes happen that the site changes its URL scheme (say, from https://example.com/video/1234567 to https://example.com/v/1234567 ) and youtube-dl reports an URL of a service in that list as unsupported. In that case, simply report a bug.

It is *not* possible to detect whether a URL is supported or not. That's because youtube-dl contains a generic extractor which matches **all** URLs. You may be tempted to disable, exclude, or remove the generic extractor, but the generic extractor not only allows users to extract videos from lots of websites that embed a video from another service, but may also be used to extract video from a service that it's hosting itself. Therefore, we neither recommend nor support disabling, excluding, or removing the generic extractor.

If you want to find out whether a given URL is supported, simply call youtube-dl with it. If you get no videos back, chances are the URL is either not referring to a video or unsupported. You can find out which by examining the output (if you run youtube-dl on the console) or catching an `UnsupportedError` exception if you run it from a Python program.

## Bugs

*   **Reporting Bugs:** Report bugs and suggestions in the issue tracker: <https://github.com/ytdl-org/youtube-dl/issues> (<https://yt-dl.org/bug> is an alias for this).

    *   **Follow Instructions:** Provide the full output of `youtube-dl -v YOUR_URL_HERE`.
    *   **Sufficient Detail:** Describe the issue clearly, provide a URL, and the expected behavior.
    *   **Current Version:** Always update to the latest version (`youtube-dl -U`).
*   **Bug Reporting Checklist:**
    *   Description of the issue should be sufficient
    *   The issue isn't already documented
    *   You're using the latest version
    *   Existing options aren't enough
    *   The issue has enough context
    *   The issue has only one problem
    *   The feature is useful to others
    *   Your question is about youtube-dl
*   **Contributing to the project's code and workflow can be found in the developer instructions.**

## Copyright

The youtube-dl code is released into the public domain by the copyright holders.