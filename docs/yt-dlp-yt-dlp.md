<div align="center">

[![YT-DLP](https://raw.githubusercontent.com/yt-dlp/yt-dlp/master/.github/banner.svg)](#)

[![Release version](https://img.shields.io/github/v/release/yt-dlp/yt-dlp?color=brightgreen&label=Download&style=for-the-badge)](#installation "Installation")
[![PyPI](https://img.shields.io/badge/-PyPI-blue.svg?logo=pypi&labelColor=555555&style=for-the-badge)](https://pypi.org/project/yt-dlp "PyPI")
[![Donate](https://img.shields.io/badge/_-Donate-red.svg?logo=githubsponsors&labelColor=555555&style=for-the-badge)](Collaborators.md#collaborators "Donate")
[![Discord](https://img.shields.io/discord/807245652072857610?color=blue&labelColor=555555&label=&logo=discord&style=for-the-badge)](https://discord.gg/H5MNcFW63r "Discord")
[![Supported Sites](https://img.shields.io/badge/-Supported_Sites-brightgreen.svg?style=for-the-badge)](supportedsites.md "Supported Sites")
[![License: Unlicense](https://img.shields.io/badge/-Unlicense-blue.svg?style=for-the-badge)](LICENSE "License")
[![CI Status](https://img.shields.io/github/actions/workflow/status/yt-dlp/yt-dlp/core.yml?branch=master&label=Tests&style=for-the-badge)](https://github.com/yt-dlp/yt-dlp/actions "CI Status")
[![Commits](https://img.shields.io/github/commit-activity/m/yt-dlp/yt-dlp?label=commits&style=for-the-badge)](https://github.com/yt-dlp/yt-dlp/commits "Commit History")
[![Last Commit](https://img.shields.io/github/last-commit/yt-dlp/yt-dlp/master?label=&style=for-the-badge&display_timestamp=committer)](https://github.com/yt-dlp/yt-dlp/pulse/monthly "Last activity")

</div>

# yt-dlp: The Powerful Command-Line Video and Audio Downloader

yt-dlp is a feature-rich command-line utility for downloading audio and video from thousands of websites.  Forked from youtube-dl and built upon yt-dlc, it provides enhanced functionality and supports a vast range of online platforms.  For more details on how to contribute, check out the [original repository](https://github.com/yt-dlp/yt-dlp).

**Key Features:**

*   **Extensive Site Support:** Download from [thousands of sites](supportedsites.md).
*   **Format Selection:** Fine-grained control over video and audio formats with advanced filtering and sorting.
*   **Playlist Support:** Download entire playlists or select specific items.
*   **Subtitle Download:** Download and embed subtitles in various formats.
*   **Metadata Handling:** Embed metadata, including titles, descriptions, and cover art.
*   **Post-Processing:** Convert videos to audio, remux, and apply other processing steps.
*   **SponsorBlock Integration:** Remove or mark sponsor segments in YouTube videos.
*   **Plugin Support:** Extend functionality with custom extractors and postprocessors.
*   **Cross-Platform:** Works on Windows, Linux, and macOS.
*   **Self-Updating:** Easily update to the latest version with a single command.

## Table of Contents

*   [Installation](#installation)
*   [Usage and Options](#usage-and-options)
*   [Configuration](#configuration)
*   [Output Template](#output-template)
*   [Format Selection](#format-selection)
*   [Modifying Metadata](#modifying-metadata)
*   [Extractor Arguments](#extractor-arguments)
*   [Plugins](#plugins)
*   [Embedding yt-dlp](#embedding-yt-dlp)
*   [Changes from youtube-dl](#changes-from-youtube-dl)
*   [Contributing](#contributing)
*   [Wiki](#wiki)

## Installation

You can install yt-dlp using [the binaries](#release-files), [pip](https://pypi.org/project/yt-dlp) or one using a third-party package manager. See [the wiki](https://github.com/yt-dlp/yt-dlp/wiki/Installation) for detailed instructions

### Release Files

#### Recommended

File|Description
:---|:---
[yt-dlp](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp)|Platform-independent [zipimport](https://docs.python.org/3/library/zipimport.html) binary. Needs Python (recommended for **Linux/BSD**)
[yt-dlp.exe](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe)|Windows (Win8+) standalone x64 binary (recommended for **Windows**)
[yt-dlp_macos](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos)|Universal MacOS (10.15+) standalone executable (recommended for **MacOS**)

#### Alternatives

File|Description
:---|:---
[yt-dlp_x86.exe](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_x86.exe)|Windows (Win8+) standalone x86 (32-bit) binary
[yt-dlp_linux](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux)|Linux standalone x64 binary
[yt-dlp_linux_armv7l](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux_armv7l)|Linux standalone armv7l (32-bit) binary
[yt-dlp_linux_aarch64](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux_aarch64)|Linux standalone aarch64 (64-bit) binary
[yt-dlp_win.zip](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_win.zip)|Unpackaged Windows executable (no auto-update)
[yt-dlp_macos.zip](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos.zip)|Unpackaged MacOS (10.15+) executable (no auto-update)
[yt-dlp_macos_legacy](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos_legacy)|MacOS (10.9+) standalone x64 executable

#### Misc

File|Description
:---|:---
[yt-dlp.tar.gz](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.tar.gz)|Source tarball
[SHA2-512SUMS](https://github.com/yt-dlp/yt-dlp/releases/latest/download/SHA2-512SUMS)|GNU-style SHA512 sums
[SHA2-512SUMS.sig](https://github.com/yt-dlp/yt-dlp/releases/latest/download/SHA2-512SUMS.sig)|GPG signature file for SHA512 sums
[SHA2-256SUMS](https://github.com/yt-dlp/yt-dlp/releases/latest/download/SHA2-256SUMS)|GNU-style SHA256 sums
[SHA2-256SUMS.sig](https://github.com/yt-dlp/yt-dlp/releases/latest/download/SHA2-256SUMS.sig)|GPG signature file for SHA256 sums

The public key that can be used to verify the GPG signatures is [available here](https://github.com/yt-dlp/yt-dlp/blob/master/public.key)
Example usage:
```
curl -L https://github.com/yt-dlp/yt-dlp/raw/master/public.key | gpg --import
gpg --verify SHA2-256SUMS.sig SHA2-256SUMS
gpg --verify SHA2-512SUMS.sig SHA2-512SUMS
```
<!-- MANPAGE: END EXCLUDED SECTION -->

**Note**: The manpages, shell completion (autocomplete) files etc. are available inside the [source tarball](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.tar.gz)

## UPDATE

You can use `yt-dlp -U` to update if you are using the [release binaries](#release-files)

If you [installed with pip](https://github.com/yt-dlp/yt-dlp/wiki/Installation#with-pip), simply re-run the same command that was used to install the program

For other third-party package managers, see [the wiki](https://github.com/yt-dlp/yt-dlp/wiki/Installation#third-party-package-managers) or refer to their documentation

<a id="update-channels"></a>

There are currently three release channels for binaries: `stable`, `nightly` and `master`.

* `stable` is the default channel, and many of its changes have been tested by users of the `nightly` and `master` channels.
* The `nightly` channel has releases scheduled to build every day around midnight UTC, for a snapshot of the project's new patches and changes. This is the **recommended channel for regular users** of yt-dlp. The `nightly` releases are available from [yt-dlp/yt-dlp-nightly-builds](https://github.com/yt-dlp/yt-dlp-nightly-builds/releases) or as development releases of the `yt-dlp` PyPI package (which can be installed with pip's `--pre` flag).
* The `master` channel features releases that are built after each push to the master branch, and these will have the very latest fixes and additions, but may also be more prone to regressions. They are available from [yt-dlp/yt-dlp-master-builds](https://github.com/yt-dlp/yt-dlp-master-builds/releases).

When using `--update`/`-U`, a release binary will only update to its current channel.
`--update-to CHANNEL` can be used to switch to a different channel when a newer version is available. `--update-to [CHANNEL@]TAG` can also be used to upgrade or downgrade to specific tags from a channel.

You may also use `--update-to <repository>` (`<owner>/<repository>`) to update to a channel on a completely different repository. Be careful with what repository you are updating to though, there is no verification done for binaries from different repositories.

Example usage:

* `yt-dlp --update-to master` switch to the `master` channel and update to its latest release
* `yt-dlp --update-to stable@2023.07.06` upgrade/downgrade to release to `stable` channel tag `2023.07.06`
* `yt-dlp --update-to 2023.10.07` upgrade/downgrade to tag `2023.10.07` if it exists on the current channel
* `yt-dlp --update-to example/yt-dlp@2023.09.24` upgrade/downgrade to the release from the `example/yt-dlp` repository, tag `2023.09.24`

**Important**: Any user experiencing an issue with the `stable` release should install or update to the `nightly` release before submitting a bug report:
```
# To update to nightly from stable executable/binary:
yt-dlp --update-to nightly

# To install nightly with pip:
python3 -m pip install -U --pre "yt-dlp[default]"
```

## DEPENDENCIES
Python versions 3.9+ (CPython) and 3.10+ (PyPy) are supported. Other versions and implementations may or may not work correctly.

<!-- Python 3.5+ uses VC++14 and it is already embedded in the binary created
<!x-- https://www.microsoft.com/en-us/download/details.aspx?id=26999 --x>
On Windows, [Microsoft Visual C++ 2010 SP1 Redistributable Package (x86)](https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x86.exe) is also necessary to run yt-dlp. You probably already have this, but if the executable throws an error due to missing `MSVCR100.dll` you need to install it manually.
-->

While all the other dependencies are optional, `ffmpeg` and `ffprobe` are highly recommended

### Strongly recommended

* [**ffmpeg** and **ffprobe**](https://www.ffmpeg.org) - Required for [merging separate video and audio files](#format-selection), as well as for various [post-processing](#post-processing-options) tasks. License [depends on the build](https://www.ffmpeg.org/legal.html)

    There are bugs in ffmpeg that cause various issues when used alongside yt-dlp. Since ffmpeg is such an important dependency, we provide [custom builds](https://github.com/yt-dlp/FFmpeg-Builds#ffmpeg-static-auto-builds) with patches for some of these issues at [yt-dlp/FFmpeg-Builds](https://github.com/yt-dlp/FFmpeg-Builds). See [the readme](https://github.com/yt-dlp/FFmpeg-Builds#patches-applied) for details on the specific issues solved by these builds

    **Important**: What you need is ffmpeg *binary*, **NOT** [the Python package of the same name](https://pypi.org/project/ffmpeg)

### Networking
* [**certifi**](https://github.com/certifi/python-certifi)\* - Provides Mozilla's root certificate bundle. Licensed under [MPLv2](https://github.com/certifi/python-certifi/blob/master/LICENSE)
* [**brotli**](https://github.com/google/brotli)\* or [**brotlicffi**](https://github.com/python-hyper/brotlicffi) - [Brotli](https://en.wikipedia.org/wiki/Brotli) content encoding support. Both licensed under MIT <sup>[1](https://github.com/google/brotli/blob/master/LICENSE) [2](https://github.com/python-hyper/brotlicffi/blob/master/LICENSE) </sup>
* [**websockets**](https://github.com/aaugustin/websockets)\* - For downloading over websocket. Licensed under [BSD-3-Clause](https://github.com/aaugustin/websockets/blob/main/LICENSE)
* [**requests**](https://github.com/psf/requests)\* - HTTP library. For HTTPS proxy and persistent connections support. Licensed under [Apache-2.0](https://github.com/psf/requests/blob/main/LICENSE)

#### Impersonation

The following provide support for impersonating browser requests. This may be required for some sites that employ TLS fingerprinting.

* [**curl_cffi**](https://github.com/lexiforest/curl_cffi) (recommended) - Python binding for [curl-impersonate](https://github.com/lexiforest/curl-impersonate). Provides impersonation targets for Chrome, Edge and Safari. Licensed under [MIT](https://github.com/lexiforest/curl_cffi/blob/main/LICENSE)
  * Can be installed with the `curl-cffi` group, e.g. `pip install "yt-dlp[default,curl-cffi]"`
  * Currently included in `yt-dlp.exe`, `yt-dlp_linux` and `yt-dlp_macos` builds


### Metadata

* [**mutagen**](https://github.com/quodlibet/mutagen)\* - For `--embed-thumbnail` in certain formats. Licensed under [GPLv2+](https://github.com/quodlibet/mutagen/blob/master/COPYING)
* [**AtomicParsley**](https://github.com/wez/atomicparsley) - For `--embed-thumbnail` in `mp4`/`m4a` files when `mutagen`/`ffmpeg` cannot. Licensed under [GPLv2+](https://github.com/wez/atomicparsley/blob/master/COPYING)
* [**xattr**](https://github.com/xattr/xattr), [**pyxattr**](https://github.com/iustin/pyxattr) or [**setfattr**](http://savannah.nongnu.org/projects/attr) - For writing xattr metadata (`--xattr`) on **Mac** and **BSD**. Licensed under [MIT](https://github.com/xattr/xattr/blob/master/LICENSE.txt), [LGPL2.1](https://github.com/iustin/pyxattr/blob/master/COPYING) and [GPLv2+](http://git.savannah.nongnu.org/cgit/attr.git/tree/doc/COPYING) respectively

### Misc

* [**pycryptodomex**](https://github.com/Legrandin/pycryptodome)\* - For decrypting AES-128 HLS streams and various other data. Licensed under [BSD-2-Clause](https://github.com/Legrandin/pycryptodome/blob/master/LICENSE.rst)
* [**phantomjs**](https://github.com/ariya/phantomjs) - Used in extractors where javascript needs to be run. Licensed under [BSD-3-Clause](https://github.com/ariya/phantomjs/blob/master/LICENSE.BSD)
* [**secretstorage**](https://github.com/mitya57/secretstorage)\* - For `--cookies-from-browser` to access the **Gnome** keyring while decrypting cookies of **Chromium**-based browsers on **Linux**. Licensed under [BSD-3-Clause](https://github.com/mitya57/secretstorage/blob/master/LICENSE)
* Any external downloader that you want to use with `--downloader`

### Deprecated

* [**avconv** and **avprobe**](https://www.libav.org) - Now **deprecated** alternative to ffmpeg. License [depends on the build](https://libav.org/legal)
* [**sponskrub**](https://github.com/faissaloo/SponSkrub) - For using the now **deprecated** [sponskrub options](#sponskrub-options). Licensed under [GPLv3+](https://github.com/faissaloo/SponSkrub/blob/master/LICENCE.md)
* [**rtmpdump**](http://rtmpdump.mplayerhq.hu) - For downloading `rtmp` streams. ffmpeg can be used instead with `--downloader ffmpeg`. Licensed under [GPLv2+](http://rtmpdump.mplayerhq.hu)
* [**mplayer**](http://mplayerhq.hu/design7/info.html) or [**mpv**](https://mpv.io) - For downloading `rstp`/`mms` streams. ffmpeg can be used instead with `--downloader ffmpeg`. Licensed under [GPLv2+](https://github.com/mpv-player/mpv/blob/master/Copyright)

To use or redistribute the dependencies, you must agree to their respective licensing terms.

The standalone release binaries are built with the Python interpreter and the packages marked with **\*** included.

If you do not have the necessary dependencies for a task you are attempting, yt-dlp will warn you. All the currently available dependencies are visible at the top of the `--verbose` output

## USAGE AND OPTIONS

```
yt-dlp [OPTIONS] [--] URL [URL...]
```

For a full list of options, run `yt-dlp --help`.

## Configuration

yt-dlp supports configuration files for customizing your downloads. You can specify options in a configuration file, loaded from:

1.  **Main Configuration**: Specified by `--config-location`
2.  **Portable Configuration**: `yt-dlp.conf` in the same directory as the binary.
3.  **Home Configuration**: `yt-dlp.conf` in your home directory.
4.  **User Configuration**: `${XDG_CONFIG_HOME}/yt-dlp.conf`,  `${XDG_CONFIG_HOME}/yt-dlp/config` (recommended on Linux/macOS), `${XDG_CONFIG_HOME}/yt-dlp/config.txt`, `${APPDATA}/yt-dlp.conf`, `${APPDATA}/yt-dlp/config` (recommended on Windows), `${APPDATA}/yt-dlp/config.txt`, `~/yt-dlp.conf`, `~/yt-dlp.conf.txt`, `~/.yt-dlp/config`, `~/.yt-dlp/config.txt`
5.  **System Configuration**: `/etc/yt-dlp.conf`, `/etc/yt-dlp/config`, `/etc/yt-dlp/config.txt`

## OUTPUT TEMPLATE

Use the `-o` option to define the output filename template. The output template allows you to control the naming and organization of your downloaded files.

**Output Template Field Examples:**

*   `%(title)s`: Video title
*   `%(id)s`: Video ID
*   `%(ext)s`: File extension
*   `%(upload_date>%Y-%m-%d)s`:  Upload date formatted as YYYY-MM-DD
*   `%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s`: Playlist downloads in a directory structure

## FORMAT SELECTION

Use the `-f` option to select the desired video format.

**Format Selection Options:**

*   `-f best`: Download the best available quality (video and audio).
*   `-f bestvideo`: Download the best video-only format.
*   `-f bestaudio`: Download the best audio-only format.
*   `-f <format_code>`:  Specify a format code (use `-F` to list available codes).
*   `-f <format1>+<format2>`: Merge video and audio formats (requires ffmpeg).
*   `-S <sort_criteria>`: Sort formats based on specified criteria (e.g., `-S "res,fps"`).

## MODIFYING METADATA

Use the `--parse-metadata` and `--replace-in-metadata` to modify the metadata.

*   `--parse-metadata "<field_to_parse>:<extraction_pattern>"`: Extract information from a field and transform or set it as a different field.
*   `--replace-in-metadata "<fields> <regex> <replacement>"`:  Search and replace text in metadata fields.

## EXTRACTOR ARGUMENTS

Some extractors support additional arguments that can be passed using `--extractor-args KEY:ARGS`.

## PLUGINS

yt-dlp supports plugins to extend functionality. Place plugin files in the correct locations, as described in [Installing Plugins](#installing-plugins).  Then you can use `--use-postprocessor NAME` to use postprocessor plugins.

## EMBEDDING YT-DLP

yt-dlp can be easily embedded in your Python programs:

```python
from yt_dlp import YoutubeDL

ydl_opts = {
    'outtmpl': '%(title)s-%(id)s.%(ext)s',
}
with YoutubeDL(ydl_opts) as ydl:
    ydl.download(['VIDEO_URL'])
```

## CHANGES FROM YOUTUBE-DL

yt-dlp is actively maintained fork of youtube-dl, incorporating improvements and features not available in the original.
Key changes from youtube-dl include:

*   Improved Support and Maintenance
*   SponsorBlock Integration
*   Enhanced Format Sorting
*   New and Updated Extractors
*   Cookie from Browser
*   Download Time Range
*   Many other new features and improvements

See the [Changes from youtube-dl](#changes-from-youtube-dl) section for a more detailed overview.

## CONTRIBUTING

Contribute to the project by reporting issues, suggesting features, or submitting code.  See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## WIKI

Access the [Wiki](https://github.com/yt-dlp/yt-dlp/wiki) for additional information and resources.