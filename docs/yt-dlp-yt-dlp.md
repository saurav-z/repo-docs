<div align="center">

[![YT-DLP](https://raw.githubusercontent.com/yt-dlp/yt-dlp/master/.github/banner.svg)](#introduction)

[![Release version](https://img.shields.io/github/v/release/yt-dlp/yt-dlp?color=brightgreen&label=Download&style=for-the-badge)](#installation)
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

yt-dlp is a feature-rich command-line program that can download audio and video from thousands of sites.  Forked from youtube-dl and built on the foundation of yt-dlc, yt-dlp is a versatile tool for all your media downloading needs.  Visit the [original repository](https://github.com/yt-dlp/yt-dlp) for more information.

## Key Features

*   **Broad Site Support**: Download from a vast library of [supported sites](supportedsites.md).
*   **Format Selection**:  Choose from a variety of video and audio formats with advanced filtering and sorting options.
*   **Playlist and Channel Downloads**: Easily download entire playlists and channels.
*   **Subtitle and Thumbnail Support**: Download subtitles and thumbnails.
*   **Metadata Embedding**: Embed metadata, including thumbnails and chapters, into your downloaded files.
*   **Post-Processing**: Convert videos to audio, remux, or recode files with ffmpeg.
*   **Customization**: Use plugins, output templates, and extractor arguments for flexibility.
*   **SponsorBlock Integration**: Remove sponsor segments from YouTube videos using the SponsorBlock API.
*   **Active Development**: Benefit from ongoing updates and new features.

## Installation

yt-dlp can be installed using various methods.  See [the wiki](https://github.com/yt-dlp/yt-dlp/wiki/Installation) for detailed installation instructions.

*   **Binaries**: Download pre-built binaries for [Windows](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe), [Linux/BSD](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp), and [MacOS](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos).
*   **pip**: Install via `pip install yt-dlp`.
*   **Package Managers**: Use third-party package managers for your operating system.

### Release Files

*   **Recommended**
    *   [yt-dlp](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp) (Linux/BSD): Platform-independent [zipimport](https://docs.python.org/3/library/zipimport.html) binary.  Needs Python.
    *   [yt-dlp.exe](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe) (Windows): Windows (Win8+) standalone x64 binary.
    *   [yt-dlp_macos](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos) (MacOS): Universal MacOS (10.15+) standalone executable.
*   **Alternatives**
    *   [yt-dlp_x86.exe](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_x86.exe): Windows (Win8+) standalone x86 (32-bit) binary
    *   [yt-dlp_linux](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux): Linux standalone x64 binary
    *   [yt-dlp_linux_armv7l](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux_armv7l): Linux standalone armv7l (32-bit) binary
    *   [yt-dlp_linux_aarch64](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux_aarch64): Linux standalone aarch64 (64-bit) binary
    *   [yt-dlp_win.zip](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_win.zip): Unpackaged Windows executable (no auto-update)
    *   [yt-dlp_macos.zip](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos.zip): Unpackaged MacOS (10.15+) executable (no auto-update)
    *   [yt-dlp_macos_legacy](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos_legacy): MacOS (10.9+) standalone x64 executable

*   **Misc**
    *   [yt-dlp.tar.gz](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.tar.gz): Source tarball
    *   [SHA2-512SUMS](https://github.com/yt-dlp/yt-dlp/releases/latest/download/SHA2-512SUMS): GNU-style SHA512 sums
    *   [SHA2-512SUMS.sig](https://github.com/yt-dlp/yt-dlp/releases/latest/download/SHA2-512SUMS.sig): GPG signature file for SHA512 sums
    *   [SHA2-256SUMS](https://github.com/yt-dlp/yt-dlp/releases/latest/download/SHA2-256SUMS): GNU-style SHA256 sums
    *   [SHA2-256SUMS.sig](https://github.com/yt-dlp/yt-dlp/releases/latest/download/SHA2-256SUMS.sig): GPG signature file for SHA256 sums

The public key that can be used to verify the GPG signatures is [available here](https://github.com/yt-dlp/yt-dlp/blob/master/public.key)

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

**Note**: The manpages, shell completion (autocomplete) files etc. are available inside the [source tarball](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.tar.gz)

## Update

*   Use `yt-dlp -U` to update the program.
*   If you installed with `pip`, re-run the installation command.
*   See the [wiki](https://github.com/yt-dlp/yt-dlp/wiki/Installation#third-party-package-managers) for third-party package manager updates.

**Channels:**

*   `stable`:  The default channel.
*   `nightly`:  Releases built daily, recommended for most users.  Available at [yt-dlp/yt-dlp-nightly-builds](https://github.com/yt-dlp/yt-dlp-nightly-builds/releases) or as development releases of the `yt-dlp` PyPI package (which can be installed with pip's `--pre` flag).
*   `master`:  Releases built after each commit to the master branch. Available at [yt-dlp/yt-dlp-master-builds](https://github.com/yt-dlp/yt-dlp-master-builds/releases).

Use `--update-to CHANNEL[@TAG]` to upgrade/downgrade.

**Important:** Report issues with the `stable` release to the `nightly` release first.

```bash
# To update to nightly from stable executable/binary:
yt-dlp --update-to nightly

# To install nightly with pip:
python3 -m pip install -U --pre "yt-dlp[default]"
```

## Dependencies

*   **Python:** Requires Python 3.9+ or PyPy 3.10+.
*   **Recommended**: `ffmpeg` and `ffprobe` (highly recommended for merging, post-processing). See [yt-dlp/FFmpeg-Builds](https://github.com/yt-dlp/FFmpeg-Builds#ffmpeg-static-auto-builds) for custom builds with patches for some issues.

*   **Networking**
    *   [**certifi**](https://github.com/certifi/python-certifi)
    *   [**brotli**](https://github.com/google/brotli) or [**brotlicffi**](https://github.com/python-hyper/brotlicffi)
    *   [**websockets**](https://github.com/aaugustin/websockets)
    *   [**requests**](https://github.com/psf/requests)
*   **Impersonation**
    *   [**curl_cffi**](https://github.com/lexiforest/curl_cffi) (recommended)
*   **Metadata**
    *   [**mutagen**](https://github.com/quodlibet/mutagen)
    *   [**AtomicParsley**](https://github.com/wez/atomicparsley)
    *   [**xattr**](https://github.com/xattr/xattr), [**pyxattr**](https://github.com/iustin/pyxattr) or [**setfattr**](http://savannah.nongnu.org/projects/attr)
*   **Misc**
    *   [**pycryptodomex**](https://github.com/Legrandin/pycryptodome)
    *   [**phantomjs**](https://github.com/ariya/phantomjs)
    *   [**secretstorage**](https://github.com/mitya57/secretstorage)
    *   Any external downloader that you want to use with `--downloader`
*   **Deprecated**
    *   [**avconv** and **avprobe**](https://www.libav.org)
    *   [**sponskrub**](https://github.com/faissaloo/SponSkrub)
    *   [**rtmpdump**](http://rtmpdump.mplayerhq.hu)
    *   [**mplayer**](http://mplayerhq.hu/design7/info.html) or [**mpv**](https://mpv.io)

## Contribution

See [CONTRIBUTING.md](CONTRIBUTING.md#contributing-to-yt-dlp) for information on how to contribute to the project.

## Documentation

*   [Usage and Options](https://github.com/yt-dlp/yt-dlp/blob/master/README.md#usage-and-options)
*   [Wiki](https://github.com/yt-dlp/yt-dlp/wiki)