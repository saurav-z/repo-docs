<!-- MANPAGE: BEGIN EXCLUDED SECTION -->
<div align="center">

[![YT-DLP](https://raw.githubusercontent.com/yt-dlp/yt-dlp/master/.github/banner.svg)](#readme)

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
<!-- MANPAGE: END EXCLUDED SECTION -->

# yt-dlp: The Powerful Command-Line Video Downloader for Thousands of Sites

[yt-dlp](https://github.com/yt-dlp/yt-dlp) is a feature-rich, open-source command-line program for downloading audio and video from a vast range of websites. A fork of youtube-dl, it builds upon its predecessor with enhanced features, updated extractors, and improved performance.

**Key Features:**

*   **Extensive Site Support:** Download from thousands of websites ([Supported Sites](supportedsites.md)).
*   **Format Selection:** Flexible options for choosing video and audio formats, with advanced filtering and sorting.
*   **Playlist and Channel Downloads:** Easily download entire playlists and channels.
*   **Subtitle and Thumbnail Handling:** Download and embed subtitles; write thumbnails to disk.
*   **Metadata Management:** Modify and embed metadata, including title, description, and more.
*   **Post-Processing:** Convert videos to audio, remux into different containers, and add metadata.
*   **SponsorBlock Integration:** Automatically remove or mark sponsored segments in YouTube videos using the [SponsorBlock](https://sponsor.ajay.app) API.
*   **Browser Cookie Support:**  Import cookies from various web browsers for authenticated downloads.
*   **Plugin Support:** Extend functionality with custom extractor and post-processing plugins.
*   **Automatic Updates:** Stay up-to-date with the latest features and fixes.
*   **Cross-Platform:** Available for Windows, Linux, and macOS.

**Installation:**

You can install yt-dlp using [the binaries](#release-files), [pip](https://pypi.org/project/yt-dlp) or one using a third-party package manager. See [the wiki](https://github.com/yt-dlp/yt-dlp/wiki/Installation) for detailed instructions

<!-- MANPAGE: BEGIN EXCLUDED SECTION -->
## RELEASE FILES

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

**Update:**

Update yt-dlp easily using `yt-dlp -U`. See the [wiki](https://github.com/yt-dlp/yt-dlp/wiki/Installation) for further details on updating, including information on release channels.

*   [General Options](#general-options)
*   [Network Options](#network-options)
*   [Geo-restriction](#geo-restriction)
*   [Video Selection](#video-selection)
*   [Download Options](#download-options)
*   [Filesystem Options](#filesystem-options)
*   [Thumbnail Options](#thumbnail-options)
*   [Internet Shortcut Options](#internet-shortcut-options)
*   [Verbosity and Simulation Options](#verbosity-and-simulation-options)
*   [Workarounds](#workarounds)
*   [Video Format Options](#video-format-options)
*   [Subtitle Options](#subtitle-options)
*   [Authentication Options](#authentication-options)
*   [Post-processing Options](#post-processing-options)
*   [SponsorBlock Options](#sponsorblock-options)
*   [Extractor Options](#extractor-options)
*   [Preset Aliases](#preset-aliases)

*   [Configuration](#configuration)
*   [Output Template](#output-template)
*   [Format Selection](#format-selection)
*   [Modifying Metadata](#modifying-metadata)
*   [Extractor Arguments](#extractor-arguments)
*   [Plugins](#plugins)
*   [Embedding yt-dlp](#embedding-yt-dlp)
*   [Changes from youtube-dl](#changes-from-youtube-dl)
*   [Contributing](CONTRIBUTING.md#contributing-to-yt-dlp)
*   [Wiki](https://github.com/yt-dlp/yt-dlp/wiki)

```