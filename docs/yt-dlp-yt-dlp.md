# yt-dlp: The Ultimate Command-Line Video and Audio Downloader

Easily download your favorite videos and music from thousands of sites with yt-dlp, the powerful command-line tool that's a fork of youtube-dl.  [Explore the original repository](https://github.com/yt-dlp/yt-dlp).

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


**Key Features:**

*   **Wide Site Support:** Download from thousands of video and audio streaming sites ([See Supported Sites](supportedsites.md)).
*   **Format Selection:** Easily choose your preferred video and audio formats, including best quality.
*   **Playlist and Channel Downloads:** Download entire playlists and channels with ease.
*   **Subtitle Support:** Download and embed subtitles in various formats.
*   **Metadata Handling:** Add and modify video metadata for better organization.
*   **Post-Processing:** Convert videos to audio, embed thumbnails, and more using FFmpeg.
*   **SponsorBlock Integration:** Automatically skip sponsored segments in YouTube videos.
*   **Browser Cookie Import:**  Import cookies from your browser for authenticated downloads.
*   **Customizable Output:**  Flexible output templates allow for customized file naming and organization.
*   **Regular Updates:** Stay up-to-date with frequent updates and new feature additions.

## Installation

yt-dlp can be installed via binaries, pip, or a third-party package manager. See the [wiki](https://github.com/yt-dlp/yt-dlp/wiki/Installation) for detailed instructions.

### Installation Options:

*   **Binaries**: Download platform-specific executables directly for Windows, macOS, and Linux.
*   **PyPI**: Install using `pip install yt-dlp`.
*   **Package Managers**: Install via various third-party package managers.

### Release Files:

#### Recommended
| File | Description |
|---|---|
| [yt-dlp](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp) | Platform-independent zipimport binary. Needs Python (recommended for **Linux/BSD**) |
| [yt-dlp.exe](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe) | Windows (Win8+) standalone x64 binary (recommended for **Windows**) |
| [yt-dlp_macos](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos) | Universal MacOS (10.15+) standalone executable (recommended for **MacOS**) |
#### Alternatives
| File | Description |
|---|---|
| [yt-dlp_x86.exe](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_x86.exe) | Windows (Win8+) standalone x86 (32-bit) binary |
| [yt-dlp_linux](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux) | Linux standalone x64 binary |
| [yt-dlp_linux_armv7l](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux_armv7l) | Linux standalone armv7l (32-bit) binary |
| [yt-dlp_linux_aarch64](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux_aarch64) | Linux standalone aarch64 (64-bit) binary |
| [yt-dlp_win.zip](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_win.zip) | Unpackaged Windows executable (no auto-update) |
| [yt-dlp_macos.zip](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos.zip) | Unpackaged MacOS (10.15+) executable (no auto-update) |
| [yt-dlp_macos_legacy](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos_legacy) | MacOS (10.9+) standalone x64 executable |
#### Misc
| File | Description |
|---|---|
| [yt-dlp.tar.gz](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.tar.gz) | Source tarball |
| [SHA2-512SUMS](https://github.com/yt-dlp/yt-dlp/releases/latest/download/SHA2-512SUMS) | GNU-style SHA512 sums |
| [SHA2-512SUMS.sig](https://github.com/yt-dlp/yt-dlp/releases/latest/download/SHA2-512SUMS.sig) | GPG signature file for SHA512 sums |
| [SHA2-256SUMS](https://github.com/yt-dlp/yt-dlp/releases/latest/download/SHA2-256SUMS) | GNU-style SHA256 sums |
| [SHA2-256SUMS.sig](https://github.com/yt-dlp/yt-dlp/releases/latest/download/SHA2-256SUMS.sig) | GPG signature file for SHA256 sums |

The public key that can be used to verify the GPG signatures is [available here](https://github.com/yt-dlp/yt-dlp/blob/master/public.key)
Example usage:
```
curl -L https://github.com/yt-dlp/yt-dlp/raw/master/public.key | gpg --import
gpg --verify SHA2-256SUMS.sig SHA2-256SUMS
gpg --verify SHA2-512SUMS.sig SHA2-512SUMS
```

**Note**: The manpages, shell completion (autocomplete) files etc. are available inside the [source tarball](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.tar.gz)


## Updating yt-dlp
Keep yt-dlp up-to-date with the latest features and fixes!  Use `yt-dlp -U` to update if you're using the release binaries. If you installed with `pip`, simply re-run the same command used to install.

## Usage and Options

Explore a comprehensive list of [yt-dlp's command-line options in the documentation](https://github.com/yt-dlp/yt-dlp#usage-and-options).