<div align="center">

[![YT-DLP](https://raw.githubusercontent.com/yt-dlp/yt-dlp/master/.github/banner.svg)](#yt-dlp-audio-video-downloader)

[![Release version](https://img.shields.io/github/v/release/yt-dlp/yt-dlp?color=brightgreen&label=Download&style=for-the-badge)](#installation)
[![PyPI](https://img.shields.io/badge/-PyPI-blue.svg?logo=pypi&labelColor=555555&style=for-the-badge)](https://pypi.org/project/yt-dlp)
[![Donate](https://img.shields.io/badge/_-Donate-red.svg?logo=githubsponsors&labelColor=555555&style=for-the-badge)](Collaborators.md#collaborators)
[![Discord](https://img.shields.io/discord/807245652072857610?color=blue&labelColor=555555&label=&logo=discord&style=for-the-badge)](https://discord.gg/H5MNcFW63r)
[![Supported Sites](https://img.shields.io/badge/-Supported_Sites-brightgreen.svg?style=for-the-badge)](supportedsites.md)
[![License: Unlicense](https://img.shields.io/badge/-Unlicense-blue.svg?style=for-the-badge)](LICENSE)
[![CI Status](https://img.shields.io/github/actions/workflow/status/yt-dlp/yt-dlp/core.yml?branch=master&label=Tests&style=for-the-badge)](https://github.com/yt-dlp/yt-dlp/actions)
[![Commits](https://img.shields.io/github/commit-activity/m/yt-dlp/yt-dlp?label=commits&style=for-the-badge)](https://github.com/yt-dlp/yt-dlp/commits)
[![Last Commit](https://img.shields.io/github/last-commit/yt-dlp/yt-dlp/master?label=&style=for-the-badge&display_timestamp=committer)](https://github.com/yt-dlp/yt-dlp/pulse/monthly)

</div>

## yt-dlp: Your Go-To Command-Line Audio/Video Downloader

yt-dlp is a powerful command-line tool for downloading audio and video from [thousands of websites](supportedsites.md), offering unparalleled flexibility and features.  Find the original repo [here](https://github.com/yt-dlp/yt-dlp).

**Key Features:**

*   **Extensive Site Support:**  Download from thousands of video and audio sites.
*   **Format Selection:** Fine-tune your downloads with advanced format options and sorting.
*   **Playlist Support:** Download entire playlists with ease, and customize playlist items.
*   **Subtitle & Thumbnail Handling:** Download and embed subtitles, and extract thumbnails.
*   **Post-Processing:** Convert, remux, and embed metadata with robust post-processing options.
*   **Metadata Manipulation:**  Modify video metadata before or after download for customization.
*   **SponsorBlock Integration:**  Automatically remove sponsor segments from YouTube videos, utilizing [SponsorBlock](https://sponsor.ajay.app).
*   **Browser Cookie Support:** Download by using the existing cookies from all major web browsers.
*   **Plugins:** Extend functionality with custom extractors and post-processors.
*   **Self-Updating:**  Keep up-to-date with easy built-in update functionality.

**Table of Contents:**

*   [Installation](#installation)
    *   [Detailed instructions](https://github.com/yt-dlp/yt-dlp/wiki/Installation)
    *   [Release Files](#release-files)
    *   [Update](#update)
    *   [Dependencies](#dependencies)
    *   [Compile](#compile)
*   [USAGE AND OPTIONS](#usage-and-options)
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
*   [CONFIGURATION](#configuration)
    *   [Configuration file encoding](#configuration-file-encoding)
    *   [Authentication with netrc](#authentication-with-netrc)
    *   [Notes about environment variables](#notes-about-environment-variables)
*   [OUTPUT TEMPLATE](#output-template)
    *   [Output template examples](#output-template-examples)
*   [FORMAT SELECTION](#format-selection)
    *   [Filtering Formats](#filtering-formats)
    *   [Sorting Formats](#sorting-formats)
    *   [Format Selection examples](#format-selection-examples)
*   [MODIFYING METADATA](#modifying-metadata)
    *   [Modifying metadata examples](#modifying-metadata-examples)
*   [EXTRACTOR ARGUMENTS](#extractor-arguments)
*   [PLUGINS](#plugins)
    *   [Installing Plugins](#installing-plugins)
    *   [Developing Plugins](#developing-plugins)
*   [EMBEDDING YT-DLP](#embedding-yt-dlp)
    *   [Embedding examples](#embedding-examples)
*   [CHANGES FROM YOUTUBE-DL](#changes-from-youtube-dl)
    *   [New features](#new-features)
    *   [Differences in default behavior](#differences-in-default-behavior)
    *   [Deprecated options](#deprecated-options)
*   [CONTRIBUTING](CONTRIBUTING.md#contributing-to-yt-dlp)
    *   [Opening an Issue](CONTRIBUTING.md#opening-an-issue)
    *   [Developer Instructions](CONTRIBUTING.md#developer-instructions)
*   [WIKI](https://github.com/yt-dlp/yt-dlp/wiki)
    *   [FAQ](https://github.com/yt-dlp/yt-dlp/wiki/FAQ)

## Installation

Choose your preferred installation method for yt-dlp:

<!-- MANPAGE: BEGIN EXCLUDED SECTION -->
[![Windows](https://img.shields.io/badge/-Windows_x64-blue.svg?style=for-the-badge&logo=windows)](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe)
[![Unix](https://img.shields.io/badge/-Linux/BSD-red.svg?style=for-the-badge&logo=linux)](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp)
[![MacOS](https://img.shields.io/badge/-MacOS-lightblue.svg?style=for-the-badge&logo=apple)](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos)
[![PyPI](https://img.shields.io/badge/-PyPI-blue.svg?logo=pypi&labelColor=555555&style=for-the-badge)](https://pypi.org/project/yt-dlp)
[![Source Tarball](https://img.shields.io/badge/-Source_tar-green.svg?style=for-the-badge)](https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.tar.gz)
[![Other variants](https://img.shields.io/badge/-Other-grey.svg?style=for-the-badge)](#release-files)
[![All versions](https://img.shields.io/badge/-All_Versions-lightgrey.svg?style=for-the-badge)](https://github.com/yt-dlp/yt-dlp/releases)
<!-- MANPAGE: END EXCLUDED SECTION -->

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

*   [**ffmpeg** and **ffprobe**](https://www.ffmpeg.org) - Required for [merging separate video and audio files](#format-selection), as well as for various [post-processing](#post-processing-options) tasks. License [depends on the build](https://www.ffmpeg.org/legal.html)

    There are bugs in ffmpeg that cause various issues when used alongside yt-dlp. Since ffmpeg is such an important dependency, we provide [custom builds](https://github.com/yt-dlp/FFmpeg-Builds#ffmpeg-static-auto-builds) with patches for some of these issues at [yt-dlp/FFmpeg-Builds](https://github.com/yt-dlp/FFmpeg-Builds). See [the readme](https://github.com/yt-dlp/FFmpeg-Builds#patches-applied) for details on the specific issues solved by these builds

    **Important**: What you need is ffmpeg *binary*, **NOT** [the Python package of the same name](https://pypi.org/project/ffmpeg)

### Networking

*   [**certifi**](https://github.com/certifi/python-certifi)\* - Provides Mozilla's root certificate bundle. Licensed under [MPLv2](https://github.com/certifi/python-certifi/blob/master/LICENSE)
*   [**brotli**](https://github.com/google/brotli)\* or [**brotlicffi**](https://github.com/python-hyper/brotlicffi) - [Brotli](https://en.wikipedia.org/wiki/Brotli) content encoding support. Both licensed under MIT <sup>[1](https://github.com/google/brotli/blob/master/LICENSE) [2](https://github.com/python-hyper/brotlicffi/blob/master/LICENSE) </sup>
*   [**websockets**](https://github.com/aaugustin/websockets)\* - For downloading over websocket. Licensed under [BSD-3-Clause](https://github.com/aaugustin/websockets/blob/main/LICENSE)
*   [**requests**](https://github.com/psf/requests)\* - HTTP library. For HTTPS proxy and persistent connections support. Licensed under [Apache-2.0](https://github.com/psf/requests/blob/main/LICENSE)

#### Impersonation

The following provide support for impersonating browser requests. This may be required for some sites that employ TLS fingerprinting.

*   [**curl_cffi**](https://github.com/lexiforest/curl_cffi) (recommended) - Python binding for [curl-impersonate](https://github.com/lexiforest/curl_cffi). Provides impersonation targets for Chrome, Edge and Safari. Licensed under [MIT](https://github.com/lexiforest/curl_cffi/blob/main/LICENSE)
    *   Can be installed with the `curl-cffi` group, e.g. `pip install "yt-dlp[default,curl-cffi]"`
    *   Currently included in `yt-dlp.exe`, `yt-dlp_linux` and `yt-dlp_macos` builds

### Metadata

*   [**mutagen**](https://github.com/quodlibet/mutagen)\* - For `--embed-thumbnail` in certain formats. Licensed under [GPLv2+](https://github.com/quodlibet/mutagen/blob/master/COPYING)
*   [**AtomicParsley**](https://github.com/wez/atomicparsley) - For `--embed-thumbnail` in `mp4`/`m4a` files when `mutagen`/`ffmpeg` cannot. Licensed under [GPLv2+](https://github.com/wez/atomicparsley/blob/master/COPYING)
*   [**xattr**](https://github.com/xattr/xattr), [**pyxattr**](https://github.com/iustin/pyxattr) or [**setfattr**](http://savannah.nongnu.org/projects/attr) - For writing xattr metadata (`--xattr`) on **Mac** and **BSD**. Licensed under [MIT](https://github.com/xattr/xattr/blob/master/LICENSE.txt), [LGPL2.1](https://github.com/iustin/pyxattr/blob/master/COPYING) and [GPLv2+](http://git.savannah.nongnu.org/cgit/attr.git/tree/doc/COPYING) respectively

### Misc

*   [**pycryptodomex**](https://github.com/Legrandin/pycryptodome)\* - For decrypting AES-128 HLS streams and various other data. Licensed under [BSD-2-Clause](https://github.com/Legrandin/pycryptodome/blob/master/LICENSE.rst)
*   [**phantomjs**](https://github.com/ariya/phantomjs) - Used in extractors where javascript needs to be run. Licensed under [BSD-3-Clause](https://github.com/ariya/phantomjs/blob/master/LICENSE.BSD)
*   [**secretstorage**](https://github.com/mitya57/secretstorage)\* - For `--cookies-from-browser` to access the **Gnome** keyring while decrypting cookies of **Chromium**-based browsers on **Linux**. Licensed under [BSD-3-Clause](https://github.com/mitya57/secretstorage/blob/master/LICENSE)
*   Any external downloader that you want to use with `--downloader`

### Deprecated

*   [**avconv** and **avprobe**](https://www.libav.org) - Now **deprecated** alternative to ffmpeg. License [depends on the build](https://libav.org/legal)
*   [**sponskrub**](https://github.com/faissaloo/SponSkrub) - For using the now **deprecated** [sponskrub options](#sponskrub-options). Licensed under [GPLv3+](https://github.com/faissaloo/SponSkrub/blob/master/LICENCE.md)
*   [**rtmpdump**](http://rtmpdump.mplayerhq.hu) - For downloading `rtmp` streams. ffmpeg can be used instead with `--downloader ffmpeg`. Licensed under [GPLv2+](http://rtmpdump.mplayerhq.hu)
*   [**mplayer**](http://mplayerhq.hu/design7/info.html) or [**mpv**](https://mpv.io) - For downloading `rstp`/`mms` streams. ffmpeg can be used instead with `--downloader ffmpeg`. Licensed under [GPLv2+](https://github.com/mpv-player/mpv/blob/master/Copyright)

To use or redistribute the dependencies, you must agree to their respective licensing terms.

The standalone release binaries are built with the Python interpreter and the packages marked with **\*** included.

If you do not have the necessary dependencies for a task you are attempting, yt-dlp will warn you. All the currently available dependencies are visible at the top of the `--verbose` output

## COMPILE

### Standalone PyInstaller Builds
To build the standalone executable, you must have Python and `pyinstaller` (plus any of yt-dlp's [optional dependencies](#dependencies) if needed). The executable will be built for the same CPU architecture as the Python used.

You can run the following commands:

```
python3 devscripts/install_deps.py --include pyinstaller
python3 devscripts/make_lazy_extractors.py
python3 -m bundle.pyinstaller
```

On some systems, you may need to use `py` or `python` instead of `python3`.

`python -m bundle.pyinstaller` accepts any arguments that can be passed to `pyinstaller`, such as `--onefile/-F` or `--onedir/-D`, which is further [documented here](https://pyinstaller.org/en/stable/usage.html#what-to-generate).

**Note**: Pyinstaller versions below 4.4 [do not support](https://github.com/pyinstaller/pyinstaller#requirements-and-tested-platforms) Python installed from the Windows store without using a virtual environment.

**Important**: Running `pyinstaller` directly **instead of** using `python -m bundle.pyinstaller` is **not** officially supported. This may or may not work correctly.

### Platform-independent Binary (UNIX)
You will need the build tools `python` (3.9+), `zip`, `make` (GNU), `pandoc`\* and `pytest`\*.

After installing these, simply run `make`.

You can also run `make yt-dlp` instead to compile only the binary without updating any of the additional files. (The build tools marked with **\*** are not needed for this)

### Related scripts

*   **`devscripts/install_deps.py`** - Install dependencies for yt-dlp.
*   **`devscripts/update-version.py`** - Update the version number based on the current date.
*   **`devscripts/set-variant.py`** - Set the build variant of the executable.
*   **`devscripts/make_changelog.py`** - Create a markdown changelog using short commit messages and update `CONTRIBUTORS` file.
*   **`devscripts/make_lazy_extractors.py`** - Create lazy extractors. Running this before building the binaries (any variant) will improve their startup performance. Set the environment variable `YTDLP_NO_LAZY_EXTRACTORS` to something nonempty to forcefully disable lazy extractor loading.

Note: See their `--help` for more info.

### Forking the project
If you fork the project on GitHub, you can run your fork's [build workflow](.github/workflows/build.yml) to automatically build the selected version(s) as artifacts. Alternatively, you can run the [release workflow](.github/workflows/release.yml) or enable the [nightly workflow](.github/workflows/release-nightly.yml) to create full (pre-)releases.

<!-- MANPAGE: MOVE "USAGE AND OPTIONS" SECTION HERE -->

## USAGE AND OPTIONS

    yt-dlp [OPTIONS] [--] URL [URL...]

`Ctrl+F` is your friend :D

<!-- Auto generated -->
### General Options:

    -h, --help                      Print this help text and exit
    --version                       Print program version and exit
    -U, --update                    Update this program to the latest version
    --no-update                     Do not check for updates (default)
    --update-to [CHANNEL]@[TAG]     Upgrade/downgrade to a specific version.
                                    CHANNEL can be a repository as well. CHANNEL
                                    and TAG default to "stable" and "latest"
                                    respectively if omitted; See "UPDATE" for
                                    details. Supported channels: stable,
                                    nightly, master
    -i, --ignore-errors             Ignore download and postprocessing errors.
                                    The download will be considered successful
                                    even if the postprocessing fails
    --no-abort-on-error             Continue with next video on download errors;
                                    e.g. to skip unavailable videos in a
                                    playlist (default)
    --abort-on-error                Abort downloading of further videos if an
                                    error occurs (Alias: --no-ignore-errors)
    --dump-user-agent               Display the current user-agent and exit
    --list-extractors               List all supported extractors and exit
    --extractor-descriptions        Output descriptions of all supported
                                    extractors and exit
    --use-extractors NAMES          Extractor names to use separated by commas.
                                    You can also use regexes, "all", "default"
                                    and "end" (end URL matching); e.g. --ies
                                    "holodex.*,end,youtube". Prefix the name
                                    with a "-" to exclude it, e.g. --ies
                                    default,-generic. Use --list-extractors for
                                    a list of extractor names. (Alias: --ies)
    --default-search PREFIX         Use this prefix for unqualified URLs. E.g.
                                    "gvsearch2:python" downloads two videos from
                                    google videos for the search term "python".
                                    Use the value "auto" to let yt-dlp guess
                                    ("auto_warning" to emit a warning when
                                    guessing). "error" just throws an error. The
                                    default value "fixup_error" repairs broken
                                    URLs, but emits an error if this is not
                                    possible instead of searching
    --ignore-config                 Don't load any more configuration files
                                    except those given to --config-locations.
                                    For backward compatibility, if this option
                                    is found inside the system configuration
                                    file, the user configuration is not loaded.
                                    (Alias: --no-config)
    --no-config-locations           Do not load any custom configuration files
                                    (default). When given inside a configuration
                                    file, ignore all previous --config-locations
                                    defined in the current file
    --config-locations PATH         Location of the main configuration file;
                                    either the path to the config or its
                                    containing directory ("-" for stdin). Can be
                                    used multiple times and inside other
                                    configuration files
    --plugin-dirs PATH              Path to an additional directory to search
                                    for plugins. This option can be used
                                    multiple times to add multiple directories.
                                    Use "default" to search the default plugin
                                    directories (default)
    --no-plugin-dirs                Clear plugin directories to search,
                                    including defaults and those provided by
                                    previous --plugin-dirs
    --flat-playlist                 Do not extract a playlist's URL result
                                    entries; some entry metadata may be missing
                                    and downloading may be bypassed
    --no-flat-playlist              Fully extract the videos of a playlist
                                    (default)
    --live-from-start               Download livestreams from the start.
                                    Currently experimental and only supported
                                    for YouTube and Twitch
    --no-live-from-start            Download livestreams from the current time
                                    (default)
    --wait-for-video MIN[-MAX]      Wait for scheduled streams to become
                                    available. Pass the minimum number of
                                    seconds (or range) to wait between retries
    --no-wait-for-video             Do not wait for scheduled streams (default)
    --mark-watched                  Mark videos watched (even with --simulate)
    --no-mark-watched               Do not mark videos watched (default)
    --color [STREAM:]POLICY         Whether to emit color codes in output,
                                    optionally prefixed by the STREAM (stdout or
                                    stderr) to apply the setting to. Can be one
                                    of "always", "auto" (default), "never", or
                                    "no_color" (use non color terminal
                                    sequences). Use "auto-tty" or "no_color-tty"
                                    to decide based on terminal support only.
                                    Can be used multiple times
    --compat-options OPTS           Options that can help keep compatibility
                                    with youtube-dl or youtube-dlc
                                    configurations by reverting some of the
                                    changes made in yt-dlp. See "Differences in
                                    default behavior" for details
    --alias ALIASES OPTIONS         Create aliases for an option string. Unless
                                    an alias starts with a dash "-", it is
                                    prefixed with "--". Arguments are parsed
                                    according to the Python string formatting
                                    mini-language. E.g. --alias get-audio,-X "-S
                                    aext:{0},abr -x --audio-format {0}" creates
                                    options "--get-audio" and "-X" that takes an
                                    argument (ARG0) and expands to "-S
                                    aext:ARG0,abr -x --audio-format ARG0". All
                                    defined aliases are listed in the --help
                                    output. Alias options can trigger more
                                    aliases; so be careful to avoid defining
                                    recursive options. As a safety measure, each
                                    alias may be triggered a maximum of 100
                                    times. This option can be used multiple times
    -t, --preset-alias PRESET       Applies a predefined set of options. e.g.
                                    --preset-alias mp3. The following presets
                                    are available: mp3, aac, mp4, mkv, sleep.
                                    See the "Preset Aliases" section at the end
                                    for more info. This option can be used
                                    multiple times

### Network Options:

    --proxy URL                     Use the specified HTTP/HTTPS/SOCKS proxy. To
                                    enable SOCKS proxy, specify a proper scheme,
                                    e.g. socks5://user:pass@127.0.0.1:1080/.
                                    Pass in an empty string (--proxy "") for
                                    direct connection
    --socket-timeout SECONDS        Time to wait before giving up, in seconds
    --source-address IP             Client-side IP address to bind to
    --impersonate CLIENT[:OS]       Client to impersonate for requests. E.g.
                                    chrome, chrome-1