# Buzz: Offline Audio Transcription and Translation Powered by Whisper

**Buzz** is a powerful, offline audio transcription and translation tool that leverages the cutting-edge [Whisper](https://github.com/openai/whisper) technology from OpenAI. Transcribe and translate your audio files directly on your computer without an internet connection, ensuring privacy and efficiency.

[**View the original Buzz repository on GitHub**](https://github.com/chidiwilliams/buzz)

**Key Features:**

*   **Offline Transcription & Translation:** Process audio files without needing an internet connection, ensuring privacy and faster results.
*   **Powered by OpenAI Whisper:** Utilizes the advanced speech recognition capabilities of Whisper for accurate transcriptions.
*   **Cross-Platform Compatibility:** Available for macOS, Windows, and Linux, offering flexibility across different operating systems.
*   **Multiple Installation Options:** Install via PyPI, Homebrew, Winget, Flatpak, or Snap for a seamless setup experience.
*   **GPU Support:** Take advantage of GPU acceleration on Windows for faster processing (with specific installation instructions).
*   **Mac App Store Version:** For a more refined and feature-rich experience, consider the native [Buzz Captions](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200) on the Mac App Store.

**[Documentation](https://chidiwilliams.github.io/buzz/)** | **[Buzz Captions on the Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)**

![MIT License](https://img.shields.io/badge/license-MIT-green)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

> Buzz is better on the App Store. Get a Mac-native version of Buzz with a cleaner look, audio playback, drag-and-drop import, transcript editing, search, and much more.
>
> <a href="https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&amp;itsct=apps_box_badge&amp;itscg=30200"><img src="https://toolbox.marketingtools.apple.com/api/badges/download-on-the-mac-app-store/black/en-us?size=250x83&amp;releaseDate=1679529600" alt="Download on the Mac App Store" /></a>

![Buzz](./buzz/assets/buzz-banner.jpg)

## Installation

Choose your preferred installation method:

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
2.  Install Buzz:

    ```shell
    pip install buzz-captions
    python -m buzz
    ```

### macOS

*   Using [Homebrew](https://brew.sh/):

    ```shell
    brew install --cask buzz
    ```

*   Or download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).
    *   *Note: The app is not signed, so you may receive a warning during installation. Select "More info" -> "Run anyway."*

*   **Alternatively, install with [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/):**

    ```shell
    winget install ChidiWilliams.Buzz
    ```

    **GPU Support for PyPI (Windows):**

    To enable GPU support for Nvidia GPUs on the PyPI-installed version, ensure CUDA support for [PyTorch](https://pytorch.org/get-started/locally/) and run the following:

    ```
    pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
    ```

### Linux

Buzz is available as a [Flatpak](https://flathub.org/apps/io.github.chidiwilliams.Buzz) or a [Snap](https://snapcraft.io/buzz).

*   **Flatpak:**

    ```shell
    flatpak install flathub io.github.chidiwilliams.Buzz
    ```

*   **Snap:**

    ```shell
    sudo apt-get install libportaudio2 libcanberra-gtk-module libcanberra-gtk3-module
    sudo snap install buzz
    sudo snap connect buzz:password-manager-service
    ```

### Latest Development Version

For access to the latest features and bug fixes, refer to the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

### Screenshots

<div style="display: flex; flex-wrap: wrap;">
    <img alt="File import" src="share/screenshots/buzz-1-import.png" style="max-width: 18%; margin-right: 1%;" />
    <img alt="Main screen" src="share/screenshots/buzz-2-main_screen.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Preferences" src="share/screenshots/buzz-3-preferences.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Model preferences" src="share/screenshots/buzz-3.2-model-preferences.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Transcript" src="share/screenshots/buzz-4-transcript.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Live recording" src="share/screenshots/buzz-5-live_recording.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Resize" src="share/screenshots/buzz-6-resize.png" style="max-width: 18%;" />
</div>