# Buzz: Offline Audio Transcription & Translation with Whisper

**Effortlessly transcribe and translate audio offline on your computer with Buzz, powered by OpenAI's Whisper.**  [Learn more at the original repository](https://github.com/chidiwilliams/buzz).

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chidiwilliams/buzz)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

**Key Features:**

*   **Offline Transcription:** Transcribe audio files directly on your computer, ensuring privacy and speed.
*   **Translation Capabilities:** Translate your transcriptions into other languages.
*   **Powered by Whisper:** Utilizes the robust and accurate OpenAI Whisper model.
*   **Cross-Platform:** Available for macOS, Windows, and Linux.
*   **Multiple Installation Options:** Install via PyPI, brew, winget, Flatpak, or Snap.

>  **Get the Mac-native version of Buzz on the App Store for an even better experience with a cleaner look, audio playback, drag-and-drop import, transcript editing, search, and more!**
>  <a href="https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&amp;itsct=apps_box_badge&amp;itscg=30200"><img src="https://toolbox.marketingtools.apple.com/api/badges/download-on-the-mac-app-store/black/en-us?size=250x83&amp;releaseDate=1679529600" alt="Download on the Mac App Store" /></a>

![Buzz](./buzz/assets/buzz-banner.jpg)

## Installation

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
2.  Install Buzz:
    ```bash
    pip install buzz-captions
    python -m buzz
    ```

    **GPU Support (for PyPI on Windows):**
    To enable GPU support for Nvidia GPUs, ensure CUDA support for [torch](https://pytorch.org/get-started/locally/) is installed, then install the following packages:
    ```bash
    pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
    ```

### macOS

*   Install using [brew utility](https://brew.sh/):

    ```bash
    brew install --cask buzz
    ```

*   Alternatively, download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  You may encounter a warning; select `More info` -> `Run anyway`.

*   **Alternatively, install with [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/)**

    ```bash
    winget install ChidiWilliams.Buzz
    ```

### Linux

*   **Flatpak:**

    ```bash
    flatpak install flathub io.github.chidiwilliams.Buzz
    ```

*   **Snap:**

    ```bash
    sudo apt-get install libportaudio2 libcanberra-gtk-module libcanberra-gtk3-module
    sudo snap install buzz
    sudo snap connect buzz:password-manager-service
    ```

## Development Version

For the latest features and bug fixes, see the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

## Screenshots

<div style="display: flex; flex-wrap: wrap;">
    <img alt="File import" src="share/screenshots/buzz-1-import.png" style="max-width: 18%; margin-right: 1%;" />
    <img alt="Main screen" src="share/screenshots/buzz-2-main_screen.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Preferences" src="share/screenshots/buzz-3-preferences.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Model preferences" src="share/screenshots/buzz-3.2-model-preferences.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Transcript" src="share/screenshots/buzz-4-transcript.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Live recording" src="share/screenshots/buzz-5-live_recording.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Resize" src="share/screenshots/buzz-6-resize.png" style="max-width: 18%;" />
</div>