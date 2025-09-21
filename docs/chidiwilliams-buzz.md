# Buzz: Offline Audio Transcription & Translation Powered by Whisper

**Effortlessly transcribe and translate audio files offline with Buzz, a powerful tool leveraging the cutting-edge OpenAI Whisper technology.**  [Learn more at the original repository](https://github.com/chidiwilliams/buzz).

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chidiwilliams/buzz)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

> **Get a Mac-native experience with Buzz Captions on the App Store!** Enjoy a cleaner interface, audio playback, drag-and-drop functionality, transcript editing, search, and more.
> <a href="https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&amp;itsct=apps_box_badge&amp;itscg=30200"><img src="https://toolbox.marketingtools.apple.com/api/badges/download-on-the-mac-app-store/black/en-us?size=250x83&amp;releaseDate=1679529600" alt="Download on the Mac App Store" /></a>

![Buzz](./buzz/assets/buzz-banner.jpg)

## Key Features

*   **Offline Transcription & Translation:** Process audio files directly on your computer, ensuring privacy and speed.
*   **Powered by OpenAI Whisper:** Utilizes state-of-the-art speech recognition technology.
*   **Cross-Platform Compatibility:** Available for macOS, Windows, and Linux.
*   **Multiple Installation Options:** Install via PyPI, Homebrew, Winget, Flatpak, or Snap.
*   **GPU Support (for PyPI):**  Leverage the power of your NVIDIA GPU for faster processing (with proper setup).
*   **Mac App Store Version:**  Benefit from a native Mac app with enhanced features and a streamlined user experience (Buzz Captions).

## Installation

### Prerequisites

*   [ffmpeg](https://www.ffmpeg.org/download.html) must be installed.

### macOS

*   **Homebrew:**

    ```shell
    brew install --cask buzz
    ```

*   **Direct Download:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Direct Download:** Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  *(Note:  You may need to bypass a security warning.)*
*   **Winget:**

    ```shell
    winget install ChidiWilliams.Buzz
    ```

*   **GPU Support (PyPI only):**  Ensure CUDA support for [torch](https://pytorch.org/get-started/locally/) and run the following:

    ```shell
    pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
    ```

### Linux

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

### PyPI

```shell
pip install buzz-captions
python -m buzz
```

### Latest Development Version

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