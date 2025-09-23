# Buzz: Offline Audio Transcription and Translation

**Effortlessly transcribe and translate audio on your computer with Buzz, powered by the cutting-edge OpenAI Whisper technology.**  For more features and a native Mac experience, check out [Buzz Captions on the Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200).

[View the original repository on GitHub](https://github.com/chidiwilliams/buzz)

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chidiwilliams/buzz)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

![Buzz](./buzz/assets/buzz-banner.jpg)

## Key Features

*   **Offline Transcription:** Transcribe audio files locally on your computer, ensuring privacy and speed.
*   **Offline Translation:** Translate audio into multiple languages without relying on an internet connection.
*   **Powered by Whisper:** Leverages the powerful and accurate OpenAI Whisper model.
*   **Cross-Platform Compatibility:** Available for macOS, Windows, and Linux.
*   **Multiple Installation Options:** Install via PyPI, brew, winget, Flatpak, or Snap.
*   **GPU Support:**  Utilize your NVIDIA GPU for accelerated transcription (PyPI version).

## Installation

Choose your preferred installation method:

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html).
2.  Install Buzz:

    ```shell
    pip install buzz-captions
    python -m buzz
    ```

    **GPU Support (NVIDIA):** For improved performance on Windows, install the CUDA dependencies:

    ```
    pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
    ```

### macOS

*   **Homebrew:**
    ```shell
    brew install --cask buzz
    ```
*   **Direct Download:** Download the `.dmg` file from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Direct Download:** Download and run the `.exe` file from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  You may need to select `More info` -> `Run anyway` due to the app not being signed.
*   **Winget:**
    ```shell
    winget install ChidiWilliams.Buzz
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

## Latest Development Version

For information on the latest features and bug fixes in the development version, see the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

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