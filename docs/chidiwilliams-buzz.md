# Buzz: Offline Audio Transcription & Translation Powered by Whisper

**Effortlessly transcribe and translate audio offline on your computer with Buzz, leveraging the power of OpenAI's Whisper.**  ([Original Repo](https://github.com/chidiwilliams/buzz))

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chidiwilliams/buzz)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

[Documentation](https://chidiwilliams.github.io/buzz/) | [Buzz Captions on the App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)

**Key Features:**

*   **Offline Transcription & Translation:** Process audio files locally without relying on internet connectivity.
*   **Powered by OpenAI Whisper:** Utilizes the cutting-edge AI technology for accurate transcriptions.
*   **Cross-Platform Compatibility:** Available for macOS, Windows, and Linux.
*   **Multiple Installation Options:** Choose from PyPI, macOS (Brew, DMG), Windows (EXE, Winget), and Linux (Flatpak, Snap).
*   **Open Source:** Licensed under the MIT license.
*   **GPU Acceleration:** Supports GPU acceleration for faster processing (Nvidia GPUs on Windows).
*   **Regular Updates:** Stay up-to-date with the latest features and bug fixes by checking the FAQ for development versions.

> **For a richer experience, check out the Mac-native version of Buzz on the App Store, featuring a cleaner interface, audio playback, transcript editing, and more!**
> <a href="https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&amp;itsct=apps_box_badge&amp;itscg=30200"><img src="https://toolbox.marketingtools.apple.com/api/badges/download-on-the-mac-app-store/black/en-us?size=250x83&amp;releaseDate=1679529600" alt="Download on the Mac App Store" /></a>

![Buzz](./buzz/assets/buzz-banner.jpg)

## Installation

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html).
2.  Install Buzz:

    ```shell
    pip install buzz-captions
    python -m buzz
    ```

    *   **GPU Support for PyPI (Windows with Nvidia GPUs):**  Ensure CUDA support for [torch](https://pytorch.org/get-started/locally/) and run the following:

    ```
    pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
    ```

### macOS

*   **Homebrew:**

    ```shell
    brew install --cask buzz
    ```
*   **DMG:** Download the `.dmg` file from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **EXE:** Download and run the `.exe` file from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  You may need to select `More info` -> `Run anyway` if you get a warning.
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

### Latest Development Version

For information on accessing the latest development version with the newest features and bug fixes, please refer to the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

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
```
Key improvements and SEO considerations:

*   **Clear, Concise Hook:**  The one-sentence hook immediately grabs attention and highlights the core functionality.
*   **SEO Keywords:** Included relevant keywords like "offline transcription," "audio translation," and "Whisper" in the headings and body.
*   **Well-Structured Headings:** Organized information logically with clear headings and subheadings.
*   **Bulleted Key Features:**  Makes it easy to scan and understand the main benefits.
*   **Links:**  Added links to the documentation, app store, and original repo.
*   **Concise Installation Instructions:** Simplified installation steps for each platform.
*   **Screenshots:** Added the screenshots section for better understanding.
*   **Focus on Benefits:**  Emphasized *what* Buzz does and *why* it's useful.
*   **Call to action:** Added a call to action to try the Mac App Store version.