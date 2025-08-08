# Buzz: Offline Audio Transcription and Translation

**Buzz is a powerful, offline application that uses OpenAI's Whisper to transcribe and translate your audio files directly on your computer.**  ([View the original repo](https://github.com/chidiwilliams/buzz))

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chidiwilliams/buzz)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

**Key Features:**

*   **Offline Transcription:** Transcribe audio without needing an internet connection, ensuring privacy and speed.
*   **Powered by Whisper:** Leverages the advanced speech recognition capabilities of OpenAI's Whisper.
*   **Multi-Platform Support:** Available for macOS, Windows, and Linux.
*   **Easy Installation:** Install with pip, brew, winget, Flatpak, or Snap.
*   **Translation:** Translate your transcriptions into multiple languages.

**Get the Mac-native version with even more features on the App Store!**

>   Buzz is better on the App Store. Get a Mac-native version of Buzz with a cleaner look, audio playback, drag-and-drop import, transcript editing, search, and much more.
    <a href="https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&amp;itsct=apps_box_badge&amp;itscg=30200"><img src="https://toolbox.marketingtools.apple.com/api/badges/download-on-the-mac-app-store/black/en-us?size=250x83&amp;releaseDate=1679529600" alt="Download on the Mac App Store" /></a>

![Buzz](./buzz/assets/buzz-banner.jpg)

## Installation Guide

Choose your operating system to get started:

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
2.  Install Buzz:

    ```shell
    pip install buzz-captions
    python -m buzz
    ```

    **GPU Support (PyPI - Windows):** For Nvidia GPUs, ensure CUDA support for [torch](https://pytorch.org/get-started/locally/) and run:

    ```bash
    pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
    ```

### macOS

*   Install using [brew utility](https://brew.sh/):

    ```shell
    brew install --cask buzz
    ```
*   Alternatively, download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  You may need to select `More info` -> `Run anyway` due to the app not being signed.
*   **Alternatively, install with [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/)**:

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

For the latest features and bug fixes, see the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version) for information on obtaining the development version.

## Screenshots

<div style="display: flex; flex-wrap: wrap;">
    <img alt="File import" src="share/screenshots/buzz-1-import.png" style="max-width: 18%; margin-right: 1%;" />
    <img alt="Main screen" src="share/screenshots/buzz-2-main_screen.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Preferences" src="share/screenshots/buzz-3-preferences.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Transcript" src="share/screenshots/buzz-4-transcript.png" style="max-width: 18%; margin-right: 1%; height:auto;" />
    <img alt="Live recording" src="share/screenshots/buzz-5-live_recording.png" style="max-width: 18%;" />
</div>
```
Key improvements and SEO considerations:

*   **Clear Headline:**  Uses the target keyword "offline audio transcription" prominently.
*   **Hook:** A strong one-sentence introduction to grab attention.
*   **Keyword Rich:** Includes relevant keywords like "transcription," "translation," "offline," "OpenAI Whisper," and platform names.
*   **Bulleted Features:** Easy to scan and highlights key benefits.
*   **Installation Section Improved:** Organized installation instructions for different platforms, making it user-friendly.
*   **SEO-Friendly Structure:** Uses headings (H2) to break up the content and improve readability.
*   **Call to Action:** Encourages users to explore the Mac App Store version.
*   **Internal Links:** Added links to the original repository, documentation and FAQ.
*   **Concise Language:** The text is more focused and avoids unnecessary wordiness.
*   **Contextual Alt Text:** Descriptive alt text for images.
*   **Mobile Responsiveness:** Using `style="max-width: 18%; margin-right: 1%; height:auto;"` for the images in the screenshots ensures a responsive layout across different screen sizes.