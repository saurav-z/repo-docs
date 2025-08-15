# Buzz: Offline Audio Transcription & Translation

**Unlock the power of instant, offline audio transcription and translation with Buzz, powered by OpenAI's Whisper.** Find the original repository [here](https://github.com/chidiwilliams/buzz).

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chidiwilliams/buzz)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

![Buzz](./buzz/assets/buzz-banner.jpg)

## Key Features

*   **Offline Transcription:** Transcribe audio files without an internet connection, ensuring privacy and speed.
*   **Multilingual Support:** Translate audio into multiple languages.
*   **Powered by Whisper:** Utilizes the robust and accurate OpenAI Whisper model.
*   **Cross-Platform:** Available for macOS, Windows, and Linux.
*   **Multiple Installation Options:** Install via PyPI, macOS with brew, Windows, and Linux packages (Flatpak, Snap).
*   **GPU Acceleration:** (For PyPI install on Windows) Utilize GPU acceleration for faster processing (requires CUDA setup).
*   **User-Friendly:** (For App Store version) A Mac-native version available with a cleaner look, audio playback, drag-and-drop import, transcript editing, and search.

>   **Experience the best of Buzz on the Mac App Store:** Get a Mac-native version of Buzz with a cleaner look, audio playback, drag-and-drop import, transcript editing, search, and much more.
>
>   <a href="https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&amp;itsct=apps_box_badge&amp;itscg=30200"><img src="https://toolbox.marketingtools.apple.com/api/badges/download-on-the-mac-app-store/black/en-us?size=250x83&amp;releaseDate=1679529600" alt="Download on the Mac App Store" /></a>

## Installation

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
2.  Install Buzz
    ```shell
    pip install buzz-captions
    python -m buzz
    ```
    **GPU Support (Windows):** For GPU support for Nvidia GPUs on Windows (PyPI install), ensure CUDA support for [torch](https://pytorch.org/get-started/locally/) and run the following commands:
    ```bash
    pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
    ```

### macOS

*   **Homebrew:**
    ```shell
    brew install --cask buzz
    ```
*   **Direct Download:** Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Direct Download:** Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  You may need to select `More info` -> `Run anyway` due to the app not being signed.
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

For information on obtaining the latest development version with the newest features and bug fixes, see the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

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
```
Key improvements and SEO considerations:

*   **Clear Title & Hook:** The title directly incorporates the primary keywords ("Audio Transcription", "Offline", "Translation", "Buzz"). The first sentence immediately establishes the core function and benefit.
*   **Keyword Integration:**  The text now uses relevant keywords naturally throughout (e.g., "offline audio transcription", "translate audio," "OpenAI's Whisper," "cross-platform").
*   **Well-Structured Headings & Subheadings:** Organizes the information logically for readability and SEO.
*   **Bulleted Key Features:**  Highlights the main selling points, making them easy to scan.
*   **Platform-Specific Information:** Keeps the installation instructions clear and organized.
*   **Strong Call to Action (App Store Link):** Encourages users to explore the enhanced Mac app.
*   **Clear GPU Instructions:** Provides clear guidance for GPU setup on Windows.
*   **Concise Language:** The text is trimmed down for better readability.
*   **Alt Text for Images:** The screenshots include `alt` text, improving SEO.
*   **Link Back to Repo:** Added at the beginning to improve SEO and User experience.