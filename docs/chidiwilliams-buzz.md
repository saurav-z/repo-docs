# Buzz: Offline Audio Transcription and Translation

**Buzz is a powerful, offline audio transcription and translation tool, leveraging the capabilities of OpenAI's Whisper, and available across multiple platforms.**  [Visit the original repository](https://github.com/chidiwilliams/buzz).

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chidiwilliams/buzz)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

*   **[Documentation](https://chidiwilliams.github.io/buzz/)**
*   **[Buzz Captions on the Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200)**

## Key Features

*   **Offline Transcription:** Transcribe audio files directly on your computer without relying on internet connectivity.
*   **Offline Translation:** Translate your audio transcriptions offline, supporting various languages.
*   **Powered by Whisper:** Utilizes the robust and accurate OpenAI Whisper model for high-quality transcription and translation.
*   **Cross-Platform Support:** Available for macOS, Windows, and Linux.
*   **Mac App Store Version:**  Consider the Mac-native version for improved features and user experience (audio playback, editing, search, etc.) - see link above.

## Installation

Choose your operating system:

### macOS

*   **Homebrew:**
    ```bash
    brew install --cask buzz
    ```
*   **Direct Download:** Download the `.dmg` file from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Direct Download:** Download the `.exe` file from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).  *Note: You may need to select "More info" -> "Run anyway" if the app is not signed.*
*   **Winget:**
    ```bash
    winget install ChidiWilliams.Buzz
    ```
*   **GPU Support for PyPI:** For the PyPI installed version, you may enable GPU support for Nvidia GPUs with the following commands. Be sure to have CUDA support for [torch](https://pytorch.org/get-started/locally/) installed first:
    ```bash
    pip3 install -U torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    pip3 install nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57 nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57 nvidia-cudnn-cu12==9.7.1.26 nvidia-cufft-cu12==11.3.3.41 nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.6.3 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55 --extra-index-url https://pypi.ngc.nvidia.com
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

### PyPI

*   Install [ffmpeg](https://www.ffmpeg.org/download.html)
*   Install Buzz:
    ```bash
    pip install buzz-captions
    python -m buzz
    ```
## Latest Development Version

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
```
Key improvements and SEO considerations:

*   **Clear Title and Hook:**  The title is more descriptive and includes keywords. The hook immediately tells users what Buzz does.
*   **Keyword Optimization:** Includes keywords like "offline audio transcription," "offline translation," "OpenAI Whisper," "macOS," "Windows," and "Linux" to improve searchability.
*   **Structured Content:** Uses headings, subheadings, and bullet points for readability and SEO benefits.  This makes it easy for search engines to understand the content.
*   **Platform-Specific Sections:** Clearly separates installation instructions for each platform (macOS, Windows, Linux), which is crucial for user experience.
*   **Mac App Store Callout:**  Highlights the benefits of the Mac App Store version to drive users there.
*   **Concise Language:** Uses clear and direct language, avoiding unnecessary jargon.
*   **Call to Action:**  The "Visit the original repository" link encourages users to explore the project further.
*   **Links and Badges Maintained:** All links to documentation, App Store, and the original repo are maintained.
*   **GPU Installation Information:** Included the necessary installation instructions for Nvidia GPU support.
*   **Complete Code Examples:** Ensure users have functional code examples.