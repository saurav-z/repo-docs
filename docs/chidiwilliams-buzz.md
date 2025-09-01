# Buzz: Offline Audio Transcription and Translation Powered by Whisper

**Effortlessly transcribe and translate audio on your computer with Buzz, utilizing the power of OpenAI's Whisper.**  [View the original repository on GitHub](https://github.com/chidiwilliams/buzz).

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chidiwilliams/buzz)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

**Key Features:**

*   **Offline Transcription and Translation:** Process audio files directly on your computer, ensuring privacy and speed.
*   **Powered by OpenAI Whisper:** Leverages the advanced speech-to-text capabilities of Whisper for accurate results.
*   **Cross-Platform Support:** Available for macOS, Windows, and Linux, offering flexibility for all users.
*   **Multiple Installation Options:** Install via PyPI, Brew, Winget, Flatpak, and Snap for ease of use.
*   **GPU Support:**  Utilize GPU acceleration for faster processing on compatible systems.
*   **App Store Version:** For a native Mac experience, check out the [Buzz Captions app on the Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200).

**Why Choose Buzz?**

*   **Privacy:** Transcribe and translate your audio without sending data to the cloud.
*   **Accuracy:** Benefit from the state-of-the-art Whisper model.
*   **Convenience:** Simple installation and user-friendly interface.
*   **Speed:** Benefit from GPU processing on compatible systems.

**Get the Mac-native version of Buzz:**
<blockquote>
<p>Buzz is better on the App Store. Get a Mac-native version of Buzz with a cleaner look, audio playback, drag-and-drop import, transcript editing, search, and much more.</p>
<a href="https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&amp;itsct=apps_box_badge&amp;itscg=30200"><img src="https://toolbox.marketingtools.apple.com/api/badges/download-on-the-mac-app-store/black/en-us?size=250x83&amp;releaseDate=1679529600" alt="Download on the Mac App Store" /></a>
</blockquote>

![Buzz](./buzz/assets/buzz-banner.jpg)

## Installation

### PyPI

1.  Install [ffmpeg](https://www.ffmpeg.org/download.html)
2.  Install Buzz:
    ```shell
    pip install buzz-captions
    python -m buzz
    ```

### macOS

*   Using [brew utility](https://brew.sh/)
    ```shell
    brew install --cask buzz
    ```
*   Download the `.dmg` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   Download and run the `.exe` from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).
    *   The app is not signed.  You may get a warning during installation. Select `More info` -> `Run anyway`.
*   **Alternatively, install with [winget](https://learn.microsoft.com/en-us/windows/package-manager/winget/)**
    ```shell
    winget install ChidiWilliams.Buzz
    ```

**GPU support for PyPI**

To have GPU support for Nvidia GPUS on Windows, for PyPI installed version ensure, CUDA support for [torch](https://pytorch.org/get-started/locally/) 

```
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

### Latest Development Version

For info on how to get latest development version with latest features and bug fixes see [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

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

*   **Clear Heading:** The main heading now includes relevant keywords like "Offline Audio Transcription" and "Whisper."
*   **Concise Hook:** The one-sentence hook immediately explains what Buzz does and its core benefit.
*   **Bulleted Key Features:** Uses bullet points for easy readability and highlights key benefits.
*   **Keyword Optimization:** Uses relevant keywords like "transcribe," "translate," "offline," "audio," and "Whisper" naturally throughout the description.
*   **Platform Emphasis:** Specifically mentions macOS, Windows, and Linux to improve searchability.
*   **Call to Action:** Encourages users to explore the project and the Mac App Store version.
*   **Installation Section:** Updated the installation sections for better readability.
*   **Links:** Provides clear links to the original repository and other resources.
*   **Clean Formatting:**  Uses markdown formatting for readability, which is important for search engines.
*   **Emphasis on Benefits:**  Focuses on the value proposition for users (privacy, accuracy, speed).
*   **Mac App Store:** Highlighted the Mac App Store version to get extra user engagement.