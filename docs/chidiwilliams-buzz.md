# Buzz: Offline Audio Transcription and Translation Powered by OpenAI Whisper

**Effortlessly transcribe and translate audio offline on your computer using Buzz, a powerful tool leveraging the cutting-edge OpenAI Whisper model.** ([Original Repo](https://github.com/chidiwilliams/buzz))

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/chidiwilliams/buzz)
[![CI](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml/badge.svg)](https://github.com/chidiwilliams/buzz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/chidiwilliams/buzz/branch/main/graph/badge.svg?token=YJSB8S2VEP)](https://codecov.io/github/chidiwilliams/buzz)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/chidiwilliams/buzz)
[![Github all releases](https://img.shields.io/github/downloads/chidiwilliams/buzz/total.svg)](https://GitHub.com/chidiwilliams/buzz/releases/)

<br>

Get the **Mac-native** version with enhanced features on the [Mac App Store](https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&itsct=apps_box_badge&itscg=30200):
<a href="https://apps.apple.com/us/app/buzz-captions/id6446018936?mt=12&amp;itsct=apps_box_badge&amp;itscg=30200"><img src="https://toolbox.marketingtools.apple.com/api/badges/download-on-the-mac-app-store/black/en-us?size=250x83&amp;releaseDate=1679529600" alt="Download on the Mac App Store" /></a>

<br>

![Buzz](./buzz/assets/buzz-banner.jpg)

## Key Features

*   **Offline Transcription & Translation:** Transcribe and translate audio directly on your device without internet dependency.
*   **Powered by OpenAI Whisper:** Utilize the advanced capabilities of OpenAI's Whisper for highly accurate results.
*   **Cross-Platform Support:** Available for macOS, Windows, and Linux, with flexible installation options.
*   **Multiple Installation Methods:** Install via PyPI, brew (macOS), winget (Windows), Flatpak & Snap (Linux), or direct downloads.
*   **GPU Acceleration (Optional):** Leverage your GPU for faster processing (available for PyPI installation on compatible systems).
*   **Mac App Store Version:** Experience a feature-rich, native macOS application with audio playback, editing, and more.
*   **Regular Updates:** Benefit from ongoing improvements and the latest features.

## Installation Guide

### PyPI

1.  **Install ffmpeg:**  Download from [ffmpeg](https://www.ffmpeg.org/download.html)
2.  **Install Buzz:**
    ```bash
    pip install buzz-captions
    python -m buzz
    ```

### macOS

*   **Using Homebrew:**
    ```bash
    brew install --cask buzz
    ```
*   **Download from Releases:** Download the `.dmg` file from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest).

### Windows

*   **Download from Releases:** Download the `.exe` file from the [releases page](https://github.com/chidiwilliams/buzz/releases/latest). *Note: You may need to bypass a security warning.*
*   **Using winget:**
    ```bash
    winget install ChidiWilliams.Buzz
    ```

    **Enable GPU Support (PyPI only):**
    To enable GPU support for Nvidia GPUs when using the PyPI installation on Windows, ensure you have CUDA support for [torch](https://pytorch.org/get-started/locally/) installed:

    ```
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

## Resources

*   **Documentation:** [Documentation](https://chidiwilliams.github.io/buzz/)
*   **FAQ:** For information on obtaining the latest development version, see the [FAQ](https://chidiwilliams.github.io/buzz/docs/faq#9-where-can-i-get-latest-development-version).

## Screenshots

*(Screenshots have been maintained)*

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
Key improvements:

*   **SEO-Optimized Title & Description:** Uses keywords like "audio transcription," "offline," and "OpenAI Whisper" to improve search visibility.
*   **Clear Headings:** Organized content for readability and easy navigation.
*   **Bulleted Key Features:** Highlights the most important aspects of the software.
*   **One-Sentence Hook:** Grabs the reader's attention immediately.
*   **Mac App Store Emphasis:**  Highlights the enhanced features of the Mac version.
*   **Concise Installation Instructions:**  Provides clear, step-by-step instructions for all supported platforms.
*   **Clear GPU instructions:**  Includes GPU installation steps as a key feature and provides clear steps.
*   **Links & Resources:**  Includes important links for further information and support.
*   **Updated formatting and styling:**  Added formatting to make the information more visually appealing.